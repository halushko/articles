from __future__ import annotations

import io
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool

from .archive import InputValidationError, documents_from_directory, documents_from_zip
from .control_process import (
    ControlProcessModelBuilder,
    ProcessModelSummary,
    ProcessModelUnavailableError,
)
from .database import create_database_engine, create_session_factory
from .deterministic_process import DeterministicProcessModelBuilder
from .documentation_graph import (
    DocumentationGraphBuilder,
    DocumentationGraphSummary,
)
from .exporter import build_result_zip
from .llm_process import (
    LLMConfigurationError,
    LLMProcessModelBuilder,
    LLMProviderError,
    LLMUnavailableError,
    OpenAICompatibleClient,
    ProcessExtractionError,
)
from .pipeline import SegmentationPipeline, get_run_summary
from .settings import Settings

PACKAGE_DIR = Path(__file__).resolve().parent
WITHOUT_LLM_MESSAGE = "Без LLM"


def _graph_response(summary: DocumentationGraphSummary) -> dict[str, object]:
    return {
        "graph_id": summary.id,
        "run_id": summary.run_id,
        "node_count": summary.node_count,
        "edge_count": summary.edge_count,
        "cache_hit": summary.cache_hit,
        "graph": summary.payload,
    }


def _process_model_response(
    summary: ProcessModelSummary,
    *,
    llm_status: str | None = None,
) -> dict[str, object]:
    response: dict[str, object] = {
        "process_model_id": summary.id,
        "run_id": summary.run_id,
        "level_count": summary.level_count,
        "atomic_node_count": summary.atomic_node_count,
        "cache_hit": summary.cache_hit,
        "process_model": summary.payload,
    }
    if llm_status:
        response["llm_status"] = llm_status
    return response


def create_app(
    *,
    settings: Settings | None = None,
    session_factory: sessionmaker[Session] | None = None,
    process_model_builder: Any | None = None,
) -> FastAPI:
    app_settings = settings or Settings.from_env()
    builder_was_injected = process_model_builder is not None
    if session_factory is None:
        engine = create_database_engine(app_settings.database_url)
        session_factory = create_session_factory(engine)

    deterministic_builder = DeterministicProcessModelBuilder(session_factory)
    llm_builder: Any | None = None
    if process_model_builder is None:
        if app_settings.process_model_mode == "control":
            process_model_builder = ControlProcessModelBuilder(
                session_factory,
                app_settings.control_process_path,
            )
        elif app_settings.process_model_mode in {"auto", "llm"} and (
            app_settings.llm_configured
        ):
            llm_builder = LLMProcessModelBuilder(
                session_factory,
                OpenAICompatibleClient(
                    base_url=app_settings.llm_base_url,
                    api_key=app_settings.llm_api_key,
                    model=app_settings.llm_model,
                    timeout_seconds=app_settings.llm_timeout_seconds,
                    max_output_tokens=app_settings.llm_max_output_tokens,
                    response_format=app_settings.llm_response_format,
                    reasoning_effort=app_settings.llm_reasoning_effort,
                ),
                max_input_chars=app_settings.llm_max_input_chars,
            )
            process_model_builder = llm_builder
        elif app_settings.process_model_mode in {"auto", "llm", "deterministic"}:
            process_model_builder = deterministic_builder
        else:
            raise ValueError(
                "PROCESS_MODEL_MODE must be 'auto', 'llm', 'deterministic' or 'control'"
            )
    elif app_settings.process_model_mode in {"auto", "llm"}:
        llm_builder = process_model_builder

    app = FastAPI(
        title="Documentation to Process Hierarchy",
        version="0.4.0",
        description=(
            "Segment Markdown documentation, inspect its source structure, "
            "and build a traceable hierarchical process model with or without an LLM."
        ),
    )
    app.state.settings = app_settings
    app.state.session_factory = session_factory
    app.state.process_model_builder = process_model_builder
    app.state.deterministic_process_model_builder = deterministic_builder
    app.state.llm_process_model_builder = llm_builder
    app.state.llm_disabled_reason = (
        WITHOUT_LLM_MESSAGE
        if app_settings.process_model_mode in {"auto", "llm", "deterministic"}
        and (
            app_settings.process_model_mode == "deterministic"
            or not app_settings.llm_enabled
            or (not builder_was_injected and not app_settings.llm_configured)
        )
        else None
    )

    def process_model_capability(source_type: str) -> tuple[bool, str | None]:
        if app_settings.process_model_mode == "control" and source_type != "example":
            return (
                False,
                "Control mode supports only the built-in D1-D5 corpus.",
            )
        return True, None

    def process_model_strategy() -> str:
        if app_settings.process_model_mode == "control":
            return "control_baseline"
        if llm_builder is not None:
            return "llm_with_deterministic_fallback"
        return "deterministic_rules"

    templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")
    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        example_count = len(list(app_settings.example_docs_dir.glob("D*.md")))
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "example_count": example_count,
                "max_archive_mb": app_settings.max_archive_bytes // (1024 * 1024),
            },
        )

    @app.post("/api/v1/runs")
    async def create_run(
        source: Annotated[str, Form()],
        archive: Annotated[UploadFile | None, File()] = None,
    ) -> JSONResponse:
        try:
            if source == "example":
                if archive is not None and archive.filename:
                    raise InputValidationError(
                        "Do not upload an archive when the built-in example is selected"
                    )
                documents = documents_from_directory(
                    app_settings.example_docs_dir,
                    max_documents=app_settings.max_documents,
                )
            elif source == "upload":
                if archive is None or not archive.filename:
                    raise InputValidationError("Select a ZIP archive to process")
                raw_archive = await archive.read(app_settings.max_archive_bytes + 1)
                documents = documents_from_zip(
                    raw_archive,
                    max_archive_bytes=app_settings.max_archive_bytes,
                    max_uncompressed_bytes=app_settings.max_uncompressed_bytes,
                    max_documents=app_settings.max_documents,
                )
            else:
                raise InputValidationError(
                    "source must be either 'example' or 'upload'"
                )

            summary = await run_in_threadpool(
                SegmentationPipeline(session_factory).process,
                documents,
                source_type=source,
            )
        except InputValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        process_available, process_reason = process_model_capability(
            summary.source_type
        )
        return JSONResponse(
            {
                "run_id": summary.run_id,
                "source_type": summary.source_type,
                "status": summary.status,
                "corpus_sha256": summary.corpus_sha256,
                "document_count": summary.document_count,
                "fragment_count": summary.fragment_count,
                "cache_hits": summary.cache_hits,
                "result_url": f"/api/v1/runs/{summary.run_id}/result",
                "documentation_graph_url": (
                    f"/api/v1/runs/{summary.run_id}/documentation-graph"
                ),
                "process_model_available": process_available,
                "process_model_unavailable_reason": process_reason,
                "process_model_strategy": process_model_strategy(),
                "llm_status": app.state.llm_disabled_reason,
                "process_model_url": f"/api/v1/runs/{summary.run_id}/process-model",
            },
            status_code=201,
        )

    @app.get("/api/v1/runs/{run_id}")
    async def get_run(run_id: str) -> JSONResponse:
        with session_factory() as session:
            summary = get_run_summary(session, run_id)
        if summary is None:
            raise HTTPException(status_code=404, detail="Segmentation run not found")
        process_available, process_reason = process_model_capability(
            summary.source_type
        )
        return JSONResponse(
            {
                "run_id": summary.run_id,
                "source_type": summary.source_type,
                "status": summary.status,
                "corpus_sha256": summary.corpus_sha256,
                "document_count": summary.document_count,
                "fragment_count": summary.fragment_count,
                "cache_hits": summary.cache_hits,
                "result_url": f"/api/v1/runs/{summary.run_id}/result",
                "documentation_graph_url": (
                    f"/api/v1/runs/{summary.run_id}/documentation-graph"
                ),
                "process_model_available": process_available,
                "process_model_unavailable_reason": process_reason,
                "process_model_strategy": process_model_strategy(),
                "llm_status": app.state.llm_disabled_reason,
                "process_model_url": f"/api/v1/runs/{summary.run_id}/process-model",
            }
        )

    @app.post("/api/v1/runs/{run_id}/documentation-graph")
    async def build_documentation_graph(run_id: str) -> JSONResponse:
        try:
            summary = await run_in_threadpool(
                DocumentationGraphBuilder(session_factory).build,
                run_id,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Segmentation run not found"
            ) from exc

        return JSONResponse(
            _graph_response(summary),
            status_code=200 if summary.cache_hit else 201,
        )

    @app.get("/api/v1/runs/{run_id}/documentation-graph")
    async def get_documentation_graph(run_id: str) -> JSONResponse:
        summary = await run_in_threadpool(
            DocumentationGraphBuilder(session_factory).get,
            run_id,
        )
        if summary is None:
            raise HTTPException(
                status_code=404,
                detail="Documentation graph has not been built for this run",
            )
        return JSONResponse(_graph_response(summary))

    @app.post("/api/v1/runs/{run_id}/process-model")
    async def build_process_model(run_id: str) -> JSONResponse:
        llm_status = app.state.llm_disabled_reason
        try:
            use_llm = (
                llm_builder is not None
                and process_model_builder is llm_builder
                and app.state.llm_disabled_reason is None
            )
            if use_llm:
                try:
                    summary = await run_in_threadpool(llm_builder.build, run_id)
                except LLMUnavailableError:
                    app.state.llm_disabled_reason = WITHOUT_LLM_MESSAGE
                    llm_status = WITHOUT_LLM_MESSAGE
                    summary = await run_in_threadpool(
                        deterministic_builder.build,
                        run_id,
                    )
                except (
                    LLMConfigurationError,
                    LLMProviderError,
                    ProcessExtractionError,
                ):
                    llm_status = WITHOUT_LLM_MESSAGE
                    summary = await run_in_threadpool(
                        deterministic_builder.build,
                        run_id,
                    )
            else:
                active_builder = (
                    deterministic_builder
                    if llm_builder is not None
                    and app.state.llm_disabled_reason is not None
                    else process_model_builder
                )
                summary = await run_in_threadpool(active_builder.build, run_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Segmentation run not found"
            ) from exc
        except ProcessModelUnavailableError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LLMConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LLMProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ProcessExtractionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return JSONResponse(
            _process_model_response(summary, llm_status=llm_status),
            status_code=200 if summary.cache_hit else 201,
        )

    @app.get("/api/v1/runs/{run_id}/process-model")
    async def get_process_model(run_id: str) -> JSONResponse:
        summary = await run_in_threadpool(
            DeterministicProcessModelBuilder.latest,
            session_factory,
            run_id,
        )
        if summary is None:
            raise HTTPException(
                status_code=404,
                detail="Process model has not been built for this run",
            )
        return JSONResponse(
            _process_model_response(
                summary,
                llm_status=app.state.llm_disabled_reason,
            )
        )

    @app.get("/api/v1/runs/{run_id}/result")
    async def download_result(run_id: str) -> StreamingResponse:
        def make_result() -> bytes:
            with session_factory() as session:
                return build_result_zip(session, run_id)

        try:
            result = await run_in_threadpool(make_result)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Segmentation run not found"
            ) from exc

        return StreamingResponse(
            io.BytesIO(result),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="segmentation-{run_id}.zip"'
            },
        )

    @app.get("/health/live")
    async def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def readiness() -> JSONResponse:
        try:
            with session_factory() as session:
                session.execute(text("SELECT 1"))
        except SQLAlchemyError:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ok"})

    return app


app = create_app()

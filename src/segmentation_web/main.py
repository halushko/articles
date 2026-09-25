from __future__ import annotations

import io
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool

from .archive import InputValidationError, documents_from_directory, documents_from_zip
from .database import create_database_engine, create_session_factory
from .documentation_graph import (
    DocumentationGraphBuilder,
    DocumentationGraphSummary,
)
from .exporter import build_result_zip
from .pipeline import SegmentationPipeline, get_run_summary
from .settings import Settings

PACKAGE_DIR = Path(__file__).resolve().parent


def _graph_response(summary: DocumentationGraphSummary) -> dict[str, object]:
    return {
        "graph_id": summary.id,
        "run_id": summary.run_id,
        "node_count": summary.node_count,
        "edge_count": summary.edge_count,
        "cache_hit": summary.cache_hit,
        "graph": summary.payload,
    }


def create_app(
    *,
    settings: Settings | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> FastAPI:
    app_settings = settings or Settings.from_env()
    if session_factory is None:
        engine = create_database_engine(app_settings.database_url)
        session_factory = create_session_factory(engine)

    app = FastAPI(
        title="Documentation Fragmentation MVP",
        version="0.1.0",
        description="Segment Markdown documentation into versioned fragment cards.",
    )
    app.state.settings = app_settings
    app.state.session_factory = session_factory

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

        return JSONResponse(
            {
                "run_id": summary.run_id,
                "status": summary.status,
                "corpus_sha256": summary.corpus_sha256,
                "document_count": summary.document_count,
                "fragment_count": summary.fragment_count,
                "cache_hits": summary.cache_hits,
                "result_url": f"/api/v1/runs/{summary.run_id}/result",
                "documentation_graph_url": (
                    f"/api/v1/runs/{summary.run_id}/documentation-graph"
                ),
            },
            status_code=201,
        )

    @app.get("/api/v1/runs/{run_id}")
    async def get_run(run_id: str) -> JSONResponse:
        with session_factory() as session:
            summary = get_run_summary(session, run_id)
        if summary is None:
            raise HTTPException(status_code=404, detail="Segmentation run not found")
        return JSONResponse(
            {
                "run_id": summary.run_id,
                "status": summary.status,
                "corpus_sha256": summary.corpus_sha256,
                "document_count": summary.document_count,
                "fragment_count": summary.fragment_count,
                "cache_hits": summary.cache_hits,
                "result_url": f"/api/v1/runs/{summary.run_id}/result",
                "documentation_graph_url": (
                    f"/api/v1/runs/{summary.run_id}/documentation-graph"
                ),
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

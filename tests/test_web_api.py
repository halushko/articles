import asyncio
import io
import zipfile
from pathlib import Path

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from segmentation_web.database import Base
from segmentation_web.main import create_app
from segmentation_web.settings import Settings


def test_runtime_does_not_embed_curated_corpus_identifiers():
    package = Path("src/segmentation_web")
    runtime_source = "\n".join(
        file.read_text(encoding="utf-8")
        for suffix in ("*.py", "*.js", "*.html")
        for file in package.rglob(suffix)
    )

    assert "D1_access_recovery_support_policy.md" not in runtime_source
    assert "home_internet_order_fulfilment_guide.md" not in runtime_source
    assert "Access Recovery Support Policy" not in runtime_source
    assert "Home Internet Order Fulfilment Guide" not in runtime_source


def make_client(tmp_path):
    example_dir = tmp_path / "examples"
    example_dir.mkdir()
    (example_dir / "D1_policy.md").write_text(
        "# Policy\n\nReceive the report and then register the incident.\n",
        encoding="utf-8",
    )
    (example_dir / "D2_runbook.md").write_text(
        "# Runbook\n\nIf the account is locked, unlock it.\n",
        encoding="utf-8",
    )

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    settings = Settings(
        database_url="sqlite://",
        example_docs_dir=example_dir,
        max_archive_bytes=1024 * 1024,
        max_uncompressed_bytes=2 * 1024 * 1024,
        max_documents=10,
    )
    return create_app(settings=settings, session_factory=sessions)


def test_builtin_example_run_and_result_download(tmp_path):
    app = make_client(tmp_path)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post("/api/v1/runs", data={"source": "example"})
            assert response.status_code == 201
            body = response.json()
            assert body["status"] == "completed"
            assert body["document_count"] == 2
            assert body["fragment_count"] > 0
            assert body["process_model_available"] is True
            assert body["process_model_unavailable_reason"] is None
            assert body["process_model_strategy"] == "deterministic_rules"
            assert body["llm_status"] == "Без LLM"

            process = await client.post(body["process_model_url"])
            assert process.status_code == 201
            assert process.json()["llm_status"] == "Без LLM"
            assert process.json()["process_model"]["derivation"]["mode"] == (
                "deterministic_rules"
            )

            status = await client.get(f"/api/v1/runs/{body['run_id']}")
            assert status.status_code == 200
            assert status.json()["corpus_sha256"] == body["corpus_sha256"]

            graph = await client.post(body["documentation_graph_url"])
            assert graph.status_code == 201
            graph_body = graph.json()
            assert graph_body["node_count"] > body["document_count"]
            assert graph_body["graph"]["semantic_analysis"]["status"] == ("not_started")

            cached_graph = await client.post(body["documentation_graph_url"])
            assert cached_graph.status_code == 200
            assert cached_graph.json()["cache_hit"] is True

            result = await client.get(body["result_url"])
            assert result.status_code == 200
            assert result.headers["content-type"] == "application/zip"
            with zipfile.ZipFile(io.BytesIO(result.content)) as archive:
                assert "manifest.json" in archive.namelist()
                assert "documentation_graph.json" in archive.namelist()
                assert "process_model_l0.json" in archive.namelist()
                assert "provenance.json" in archive.namelist()

    asyncio.run(scenario())


def test_second_builtin_example_uses_the_same_api_and_builder(tmp_path):
    base_app = make_client(tmp_path)
    app = create_app(
        settings=Settings(
            database_url="sqlite://",
            example_docs_dir=Path("examples/access_recovery_source_docs"),
            example_catalog_dir=Path("examples"),
            process_model_mode="deterministic",
            max_archive_bytes=1024 * 1024,
            max_uncompressed_bytes=2 * 1024 * 1024,
            max_documents=20,
        ),
        session_factory=base_app.state.session_factory,
    )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            index = await client.get("/")
            assert index.status_code == 200
            assert "Corporate access recovery" in index.text
            assert "Home internet connection" in index.text

            response = await client.post(
                "/api/v1/runs",
                data={
                    "source": "example",
                    "example_id": "home-internet-connection",
                },
            )
            assert response.status_code == 201
            body = response.json()
            assert body["document_count"] == 8
            assert body["process_model_strategy"] == "deterministic_rules"

            process = await client.post(body["process_model_url"])
            assert process.status_code == 201
            process_body = process.json()["process_model"]
            assert process_body["process_title"] == (
                "Home Internet Order Fulfilment Guide"
            )
            assert process_body["analysis"]["procedural_document_count"] == 7

    asyncio.run(scenario())


def test_upload_requires_a_zip_archive(tmp_path):
    app = make_client(tmp_path)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post("/api/v1/runs", data={"source": "upload"})
            assert response.status_code == 400
            assert response.json()["detail"] == "Select a ZIP archive to process"

    asyncio.run(scenario())


def test_uploaded_zip_is_segmented(tmp_path):
    app = make_client(tmp_path)
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("policy.md", "# Policy\n\nRegister the incident.\n")
        archive.writestr("runbook.md", "# Runbook\n\nCheck the account status.\n")

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/runs",
                data={"source": "upload"},
                files={
                    "archive": (
                        "documentation.zip",
                        archive_buffer.getvalue(),
                        "application/zip",
                    )
                },
            )
            assert response.status_code == 201
            assert response.json()["document_count"] == 2

    asyncio.run(scenario())


def test_unknown_run_returns_not_found(tmp_path):
    app = make_client(tmp_path)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/api/v1/runs/not-a-real-id")
            assert response.status_code == 404

    asyncio.run(scenario())


def test_llm_flag_disables_an_injected_builder(tmp_path):
    class UnexpectedBuilder:
        def build(self, run_id):
            raise AssertionError("The disabled LLM builder must not be called")

    app = make_client(tmp_path)
    app.state.settings = Settings(
        database_url="sqlite://",
        example_docs_dir=tmp_path / "examples",
        process_model_mode="llm",
        llm_enabled=False,
    )

    sessions = app.state.session_factory
    disabled_app = create_app(
        settings=app.state.settings,
        session_factory=sessions,
        process_model_builder=UnexpectedBuilder(),
    )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=disabled_app),
            base_url="http://testserver",
        ) as client:
            response = await client.post("/api/v1/runs", data={"source": "example"})
            assert response.status_code == 201
            body = response.json()
            assert body["process_model_available"] is True
            assert body["process_model_unavailable_reason"] is None
            assert body["llm_status"] == "Без LLM"

            process = await client.post(body["process_model_url"])
            assert process.status_code == 201
            assert process.json()["llm_status"] == "Без LLM"
            assert process.json()["process_model"]["derivation"]["mode"] == (
                "deterministic_rules"
            )

    asyncio.run(scenario())

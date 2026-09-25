import asyncio
import io
import zipfile

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from segmentation_web.database import Base
from segmentation_web.main import create_app
from segmentation_web.settings import Settings


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

            status = await client.get(f"/api/v1/runs/{body['run_id']}")
            assert status.status_code == 200
            assert status.json()["corpus_sha256"] == body["corpus_sha256"]

            result = await client.get(body["result_url"])
            assert result.status_code == 200
            assert result.headers["content-type"] == "application/zip"
            with zipfile.ZipFile(io.BytesIO(result.content)) as archive:
                assert "manifest.json" in archive.namelist()

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

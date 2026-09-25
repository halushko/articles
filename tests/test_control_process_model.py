import asyncio
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from segmentation_web.archive import documents_from_directory
from segmentation_web.control_process import (
    ControlProcessModelBuilder,
    ProcessModelUnavailableError,
)
from segmentation_web.database import Base
from segmentation_web.db_models import ProcessModelResult
from segmentation_web.main import create_app
from segmentation_web.pipeline import SegmentationPipeline
from segmentation_web.settings import Settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DOCS = REPOSITORY_ROOT / "examples" / "access_recovery_source_docs"
PROCESS_DEFINITION = REPOSITORY_ROOT / "examples" / "access_recovery_process.md"


def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def control_run(sessions, *, source_type="example"):
    documents = documents_from_directory(EXAMPLE_DOCS, max_documents=100)
    return SegmentationPipeline(sessions).process(
        documents,
        source_type=source_type,
    )


def test_control_model_builds_three_traceable_levels_and_is_cached():
    sessions = session_factory()
    run = control_run(sessions)
    builder = ControlProcessModelBuilder(sessions, PROCESS_DEFINITION)

    first = builder.build(run.run_id)
    second = builder.build(run.run_id)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.id == first.id
    assert first.payload["derivation"]["mode"] == "control_baseline"
    assert first.payload["derivation"]["universal_extraction"] is False
    assert first.payload["summary"] == [
        {"level": 0, "node_count": 18, "edge_count": 21},
        {"level": 1, "node_count": 5, "edge_count": 5},
        {"level": 2, "node_count": 3, "edge_count": 2},
    ]
    assert set(first.payload["provenance"]) == {f"v{index}" for index in range(1, 19)}
    assert all(first.payload["provenance"].values())

    diagnostic = first.payload["hierarchy"]["levels"][1]["rejected_candidates"]
    assert len(diagnostic) == 1
    assert diagnostic[0]["candidate_id"] == "Cx"
    assert "partially covered" in diagnostic[0]["rejection_reason"]

    with sessions() as session:
        count = session.scalar(select(func.count()).select_from(ProcessModelResult))
    assert count == 1


def test_control_model_is_not_claimed_for_an_uploaded_corpus():
    sessions = session_factory()
    run = control_run(sessions, source_type="upload")

    with pytest.raises(ProcessModelUnavailableError, match="LLM extraction"):
        ControlProcessModelBuilder(sessions, PROCESS_DEFINITION).build(run.run_id)


def test_process_model_api_and_result_zip():
    sessions = session_factory()
    settings = Settings(
        database_url="sqlite://",
        example_docs_dir=EXAMPLE_DOCS,
        control_process_path=PROCESS_DEFINITION,
        process_model_mode="control",
        max_archive_bytes=1024 * 1024,
        max_uncompressed_bytes=2 * 1024 * 1024,
        max_documents=10,
    )
    app = create_app(settings=settings, session_factory=sessions)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            run_response = await client.post(
                "/api/v1/runs",
                data={"source": "example"},
            )
            assert run_response.status_code == 201
            run_body = run_response.json()
            assert run_body["process_model_available"] is True

            model_response = await client.post(run_body["process_model_url"])
            assert model_response.status_code == 201
            model_body = model_response.json()
            assert model_body["level_count"] == 3
            assert model_body["atomic_node_count"] == 18

            cached_response = await client.post(run_body["process_model_url"])
            assert cached_response.status_code == 200
            assert cached_response.json()["cache_hit"] is True

            result = await client.get(run_body["result_url"])
            assert result.status_code == 200
            with zipfile.ZipFile(io.BytesIO(result.content)) as archive:
                assert "process_model_hierarchy.json" in archive.namelist()
                manifest = json.loads(archive.read("manifest.json"))
                hierarchy = json.loads(archive.read("process_model_hierarchy.json"))
            assert manifest["process_model"]["derivation_mode"] == ("control_baseline")
            assert hierarchy["summary"][2]["node_count"] == 3

    asyncio.run(scenario())

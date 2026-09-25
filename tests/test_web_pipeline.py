import io
import json
import zipfile

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from segmentation_web.archive import SourceDocument
from segmentation_web.database import Base
from segmentation_web.db_models import (
    DocumentVersion,
    SegmentationResult,
    SegmentationRun,
)
from segmentation_web.exporter import build_result_zip
from segmentation_web.pipeline import SegmentationPipeline


def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_pipeline_versions_documents_and_reuses_cached_results():
    sessions = session_factory()
    documents = [
        SourceDocument(
            "guide.md", b"# Guide\n\nReceive the request and then register it.\n"
        ),
        SourceDocument(
            "runbook.md", b"# Runbook\n\nIf the account is locked, unlock it.\n"
        ),
    ]

    first = SegmentationPipeline(sessions).process(documents, source_type="upload")
    second = SegmentationPipeline(sessions).process(documents, source_type="upload")

    assert first.status == "completed"
    assert first.cache_hits == 0
    assert second.cache_hits == 2
    assert first.corpus_sha256 == second.corpus_sha256

    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(DocumentVersion)) == 2
        assert session.scalar(select(func.count()).select_from(SegmentationResult)) == 2
        assert session.scalar(select(func.count()).select_from(SegmentationRun)) == 2


def test_result_zip_contains_manifest_per_document_results_and_combined_output():
    sessions = session_factory()
    summary = SegmentationPipeline(sessions).process(
        [SourceDocument("handbook/guide.md", b"# Guide\n\nRegister the incident.\n")],
        source_type="upload",
    )

    with sessions() as session:
        payload = build_result_zip(session, summary.run_id)

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        assert "manifest.json" in names
        assert "all_fragments.json" in names
        assert "errors.json" in names
        result_name = next(name for name in names if name.startswith("documents/"))
        result = json.loads(archive.read(result_name))
        manifest = json.loads(archive.read("manifest.json"))

    assert result["source"]["path"] == "handbook/guide.md"
    assert len(result["source"]["sha256"]) == 64
    assert manifest["run"]["id"] == summary.run_id
    assert manifest["documents"][0]["result_file"] == result_name

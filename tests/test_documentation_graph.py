import io
import json
import zipfile

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from segmentation_web.archive import SourceDocument
from segmentation_web.database import Base
from segmentation_web.db_models import DocumentationGraphResult
from segmentation_web.documentation_graph import DocumentationGraphBuilder
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


def segmented_run(sessions):
    documents = [
        SourceDocument(
            "D1_policy.md",
            (
                b"# Access Policy\n\n"
                b"## Handling\n\n"
                b"Register the request. Continue with the Recovery Runbook.\n"
            ),
        ),
        SourceDocument(
            "D2_runbook.md",
            (
                b"# Recovery Runbook\n\n"
                b"## Restore access\n\n"
                b"Check the account and then unlock it.\n"
            ),
        ),
    ]
    return SegmentationPipeline(sessions).process(documents, source_type="upload")


def test_graph_contains_documents_sections_leaf_fragments_and_explicit_references():
    sessions = session_factory()
    run = segmented_run(sessions)

    result = DocumentationGraphBuilder(sessions).build(run.run_id)
    nodes = result.payload["nodes"]
    edges = result.payload["edges"]

    assert result.cache_hit is False
    assert result.payload["semantic_analysis"]["status"] == "not_started"
    assert {node["label"] for node in nodes if node["type"] == "document"} == {
        "Access Policy",
        "Recovery Runbook",
    }
    assert {node["label"] for node in nodes if node["type"] == "section"} == {
        "Handling",
        "Restore access",
    }
    assert any(node["type"] == "fragment" for node in nodes)
    assert {edge["type"] for edge in edges} >= {
        "contains",
        "document_order",
        "references",
    }

    reference = next(edge for edge in edges if edge["type"] == "references")
    assert reference["confidence"] == 1.0
    assert reference["evidence"][0]["fragment_id"]
    assert "Recovery Runbook" in reference["evidence"][0]["quote"]


def test_graph_build_is_cached_by_run_builder_version_and_configuration():
    sessions = session_factory()
    run = segmented_run(sessions)
    builder = DocumentationGraphBuilder(sessions)

    first = builder.build(run.run_id)
    second = builder.build(run.run_id)

    assert second.cache_hit is True
    assert second.id == first.id
    with sessions() as session:
        count = session.scalar(
            select(func.count()).select_from(DocumentationGraphResult)
        )
    assert count == 1


def test_result_zip_includes_graph_only_after_it_has_been_built():
    sessions = session_factory()
    run = segmented_run(sessions)

    with sessions() as session:
        before = build_result_zip(session, run.run_id)
    with zipfile.ZipFile(io.BytesIO(before)) as archive:
        assert "documentation_graph.json" not in archive.namelist()

    DocumentationGraphBuilder(sessions).build(run.run_id)
    with sessions() as session:
        after = build_result_zip(session, run.run_id)
    with zipfile.ZipFile(io.BytesIO(after)) as archive:
        assert "documentation_graph.json" in archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
        graph = json.loads(archive.read("documentation_graph.json"))

    assert manifest["documentation_graph"]["result_file"] == (
        "documentation_graph.json"
    )
    assert graph["source_run"]["id"] == run.run_id

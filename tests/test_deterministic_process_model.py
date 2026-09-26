import asyncio
import io
import zipfile
from pathlib import Path

import httpx
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from segmentation_web.archive import SourceDocument
from segmentation_web.database import Base
from segmentation_web.db_models import ProcessModelResult
from segmentation_web.deterministic_process import DeterministicProcessModelBuilder
from segmentation_web.main import create_app
from segmentation_web.pipeline import SegmentationPipeline
from segmentation_web.settings import Settings


def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def invoice_documents():
    return [
        SourceDocument(
            "operations/invoice_intake.md",
            b"""# Supplier invoice handling

## Intake

1. The accounts payable clerk records the invoice in ERP.
2. The clerk validates the purchase order in ERP.
3. Then the clerk checks the invoice total.

## Decision

If the totals match, approve the invoice in ERP.
If the totals do not match, send the invoice to the procurement analyst.

## Handoff

- Continue with Payment execution.
""",
        ),
        SourceDocument(
            "finance/payment_execution.md",
            b"""# Payment execution

## Payment run

1. The finance manager reviews approved invoices in ERP.
2. The manager authorizes the payment in ERP.
3. The treasury system sends the bank transfer through the banking API.
""",
        ),
    ]


def test_unrelated_uploaded_corpus_builds_grounded_deterministic_hierarchy():
    sessions = session_factory()
    run = SegmentationPipeline(sessions).process(
        invoice_documents(),
        source_type="upload",
    )
    builder = DeterministicProcessModelBuilder(sessions)

    first = builder.build(run.run_id)
    cached = builder.build(run.run_id)

    assert first.cache_hit is False
    assert cached.cache_hit is True
    assert cached.id == first.id
    assert first.payload["derivation"]["mode"] == "deterministic_rules"
    assert first.payload["derivation"]["llm_used"] is False
    assert first.payload["derivation"]["universal_extraction"] is True
    assert first.atomic_node_count >= 10
    assert first.level_count >= 2

    graph = first.payload["hierarchy"]["base_graph"]
    operations = {node["operation"] for node in graph["nodes"]}
    assert "The accounts payable clerk records the invoice in ERP" in operations
    assert "The finance manager reviews approved invoices in ERP" in operations
    assert any(node["operation"].startswith("Decision:") for node in graph["nodes"])
    assert any(edge["edge_type"] == "conditional" for edge in graph["edges"])
    assert any(
        "target document" in item["reason"]
        for item in first.payload["transition_provenance"].values()
    )
    assert all(first.payload["provenance"].values())
    assert all(
        item["evidence"] for item in first.payload["transition_provenance"].values()
    )
    assert all(
        "document order" not in item["reason"].casefold()
        for item in first.payload["transition_provenance"].values()
    )

    with sessions() as session:
        count = session.scalar(select(func.count()).select_from(ProcessModelResult))
    assert count == 1


def test_plain_sentence_adjacency_is_only_a_very_low_confidence_hypothesis():
    sessions = session_factory()
    run = SegmentationPipeline(sessions).process(
        [
            SourceDocument(
                "policy.md",
                b"""# Retention policy

The records officer reviews retention requests. The legal adviser checks policy compliance.
""",
            )
        ],
        source_type="upload",
    )

    model = DeterministicProcessModelBuilder(sessions).build(run.run_id)
    graph = model.payload["hierarchy"]["base_graph"]

    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1
    provenance = model.payload["transition_provenance"][graph["edges"][0]["id"]]
    assert provenance["confidence"] == 0.35
    assert provenance["reason"].startswith("Low-confidence hypothesis:")


def test_separate_action_blocks_create_reviewable_low_confidence_hypothesis():
    sessions = session_factory()
    run = SegmentationPipeline(sessions).process(
        [
            SourceDocument(
                "procedure.md",
                b"""# Account closure

## Procedure

The account manager reviews the closure request.

The operations specialist archives the account record.
""",
            )
        ],
        source_type="upload",
    )

    model = DeterministicProcessModelBuilder(sessions).build(run.run_id)
    graph = model.payload["hierarchy"]["base_graph"]

    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1
    provenance = model.payload["transition_provenance"][graph["edges"][0]["id"]]
    assert provenance["confidence"] == 0.55
    assert provenance["reason"].startswith("Low-confidence hypothesis:")
    assert model.payload["analysis"]["inferred_transition_count"] == 1
    assert any("low-confidence sequence" in item for item in model.payload["warnings"])


def test_numbered_steps_create_edges_with_source_evidence():
    sessions = session_factory()
    run = SegmentationPipeline(sessions).process(
        [
            SourceDocument(
                "release.md",
                b"""# Release procedure

1. The release engineer creates a deployment tag.
2. The engineer deploys the image to staging.
3. The engineer verifies the health endpoint.
""",
            )
        ],
        source_type="upload",
    )

    model = DeterministicProcessModelBuilder(sessions).build(run.run_id)
    graph = model.payload["hierarchy"]["base_graph"]

    assert len(graph["nodes"]) == 3
    assert len(graph["edges"]) == 2
    for edge in graph["edges"]:
        provenance = model.payload["transition_provenance"][edge["id"]]
        assert provenance["confidence"] == 0.90
        assert len(provenance["evidence"]) == 2


def test_uploaded_unrelated_corpus_works_end_to_end_without_llm():
    sessions = session_factory()
    settings = Settings(
        database_url="sqlite://",
        example_docs_dir=Path("examples/access_recovery_source_docs"),
        process_model_mode="auto",
        llm_api_key="",
        max_archive_bytes=1024 * 1024,
        max_uncompressed_bytes=2 * 1024 * 1024,
        max_documents=10,
    )
    app = create_app(settings=settings, session_factory=sessions)
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for document in invoice_documents():
            archive.writestr(document.relative_path, document.content)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            run_response = await client.post(
                "/api/v1/runs",
                data={"source": "upload"},
                files={
                    "archive": (
                        "invoice-documentation.zip",
                        archive_buffer.getvalue(),
                        "application/zip",
                    )
                },
            )
            assert run_response.status_code == 201
            run_body = run_response.json()
            assert run_body["process_model_available"] is True
            assert run_body["process_model_strategy"] == "deterministic_rules"
            assert run_body["llm_status"] == "Без LLM"

            process_response = await client.post(run_body["process_model_url"])
            assert process_response.status_code == 201
            process_body = process_response.json()
            assert process_body["atomic_node_count"] >= 10
            assert process_body["process_model"]["derivation"]["mode"] == (
                "deterministic_rules"
            )

            cached_response = await client.get(run_body["process_model_url"])
            assert cached_response.status_code == 200
            assert (
                cached_response.json()["process_model_id"]
                == (process_body["process_model_id"])
            )

            result = await client.get(run_body["result_url"])
            with zipfile.ZipFile(io.BytesIO(result.content)) as result_archive:
                names = set(result_archive.namelist())
            assert {
                "process_model_hierarchy.json",
                "process_model_l0.json",
                "aggregation_results.json",
                "provenance.json",
                "process_model_warnings.json",
            } <= names

    asyncio.run(scenario())

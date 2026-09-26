import asyncio
import io
import zipfile
from pathlib import Path

import httpx
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from segmentation_web.archive import (
    SourceDocument,
    discover_example_corpora,
    documents_from_directory,
)
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


def access_recovery_documents():
    return [
        SourceDocument(path.name, path.read_bytes())
        for path in sorted(Path("examples/access_recovery_source_docs").glob("D*.md"))
    ]


def home_internet_documents():
    corpus = next(
        corpus
        for corpus in discover_example_corpora(
            catalog_dir=Path("examples"),
            fallback_dir=Path("examples/access_recovery_source_docs"),
            max_documents=20,
        )
        if corpus.id == "home-internet-connection"
    )
    return documents_from_directory(
        corpus.directory,
        max_documents=20,
        document_paths=corpus.document_paths,
    )


def test_home_internet_corpus_uses_the_generic_deterministic_pipeline():
    sessions = session_factory()
    run = SegmentationPipeline(sessions).process(
        home_internet_documents(),
        source_type="example",
    )

    result = DeterministicProcessModelBuilder(sessions).build(run.run_id)
    payload = result.payload
    graph = payload["hierarchy"]["base_graph"]
    metadata = payload["node_metadata"]
    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    outgoing = {
        node_id: [edge for edge in graph["edges"] if edge["source"] == node_id]
        for node_id in nodes_by_id
    }

    assert payload["process_title"] == "Home Internet Order Fulfilment Guide"
    assert payload["derivation"]["mode"] == "deterministic_rules"
    assert payload["derivation"]["universal_extraction"] is True
    assert payload["analysis"]["procedural_document_count"] == 7
    assert payload["analysis"]["supporting_document_count"] == 1
    assert len(payload["process_structure"]["section_calls"]) == 6
    assert len(payload["summary"]) == 3
    assert all(level["node_count"] > 0 for level in payload["summary"])
    assert all(payload["provenance"].values())
    assert {
        evidence["document_path"]
        for evidence_items in payload["provenance"].values()
        for evidence in evidence_items
    } <= {
        document.relative_path for document in home_internet_documents()
    }

    gateways = [
        node
        for node in graph["nodes"]
        if metadata[node["id"]]["node_type"] == "gateway"
    ]
    assert len(gateways) >= 4
    assert all(len(outgoing[node["id"]]) >= 2 for node in gateways)
    assert not any(
        "fewer than two outgoing" in warning for warning in payload["warnings"]
    )

    reachable = {graph["nodes"][0]["id"]}
    while True:
        expanded = reachable | {
            endpoint
            for edge in graph["edges"]
            if edge["source"] in reachable or edge["target"] in reachable
            for endpoint in (edge["source"], edge["target"])
        }
        if expanded == reachable:
            break
        reachable = expanded
    assert reachable == set(nodes_by_id)


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


def test_named_section_does_not_silently_drop_actions_after_the_fourth():
    sessions = session_factory()
    run = SegmentationPipeline(sessions).process(
        [
            SourceDocument(
                "onboarding_procedure.md",
                b"""# Onboarding procedure

## Provision the account

1. Record the approved request.
2. Create the user account.
3. Assign the standard role.
4. Enable multifactor authentication.
5. Send the activation notice.
6. Archive the approval evidence.
""",
            )
        ],
        source_type="upload",
    )

    payload = DeterministicProcessModelBuilder(sessions).build(run.run_id).payload
    graph = payload["hierarchy"]["base_graph"]

    assert len(graph["nodes"]) == 6
    assert len(graph["edges"]) == 5
    assert graph["nodes"][-1]["operation"] == "Archive the approval evidence"


def test_access_recovery_corpus_builds_a_split_join_process_and_readable_l2():
    sessions = session_factory()
    run = SegmentationPipeline(sessions).process(
        access_recovery_documents(),
        source_type="example",
    )

    model = DeterministicProcessModelBuilder(sessions).build(run.run_id)
    payload = model.payload
    base = payload["hierarchy"]["base_graph"]

    classifications = {
        document["path"]: document["classification"]
        for document in payload["process_structure"]["documents"]
    }
    assert classifications["D1_access_recovery_support_policy.md"] == "supporting"
    assert all(
        classifications[path] == "procedural"
        for path in classifications
        if path.startswith(("D2_", "D3_", "D4_", "D5_"))
    )
    assert all(
        evidence["document_path"] != "D1_access_recovery_support_policy.md"
        for evidence_items in payload["provenance"].values()
        for evidence in evidence_items
    )

    outgoing = {node["id"]: [] for node in base["nodes"]}
    incoming = {node["id"]: [] for node in base["nodes"]}
    for edge in base["edges"]:
        outgoing[edge["source"]].append(edge["target"])
        incoming[edge["target"]].append(edge["source"])
    remaining = {node["id"] for node in base["nodes"]}
    visited = {remaining.pop()}
    while True:
        connected = {
            neighbor
            for node_id in visited
            for neighbor in outgoing[node_id] + incoming[node_id]
            if neighbor not in visited
        }
        if not connected:
            break
        visited.update(connected)
        remaining.difference_update(connected)
    assert not remaining
    assert any(len(targets) >= 2 for targets in outgoing.values())
    assert any(len(sources) >= 2 for sources in incoming.values())
    assert any(
        edge["edge_type"] == "conditional"
        and "account or authentication" in (edge["condition"] or "")
        for edge in base["edges"]
    )
    assert any(
        edge["edge_type"] == "conditional"
        and "service or application" in (edge["condition"] or "")
        for edge in base["edges"]
    )
    route_gateway_id = next(
        edge["source"]
        for edge in base["edges"]
        if "account or authentication" in (edge["condition"] or "")
    )
    route_edges = [
        edge for edge in base["edges"] if edge["source"] == route_gateway_id
    ]
    called_document_by_condition = {
        edge["condition"]: payload["provenance"][edge["target"]][0]["document_path"]
        for edge in route_edges
    }
    assert called_document_by_condition[
        "the evidence points to an account or authentication problem"
    ] == "D3_identity_and_authentication_recovery_runbook.md"
    assert called_document_by_condition[
        "the evidence points to a service or application problem"
    ] == "D4_software_platform_recovery_runbook.md"

    outcome_gateway = next(
        node
        for node in base["nodes"]
        if node["operation"] == "Decision: Incident outcome?"
    )
    outcome_edges = [
        edge for edge in base["edges"] if edge["source"] == outcome_gateway["id"]
    ]
    assert len(outcome_edges) == 2
    assert {edge["condition"] for edge in outcome_edges} == {
        "Recovered",
        "Unresolved",
    }
    outcome_targets = {
        edge["condition"]: next(
            node for node in base["nodes"] if node["id"] == edge["target"]
        )["operation"]
        for edge in outcome_edges
    }
    assert outcome_targets["Unresolved"].startswith("Create an escalation")
    assert outcome_targets["Recovered"].startswith("Tell the user")
    for edge in outcome_edges:
        evidence = payload["transition_provenance"][edge["id"]]["evidence"]
        assert any(
            f"as {edge['condition'].casefold()} when" in item["quote"].casefold()
            for item in evidence
        )

    assert [item["level"] for item in payload["summary"]] == [0, 1, 2]
    assert payload["summary"][1]["node_count"] == 5
    assert payload["summary"][1]["edge_count"] == 5
    assert payload["summary"][-1]["node_count"] == 3
    l1 = payload["hierarchy"]["levels"][0]
    l1_names = [node["operation"] for node in l1["graph"]["nodes"]]
    assert all(not name.startswith("Stage:") for name in l1_names)
    assert any("Identity and Authentication" in name for name in l1_names)
    assert any("Software Platform" in name for name in l1_names)
    assert all(
        candidate["selection_basis"] == "explicit_source_structure"
        for candidate in l1["accepted_candidates"]
    )
    l2 = payload["hierarchy"]["levels"][-1]
    l2_names = [node["operation"] for node in l2["graph"]["nodes"]]
    assert all(not name.startswith("Stage:") and len(name) <= 90 for name in l2_names)
    assert any("Identity and Authentication" in name for name in l2_names)
    assert any("Close the incident" in name for name in l2_names)
    assert all(
        candidate["candidate_name"]
        for level in payload["hierarchy"]["levels"]
        for candidate in level["accepted_candidates"]
    )


def test_if_then_without_else_has_true_and_implicit_false_paths():
    sessions = session_factory()
    run = SegmentationPipeline(sessions).process(
        [
            SourceDocument(
                "review_procedure.md",
                b"""# Request review procedure

## Decision

If the request is incomplete, return the request to the submitter.

## Record outcome

Record the review result in the case system.
""",
            )
        ],
        source_type="upload",
    )

    payload = DeterministicProcessModelBuilder(sessions).build(run.run_id).payload
    graph = payload["hierarchy"]["base_graph"]
    metadata = payload["node_metadata"]
    gateway = next(
        node for node in graph["nodes"] if metadata[node["id"]]["node_type"] == "gateway"
    )
    outgoing = [edge for edge in graph["edges"] if edge["source"] == gateway["id"]]

    assert len(outgoing) == 2
    assert {edge["condition"] for edge in outgoing} == {
        "the request is incomplete",
        "Otherwise",
    }
    assert not any("fewer than two outgoing" in item for item in payload["warnings"])


def test_if_then_else_creates_two_explicit_branches_and_a_join():
    sessions = session_factory()
    run = SegmentationPipeline(sessions).process(
        [
            SourceDocument(
                "review_procedure.md",
                b"""# Request review procedure

## Decision

If the request is complete then approve the request else return it to the submitter.

## Record outcome

Record the review result in the case system.
""",
            )
        ],
        source_type="upload",
    )

    payload = DeterministicProcessModelBuilder(sessions).build(run.run_id).payload
    graph = payload["hierarchy"]["base_graph"]
    by_id = {node["id"]: node for node in graph["nodes"]}
    metadata = payload["node_metadata"]
    gateway = next(
        node for node in graph["nodes"] if metadata[node["id"]]["node_type"] == "gateway"
    )
    outgoing = [edge for edge in graph["edges"] if edge["source"] == gateway["id"]]

    assert len(outgoing) == 2
    assert {edge["condition"] for edge in outgoing} == {
        "the request is complete",
        "Otherwise",
    }
    assert {
        by_id[edge["target"]]["operation"] for edge in outgoing
    } == {
        "approve the request",
        "return it to the submitter",
    }
    continuation = next(
        node
        for node in graph["nodes"]
        if node["operation"].startswith("Record the review result")
    )
    incoming = [
        edge for edge in graph["edges"] if edge["target"] == continuation["id"]
    ]
    assert {edge["source"] for edge in incoming} == {
        edge["target"] for edge in outgoing
    }
    assert not any("fewer than two outgoing" in item for item in payload["warnings"])


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

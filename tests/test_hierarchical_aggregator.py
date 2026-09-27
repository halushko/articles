from pathlib import Path

import pytest

from process_hierarchy.aggregator import HierarchicalAggregator
from process_hierarchy.candidates import (
    CandidateGenerationLimitError,
    generate_connected_candidates,
)
from process_hierarchy.markdown_graph import MarkdownProcessGraphParser
from process_hierarchy.models import (
    AggregationConfig,
    AggregationLevelConfig,
    CandidateDefinition,
    ProcessEdge,
    ProcessGraph,
    ProcessNode,
    ScoreWeights,
)
from process_hierarchy.report import render_markdown_report


def test_aggregation_produces_a_partition_and_preserves_atomic_members():
    graph = MarkdownProcessGraphParser().parse_file(
        "examples/access_recovery_process.md"
    )
    config = AggregationConfig(
        weights=ScoreWeights(),
        levels=(AggregationLevelConfig(q_min=0.0, max_candidate_nodes=4, radius=3),),
    )

    result = HierarchicalAggregator(config).run(graph)

    assert len(result.levels) == 1
    level = result.levels[0]
    assert set(level.mapping) == set(graph.nodes)

    mapped_members = [
        member_id
        for node in level.graph.nodes.values()
        for member_id in node.member_ids
    ]
    assert sorted(mapped_members) == sorted(graph.nodes)
    assert len(mapped_members) == len(set(mapped_members))

    for edge in level.graph.edges:
        assert edge.source != edge.target
        assert edge.original_edge_ids


def test_control_candidate_set_reproduces_five_and_three_node_levels():
    source = Path("examples/access_recovery_process.md").read_text(encoding="utf-8")
    parser = MarkdownProcessGraphParser()
    graph = parser.parse(source)
    candidates = parser.parse_candidates(source, graph)
    config = AggregationConfig(
        weights=ScoreWeights(),
        levels=(
            AggregationLevelConfig(q_min=0.65, max_candidate_nodes=5, radius=3),
            AggregationLevelConfig(q_min=0.50, max_candidate_nodes=5, radius=3),
        ),
    )

    result = HierarchicalAggregator(config).run(graph, candidates)

    assert [len(level.graph.nodes) for level in result.levels] == [5, 3]
    assert [len(level.graph.edges) for level in result.levels] == [5, 2]
    assert {item.candidate_id for item in result.levels[0].accepted_candidates} == {
        "C1",
        "C2",
        "C3",
        "C4",
    }
    diagnostic = next(
        item
        for item in result.levels[1].rejected_candidates
        if item.candidate_id == "Cx"
    )
    assert diagnostic.q >= 0.5
    assert not diagnostic.branch_integrity

    expected_report = Path("examples/access_recovery_aggregation.md").read_text(
        encoding="utf-8"
    )
    assert render_markdown_report(result) == expected_report


def test_candidate_generation_stops_at_configured_safety_limit():
    nodes = {
        f"v{index}": ProcessNode(
            id=f"v{index}",
            operation=f"Operation {index}",
            role="Role",
            system="System",
        )
        for index in range(1, 7)
    }
    edges = tuple(
        ProcessEdge(id=f"e{index}", source="v1", target=f"v{index}")
        for index in range(2, 7)
    )

    with pytest.raises(CandidateGenerationLimitError, match="limit of 5"):
        generate_connected_candidates(
            ProcessGraph(nodes=nodes, edges=edges),
            max_candidate_nodes=4,
            radius=3,
            max_candidates=5,
        )


def test_explicit_structural_region_is_kept_but_still_receives_a_score():
    graph = ProcessGraph(
        nodes={
            "v1": ProcessNode(
                id="v1", operation="Review request", role="Unknown", system="Unknown"
            ),
            "v2": ProcessNode(
                id="v2", operation="Record outcome", role="Unknown", system="Unknown"
            ),
        },
        edges=(ProcessEdge(id="e1", source="v1", target="v2"),),
    )
    config = AggregationConfig(
        weights=ScoreWeights(),
        levels=(AggregationLevelConfig(q_min=0.99),),
    )
    definition = CandidateDefinition(
        id="S1",
        name="Documented stage",
        target_level=1,
        atomic_node_ids=("v1", "v2"),
        purpose="structural",
    )

    level = HierarchicalAggregator(config).run(graph, (definition,)).levels[0]

    assert len(level.accepted_candidates) == 1
    assert level.accepted_candidates[0].q < 0.99
    assert level.accepted_candidates[0].selection_basis == ("explicit_source_structure")

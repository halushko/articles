from pathlib import Path

from process_hierarchy.aggregator import HierarchicalAggregator
from process_hierarchy.markdown_graph import MarkdownProcessGraphParser
from process_hierarchy.models import (
    AggregationConfig,
    AggregationLevelConfig,
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

import pytest

from process_hierarchy.branching import BranchIntegrityValidator, detect_branch_regions
from process_hierarchy.markdown_graph import MarkdownProcessGraphParser
from process_hierarchy.models import ProcessGraph, ProcessNode, ScoreWeights
from process_hierarchy.scoring import CandidateScorer


@pytest.fixture
def control_graph():
    return MarkdownProcessGraphParser().parse_file(
        "examples/access_recovery_process.md"
    )


def test_control_candidate_components(control_graph):
    scorer = CandidateScorer(control_graph, ScoreWeights())
    score = scorer.score(
        ("v6", "v7", "v8", "v9"),
        ("v6", "v7", "v8", "v9"),
        branch_integrity=True,
    )

    assert 0 <= score.s_txt <= 1
    assert score.s_ctx == pytest.approx(0.875)
    assert score.s_flow == pytest.approx(0.8)
    assert score.q == pytest.approx(
        0.2 * score.s_txt + 0.4 * score.s_ctx + 0.4 * score.s_flow
    )


def test_missing_context_does_not_create_false_similarity():
    graph = ProcessGraph(
        nodes={
            "v1": ProcessNode(
                id="v1", operation="Review request", role="Unknown", system="Unknown"
            ),
            "v2": ProcessNode(
                id="v2", operation="Approve request", role="Unknown", system="Unknown"
            ),
        },
        edges=(),
    )

    score = CandidateScorer(graph, ScoreWeights()).score(
        ("v1", "v2"),
        ("v1", "v2"),
        branch_integrity=True,
    )

    assert score.s_ctx == 0.0


def test_partially_known_context_is_weighted_by_coverage():
    graph = ProcessGraph(
        nodes={
            "v1": ProcessNode(
                id="v1", operation="Review request", role="Analyst", system="CRM"
            ),
            "v2": ProcessNode(
                id="v2", operation="Approve request", role="Unknown", system="Unknown"
            ),
        },
        edges=(),
    )

    score = CandidateScorer(graph, ScoreWeights()).score(
        ("v1", "v2"),
        ("v1", "v2"),
        branch_integrity=True,
    )

    assert score.s_ctx == 0.5


def test_missing_system_dimension_does_not_count_as_matching_context():
    graph = ProcessGraph(
        nodes={
            "v1": ProcessNode(
                id="v1", operation="Review request", role="Analyst", system="Unknown"
            ),
            "v2": ProcessNode(
                id="v2", operation="Approve request", role="Analyst", system="Unknown"
            ),
        },
        edges=(),
    )

    score = CandidateScorer(graph, ScoreWeights()).score(
        ("v1", "v2"),
        ("v1", "v2"),
        branch_integrity=True,
    )

    assert score.s_ctx == 0.5


def test_split_join_region_is_detected(control_graph):
    regions = detect_branch_regions(control_graph)
    region = next(item for item in regions if item.split == "v5")

    assert region.join == "v14"
    assert frozenset({"v6", "v7", "v8", "v9"}) in region.branches
    assert frozenset({"v10", "v11", "v12", "v13"}) in region.branches


def test_complete_branches_are_valid_but_partial_crossing_is_not(control_graph):
    validator = BranchIntegrityValidator(control_graph)

    valid, _ = validator.validate({"v6", "v7", "v8", "v9"})
    assert valid

    valid, _ = validator.validate({"v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13"})
    assert valid

    valid, reason = validator.validate({"v5", "v6", "v7", "v8", "v9", "v10"})
    assert not valid
    assert "partially covered" in reason

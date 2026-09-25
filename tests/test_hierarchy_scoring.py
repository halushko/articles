import pytest

from process_hierarchy.branching import BranchIntegrityValidator, detect_branch_regions
from process_hierarchy.markdown_graph import MarkdownProcessGraphParser
from process_hierarchy.models import ScoreWeights
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

import pytest

from process_hierarchy.branching import BranchIntegrityValidator, detect_branch_regions
from process_hierarchy.models import ProcessEdge, ProcessGraph, ProcessNode


def process_graph(edges):
    node_ids = {node_id for edge in edges for node_id in edge}
    return ProcessGraph(
        nodes={
            node_id: ProcessNode(
                id=node_id, operation=node_id, role="Operator", system="Workflow"
            )
            for node_id in node_ids
        },
        edges=tuple(
            ProcessEdge(id=f"e{index}", source=source, target=target)
            for index, (source, target) in enumerate(edges)
        ),
    )


@pytest.mark.parametrize("retry_target", ["split", "prepare", "left_first"])
def test_retry_loop_does_not_mix_alternatives(retry_target):
    graph = process_graph(
        [
            ("prepare", "split"),
            ("split", "left_first"),
            ("split", "right_first"),
            ("left_first", "left_last"),
            ("right_first", "right_last"),
            ("left_last", "join"),
            ("right_last", "join"),
            ("left_last", retry_target),
            ("join", "finish"),
        ]
    )

    region = next(item for item in detect_branch_regions(graph) if item.split == "split")

    assert region.join == "join"
    assert set(region.branches) == {
        frozenset({"left_first", "left_last"}),
        frozenset({"right_first", "right_last"}),
    }
    validator = BranchIntegrityValidator(graph)
    assert validator.validate({"left_first", "left_last"}) == (True, None)
    assert validator.validate({"prepare"}) == (True, None)
    assert validator.validate({"left_first", "right_first"})[0] is False
    assert validator.validate({"split", "left_first", "left_last"})[0] is False


def test_join_cannot_be_reached_only_by_reentering_the_split():
    graph = process_graph(
        [
            ("prepare", "split"),
            ("split", "left"),
            ("split", "right"),
            ("left", "join"),
            ("join", "finish"),
            ("right", "prepare"),
            ("external", "join"),
        ]
    )

    assert not any(region.split == "split" for region in detect_branch_regions(graph))


def test_overlapping_paths_are_not_reported_as_independent_alternatives():
    graph = process_graph(
        [
            ("split", "left"),
            ("split", "right"),
            ("left", "join"),
            ("right", "join"),
            ("left", "shared"),
            ("right", "shared"),
            ("shared", "tail"),
            ("tail", "join"),
            ("join", "finish"),
        ]
    )

    assert not any(region.split == "split" for region in detect_branch_regions(graph))

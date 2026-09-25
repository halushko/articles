from pathlib import Path

import pytest

from process_hierarchy.markdown_graph import MarkdownProcessGraphParser


def test_parse_control_process_markdown():
    source = Path("examples/access_recovery_process.md").read_text(encoding="utf-8")
    parser = MarkdownProcessGraphParser()
    graph = parser.parse(source)
    candidates = parser.parse_candidates(source, graph)

    assert graph.level == 0
    assert len(graph.nodes) == 18
    assert len(graph.edges) == 21
    assert graph.nodes["v8"].role == "Identity Analyst"
    assert graph.nodes["v8"].system == "External IdP"
    assert graph.nodes["v14"].source_fragment_ids == ("D1-F06", "D5-F01")
    assert graph.edges[4].source == "v5"
    assert graph.edges[4].target == "v6"
    assert [candidate.id for candidate in candidates] == [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "Cx",
    ]
    assert candidates[-1].purpose == "diagnostic"


def test_parser_rejects_edge_to_unknown_node():
    text = """
| id | operation | role | system |
|---|---|---|---|
| v1 | Start process | User | Portal |

| source | target |
|---|---|
| v1 | missing |
"""

    with pytest.raises(ValueError, match="unknown nodes"):
        MarkdownProcessGraphParser().parse(text)

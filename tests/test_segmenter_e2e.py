from pathlib import Path

from fragment_segmenter.segmenter import Segmenter


def test_segmenter_end_to_end_checkout_example():
    text = Path("examples/checkout.md").read_text(encoding="utf-8")
    fragments = Segmenter(artifact_format="markdown").segment("checkout_doc", text)

    types = [fragment.fragment_type for fragment in fragments]
    texts = [fragment.text for fragment in fragments]

    assert "heading" in types
    assert "user_story" in types
    assert "acceptance_criterion" in types
    assert "api_operation" in types
    assert "api_description_sentence" in types
    assert "process_step" in types
    assert "condition_clause" in types
    assert "conditional_scope" in types

    assert "The system loads available payment methods, e.g. card, PayPal, etc." in texts
    assert "Returns v1.2 response format." in texts

    process_steps = [f for f in fragments if f.fragment_type == "process_step"]
    assert any(f.text == "The user submits the form" for f in process_steps)
    assert any(f.text == "the system validates the input" for f in process_steps)
    assert any(f.text == "Then the order is created" for f in process_steps)
    assert any(f.text == "confirmation email is sent" for f in process_steps)

    superseded = [f for f in fragments if f.status == "superseded"]
    assert superseded


def test_segmenter_preserves_each_markdown_table_row_as_one_fragment():
    text = """# Operations

| Node | Operation | System |
|---|---|---|
| v1 | Receive request | ITSM |
| v2 | Register incident | ITSM |
"""

    fragments = Segmenter(artifact_format="markdown").segment("table_doc", text)
    rows = [fragment for fragment in fragments if fragment.fragment_type == "table_row"]

    assert [row.structural_fields["cells"] for row in rows] == [
        ["Node", "Operation", "System"],
        ["v1", "Receive request", "ITSM"],
        ["v2", "Register incident", "ITSM"],
    ]
    assert rows[0].segmentation_trigger == "table_header"
    assert rows[1].segmentation_trigger == "table_row"
    assert not any("---" in row.text for row in rows)


def test_segmenter_keeps_if_then_else_branches_under_one_condition():
    fragments = Segmenter(artifact_format="markdown").segment(
        "decision_doc",
        """# Review procedure

## Decision

If the request is complete then approve the request else return it to the submitter.
""",
    )

    condition = next(
        item for item in fragments if item.fragment_type == "condition_clause"
    )
    scopes = [
        item for item in fragments if item.fragment_type == "conditional_scope"
    ]

    assert condition.text == "if the request is complete"
    assert [item.text for item in scopes] == [
        "approve the request",
        "return it to the submitter.",
    ]
    assert [item.structural_fields["condition_branch"] for item in scopes] == [
        "then",
        "else",
    ]
    assert len({item.parent_fragment_id for item in scopes}) == 1

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

from fragment_segmenter.process_splitter import hard_process_split, soft_process_split


def test_hard_process_split():
    text = "The user submits the form and then the system validates the input."
    result = hard_process_split(text)

    assert result["parts"] == [
        "The user submits the form",
        "the system validates the input",
    ]
    assert result["triggers"] == ["and then"]


def test_hard_process_split_ignores_initial_gherkin_then():
    text = "Then the order is created and then confirmation email is sent"
    result = hard_process_split(text, ignore_initial_then=True)

    assert result["parts"] == [
        "Then the order is created",
        "confirmation email is sent",
    ]


def test_soft_process_split():
    text = "If payment fails, the system shows an error message."
    result = soft_process_split(text)

    assert result["trigger"] == "if"
    assert result["condition_clause"] == "if payment fails"
    assert result["conditional_scope"] == "the system shows an error message."

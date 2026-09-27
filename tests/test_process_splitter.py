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


def test_hard_process_split_does_not_treat_next_as_an_adjective_boundary():
    assert (
        hard_process_split(
            "Tell the customer what remains unresolved and give the next review time."
        )
        is None
    )
    assert (
        hard_process_split("Return the fault reference and next responsible team.")
        is None
    )
    assert hard_process_split("Create a prerequisite for the next visit.") is None


def test_hard_process_split_accepts_punctuated_next_transition():
    result = hard_process_split(
        "Record the request; next, verify the installation address."
    )

    assert result == {
        "parts": ["Record the request", "verify the installation address"],
        "triggers": ["next"],
    }


def test_soft_process_split():
    text = "If payment fails, the system shows an error message."
    result = soft_process_split(text)

    assert result["trigger"] == "if"
    assert result["condition_clause"] == "if payment fails"
    assert result["conditional_scope"] == "the system shows an error message."


def test_soft_process_split_preserves_if_then_as_one_condition():
    result = soft_process_split(
        "If payment fails then the system shows an error message."
    )

    assert result == {
        "trigger": "if",
        "condition_clause": "if payment fails",
        "conditional_scope": "the system shows an error message.",
    }


def test_soft_process_split_extracts_explicit_else_branch():
    result = soft_process_split(
        "If payment fails, show an error; otherwise, continue to confirmation."
    )

    assert result == {
        "trigger": "if",
        "condition_clause": "if payment fails",
        "conditional_scope": "show an error",
        "else_scope": "continue to confirmation.",
    }


def test_soft_process_split_extracts_unpunctuated_then_else():
    result = soft_process_split(
        "If payment fails then show an error else continue to confirmation."
    )

    assert result == {
        "trigger": "if",
        "condition_clause": "if payment fails",
        "conditional_scope": "show an error",
        "else_scope": "continue to confirmation.",
    }

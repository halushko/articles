from fragment_segmenter.detectors import (
    detect_api_operation,
    detect_acceptance_criterion,
    detect_table_row,
    detect_user_story,
)


def test_user_story_detection():
    text = "As a customer, I want to pay for an order, so that I can complete the checkout."
    result = detect_user_story(text)

    assert result["role"] == "a customer"
    assert result["goal"] == "to pay for an order"
    assert result["benefit"] == "I can complete the checkout"


def test_api_operation_detection():
    text = "POST /api/v1/orders/{orderId}/payments - Creates a payment. Returns v1.2 response format."
    result = detect_api_operation(text)

    assert result["method"] == "POST"
    assert result["path"] == "/api/v1/orders/{orderId}/payments"
    assert result["description"] == "Creates a payment. Returns v1.2 response format."


def test_acceptance_criterion_gherkin():
    result = detect_acceptance_criterion("Then the order is created")

    assert result["criterion_format"] == "gherkin"
    assert result["gherkin_keyword"] == "Then"


def test_markdown_table_row_detection():
    row = detect_table_row("| v1 | Receive user request | ITSM |")
    separator = detect_table_row("|---|:---|---:|")

    assert row == {
        "cells": ["v1", "Receive user request", "ITSM"],
        "is_separator": False,
    }
    assert separator == {
        "cells": ["---", ":---", "---:"],
        "is_separator": True,
    }
    assert detect_table_row("Use A | B in prose") is None

from fragment_segmenter.sentence_splitter import split_sentences


def test_sentence_split_does_not_break_abbreviations():
    text = "The system supports card payments, e.g. Visa, Mastercard, etc. Then the user continues."
    result = split_sentences(text)

    assert result == [
        "The system supports card payments, e.g. Visa, Mastercard, etc.",
        "Then the user continues.",
    ]


def test_sentence_split_does_not_break_api_path():
    text = "Use GET /api/v1.0/orders. The response is returned."
    result = split_sentences(text)

    assert result == [
        "Use GET /api/v1.0/orders.",
        "The response is returned.",
    ]


def test_sentence_split_does_not_break_version():
    text = "Returns v1.2 response format. The request is complete."
    result = split_sentences(text)

    assert result == [
        "Returns v1.2 response format.",
        "The request is complete.",
    ]

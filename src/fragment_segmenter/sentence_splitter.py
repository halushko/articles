from __future__ import annotations

import re


PROTECTED_PATTERNS = [
    r"\be\.g\.",
    r"\bi\.e\.",
    r"\bvs\.",
    r"\bMr\.",
    r"\bMrs\.",
    r"\bMs\.",
    r"\bDr\.",
    r"\bFig\.",
    r"\bv\d+(?:\.\d+)+\b",
    r"\b\d+\.\d+\b",
    r"\b[A-Za-z0-9-]+\.[A-Za-z]{2,}\b",
]


def protect_text(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}
    result = text

    for pattern in PROTECTED_PATTERNS:
        while True:
            match = re.search(pattern, result, flags=re.IGNORECASE)
            if not match:
                break
            original = match.group(0)
            placeholder = f"__PROTECTED_{len(protected)}__"
            protected[placeholder] = original
            result = result[: match.start()] + placeholder + result[match.end() :]

    return result, protected


def restore_text(text: str, protected: dict[str, str]) -> str:
    result = text

    # Nested placeholders may appear when a protected version is located inside
    # an API path, for example /api/v1.0/orders. Restore until stable.
    changed = True
    while changed:
        changed = False
        for placeholder, original in protected.items():
            if placeholder in result:
                result = result.replace(placeholder, original)
                changed = True

    return result


def split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    protected_text, protected = protect_text(stripped)

    # Split after terminal punctuation only when the next sentence probably starts.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", protected_text)

    sentences: list[str] = []
    for part in parts:
        restored = restore_text(part, protected).strip()
        if restored:
            sentences.append(restored)

    return sentences

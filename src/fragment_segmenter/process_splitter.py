from __future__ import annotations

import re

HARD_SPLIT_RE = re.compile(
    r"\b(and then|after that|next|then)\b",
    re.IGNORECASE,
)

SOFT_CONDITION_RE = re.compile(
    r"^\s*(?P<trigger>if|when|unless)\s+"
    r"(?P<condition>.+?)"
    r"(?P<separator>,\s*(?:then\s+)?|\s+then\s+|:\s+)"
    r"(?P<scope>.+)$",
    re.IGNORECASE | re.DOTALL,
)
PUNCTUATED_ELSE_RE = re.compile(
    r"\s*(?:[;,]|\.\s+)\s*(?:else|otherwise)\s*[:,]?\s*",
    re.IGNORECASE,
)
PLAIN_ELSE_RE = re.compile(
    r"\s+(?:else|otherwise)\s*[:,]?\s+",
    re.IGNORECASE,
)


def _is_hard_sequence_marker(match: re.Match[str], text: str) -> bool:
    """Return whether a matched word actually separates two process steps.

    ``next`` is also an ordinary adjective (``the next visit`` or ``next
    review time``).  Treat it as a transition only at the start of a statement,
    after punctuation, or when ``next`` itself is followed by punctuation; the
    other markers are unambiguous enough for the existing splitter.
    """

    if match.group(1).casefold() != "next":
        return True
    prefix = text[: match.start()]
    if not prefix.strip():
        return True
    if re.search(r"[.;:,]\s*$", prefix):
        return True
    # ``and next, verify ...`` is an explicit transition, while ``and next
    # responsible team`` is a coordinated noun phrase.
    suffix = text[match.end() :]
    return bool(re.match(r"\s*[,;:]", suffix))


def hard_process_split(
    text: str,
    *,
    ignore_initial_then: bool = False,
) -> dict[str, object] | None:
    cleaned = text.strip()

    matches = [
        match
        for match in HARD_SPLIT_RE.finditer(cleaned)
        if _is_hard_sequence_marker(match, cleaned)
    ]
    if ignore_initial_then:
        matches = [
            m for m in matches if not (m.group(1).lower() == "then" and m.start() == 0)
        ]

    if not matches:
        return None

    parts: list[str] = []
    triggers: list[str] = []
    last = 0

    for match in matches:
        before = cleaned[last : match.start()].strip(" ,.;")
        if before:
            parts.append(before)
        triggers.append(match.group(1).lower())
        last = match.end()

    tail = cleaned[last:].strip(" ,.;")
    if tail:
        parts.append(tail)

    if len(parts) <= 1:
        return None

    return {
        "parts": parts,
        "triggers": triggers,
    }


def soft_process_split(text: str) -> dict[str, str] | None:
    match = SOFT_CONDITION_RE.match(text.strip())
    if not match:
        return None

    trigger = match.group("trigger").lower()
    condition_body = match.group("condition").strip()
    separator = match.group("separator")
    remaining_body = match.group("scope").strip()

    # Prefer an explicitly punctuated ELSE. A plain ``... then A else B`` form
    # is accepted only when THEN was the condition/action separator; this avoids
    # treating ordinary phrases such as "notify someone else" as a branch.
    branch_parts = PUNCTUATED_ELSE_RE.split(remaining_body, maxsplit=1)
    if len(branch_parts) == 1 and "then" in separator.casefold():
        branch_parts = PLAIN_ELSE_RE.split(remaining_body, maxsplit=1)

    true_scope = branch_parts[0].strip()
    false_scope = branch_parts[1].strip() if len(branch_parts) == 2 else ""
    if not true_scope:
        return None

    result = {
        "trigger": trigger,
        "condition_clause": f"{trigger} {condition_body}",
        "conditional_scope": true_scope,
    }
    if false_scope:
        result["else_scope"] = false_scope
    return result

from __future__ import annotations

import re
from typing import Optional


HARD_SPLIT_RE = re.compile(
    r"\b(and then|after that|next|then)\b",
    re.IGNORECASE,
)

SOFT_CONDITION_RE = re.compile(
    r"\b(if|when|unless)\b\s+(.+?)(?:,\s+|\s+then\s+)(.+)$",
    re.IGNORECASE,
)


def hard_process_split(
    text: str,
    *,
    ignore_initial_then: bool = False,
) -> Optional[dict[str, object]]:
    cleaned = text.strip()

    matches = list(HARD_SPLIT_RE.finditer(cleaned))
    if ignore_initial_then:
        matches = [
            m for m in matches
            if not (m.group(1).lower() == "then" and m.start() == 0)
        ]

    if not matches:
        return None

    parts: list[str] = []
    triggers: list[str] = []
    last = 0

    for match in matches:
        before = cleaned[last:match.start()].strip(" ,.;")
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


def soft_process_split(text: str) -> Optional[dict[str, str]]:
    match = SOFT_CONDITION_RE.search(text.strip())
    if not match:
        return None

    trigger = match.group(1).lower()
    condition_body = match.group(2).strip()
    remaining_body = match.group(3).strip()

    return {
        "trigger": trigger,
        "condition_clause": f"{trigger} {condition_body}",
        "conditional_scope": remaining_body,
    }

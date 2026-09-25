from __future__ import annotations

import re
from typing import Any, Optional


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LIST_ITEM_RE = re.compile(r"^\s*(?P<marker>[-*+]|\d+\.)\s+(?P<text>.+)$")
TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")

USER_STORY_RE = re.compile(
    r"^As\s+(?P<role>.+?),\s*I\s+want\s+(?P<goal>.+?)"
    r"(?:,\s*so\s+that\s+(?P<benefit>.+))?\.?$",
    re.IGNORECASE,
)

AC_PREFIX_RE = re.compile(
    r"^(AC\s*\d+|Acceptance\s*Criterion\s*\d+)[:.)-]\s+(.+)$",
    re.IGNORECASE,
)
GHERKIN_RE = re.compile(r"^(Given|When|Then|And|But)\s+(.+)$", re.IGNORECASE)

API_OPERATION_RE = re.compile(
    r"^(?P<method>GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+"
    r"(?P<path>/[A-Za-z0-9_./{}\-:]+)"
    r"(?:\s*[-–:]\s*(?P<description>.+))?$",
    re.IGNORECASE,
)

AC_SECTION_TITLES = {
    "acceptance criteria",
    "acceptance criterion",
    "acceptance",
    "criteria",
}


def detect_heading(line: str) -> Optional[dict[str, Any]]:
    match = HEADING_RE.match(line.strip())
    if not match:
        return None
    return {
        "level": len(match.group(1)),
        "title": match.group(2).strip(),
        "marker": match.group(1),
    }


def detect_list_item(line: str) -> Optional[dict[str, Any]]:
    match = LIST_ITEM_RE.match(line)
    if not match:
        return None
    return {
        "marker": match.group("marker"),
        "text": match.group("text").strip(),
    }


def detect_table_row(line: str) -> Optional[dict[str, Any]]:
    """Parse a pipe-delimited Markdown table row."""
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None

    cells = [cell.strip() for cell in stripped[1:-1].split("|")]
    if len(cells) < 2 or any(not cell for cell in cells):
        return None

    return {
        "cells": cells,
        "is_separator": all(TABLE_SEPARATOR_CELL_RE.fullmatch(cell) for cell in cells),
    }


def _clean_user_story_field(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().rstrip(".").strip()


def detect_user_story(text: str) -> Optional[dict[str, Any]]:
    match = USER_STORY_RE.match(text.strip())
    if not match:
        return None
    return {
        "role": _clean_user_story_field(match.group("role")),
        "goal": _clean_user_story_field(match.group("goal")),
        "benefit": _clean_user_story_field(match.group("benefit")),
    }


def detect_acceptance_criterion(
    text: str,
    *,
    inside_ac_section: bool = False,
) -> Optional[dict[str, Any]]:
    stripped = text.strip()

    ac_match = AC_PREFIX_RE.match(stripped)
    if ac_match:
        return {
            "criterion_id": ac_match.group(1).strip(),
            "criterion_text": ac_match.group(2).strip(),
            "criterion_format": "numbered_ac",
            "gherkin_keyword": None,
        }

    gherkin_match = GHERKIN_RE.match(stripped)
    if gherkin_match:
        return {
            "criterion_id": None,
            "criterion_text": stripped,
            "criterion_format": "gherkin",
            "gherkin_keyword": gherkin_match.group(1),
        }

    if inside_ac_section and stripped:
        return {
            "criterion_id": None,
            "criterion_text": stripped,
            "criterion_format": "list_item_in_ac_section",
            "gherkin_keyword": None,
        }

    return None


def detect_api_operation(text: str) -> Optional[dict[str, Any]]:
    stripped = text.strip().strip("`")
    match = API_OPERATION_RE.match(stripped)
    if not match:
        return None
    return {
        "method": match.group("method").upper(),
        "path": match.group("path"),
        "description": match.group("description").strip() if match.group("description") else None,
    }


def is_acceptance_criteria_section(hierarchy_path: list[str]) -> bool:
    return bool(hierarchy_path) and hierarchy_path[-1].strip().lower() in AC_SECTION_TITLES

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import PurePosixPath
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from process_hierarchy.aggregator import HierarchicalAggregator
from process_hierarchy.candidates import CandidateGenerationLimitError
from process_hierarchy.models import (
    AggregationConfig,
    AggregationLevelConfig,
    AggregationRun,
    CandidateDefinition,
    ProcessEdge,
    ProcessGraph,
    ProcessNode,
    ScoreWeights,
)

from .db_models import (
    ProcessModelResult,
    RunDocument,
    SegmentationResult,
    SegmentationRun,
)
from .hashing import sha256_json
from .llm_process import ProcessExtractionError
from .process_model import ProcessModelSummary

BUILDER_VERSION = "0.7.0"
DERIVATION_MODE = "deterministic_rules"
SPACE_RE = re.compile(r"\s+")
MARKDOWN_RE = re.compile(r"[*_`]+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*")
NUMBERED_MARKER_RE = re.compile(r"^(\d+)\.$")
TRANSITION_PREFIX_RE = re.compile(
    r"^(?:then|next|after\s+that|afterwards|subsequently|finally)\b[,:]?\s*",
    re.IGNORECASE,
)
MODAL_ROLE_RE = re.compile(
    r"^(?:then\s+)?(?:the\s+)?(?P<role>[A-Za-z][A-Za-z0-9 /&_-]{1,60}?)\s+"
    r"(?:must|shall|should|will|can|may|needs?\s+to|is\s+required\s+to)\s+",
    re.IGNORECASE,
)
DECLARATIVE_ROLE_RE = re.compile(
    r"^(?:then\s+)?(?:the|an?|each)\s+"
    r"(?P<role>(?:[A-Za-z][A-Za-z0-9_-]*\s+){1,5}?)"
    r"(?P<verb>[A-Za-z]+(?:s|ed))\b",
    re.IGNORECASE,
)
BY_ROLE_RE = re.compile(
    r"\bby\s+(?:the\s+)?(?P<role>[A-Za-z][A-Za-z0-9 /&_-]{1,50})"
    r"(?=[,.;]|\s+(?:in|using|via|through)\b|$)",
    re.IGNORECASE,
)
SYSTEM_PHRASE_RE = re.compile(
    r"\b(?:in|using|via|through)\s+(?:the\s+)?"
    r"(?P<system>[A-Za-z][A-Za-z0-9 _/-]{0,45}?"
    r"(?:system|platform|portal|console|application|service|api))\b",
    re.IGNORECASE,
)
SYSTEM_SUBJECT_RE = re.compile(
    r"^(?:then\s+)?(?:the|an?|each)?\s*"
    r"(?P<system>[A-Za-z][A-Za-z0-9 _/-]{0,45}?"
    r"(?:system|platform|portal|console|application|api))\s+"
    r"(?P<verb>[A-Za-z]+(?:s|ed))\b",
    re.IGNORECASE,
)
SYSTEM_MENTION_RE = re.compile(
    r"\b(?:the|an?)\s+"
    r"(?P<system>(?:[A-Za-z0-9_-]+\s+){0,2}"
    r"(?:system|platform|portal|console|application|orchestrator|api))\b",
    re.IGNORECASE,
)
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9_-]{1,11}\b")
CONDITION_PREFIX_RE = re.compile(r"^(if|when|unless)\s+", re.IGNORECASE)
REFERENCE_CUE_RE = re.compile(
    r"\b(?:continue\s+with|proceed\s+to|follow|hand\s+off\s+to|"
    r"pass\s+to|move\s+to|as\s+described\s+in|see)\b",
    re.IGNORECASE,
)
DEFINITION_RE = re.compile(
    r"\b(?:is|are)\s+(?:an?|the)\s+(?:type|kind|description|definition|concept)\b|"
    r"\b(?:refers?\s+to|means|consists?\s+of)\b",
    re.IGNORECASE,
)
MODAL_ACTION_RE = re.compile(
    r"\b(?:must|shall|should|will|can|may|needs?\s+to|is\s+required\s+to)\s+"
    r"(?:not\s+)?(?:be\s+)?(?P<verb>[A-Za-z]+)",
    re.IGNORECASE,
)

# This is a domain-independent language vocabulary, not a topology or a list of
# operations copied from a built-in example. It is intentionally conservative:
# unknown verbs are left for the LLM path or for future language adapters.
ACTION_ROOTS = {
    "acknowledge",
    "add",
    "adjust",
    "analyze",
    "append",
    "apply",
    "approve",
    "archive",
    "ask",
    "assign",
    "assess",
    "attach",
    "authorize",
    "bind",
    "calculate",
    "cancel",
    "capture",
    "check",
    "choose",
    "classify",
    "close",
    "collect",
    "compare",
    "communicate",
    "complete",
    "confirm",
    "connect",
    "contact",
    "continue",
    "correlate",
    "correct",
    "create",
    "decide",
    "delete",
    "deliver",
    "deploy",
    "determine",
    "diagnose",
    "disable",
    "document",
    "enable",
    "escalate",
    "evaluate",
    "execute",
    "explain",
    "export",
    "forward",
    "generate",
    "give",
    "identify",
    "import",
    "include",
    "increase",
    "inform",
    "inspect",
    "install",
    "investigate",
    "invoke",
    "issue",
    "keep",
    "link",
    "locate",
    "log",
    "mark",
    "maintain",
    "monitor",
    "notify",
    "open",
    "offer",
    "pack",
    "perform",
    "prepare",
    "preserve",
    "process",
    "provision",
    "publish",
    "receive",
    "read",
    "record",
    "register",
    "reject",
    "release",
    "remove",
    "reproduce",
    "request",
    "reserve",
    "reset",
    "restart",
    "retest",
    "resolve",
    "restore",
    "retain",
    "return",
    "reverse",
    "review",
    "route",
    "run",
    "save",
    "schedule",
    "search",
    "select",
    "send",
    "set",
    "sign",
    "start",
    "stop",
    "store",
    "submit",
    "summarize",
    "tell",
    "transfer",
    "treat",
    "unlock",
    "update",
    "upload",
    "use",
    "validate",
    "verify",
}
ACTOR_CUES = {
    "administrator",
    "adviser",
    "agent",
    "analyst",
    "api",
    "application",
    "clerk",
    "customer",
    "desk",
    "engineer",
    "manager",
    "officer",
    "operator",
    "operations",
    "owner",
    "platform",
    "service",
    "specialist",
    "supplier",
    "system",
    "team",
    "user",
    "worker",
}
NON_PROCESS_SECTIONS = {
    "audience",
    "background",
    "definitions",
    "document control",
    "metadata",
    "overview",
    "purpose",
    "references",
    "related documents",
    "related procedures",
    "scope",
    "terminology",
}
PROCEDURAL_DOCUMENT_CUES = {
    "checklist",
    "guide",
    "instruction",
    "manual",
    "playbook",
    "procedure",
    "process",
    "runbook",
    "standard",
    "workflow",
}
SUPPORTING_DOCUMENT_CUES = {
    "architecture",
    "catalog",
    "catalogue",
    "glossary",
    "matrix",
    "policy",
    "reference",
    "requirements",
}
NON_FLOW_SECTION_PREFIXES = (
    "audience",
    "background",
    "before you begin",
    "definitions",
    "document control",
    "evidence to return",
    "metadata",
    "overview",
    "prerequisite",
    "purpose",
    "reference",
    "related document",
    "related procedure",
    "scope",
    "terminology",
)
NEGATIVE_RULE_RE = re.compile(
    r"^(?:do\s+not|don't|never)\b|"
    r"\b(?:must|shall|should|may|can|does|do)\s+not\b|"
    r"\b(?:is|are)\s+not\s+(?:required|sufficient)\b",
    re.IGNORECASE,
)
PASSIVE_RULE_RE = re.compile(
    r"\b(?:must|shall|should|may|can)\s+be\s+[A-Za-z]+(?:ed|en)\b",
    re.IGNORECASE,
)
DECISION_SECTION_RE = re.compile(
    r"\b(?:choose|classify|decide|decision|determine|outcome|route|select)\b",
    re.IGNORECASE,
)
OPTIONAL_BRANCH_RE = re.compile(
    r"\b(?:block|error|escalat|fail|failure|missing|pending|reject|remediat|"
    r"unresolved|unserviceable|unsuccessful)\w*\b",
    re.IGNORECASE,
)
NEGATIVE_OUTCOME_RE = re.compile(
    r"\b(?:block|error|fail|failed|failure|missing|pending|rejected|remediat|"
    r"unresolved|unserviceable|unsuccessful)\w*\b",
    re.IGNORECASE,
)
DECISION_OUTCOME_RE = re.compile(
    r"^(?:treat|classify|mark|consider|regard)\s+"
    r"(?P<subject>.+?)\s+as\s+"
    r"(?P<state>[A-Za-z][A-Za-z0-9 /_-]{0,60}?)\s+"
    r"(?:when|if)\s+(?P<condition>.+?)[.!?]?$",
    re.IGNORECASE,
)

CONFIG: dict[str, Any] = {
    "schema_version": "1.0",
    "derivation_mode": DERIVATION_MODE,
    "source_interface": "normalized_fragments",
    "supported_input_adapter": "markdown",
    "edge_rules": [
        "actions_within_named_procedural_section",
        "ordered_procedural_sections",
        "explicit_if_then_else",
        "implicit_else_to_continuation",
        "explicit_document_subprocess_call",
        "subprocess_return_to_continuation",
        "decision_outcome_to_named_branch",
    ],
    "candidate_generation": "named_process_regions",
    "l1_selection": "explicit_source_structure",
    "l2_selection": "q_threshold",
    "document_order_is_hypothesis_only": True,
    "weights": {"text": 0.2, "context": 0.4, "flow": 0.4},
    "levels": [
        {
            "target_level": 1,
            "q_min": 0.65,
            "max_candidate_nodes": 5,
            "radius": 3,
            "max_candidates": 20_000,
        },
        {
            "target_level": 2,
            "q_min": 0.50,
            "max_candidate_nodes": 5,
            "radius": 3,
            "max_candidates": 20_000,
        },
    ],
    "branch_integrity": "required",
}
CONFIG_SHA256 = sha256_json(CONFIG)


@dataclass(frozen=True)
class SourceFragment:
    document_path: str
    document_position: int
    document_title: str
    ordinal: int
    fragment_id: str
    parent_fragment_id: str | None
    fragment_type: str
    text: str
    hierarchy_path: tuple[str, ...]
    line_start: int | None
    line_end: int | None
    structural_fields: dict[str, Any]
    segmentation_confidence: float

    @property
    def key(self) -> tuple[str, str]:
        return self.document_path, self.fragment_id

    def evidence(self) -> dict[str, Any]:
        return {
            "document_path": self.document_path,
            "fragment_id": self.fragment_id,
            "fragment_ref": f"{self.document_path}#{self.fragment_id}",
            "fragment_type": self.fragment_type,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "quote": self.text,
        }


@dataclass(frozen=True)
class ExtractedNode:
    id: str
    fragment: SourceFragment
    operation: str
    role: str
    system: str
    node_type: str
    confidence: float
    extraction_rule: str


@dataclass(frozen=True)
class ExtractedEdge:
    source: str
    target: str
    edge_type: str
    condition: str | None
    reason: str
    confidence: float
    evidence: tuple[SourceFragment, ...]


@dataclass(frozen=True)
class ProcessSection:
    document_path: str
    document_position: int
    document_title: str
    title: str
    ordinal: int
    fragments: tuple[SourceFragment, ...]
    role: str
    system: str

    @property
    def key(self) -> tuple[str, str]:
        return self.document_path, self.title


@dataclass(frozen=True)
class DecisionOutcome:
    subject: str
    state: str
    condition: str
    fragment: SourceFragment


@dataclass(frozen=True)
class DocumentProfile:
    path: str
    position: int
    title: str
    role: str
    system: str
    procedural: bool
    sections: tuple[ProcessSection, ...]


def _clean(value: Any) -> str:
    return SPACE_RE.sub(" ", MARKDOWN_RE.sub("", str(value or ""))).strip()


def _is_action_word(value: str) -> bool:
    word = value.casefold().strip(".,;:()[]{}")
    candidates = {word}
    if word.endswith("ies") and len(word) > 4:
        candidates.add(f"{word[:-3]}y")
    if word.endswith("ing") and len(word) > 5:
        candidates.update({word[:-3], f"{word[:-3]}e"})
    if word.endswith("ed") and len(word) > 4:
        candidates.update({word[:-2], word[:-1], f"{word[:-2]}e"})
    if word.endswith("es") and len(word) > 4:
        candidates.update({word[:-2], word[:-1]})
    if word.endswith("s") and len(word) > 3:
        candidates.add(word[:-1])
    return bool(candidates & ACTION_ROOTS)


def _starts_with_action(value: str) -> bool:
    value = TRANSITION_PREFIX_RE.sub("", value.strip())
    words = WORD_RE.findall(value)
    if not words:
        return False
    if words[0].casefold() == "do" and len(words) >= 3 and words[1].casefold() == "not":
        return _is_action_word(words[2])
    return _is_action_word(words[0])


def _actor_like(value: str) -> bool:
    words = {word.casefold() for word in WORD_RE.findall(value)}
    return bool(words & ACTOR_CUES)


def _plausible_actor_phrase(value: str) -> bool:
    words = WORD_RE.findall(value)
    folded = {word.casefold() for word in words}
    if not words or len(words) > 5 or folded & {"because", "if", "or", "that", "when"}:
        return False
    if _is_action_word(words[0]) and len(words) > 3:
        return False
    return _actor_like(value)


def _starts_with_coordinated_action(value: str) -> bool:
    if not _starts_with_action(value):
        return False
    words = WORD_RE.findall(value)
    if not words:
        return False
    first = words[0].casefold()
    if first.endswith(("ed", "ing", "s")):
        return True
    return len(words) > 1 and words[1].casefold() in {
        "a",
        "an",
        "how",
        "it",
        "its",
        "the",
        "their",
        "them",
        "whether",
    }


def _clip_operation(value: str, width: int = 180) -> str:
    value = value.strip()
    if len(value) <= width:
        return value
    prefix = value[: width - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{prefix}…"


def _contains_explicit_action(fragment: SourceFragment) -> bool:
    text = fragment.text
    if _starts_with_action(text):
        return True
    modal = MODAL_ACTION_RE.search(text)
    if modal and _is_action_word(modal.group("verb")):
        return True
    role = _extract_role(fragment, text)
    if role != "Unknown":
        words = WORD_RE.findall(text)
        return any(_is_action_word(word) for word in words[1:])
    return False


def _operation_parts(fragment: SourceFragment) -> list[str]:
    text = fragment.text.strip()
    # "But" and "or" usually express contrast or alternatives, not an ordered
    # pair of actions. Preserve those statements instead of inventing a flow.
    if re.search(r"\b(?:but|or)\b", text, re.IGNORECASE):
        return [_operation(fragment)]
    raw_parts = re.split(r"\s*;\s*|\s*,\s*|\s+and\s+", text, flags=re.IGNORECASE)
    if len(raw_parts) == 1:
        return [_operation(fragment)]

    result: list[str] = []
    current = raw_parts[0]
    for part in raw_parts[1:]:
        if _starts_with_coordinated_action(part):
            result.append(_clean(current).rstrip(".;:"))
            current = part
        else:
            current = f"{current} and {part}"
    result.append(_clean(current).rstrip(".;:"))
    cleaned = [
        _clip_operation(TRANSITION_PREFIX_RE.sub("", item)) for item in result if item
    ]
    return cleaned if len(cleaned) > 1 else [_operation(fragment)]


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _document_title(fragments: list[dict[str, Any]], path: str) -> str:
    headings = [item for item in fragments if item.get("fragment_type") == "heading"]
    headings.sort(
        key=lambda item: (
            len(item.get("hierarchy_path") or []),
            int((item.get("source_position") or {}).get("line_start") or 0),
        )
    )
    if headings and (title := _clean(headings[0].get("text"))):
        return title
    return PurePosixPath(path).stem.replace("_", " ").replace("-", " ").strip()


def _active_leaf_fragments(fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active = [item for item in fragments if item.get("status") == "active"]
    parents = {
        item.get("parent_fragment_id")
        for item in active
        if item.get("parent_fragment_id")
    }
    leaves = [
        item
        for item in active
        if item.get("fragment_type") != "heading" and item.get("id") not in parents
    ]
    return sorted(
        leaves,
        key=lambda item: (
            int((item.get("source_position") or {}).get("line_start") or 0),
            int((item.get("source_position") or {}).get("char_start") or 0),
            str(item.get("id") or ""),
        ),
    )


def _operation(fragment: SourceFragment, *, condition: bool = False) -> str:
    text = fragment.text.strip().strip("-–— ")
    fields = fragment.structural_fields
    if fragment.fragment_type == "user_story" and fields.get("goal"):
        text = _clean(fields["goal"])
    if fragment.fragment_type.startswith("acceptance_criterion"):
        text = re.sub(
            r"^(?:Given|When|Then|And|But)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )
    text = TRANSITION_PREFIX_RE.sub("", text)
    text = text.rstrip(".;:")
    if condition:
        text = CONDITION_PREFIX_RE.sub("", text).rstrip("?.,;:")
        return _clip_operation(f"Decision: {text}?")
    return _clip_operation(text)


def _extract_role(fragment: SourceFragment, text: str) -> str:
    fields = fragment.structural_fields
    if fragment.fragment_type == "user_story" and fields.get("role"):
        return _clean(fields["role"])
    for pattern in (MODAL_ROLE_RE, BY_ROLE_RE):
        match = pattern.search(text)
        if match:
            if pattern is MODAL_ROLE_RE and re.match(
                r"be\b", text[match.end() :], re.IGNORECASE
            ):
                continue
            role = re.sub(
                r"^(?:the|an?|each|every)\s+",
                "",
                _clean(match.group("role")),
                flags=re.IGNORECASE,
            )
            if _plausible_actor_phrase(role):
                return role[:80]

    match = DECLARATIVE_ROLE_RE.search(text)
    if match:
        role = re.sub(
            r"^(?:the|an?|each|every)\s+",
            "",
            _clean(match.group("role")),
            flags=re.IGNORECASE,
        )
        verb = match.group("verb").casefold()
        if (
            verb not in {"is", "was", "has", "does", "includes", "contains"}
            and (role.casefold() not in {"it", "he", "she", "they", "this", "that"})
            and _plausible_actor_phrase(role)
        ):
            return role[:80]
    return "Unknown"


def _extract_system(fragment: SourceFragment, text: str) -> str:
    if fragment.fragment_type == "api_operation":
        return "API"
    match = SYSTEM_PHRASE_RE.search(text)
    if match:
        return _clean(match.group("system"))[:80]
    match = SYSTEM_SUBJECT_RE.search(text)
    if match:
        return re.sub(
            r"^(?:the|an?)\s+",
            "",
            _clean(match.group("system")),
            flags=re.IGNORECASE,
        )[:80]
    match = SYSTEM_MENTION_RE.search(text)
    if match:
        return _clean(match.group("system"))[:80]
    acronyms = [
        value
        for value in ACRONYM_RE.findall(text)
        if value not in {"AC", "API", "HTTP", "HTTPS", "UTF"}
    ]
    if acronyms:
        return acronyms[0]
    if re.search(r"\bAPI\b", text):
        return "API"
    return "Unknown"


def _candidate_confidence(fragment: SourceFragment, text: str) -> float:
    value = {
        "process_step": 0.90,
        "conditional_scope": 0.86,
        "api_operation": 0.88,
        "api_description_sentence": 0.74,
        "acceptance_criterion": 0.76,
        "acceptance_criterion_sentence": 0.78,
        "user_story": 0.62,
        "list_item": 0.72,
        "sentence": 0.60,
        "paragraph": 0.56,
        "table_row": 0.52,
    }.get(fragment.fragment_type, 0.50)
    marker = str(fragment.structural_fields.get("list_marker") or "")
    if NUMBERED_MARKER_RE.match(marker):
        value += 0.14
    if MODAL_ROLE_RE.search(text) or TRANSITION_PREFIX_RE.search(text):
        value += 0.10
    if _extract_role(fragment, text) != "Unknown":
        value += 0.06
    return min(0.96, value * max(0.75, fragment.segmentation_confidence))


def _is_action_candidate(fragment: SourceFragment) -> bool:
    if fragment.fragment_type == "condition_clause":
        return False
    if fragment.fragment_type == "table_row":
        return False
    text = fragment.text.strip()
    words = WORD_RE.findall(text)
    if len(words) < 2 or len(text) < 5:
        return False
    if DEFINITION_RE.search(text):
        return False
    section_names = {value.casefold() for value in fragment.hierarchy_path[1:]}
    if section_names & NON_PROCESS_SECTIONS:
        return False
    if fragment.fragment_type == "list_item":
        label = re.match(r"^[A-Za-z][A-Za-z ]{1,35}:\s*", text)
        if label:
            value = text[label.end() :]
            modal = MODAL_ACTION_RE.search(value)
            if not _starts_with_action(value) and not (
                modal and _is_action_word(modal.group("verb"))
            ):
                return False
    supported = {
        "acceptance_criterion",
        "acceptance_criterion_sentence",
        "api_description_sentence",
        "api_operation",
        "conditional_scope",
        "list_item",
        "paragraph",
        "process_step",
        "sentence",
        "user_story",
    }
    if fragment.fragment_type not in supported:
        return False
    if fragment.fragment_type in {
        "api_operation",
        "conditional_scope",
        "user_story",
    }:
        return True
    return _contains_explicit_action(fragment)


def _condition_label(text: str) -> str:
    cleaned = text.strip().rstrip("?.,;:")
    match = CONDITION_PREFIX_RE.match(cleaned)
    if not match:
        return cleaned
    body = cleaned[match.end() :].strip()
    if match.group(1).casefold() == "unless":
        return f"Not ({body})"
    return body


def _decision_outcomes(section: ProcessSection) -> tuple[DecisionOutcome, ...]:
    outcomes: list[DecisionOutcome] = []
    for fragment in section.fragments:
        match = DECISION_OUTCOME_RE.match(_clean(fragment.text))
        if not match:
            continue
        outcomes.append(
            DecisionOutcome(
                subject=match.group("subject").strip(),
                state=match.group("state").strip(),
                condition=match.group("condition").strip().rstrip(".?!"),
                fragment=fragment,
            )
        )
    return tuple(outcomes)


def _decision_gateway_operation(
    section: ProcessSection,
    outcomes: tuple[DecisionOutcome, ...],
) -> str:
    subjects = {item.subject.casefold() for item in outcomes}
    if len(outcomes) >= 2 and len(subjects) == 1:
        subject = re.sub(
            r"^(?:the|an?)\s+",
            "",
            outcomes[0].subject,
            flags=re.IGNORECASE,
        ).strip()
        if subject:
            return _clip_operation(
                f"Decision: {subject[:1].upper() + subject[1:]} outcome?",
                110,
            )
    return _clip_operation(f"Decision: {section.title.rstrip('?')}?", 110)


def _decision_outcome_routes(
    section: ProcessSection,
    exceptional_title: str,
) -> tuple[DecisionOutcome, DecisionOutcome] | None:
    outcomes = _decision_outcomes(section)
    if len(outcomes) < 2:
        return None
    folded_title = exceptional_title.casefold()
    exceptional = next(
        (
            item
            for item in outcomes
            if item.state.casefold() in folded_title
        ),
        None,
    )
    if exceptional is None and OPTIONAL_BRANCH_RE.search(exceptional_title):
        exceptional = next(
            (item for item in outcomes if NEGATIVE_OUTCOME_RE.search(item.state)),
            None,
        )
    if exceptional is None:
        return None
    normal = next(
        (
            item
            for item in outcomes
            if item is not exceptional and not NEGATIVE_OUTCOME_RE.search(item.state)
        ),
        next((item for item in outcomes if item is not exceptional), None),
    )
    return None if normal is None else (exceptional, normal)


def _outcome_label(outcome: DecisionOutcome) -> str:
    state = outcome.state.strip()
    return state[:1].upper() + state[1:]


def _metadata_values(fragments: list[SourceFragment]) -> dict[str, str]:
    result: dict[str, str] = {}
    for fragment in fragments:
        if len(fragment.hierarchy_path) != 1 or fragment.fragment_type != "list_item":
            continue
        match = re.match(
            r"^(?P<label>[A-Za-z][A-Za-z ]{1,40}):\s*(?P<value>.+)$", fragment.text
        )
        if match:
            result[match.group("label").strip().casefold()] = match.group(
                "value"
            ).strip()
    return result


def _metadata_context(fragments: list[SourceFragment]) -> tuple[str, str]:
    metadata = _metadata_values(fragments)
    role = next(
        (
            metadata[label]
            for label in (
                "procedure owner",
                "process owner",
                "runbook owner",
                "document owner",
                "owner",
            )
            if label in metadata
        ),
        "Unknown",
    )
    system = next(
        (
            metadata[label]
            for label in (
                "used in",
                "primary system",
                "primary tools",
                "system",
                "platform",
            )
            if label in metadata
        ),
        "Unknown",
    )
    return _clip_operation(role, 80), _clip_operation(system, 80)


def _is_core_action_fragment(fragment: SourceFragment) -> bool:
    text = fragment.text.strip()
    if (
        NEGATIVE_RULE_RE.search(text) or PASSIVE_RULE_RE.search(text)
    ) and not _starts_with_action(text):
        return False
    if fragment.fragment_type == "conditional_scope":
        return _contains_explicit_action(fragment)
    return _is_action_candidate(fragment)


def _document_is_procedural(
    title: str,
    fragments: list[SourceFragment],
) -> bool:
    words = {word.casefold() for word in WORD_RE.findall(title)}
    if words & SUPPORTING_DOCUMENT_CUES and not (
        words & (PROCEDURAL_DOCUMENT_CUES - {"process"})
    ):
        return False
    if words & PROCEDURAL_DOCUMENT_CUES:
        return True
    action_sections = {
        fragment.hierarchy_path[1]
        for fragment in fragments
        if len(fragment.hierarchy_path) > 1 and _is_core_action_fragment(fragment)
    }
    action_count = sum(_is_core_action_fragment(fragment) for fragment in fragments)
    return bool(action_sections) and action_count >= 2


def _section_is_flow_candidate(title: str) -> bool:
    folded = title.casefold().strip()
    return not any(folded.startswith(prefix) for prefix in NON_FLOW_SECTION_PREFIXES)


def _condition_from_reference(text: str, target_title: str) -> str:
    match = re.match(
        r"^(?:if|when|unless)\s+(?P<condition>.+?),\s+",
        text.strip(),
        re.IGNORECASE,
    )
    if match:
        return _clip_operation(match.group("condition"), 100)
    return f"Follow {target_title}"


def _opposite_condition(value: str) -> str:
    folded = value.casefold()
    replacements = (
        ("unresolved", "resolved"),
        ("unsuccessful", "successful"),
        ("failure", "success"),
        ("failed", "successful"),
        ("rejected", "accepted"),
        ("missing", "available"),
        ("blocked", "clear"),
        ("pending", "completed"),
        ("error", "no error"),
    )
    for source, target in replacements:
        if source in folded:
            return re.sub(source, target, value, count=1, flags=re.IGNORECASE)
    return "Otherwise"


def _phase_range_name(sections: list[ProcessSection]) -> str:
    if not sections:
        return "Process phase"
    first = sections[0].title
    last = sections[-1].title
    return first if first == last else f"{first} → {last}"


def _short_document_label(title: str) -> str:
    words = WORD_RE.findall(title)
    removable = PROCEDURAL_DOCUMENT_CUES
    compact = [word for word in words if word.casefold() not in removable]
    if not compact:
        compact = words
    return _clip_operation(" ".join(compact), 55)


def _optional_branch_condition(title: str) -> str:
    match = NEGATIVE_OUTCOME_RE.search(title)
    if match:
        value = match.group(0).strip().rstrip(".,;:")
        return value[:1].upper() + value[1:]
    without_action = re.sub(
        r"^(?:escalate|handle|process|resolve|review)\s+(?:the|an?)?\s*",
        "",
        title.strip(),
        flags=re.IGNORECASE,
    )
    return _clip_operation(without_action or title, 80)


class DeterministicProcessModelBuilder:
    """Build a conservative graph from normalized fragments without an LLM.

    The extractor is domain-independent. It accepts a normalized fragment stream,
    currently produced by the Markdown adapter. Explicit procedural signals create
    high-confidence transitions; separate action-bearing blocks in one procedural
    section may create clearly labelled low-confidence sequence hypotheses.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def build(self, run_id: str) -> ProcessModelSummary:
        with self.session_factory() as session:
            run = session.get(SegmentationRun, run_id)
            if run is None:
                raise KeyError(run_id)
            if run.status != "completed":
                raise ProcessExtractionError(
                    "Process extraction requires a completed segmentation run"
                )
            cached = session.scalar(
                select(ProcessModelResult).where(
                    ProcessModelResult.run_id == run_id,
                    ProcessModelResult.builder_version == BUILDER_VERSION,
                    ProcessModelResult.config_sha256 == CONFIG_SHA256,
                )
            )
            if cached is not None:
                return self._summary(cached, cache_hit=True)

            payload = self._build_payload(session, run)
            result = ProcessModelResult(
                run_id=run.id,
                builder_version=BUILDER_VERSION,
                config_sha256=CONFIG_SHA256,
                derivation_mode=DERIVATION_MODE,
                payload=payload,
                level_count=len(payload["summary"]),
                atomic_node_count=len(payload["hierarchy"]["base_graph"]["nodes"]),
            )
            session.add(result)
            session.commit()
            return self._summary(result, cache_hit=False)

    def get(self, run_id: str) -> ProcessModelSummary | None:
        with self.session_factory() as session:
            result = session.scalar(
                select(ProcessModelResult)
                .where(
                    ProcessModelResult.run_id == run_id,
                    ProcessModelResult.derivation_mode == DERIVATION_MODE,
                )
                .order_by(ProcessModelResult.created_at.desc())
            )
            return None if result is None else self._summary(result, cache_hit=True)

    @staticmethod
    def latest(
        session_factory: sessionmaker[Session], run_id: str
    ) -> ProcessModelSummary | None:
        with session_factory() as session:
            result = session.scalar(
                select(ProcessModelResult)
                .where(ProcessModelResult.run_id == run_id)
                .order_by(ProcessModelResult.created_at.desc())
            )
            return (
                None
                if result is None
                else DeterministicProcessModelBuilder._summary(result, cache_hit=True)
            )

    @staticmethod
    def _summary(result: ProcessModelResult, *, cache_hit: bool) -> ProcessModelSummary:
        return ProcessModelSummary(
            id=result.id,
            run_id=result.run_id,
            level_count=result.level_count,
            atomic_node_count=result.atomic_node_count,
            cache_hit=cache_hit,
            payload=result.payload,
        )

    def _build_payload(self, session: Session, run: SegmentationRun) -> dict[str, Any]:
        fragments, document_titles = self._load_fragments(session, run.id)
        if not fragments:
            raise ProcessExtractionError(
                "No active leaf fragments are available for process extraction"
            )

        profiles = self._build_profiles(fragments, document_titles)
        procedural_profiles = [profile for profile in profiles if profile.procedural]
        if not procedural_profiles:
            raise ProcessExtractionError(
                "No procedural document or action-bearing section was found. "
                "Use headings to separate the steps of the documented process."
            )
        references = self._section_references(procedural_profiles)
        nodes, section_nodes, skipped = self._extract_skeleton_nodes(
            procedural_profiles,
            references,
        )
        if not nodes:
            raise ProcessExtractionError(
                "No explicit process actions were found. Add actionable sentences, "
                "numbered steps, conditions, or acceptance criteria to the documentation."
            )
        edges = self._extract_skeleton_edges(
            procedural_profiles,
            nodes,
            section_nodes,
            references,
        )
        inferred_edges = sum(edge.confidence < 0.8 for edge in edges)
        graph, provenance, node_metadata, transition_provenance = self._to_graph(
            nodes, edges
        )
        candidates, phase_summary = self._hierarchy_candidates(
            procedural_profiles,
            section_nodes,
            references,
        )
        aggregation, aggregation_warning = self._aggregate(graph, candidates)
        warnings = self._warnings(
            graph,
            nodes,
            skipped=skipped,
            inferred_edges=inferred_edges,
            aggregation_warning=aggregation_warning,
        )
        return {
            "schema_version": "1.0",
            "source_run": {
                "id": run.id,
                "corpus_sha256": run.corpus_sha256,
            },
            "derivation": {
                "mode": DERIVATION_MODE,
                "universal_extraction": True,
                "llm_used": False,
                "builder_version": BUILDER_VERSION,
                "config_sha256": CONFIG_SHA256,
                "message": (
                    "Без LLM: the process is reconstructed from named procedural "
                    "sections, explicit document hand-offs and conditional routes. "
                    "L1 preserves explicit subprocess and branch boundaries; L2 "
                    "groups contiguous macro-regions around detected split/join "
                    "structure. Supporting policy documents do not become process "
                    "steps."
                ),
            },
            "configuration": CONFIG,
            "process_title": self._structured_process_title(
                procedural_profiles,
                references,
            ),
            "warnings": warnings,
            "analysis": {
                "source_fragment_count": len(fragments),
                "procedural_document_count": len(procedural_profiles),
                "supporting_document_count": len(profiles) - len(procedural_profiles),
                "process_section_count": sum(
                    len(profile.sections) for profile in procedural_profiles
                ),
                "extracted_node_count": len(nodes),
                "transition_count": len(edges),
                "explicit_transition_count": len(edges) - inferred_edges,
                "inferred_transition_count": inferred_edges,
                "skipped_fragment_count": skipped,
            },
            "process_structure": {
                "documents": [
                    {
                        "path": profile.path,
                        "title": profile.title,
                        "classification": (
                            "procedural" if profile.procedural else "supporting"
                        ),
                        "role": profile.role,
                        "system": profile.system,
                        "sections": [section.title for section in profile.sections],
                    }
                    for profile in profiles
                ],
                "section_calls": [
                    {
                        "source_document": source[0],
                        "source_section": source[1],
                        "target_documents": list(targets),
                    }
                    for source, targets in sorted(references.items())
                    if targets
                ],
                "phases": phase_summary,
            },
            "summary": self._hierarchy_summary(aggregation),
            "provenance": provenance,
            "node_metadata": node_metadata,
            "transition_provenance": transition_provenance,
            "hierarchy": aggregation.to_dict(),
        }

    @staticmethod
    def _load_fragments(
        session: Session, run_id: str
    ) -> tuple[list[SourceFragment], dict[str, str]]:
        rows = session.execute(
            select(RunDocument, SegmentationResult)
            .join(
                SegmentationResult,
                SegmentationResult.id == RunDocument.segmentation_result_id,
            )
            .where(
                RunDocument.run_id == run_id,
                RunDocument.status == "completed",
            )
            .order_by(RunDocument.position)
        ).all()
        result: list[SourceFragment] = []
        titles: dict[str, str] = {}
        ordinal = 0
        for run_document, segmentation in rows:
            raw_fragments = list(segmentation.payload.get("fragments") or [])
            title = _document_title(raw_fragments, run_document.relative_path)
            titles[run_document.relative_path] = title
            for raw in _active_leaf_fragments(raw_fragments):
                text = _clean(raw.get("text"))
                if not text:
                    continue
                ordinal += 1
                position = raw.get("source_position") or {}
                result.append(
                    SourceFragment(
                        document_path=run_document.relative_path,
                        document_position=run_document.position,
                        document_title=title,
                        ordinal=ordinal,
                        fragment_id=str(raw.get("id") or ""),
                        parent_fragment_id=(
                            str(raw["parent_fragment_id"])
                            if raw.get("parent_fragment_id")
                            else None
                        ),
                        fragment_type=str(raw.get("fragment_type") or "fragment"),
                        text=text,
                        hierarchy_path=tuple(
                            _clean(item)
                            for item in raw.get("hierarchy_path") or []
                            if _clean(item)
                        ),
                        line_start=_optional_int(position.get("line_start")),
                        line_end=_optional_int(position.get("line_end")),
                        structural_fields=dict(raw.get("structural_fields") or {}),
                        segmentation_confidence=float(
                            raw.get("segmentation_confidence") or 0.5
                        ),
                    )
                )
        return result, titles

    @staticmethod
    def _build_profiles(
        fragments: list[SourceFragment],
        document_titles: dict[str, str],
    ) -> list[DocumentProfile]:
        by_document: dict[str, list[SourceFragment]] = defaultdict(list)
        for fragment in fragments:
            by_document[fragment.document_path].append(fragment)

        profiles: list[DocumentProfile] = []
        for path, document_fragments in sorted(
            by_document.items(),
            key=lambda item: (item[1][0].document_position, item[0]),
        ):
            document_fragments.sort(key=lambda item: item.ordinal)
            title = document_titles.get(path) or document_fragments[0].document_title
            role, system = _metadata_context(document_fragments)
            grouped: dict[tuple[str, ...], list[SourceFragment]] = defaultdict(list)
            for fragment in document_fragments:
                if len(fragment.hierarchy_path) > 1:
                    section_path = fragment.hierarchy_path[1:]
                else:
                    section_path = (title,)
                grouped[section_path].append(fragment)

            sections: list[ProcessSection] = []
            for section_path, section_fragments in sorted(
                grouped.items(),
                key=lambda item: min(value.ordinal for value in item[1]),
            ):
                section_title = " / ".join(section_path)
                if section_title == title:
                    flow_candidate = any(
                        _is_core_action_fragment(item) for item in section_fragments
                    )
                else:
                    flow_candidate = _section_is_flow_candidate(section_title) and (
                        any(
                            _is_core_action_fragment(item) for item in section_fragments
                        )
                        or bool(DECISION_SECTION_RE.search(section_title))
                    )
                if not flow_candidate:
                    continue
                sections.append(
                    ProcessSection(
                        document_path=path,
                        document_position=document_fragments[0].document_position,
                        document_title=title,
                        title=section_title,
                        ordinal=min(item.ordinal for item in section_fragments),
                        fragments=tuple(
                            sorted(section_fragments, key=lambda item: item.ordinal)
                        ),
                        role=role,
                        system=system,
                    )
                )

            profiles.append(
                DocumentProfile(
                    path=path,
                    position=document_fragments[0].document_position,
                    title=title,
                    role=role,
                    system=system,
                    procedural=_document_is_procedural(title, document_fragments),
                    sections=tuple(sections),
                )
            )

        # A corpus can legitimately contain one plainly written procedure whose
        # filename/title has no document-type cue. If no title-based classifier
        # succeeds, retain action-bearing documents instead of returning nothing.
        if not any(profile.procedural for profile in profiles):
            profiles = [
                replace(
                    profile,
                    procedural=sum(
                        len(
                            [
                                fragment
                                for fragment in section.fragments
                                if _is_core_action_fragment(fragment)
                            ]
                        )
                        for section in profile.sections
                    )
                    >= 2,
                )
                for profile in profiles
            ]
        return profiles

    @staticmethod
    def _section_references(
        profiles: list[DocumentProfile],
    ) -> dict[tuple[str, str], tuple[str, ...]]:
        ordered_targets = sorted(profiles, key=lambda item: (item.position, item.path))
        result: dict[tuple[str, str], tuple[str, ...]] = {}
        for profile in ordered_targets:
            for section in profile.sections:
                section_text = " ".join(
                    item.text for item in section.fragments
                ).casefold()
                targets = tuple(
                    target.path
                    for target in ordered_targets
                    if target.path != profile.path
                    and target.title.casefold() in section_text
                )
                if targets:
                    result[section.key] = targets
        return result

    @staticmethod
    def _extract_skeleton_nodes(
        profiles: list[DocumentProfile],
        references: dict[tuple[str, str], tuple[str, ...]],
    ) -> tuple[list[ExtractedNode], dict[tuple[str, str], list[str]], int]:
        nodes: list[ExtractedNode] = []
        section_nodes: dict[tuple[str, str], list[str]] = defaultdict(list)
        used_fragment_keys: set[tuple[str, str]] = set()

        for profile in sorted(profiles, key=lambda item: (item.position, item.path)):
            for section in profile.sections:
                target_count = len(references.get(section.key, ()))
                conditional_groups: dict[str, list[SourceFragment]] = defaultdict(
                    list
                )
                for fragment in section.fragments:
                    if fragment.parent_fragment_id and fragment.fragment_type in {
                        "condition_clause",
                        "conditional_scope",
                    }:
                        conditional_groups[fragment.parent_fragment_id].append(fragment)
                outcomes = _decision_outcomes(section)
                # A heading such as "Decision" is not enough to create a gateway:
                # the documentation must also expose at least one condition, two
                # named outcomes, or multiple referenced subprocesses.
                title_declares_decision = bool(DECISION_SECTION_RE.search(section.title))
                is_gateway = (
                    target_count > 1
                    or len(outcomes) >= 2
                    or (bool(conditional_groups) and title_declares_decision)
                )
                core_fragments = [
                    fragment
                    for fragment in section.fragments
                    if _is_core_action_fragment(fragment)
                ]

                if is_gateway:
                    referenced_titles = {
                        profile.path: profile.title for profile in profiles
                    }
                    referenced_evidence = next(
                        (
                            fragment
                            for fragment in section.fragments
                            if any(
                                referenced_titles[path].casefold()
                                in fragment.text.casefold()
                                for path in references.get(section.key, ())
                            )
                        ),
                        None,
                    )
                    condition_evidence = next(
                        (
                            item
                            for group in conditional_groups.values()
                            for item in group
                            if item.fragment_type == "condition_clause"
                        ),
                        None,
                    )
                    evidence = (
                        outcomes[0].fragment
                        if len(outcomes) >= 2
                        else referenced_evidence
                        or condition_evidence
                        or (core_fragments[0] if core_fragments else section.fragments[0])
                    )
                    if len(outcomes) >= 2:
                        operation = _decision_gateway_operation(section, outcomes)
                        extraction_rule = "documented_outcomes"
                        confidence = 0.94
                    elif target_count > 1:
                        operation = _clip_operation(
                            f"Decision: {section.title.rstrip('?')}?", 110
                        )
                        extraction_rule = "multi_document_route"
                        confidence = 0.88
                    elif condition_evidence is not None:
                        operation = _operation(condition_evidence, condition=True)
                        extraction_rule = "explicit_condition"
                        confidence = 0.92
                    else:
                        operation = _clip_operation(
                            f"Decision: {section.title.rstrip('?')}?", 110
                        )
                        extraction_rule = "decision_section"
                        confidence = 0.82
                    node_id = f"v{len(nodes) + 1}"
                    nodes.append(
                        ExtractedNode(
                            id=node_id,
                            fragment=evidence,
                            operation=operation,
                            role=profile.role,
                            system=profile.system,
                            node_type="gateway",
                            confidence=confidence,
                            extraction_rule=extraction_rule,
                        )
                    )
                    section_nodes[section.key].append(node_id)
                    used_fragment_keys.add(evidence.key)
                    used_fragment_keys.update(item.fragment.key for item in outcomes)
                    # When a gateway routes to multiple named documents, the
                    # referenced procedures are the branches. Creating another
                    # action node for the wording around each reference would add
                    # an artificial split inside the caller and break its explicit
                    # process-region boundary.
                    branch_groups = (
                        () if target_count > 1 else conditional_groups.values()
                    )
                    for group in branch_groups:
                        condition = next(
                            (
                                item
                                for item in group
                                if item.fragment_type == "condition_clause"
                            ),
                            None,
                        )
                        scopes = [
                            item
                            for item in group
                            if item.fragment_type == "conditional_scope"
                            and _contains_explicit_action(item)
                        ]
                        if not condition or not scopes:
                            continue
                        for scope in scopes:
                            for operation in _operation_parts(scope)[:1]:
                                branch_id = f"v{len(nodes) + 1}"
                                role = _extract_role(scope, scope.text)
                                system = _extract_system(scope, scope.text)
                                nodes.append(
                                    ExtractedNode(
                                        id=branch_id,
                                        fragment=scope,
                                        operation=_clip_operation(operation, 118),
                                        role=(
                                            profile.role if role == "Unknown" else role
                                        ),
                                        system=(
                                            profile.system
                                            if system == "Unknown"
                                            else system
                                        ),
                                        node_type="action",
                                        confidence=_candidate_confidence(
                                            scope, scope.text
                                        ),
                                        extraction_rule="conditional_branch",
                                    )
                                )
                                section_nodes[section.key].append(branch_id)
                                used_fragment_keys.update({condition.key, scope.key})
                    continue

                for fragment in core_fragments:
                    operations = _operation_parts(fragment)
                    if fragment.fragment_type == "conditional_scope":
                        condition = next(
                            (
                                item
                                for item in section.fragments
                                if item.parent_fragment_id
                                == fragment.parent_fragment_id
                                and item.fragment_type == "condition_clause"
                            ),
                            None,
                        )
                        if condition:
                            operations = [
                                f"If {_condition_label(condition.text)}: {operation}"
                                for operation in operations
                            ]
                    for operation in operations:
                        operation = _clip_operation(operation, 118)
                        if not operation or any(
                            node.operation.casefold() == operation.casefold()
                            for node in nodes
                            if node.id in section_nodes[section.key]
                        ):
                            continue
                        role = _extract_role(fragment, fragment.text)
                        system = _extract_system(fragment, fragment.text)
                        node_id = f"v{len(nodes) + 1}"
                        nodes.append(
                            ExtractedNode(
                                id=node_id,
                                fragment=fragment,
                                operation=operation,
                                role=profile.role if role == "Unknown" else role,
                                system=profile.system
                                if system == "Unknown"
                                else system,
                                node_type="action",
                                confidence=_candidate_confidence(
                                    fragment, fragment.text
                                ),
                                extraction_rule=(
                                    f"named_section:{fragment.fragment_type}"
                                ),
                            )
                        )
                        section_nodes[section.key].append(node_id)
                        used_fragment_keys.add(fragment.key)

                if not section_nodes[section.key] and _starts_with_action(
                    section.title
                ):
                    evidence = section.fragments[0]
                    node_id = f"v{len(nodes) + 1}"
                    nodes.append(
                        ExtractedNode(
                            id=node_id,
                            fragment=evidence,
                            operation=_clip_operation(section.title, 110),
                            role=profile.role,
                            system=profile.system,
                            node_type="action",
                            confidence=0.62,
                            extraction_rule="action_heading",
                        )
                    )
                    section_nodes[section.key].append(node_id)
                    used_fragment_keys.add(evidence.key)

        nodes = DeterministicProcessModelBuilder._inherit_context(nodes)
        process_fragments = {
            fragment.key
            for profile in profiles
            for section in profile.sections
            for fragment in section.fragments
        }
        skipped = len(process_fragments - used_fragment_keys)
        return nodes, dict(section_nodes), skipped

    @staticmethod
    def _extract_skeleton_edges(
        profiles: list[DocumentProfile],
        nodes: list[ExtractedNode],
        section_nodes: dict[tuple[str, str], list[str]],
        references: dict[tuple[str, str], tuple[str, ...]],
    ) -> list[ExtractedEdge]:
        nodes_by_id = {node.id: node for node in nodes}
        profiles_by_path = {profile.path: profile for profile in profiles}
        edges: list[ExtractedEdge] = []
        seen: set[tuple[str, str, str]] = set()

        def add(
            source: str,
            target: str,
            *,
            edge_type: str = "sequence",
            condition: str | None = None,
            reason: str,
            confidence: float,
            evidence: tuple[SourceFragment, ...],
        ) -> None:
            if (
                source == target
                or source not in nodes_by_id
                or target not in nodes_by_id
            ):
                return
            key = (source, target, (condition or "").casefold())
            if key in seen:
                return
            seen.add(key)
            edges.append(
                ExtractedEdge(
                    source=source,
                    target=target,
                    edge_type=edge_type,
                    condition=condition,
                    reason=reason,
                    confidence=confidence,
                    evidence=evidence,
                )
            )

        def conditional_branches(section: ProcessSection) -> list[str]:
            return [
                node_id
                for node_id in section_nodes.get(section.key, ())
                if nodes_by_id[node_id].extraction_rule == "conditional_branch"
            ]

        def section_exits(section: ProcessSection) -> list[str]:
            branches = conditional_branches(section)
            if branches:
                return branches
            ids = section_nodes.get(section.key, ())
            return [ids[-1]] if ids else []

        # The order of explicit actions inside one named section is retained,
        # but its confidence reflects the strength of the source notation.
        for profile in profiles:
            for section in profile.sections:
                ids = section_nodes.get(section.key, [])
                branches = conditional_branches(section)
                if branches and ids:
                    gateway = ids[0]
                    for branch in branches:
                        branch_fragment = nodes_by_id[branch].fragment
                        condition_fragment = next(
                            (
                                item
                                for item in section.fragments
                                if item.parent_fragment_id
                                == branch_fragment.parent_fragment_id
                                and item.fragment_type == "condition_clause"
                            ),
                            None,
                        )
                        condition = (
                            "Otherwise"
                            if branch_fragment.structural_fields.get(
                                "condition_branch"
                            )
                            == "else"
                            else (
                                _condition_label(condition_fragment.text)
                                if condition_fragment
                                else "Documented condition"
                            )
                        )
                        add(
                            gateway,
                            branch,
                            edge_type="conditional",
                            condition=condition,
                            reason=(
                                "The source documentation explicitly states this "
                                "condition and its action."
                            ),
                            confidence=0.92,
                            evidence=tuple(
                                item
                                for item in (condition_fragment, branch_fragment)
                                if item is not None
                            ),
                        )
                    continue
                for source, target in pairwise(ids):
                    left = nodes_by_id[source].fragment
                    right = nodes_by_id[target].fragment
                    left_marker = str(left.structural_fields.get("list_marker") or "")
                    right_marker = str(right.structural_fields.get("list_marker") or "")
                    left_number = NUMBERED_MARKER_RE.match(left_marker)
                    right_number = NUMBERED_MARKER_RE.match(right_marker)
                    if (
                        left_number
                        and right_number
                        and int(right_number.group(1)) == int(left_number.group(1)) + 1
                    ):
                        reason = (
                            "Consecutive numbers in the same named section explicitly "
                            "define action order."
                        )
                        confidence = 0.90
                    elif left.key == right.key:
                        reason = (
                            "The same source statement lists these coordinated actions "
                            "in this order."
                        )
                        confidence = 0.84
                    elif (
                        left.parent_fragment_id
                        and left.parent_fragment_id == right.parent_fragment_id
                    ):
                        reason = (
                            "Low-confidence hypothesis: two action sentences occur "
                            "in this order in the same source paragraph."
                        )
                        confidence = 0.35
                    else:
                        reason = (
                            "Low-confidence hypothesis: action statements occur in this "
                            "order inside the same named procedural section."
                        )
                        confidence = 0.55
                    add(
                        source,
                        target,
                        reason=reason,
                        confidence=confidence,
                        evidence=(left,) if left.key == right.key else (left, right),
                    )

        # Preserve the order of named sections inside a procedure. A section that
        # explicitly calls another document is connected through that subprocess
        # below instead of being flattened into the following section.
        for profile in profiles:
            populated = [
                section
                for section in profile.sections
                if section_nodes.get(section.key)
            ]
            optional_splits: dict[
                tuple[str, str], tuple[ProcessSection, ProcessSection]
            ] = {}
            for index, section in enumerate(populated[:-2]):
                next_section = populated[index + 1]
                following = populated[index + 2]
                is_decision = (
                    nodes_by_id[section_nodes[section.key][0]].node_type == "gateway"
                )
                if (
                    is_decision
                    and OPTIONAL_BRANCH_RE.search(next_section.title)
                    and not conditional_branches(section)
                    and not references.get(section.key)
                ):
                    optional_splits[section.key] = (next_section, following)

            for section, next_section in pairwise(populated):
                if references.get(section.key):
                    continue
                split = optional_splits.get(section.key)
                if split:
                    exceptional, normal = split
                    outcome_routes = _decision_outcome_routes(
                        section,
                        exceptional.title,
                    )
                    if outcome_routes:
                        exceptional_outcome, normal_outcome = outcome_routes
                        branch_condition = _outcome_label(exceptional_outcome)
                        normal_condition = _outcome_label(normal_outcome)
                        exceptional_reason = (
                            f'The source defines the "{exceptional_outcome.state}" '
                            f"outcome when {exceptional_outcome.condition}; the next "
                            "named section handles that outcome."
                        )
                        normal_reason = (
                            f'The source defines the "{normal_outcome.state}" outcome '
                            f"when {normal_outcome.condition}; this path bypasses the "
                            "exceptional section and continues with the normal flow."
                        )
                    else:
                        exceptional_outcome = normal_outcome = None
                        branch_condition = _optional_branch_condition(
                            exceptional.title
                        )
                        normal_condition = _opposite_condition(branch_condition)
                        exceptional_reason = (
                            "The decision section is followed by a named exceptional "
                            "branch in the same procedure."
                        )
                        normal_reason = (
                            "The documented process continues with the normal path "
                            "when the exceptional branch condition is false."
                        )
                    for source_exit in section_exits(section):
                        exceptional_fragment = (
                            exceptional_outcome.fragment
                            if exceptional_outcome
                            else nodes_by_id[source_exit].fragment
                        )
                        normal_fragment = (
                            normal_outcome.fragment
                            if normal_outcome
                            else nodes_by_id[source_exit].fragment
                        )
                        add(
                            source_exit,
                            section_nodes[exceptional.key][0],
                            edge_type="conditional",
                            condition=branch_condition,
                            reason=exceptional_reason,
                            confidence=0.94 if outcome_routes else 0.86,
                            evidence=(
                                exceptional_fragment,
                                nodes_by_id[section_nodes[exceptional.key][0]].fragment,
                            ),
                        )
                        add(
                            source_exit,
                            section_nodes[normal.key][0],
                            edge_type="conditional",
                            condition=normal_condition,
                            reason=normal_reason,
                            confidence=0.94 if outcome_routes else 0.78,
                            evidence=(
                                normal_fragment,
                                nodes_by_id[section_nodes[normal.key][0]].fragment,
                            ),
                        )
                    continue

                branches = conditional_branches(section)
                if len(branches) == 1:
                    gateway = section_nodes[section.key][0]
                    branch_fragment = nodes_by_id[branches[0]].fragment
                    condition_fragment = next(
                        (
                            item
                            for item in section.fragments
                            if item.parent_fragment_id
                            == branch_fragment.parent_fragment_id
                            and item.fragment_type == "condition_clause"
                        ),
                        None,
                    )
                    add(
                        gateway,
                        section_nodes[next_section.key][0],
                        edge_type="conditional",
                        condition="Otherwise",
                        reason=(
                            "The source states one conditional action. When its "
                            "condition is false, that action is skipped and the "
                            "documented process continues with the next section."
                        ),
                        confidence=0.84,
                        evidence=tuple(
                            item
                            for item in (
                                condition_fragment,
                                nodes_by_id[section_nodes[next_section.key][0]].fragment,
                            )
                            if item is not None
                        ),
                    )
                for source_exit in section_exits(section):
                    add(
                        source_exit,
                        section_nodes[next_section.key][0],
                        reason=(
                            "Low-confidence hypothesis: these named procedural sections "
                            "appear in this order in the same document."
                        ),
                        confidence=0.74,
                        evidence=(
                            nodes_by_id[source_exit].fragment,
                            nodes_by_id[section_nodes[next_section.key][0]].fragment,
                        ),
                    )

        # An exact title reference invokes a documented subprocess. Multiple
        # references from one section form alternatives; each subprocess returns
        # to the next section of the caller, producing an explicit split/join.
        for source_key, target_paths in sorted(references.items()):
            source_ids = section_nodes.get(source_key, [])
            source_profile = profiles_by_path.get(source_key[0])
            if not source_ids or not source_profile:
                continue
            populated_source = [
                section
                for section in source_profile.sections
                if section_nodes.get(section.key)
            ]
            source_index = next(
                (
                    index
                    for index, section in enumerate(populated_source)
                    if section.key == source_key
                ),
                -1,
            )
            continuation = (
                populated_source[source_index + 1]
                if 0 <= source_index < len(populated_source) - 1
                else None
            )
            source_section = next(
                section
                for section in source_profile.sections
                if section.key == source_key
            )
            source_branches = conditional_branches(source_section)

            for target_path in target_paths:
                target_profile = profiles_by_path[target_path]
                target_sections = [
                    section
                    for section in target_profile.sections
                    if section_nodes.get(section.key)
                ]
                if not target_sections:
                    continue
                reference_fragment = next(
                    (
                        fragment
                        for fragment in source_section.fragments
                        if target_profile.title.casefold() in fragment.text.casefold()
                    ),
                    source_section.fragments[0],
                )
                reference_condition = next(
                    (
                        fragment
                        for fragment in source_section.fragments
                        if reference_fragment.parent_fragment_id
                        and fragment.parent_fragment_id
                        == reference_fragment.parent_fragment_id
                        and fragment.fragment_type == "condition_clause"
                    ),
                    None,
                )
                matching_branch = next(
                    (
                        branch_id
                        for branch_id in source_branches
                        if target_profile.title.casefold()
                        in nodes_by_id[branch_id].fragment.text.casefold()
                    ),
                    None,
                )
                source_call = matching_branch or source_ids[-1]
                call_fragment = (
                    nodes_by_id[matching_branch].fragment
                    if matching_branch
                    else reference_fragment
                )
                target_entry = section_nodes[target_sections[0].key][0]
                condition = (
                    (
                        _condition_label(reference_condition.text)
                        if reference_condition
                        else _condition_from_reference(
                            reference_fragment.text,
                            target_profile.title,
                        )
                    )
                    if len(target_paths) > 1 and matching_branch is None
                    else None
                )
                add(
                    source_call,
                    target_entry,
                    edge_type="conditional" if condition else "sequence",
                    condition=condition,
                    reason=(
                        "The source section explicitly names the target document; "
                        "its procedure is invoked as a subprocess."
                    ),
                    confidence=0.93,
                    evidence=tuple(
                        item
                        for item in (
                            reference_condition,
                            call_fragment,
                            nodes_by_id[target_entry].fragment,
                        )
                        if item is not None
                    ),
                )
                if continuation:
                    continuation_entry = section_nodes[continuation.key][0]
                    for target_exit in section_exits(target_sections[-1]):
                        add(
                            target_exit,
                            continuation_entry,
                            reason=(
                                "The invoked subprocess returns to the next named section "
                                "of the calling procedure."
                            ),
                            confidence=0.84,
                            evidence=(
                                nodes_by_id[target_exit].fragment,
                                nodes_by_id[continuation_entry].fragment,
                            ),
                        )
        return edges

    @staticmethod
    def _hierarchy_candidates(
        profiles: list[DocumentProfile],
        section_nodes: dict[tuple[str, str], list[str]],
        references: dict[tuple[str, str], tuple[str, ...]],
    ) -> tuple[tuple[CandidateDefinition, ...], list[dict[str, Any]]]:
        definitions: list[CandidateDefinition] = []
        phase_summary: list[dict[str, Any]] = []

        incoming_paths = {
            target_path
            for target_paths in references.values()
            for target_path in target_paths
        }
        entry_profiles = [
            profile for profile in profiles if profile.path not in incoming_paths
        ]
        entry = min(
            entry_profiles or profiles,
            key=lambda item: (item.position, item.path),
        )
        split_section = next(
            (
                section
                for section in entry.sections
                if len(references.get(section.key, ())) > 1
            ),
            None,
        )

        l1_specs: list[tuple[str, list[ProcessSection]]] = []
        phase_specs: list[tuple[str, list[ProcessSection]]] = []
        if split_section:
            split_index = list(entry.sections).index(split_section)
            before = list(entry.sections[: split_index + 1])
            main_after = list(entry.sections[split_index + 1 :])
            branch_paths = set(references[split_section.key])
            branch_profiles = [
                profile for profile in profiles if profile.path in branch_paths
            ]
            branches = [
                section for profile in branch_profiles for section in profile.sections
            ]

            reachable_paths = {entry.path}
            pending = [entry.path]
            while pending:
                current = pending.pop()
                for source_key, targets in references.items():
                    if source_key[0] != current:
                        continue
                    for target in targets:
                        if target not in reachable_paths:
                            reachable_paths.add(target)
                            pending.append(target)
            downstream_profiles = [
                profile
                for profile in profiles
                if profile.path in reachable_paths
                and profile.path != entry.path
                and profile.path not in branch_paths
            ]
            after = main_after + [
                section
                for profile in downstream_profiles
                for section in profile.sections
            ]
            branch_name = " or ".join(
                _short_document_label(profile.title) for profile in branch_profiles
            )
            phase_specs = [
                (_phase_range_name(before), before),
                (branch_name or f"Branches after {split_section.title}", branches),
                (_phase_range_name(after), after),
            ]
            l1_specs = [(_phase_range_name(before), before)]
            l1_specs.extend(
                (
                    _short_document_label(profile.title),
                    list(profile.sections),
                )
                for profile in branch_profiles
            )
            if main_after:
                l1_specs.append((_phase_range_name(main_after), main_after))
            l1_specs.extend(
                (
                    _short_document_label(profile.title),
                    list(profile.sections),
                )
                for profile in downstream_profiles
            )
        elif len(profiles) > 1:
            l1_specs = [
                (_short_document_label(profile.title), list(profile.sections))
                for profile in profiles
            ]
            if len(l1_specs) >= 4:
                phase_count = 3 if len(l1_specs) >= 6 else 2
                for index in range(phase_count):
                    start = index * len(l1_specs) // phase_count
                    end = (index + 1) * len(l1_specs) // phase_count
                    chunk = l1_specs[start:end]
                    sections = [
                        section
                        for _, group_sections in chunk
                        for section in group_sections
                    ]
                    if sections:
                        phase_specs.append((_phase_range_name(sections), sections))
        else:
            populated = [
                section for section in entry.sections if section_nodes.get(section.key)
            ]
            l1_specs = [(section.title, [section]) for section in populated]
            phase_count = min(3, max(1, len(populated) // 2))
            if phase_count > 1:
                for index in range(phase_count):
                    start = index * len(populated) // phase_count
                    end = (index + 1) * len(populated) // phase_count
                    group = populated[start:end]
                    if group:
                        phase_specs.append((_phase_range_name(group), group))

        used_l1_ids: set[str] = set()
        l1_index = 0
        for name, sections in l1_specs:
            ids = tuple(
                node_id
                for section in sections
                for node_id in section_nodes.get(section.key, ())
                if node_id not in used_l1_ids
            )
            if len(ids) < 2:
                continue
            used_l1_ids.update(ids)
            l1_index += 1
            definitions.append(
                CandidateDefinition(
                    id=f"L1_REGION_{l1_index:03d}",
                    name=_clip_operation(name, 90),
                    target_level=1,
                    atomic_node_ids=ids,
                    purpose="structural",
                )
            )

        used_atomic_ids: set[str] = set()
        phase_index = 0
        for name, sections in phase_specs:
            ids = tuple(
                node_id
                for section in sections
                for node_id in section_nodes.get(section.key, ())
                if node_id not in used_atomic_ids
            )
            if len(ids) < 2:
                continue
            used_atomic_ids.update(ids)
            phase_index += 1
            definitions.append(
                CandidateDefinition(
                    id=f"L2_PHASE_{phase_index:03d}",
                    name=_clip_operation(name, 90),
                    target_level=2,
                    atomic_node_ids=ids,
                )
            )
            phase_summary.append(
                {
                    "id": f"L2_PHASE_{phase_index:03d}",
                    "name": _clip_operation(name, 90),
                    "source_sections": [section.title for section in sections],
                    "atomic_node_ids": list(ids),
                }
            )

        return tuple(definitions), phase_summary

    @staticmethod
    def _structured_process_title(
        profiles: list[DocumentProfile],
        references: dict[tuple[str, str], tuple[str, ...]],
    ) -> str:
        incoming = {
            target_path
            for target_paths in references.values()
            for target_path in target_paths
        }
        entries = [profile for profile in profiles if profile.path not in incoming]
        entry = min(entries or profiles, key=lambda item: (item.position, item.path))
        return entry.title

    @staticmethod
    def _extract_nodes(
        fragments: list[SourceFragment],
    ) -> tuple[
        list[ExtractedNode],
        dict[tuple[str, str], list[str]],
        int,
    ]:
        nodes: list[ExtractedNode] = []
        fragment_nodes: dict[tuple[str, str], list[str]] = defaultdict(list)
        skipped = 0
        conditional_groups: dict[tuple[str, str], list[SourceFragment]] = defaultdict(
            list
        )
        for fragment in fragments:
            if fragment.parent_fragment_id and fragment.fragment_type in {
                "condition_clause",
                "conditional_scope",
            }:
                conditional_groups[
                    (fragment.document_path, fragment.parent_fragment_id)
                ].append(fragment)

        handled: set[tuple[str, str]] = set()
        for fragment in fragments:
            group_key = (
                fragment.document_path,
                fragment.parent_fragment_id or "",
            )
            group = conditional_groups.get(group_key, [])
            if group and group_key not in handled:
                handled.add(group_key)
                conditions = [
                    item for item in group if item.fragment_type == "condition_clause"
                ]
                scopes = [
                    item for item in group if item.fragment_type == "conditional_scope"
                ]
                for condition, scope in zip(conditions, scopes):
                    gateway_id = f"v{len(nodes) + 1}"
                    role = _extract_role(scope, scope.text)
                    system = _extract_system(scope, scope.text)
                    nodes.append(
                        ExtractedNode(
                            id=gateway_id,
                            fragment=condition,
                            operation=_operation(condition, condition=True),
                            role=role,
                            system=system,
                            node_type="gateway",
                            confidence=0.82,
                            extraction_rule="explicit_condition",
                        )
                    )
                    fragment_nodes[condition.key].append(gateway_id)
                    for operation in _operation_parts(scope):
                        action_id = f"v{len(nodes) + 1}"
                        nodes.append(
                            ExtractedNode(
                                id=action_id,
                                fragment=scope,
                                operation=operation,
                                role=role,
                                system=system,
                                node_type="action",
                                confidence=_candidate_confidence(scope, scope.text),
                                extraction_rule="conditional_scope",
                            )
                        )
                        fragment_nodes[scope.key].append(action_id)
                continue
            if fragment.fragment_type in {"condition_clause", "conditional_scope"}:
                continue
            if not _is_action_candidate(fragment):
                skipped += 1
                continue
            role = _extract_role(fragment, fragment.text)
            system = _extract_system(fragment, fragment.text)
            for operation in _operation_parts(fragment):
                node_id = f"v{len(nodes) + 1}"
                nodes.append(
                    ExtractedNode(
                        id=node_id,
                        fragment=fragment,
                        operation=operation,
                        role=role,
                        system=system,
                        node_type="action",
                        confidence=_candidate_confidence(fragment, fragment.text),
                        extraction_rule=f"fragment_type:{fragment.fragment_type}",
                    )
                )
                fragment_nodes[fragment.key].append(node_id)
        return nodes, dict(fragment_nodes), skipped

    @staticmethod
    def _inherit_context(nodes: list[ExtractedNode]) -> list[ExtractedNode]:
        result: list[ExtractedNode] = []
        last_role: dict[tuple[str, tuple[str, ...]], str] = {}
        last_system: dict[tuple[str, tuple[str, ...]], str] = {}
        for node in nodes:
            key = (node.fragment.document_path, node.fragment.hierarchy_path)
            role = node.role
            system = node.system
            begins_with_pronoun = bool(
                re.match(
                    r"^(?:then\s+)?(?:they|he|she|it)\b",
                    node.fragment.text,
                    re.IGNORECASE,
                )
            )
            if role == "Unknown" and begins_with_pronoun and key in last_role:
                role = last_role[key]
            if (
                system == "Unknown"
                and (
                    begins_with_pronoun
                    or TRANSITION_PREFIX_RE.search(node.fragment.text)
                )
                and key in last_system
            ):
                system = last_system[key]
            if role != "Unknown":
                last_role[key] = role
            if system != "Unknown":
                last_system[key] = system
            result.append(replace(node, role=role, system=system))
        return result

    @staticmethod
    def _extract_edges(
        fragments: list[SourceFragment],
        nodes: list[ExtractedNode],
        fragment_nodes: dict[tuple[str, str], list[str]],
        document_titles: dict[str, str],
    ) -> list[ExtractedEdge]:
        nodes_by_id = {node.id: node for node in nodes}
        edges: list[ExtractedEdge] = []
        seen: set[tuple[str, str, str]] = set()

        def add(
            source: str,
            target: str,
            *,
            edge_type: str = "sequence",
            condition: str | None = None,
            reason: str,
            confidence: float,
            evidence: tuple[SourceFragment, ...],
        ) -> None:
            if (
                source == target
                or source not in nodes_by_id
                or target not in nodes_by_id
            ):
                return
            key = (source, target, (condition or "").casefold())
            if key in seen:
                return
            seen.add(key)
            edges.append(
                ExtractedEdge(
                    source=source,
                    target=target,
                    edge_type=edge_type,
                    condition=condition,
                    reason=reason,
                    confidence=confidence,
                    evidence=evidence,
                )
            )

        by_parent: dict[tuple[str, str], list[SourceFragment]] = defaultdict(list)
        by_section: dict[tuple[str, tuple[str, ...]], list[SourceFragment]] = (
            defaultdict(list)
        )
        by_document: dict[str, list[SourceFragment]] = defaultdict(list)
        for fragment in fragments:
            if fragment.parent_fragment_id:
                by_parent[(fragment.document_path, fragment.parent_fragment_id)].append(
                    fragment
                )
            by_section[(fragment.document_path, fragment.hierarchy_path)].append(
                fragment
            )
            by_document[fragment.document_path].append(fragment)

        fragment_by_key = {fragment.key: fragment for fragment in fragments}
        for fragment_key, node_ids in fragment_nodes.items():
            if len(node_ids) < 2:
                continue
            fragment = fragment_by_key[fragment_key]
            for source, target in pairwise(node_ids):
                add(
                    source,
                    target,
                    reason=(
                        "The source statement explicitly lists coordinated actions "
                        "in this order."
                    ),
                    confidence=0.84,
                    evidence=(fragment,),
                )

        # A condition and its scope are an explicit conditional transition.
        for group in by_parent.values():
            conditions = [
                item for item in group if item.fragment_type == "condition_clause"
            ]
            scopes = [
                item for item in group if item.fragment_type == "conditional_scope"
            ]
            for condition_fragment, scope_fragment in zip(conditions, scopes):
                sources = fragment_nodes.get(condition_fragment.key, [])
                targets = fragment_nodes.get(scope_fragment.key, [])
                if sources and targets:
                    add(
                        sources[-1],
                        targets[0],
                        edge_type="conditional",
                        condition=_condition_label(condition_fragment.text),
                        reason="The source sentence explicitly states this condition and its scope.",
                        confidence=0.92,
                        evidence=(condition_fragment, scope_fragment),
                    )

        # Hard process splits preserve their explicit step_index order.
        for group in by_parent.values():
            process_steps = [
                item
                for item in group
                if item.fragment_type == "process_step"
                and item.structural_fields.get("step_index") is not None
            ]
            process_steps.sort(
                key=lambda item: int(item.structural_fields.get("step_index") or 0)
            )
            for left, right in pairwise(process_steps):
                left_nodes = fragment_nodes.get(left.key, [])
                right_nodes = fragment_nodes.get(right.key, [])
                if left_nodes and right_nodes:
                    add(
                        left_nodes[-1],
                        right_nodes[0],
                        reason="The sentence uses an explicit process-order connector.",
                        confidence=0.94,
                        evidence=(left, right),
                    )

        # A numbered Markdown list explicitly defines order inside its section.
        for group in by_section.values():
            ordered = sorted(group, key=lambda item: item.ordinal)
            numbered = [
                item
                for item in ordered
                if NUMBERED_MARKER_RE.match(
                    str(item.structural_fields.get("list_marker") or "")
                )
            ]
            for left, right in pairwise(numbered):
                left_number = int(
                    NUMBERED_MARKER_RE.match(
                        str(left.structural_fields.get("list_marker"))
                    ).group(1)
                )
                right_number = int(
                    NUMBERED_MARKER_RE.match(
                        str(right.structural_fields.get("list_marker"))
                    ).group(1)
                )
                if right_number != left_number + 1:
                    continue
                left_nodes = fragment_nodes.get(left.key, [])
                right_nodes = fragment_nodes.get(right.key, [])
                if left_nodes and right_nodes:
                    add(
                        left_nodes[-1],
                        right_nodes[0],
                        reason="Consecutive numbers in the same list explicitly define step order.",
                        confidence=0.90,
                        evidence=(left, right),
                    )

        # Gherkin When/Then/And fragments define scenario order.
        for group in by_section.values():
            gherkin = [
                item
                for item in sorted(group, key=lambda item: item.ordinal)
                if str(item.structural_fields.get("criterion_format") or "")
                == "gherkin"
            ]
            for left, right in pairwise(gherkin):
                left_nodes = fragment_nodes.get(left.key, [])
                right_nodes = fragment_nodes.get(right.key, [])
                if left_nodes and right_nodes:
                    add(
                        left_nodes[-1],
                        right_nodes[0],
                        reason="The Gherkin scenario explicitly orders these statements.",
                        confidence=0.88,
                        evidence=(left, right),
                    )

        # Transition words link only to the previous extracted action in the same section.
        for group in by_section.values():
            ordered = sorted(group, key=lambda item: item.ordinal)
            previous_node: str | None = None
            previous_fragment: SourceFragment | None = None
            for fragment in ordered:
                current_nodes = fragment_nodes.get(fragment.key, [])
                if not current_nodes:
                    continue
                if (
                    previous_node
                    and previous_fragment
                    and TRANSITION_PREFIX_RE.search(fragment.text)
                ):
                    add(
                        previous_node,
                        current_nodes[0],
                        reason="The target statement begins with an explicit sequence connector.",
                        confidence=0.86,
                        evidence=(previous_fragment, fragment),
                    )
                previous_node = current_nodes[-1]
                previous_fragment = fragment

        # Turn source fragments into ordered action blocks. A condition and its
        # scope are one block with a gateway entry and action exit. This lets the
        # surrounding procedure connect to the block without flattening the
        # conditional edge itself.
        for document_fragments in by_document.values():
            blocks: list[
                tuple[
                    SourceFragment,
                    SourceFragment,
                    str,
                    str,
                    tuple[SourceFragment, ...],
                ]
            ] = []
            handled_conditional_parents: set[str] = set()
            for fragment in sorted(document_fragments, key=lambda item: item.ordinal):
                if fragment.fragment_type in {"condition_clause", "conditional_scope"}:
                    parent_id = fragment.parent_fragment_id
                    if not parent_id or parent_id in handled_conditional_parents:
                        continue
                    conditional_group = by_parent.get(
                        (fragment.document_path, parent_id), []
                    )
                    condition = next(
                        (
                            item
                            for item in conditional_group
                            if item.fragment_type == "condition_clause"
                            and fragment_nodes.get(item.key)
                        ),
                        None,
                    )
                    scope = next(
                        (
                            item
                            for item in conditional_group
                            if item.fragment_type == "conditional_scope"
                            and fragment_nodes.get(item.key)
                        ),
                        None,
                    )
                    if condition and scope:
                        handled_conditional_parents.add(parent_id)
                        blocks.append(
                            (
                                condition,
                                scope,
                                fragment_nodes[condition.key][0],
                                fragment_nodes[scope.key][-1],
                                (condition, scope),
                            )
                        )
                    continue
                current_nodes = fragment_nodes.get(fragment.key)
                if current_nodes:
                    blocks.append(
                        (
                            fragment,
                            fragment,
                            current_nodes[0],
                            current_nodes[-1],
                            (fragment,),
                        )
                    )

            blocks.sort(key=lambda item: item[0].ordinal)
            for left, right in pairwise(blocks):
                _, left_end, _, left_exit, left_evidence = left
                right_start, _, right_entry, _, right_evidence = right
                same_parent_sentences = (
                    left_end.parent_fragment_id is not None
                    and left_end.parent_fragment_id == right_start.parent_fragment_id
                    and left_end.fragment_type.endswith("sentence")
                    and right_start.fragment_type.endswith("sentence")
                )
                same_section = left_end.hierarchy_path == right_start.hierarchy_path
                if same_parent_sentences:
                    location = "the same source paragraph"
                    confidence = 0.35
                elif same_section:
                    location = "the same procedural section"
                    confidence = 0.55
                else:
                    location = "successive action-bearing sections of the same document"
                    confidence = 0.45
                add(
                    left_exit,
                    right_entry,
                    reason=(
                        "Low-confidence hypothesis: action-bearing blocks occur "
                        f"in this order within {location}."
                    ),
                    confidence=confidence,
                    evidence=left_evidence + right_evidence,
                )

        # Link documents only when a source action names the target and uses a hand-off cue.
        first_node_by_document: dict[str, str] = {}
        for node in nodes:
            first_node_by_document.setdefault(node.fragment.document_path, node.id)
        for node in nodes:
            if not REFERENCE_CUE_RE.search(node.fragment.text):
                continue
            folded = node.fragment.text.casefold()
            for path, title in document_titles.items():
                if (
                    path == node.fragment.document_path
                    or title.casefold() not in folded
                ):
                    continue
                target = first_node_by_document.get(path)
                if target:
                    add(
                        node.id,
                        target,
                        reason=(
                            "The source action explicitly names the target document "
                            "and uses a hand-off or continuation phrase."
                        ),
                        confidence=0.84,
                        evidence=(node.fragment, nodes_by_id[target].fragment),
                    )

        return edges

    @staticmethod
    def _to_graph(
        nodes: list[ExtractedNode], edges: list[ExtractedEdge]
    ) -> tuple[
        ProcessGraph,
        dict[str, list[dict[str, Any]]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        graph_nodes = {
            node.id: ProcessNode(
                id=node.id,
                operation=node.operation,
                role=node.role,
                system=node.system,
                source_fragment_ids=(
                    f"{node.fragment.document_path}#{node.fragment.fragment_id}",
                ),
            )
            for node in nodes
        }
        provenance = {node.id: [node.fragment.evidence()] for node in nodes}
        node_metadata = {
            node.id: {
                "node_type": node.node_type,
                "confidence": node.confidence,
                "extraction_rule": node.extraction_rule,
            }
            for node in nodes
        }
        graph_edges: list[ProcessEdge] = []
        transition_provenance: dict[str, dict[str, Any]] = {}
        for index, edge in enumerate(edges, start=1):
            edge_id = f"e{index}"
            graph_edges.append(
                ProcessEdge(
                    id=edge_id,
                    source=edge.source,
                    target=edge.target,
                    edge_type=edge.edge_type,
                    condition=edge.condition,
                )
            )
            transition_provenance[edge_id] = {
                "reason": edge.reason,
                "confidence": edge.confidence,
                "evidence": [item.evidence() for item in edge.evidence],
            }
        return (
            ProcessGraph(nodes=graph_nodes, edges=tuple(graph_edges), level=0),
            provenance,
            node_metadata,
            transition_provenance,
        )

    @staticmethod
    def _aggregate(
        graph: ProcessGraph,
        candidate_definitions: tuple[CandidateDefinition, ...],
    ) -> tuple[AggregationRun, str | None]:
        target_levels = {
            definition.target_level for definition in candidate_definitions
        }
        if 1 not in target_levels:
            return (
                AggregationRun(base_graph=graph, levels=()),
                (
                    "Hierarchy was limited to L0 because no named process region "
                    "contained at least two extracted actions."
                ),
            )
        levels = tuple(
            AggregationLevelConfig(
                q_min=item["q_min"],
                max_candidate_nodes=item["max_candidate_nodes"],
                radius=item["radius"],
                max_candidates=item["max_candidates"],
            )
            for item in CONFIG["levels"]
            if item["target_level"] in target_levels
        )
        try:
            return (
                HierarchicalAggregator(
                    AggregationConfig(
                        weights=ScoreWeights(**CONFIG["weights"]),
                        levels=levels,
                    )
                ).run(graph, candidate_definitions),
                None,
            )
        except CandidateGenerationLimitError as exc:
            return (
                AggregationRun(base_graph=graph, levels=()),
                f"Hierarchy was limited to L0: {exc}",
            )

    @staticmethod
    def _warnings(
        graph: ProcessGraph,
        nodes: list[ExtractedNode],
        *,
        skipped: int,
        inferred_edges: int,
        aggregation_warning: str | None,
    ) -> list[str]:
        warnings = [
            "Без LLM: implicit semantics, synonyms and unstated dependencies are not inferred."
        ]
        if skipped:
            warnings.append(
                f"{skipped} fragment(s) were skipped because they did not contain a "
                "supported explicit process statement."
            )
        if inferred_edges:
            warnings.append(
                f"{inferred_edges} low-confidence sequence hypothesis/hypotheses "
                "were inferred from action order within or between named procedural "
                "sections; review them before use."
            )
        unknown_roles = sum(node.role == "Unknown" for node in nodes)
        unknown_systems = sum(node.system == "Unknown" for node in nodes)
        if unknown_roles:
            warnings.append(
                f"Role is not explicit for {unknown_roles} of {len(nodes)} nodes."
            )
        if unknown_systems:
            warnings.append(
                f"System is not explicit for {unknown_systems} of {len(nodes)} nodes."
            )
        outgoing_counts: dict[str, int] = defaultdict(int)
        for edge in graph.edges:
            outgoing_counts[edge.source] += 1
        incomplete_gateways = [
            node.id
            for node in nodes
            if node.node_type == "gateway" and outgoing_counts[node.id] < 2
        ]
        if incomplete_gateways:
            warnings.append(
                "Decision gateway(s) with fewer than two outgoing paths require "
                f"review: {', '.join(incomplete_gateways)}."
            )
        neighbors = graph.undirected_neighbors()
        remaining = set(graph.nodes)
        components = 0
        while remaining:
            components += 1
            stack = [remaining.pop()]
            while stack:
                current = stack.pop()
                connected = neighbors[current] & remaining
                remaining -= connected
                stack.extend(connected)
        if components > 1:
            warnings.append(
                f"The documentation-derived graph has {components} disconnected "
                "components. "
                "Add sequence or hand-off statements if they belong to one process."
            )
        if aggregation_warning:
            warnings.append(aggregation_warning)
        return warnings

    @staticmethod
    def _hierarchy_summary(aggregation: AggregationRun) -> list[dict[str, int]]:
        result = [
            {
                "level": 0,
                "node_count": len(aggregation.base_graph.nodes),
                "edge_count": len(aggregation.base_graph.edges),
            }
        ]
        result.extend(
            {
                "level": level.target_level,
                "node_count": len(level.graph.nodes),
                "edge_count": len(level.graph.edges),
            }
            for level in aggregation.levels
        )
        return result

    @staticmethod
    def _process_title(document_titles: dict[str, str]) -> str:
        titles = list(dict.fromkeys(document_titles.values()))
        if len(titles) == 1:
            return titles[0]
        return "Documentation-derived process"

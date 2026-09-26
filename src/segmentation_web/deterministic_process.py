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
    ProcessEdge,
    ProcessGraph,
    ProcessNode,
    ScoreWeights,
)

from .control_process import ProcessModelSummary
from .db_models import (
    ProcessModelResult,
    RunDocument,
    SegmentationResult,
    SegmentationRun,
)
from .hashing import sha256_json
from .llm_process import ProcessExtractionError

BUILDER_VERSION = "0.4.0"
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

# This is a domain-independent language vocabulary, not a corpus topology or a
# list of D1-D5 operations. It is intentionally conservative: unknown verbs are
# left for the LLM path or for future language adapters.
ACTION_ROOTS = {
    "acknowledge",
    "add",
    "adjust",
    "analyze",
    "append",
    "apply",
    "approve",
    "archive",
    "assign",
    "assess",
    "attach",
    "authorize",
    "calculate",
    "cancel",
    "capture",
    "check",
    "classify",
    "close",
    "collect",
    "compare",
    "communicate",
    "complete",
    "confirm",
    "contact",
    "continue",
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
    "export",
    "forward",
    "generate",
    "identify",
    "import",
    "inform",
    "inspect",
    "investigate",
    "invoke",
    "issue",
    "link",
    "log",
    "mark",
    "maintain",
    "monitor",
    "notify",
    "open",
    "pack",
    "perform",
    "prepare",
    "preserve",
    "process",
    "publish",
    "receive",
    "record",
    "register",
    "reject",
    "release",
    "remove",
    "reproduce",
    "request",
    "reset",
    "restart",
    "retest",
    "resolve",
    "restore",
    "return",
    "review",
    "route",
    "save",
    "schedule",
    "search",
    "select",
    "send",
    "sign",
    "start",
    "stop",
    "store",
    "submit",
    "tell",
    "transfer",
    "unlock",
    "update",
    "upload",
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

CONFIG: dict[str, Any] = {
    "schema_version": "1.0",
    "derivation_mode": DERIVATION_MODE,
    "source_interface": "normalized_fragments",
    "supported_input_adapter": "markdown",
    "edge_rules": [
        "hard_process_split",
        "numbered_list_order",
        "explicit_transition_prefix",
        "gherkin_order",
        "conditional_scope",
        "explicit_document_handoff",
        "action_block_order_hypothesis",
    ],
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
    return CONDITION_PREFIX_RE.sub("", text).strip().rstrip("?.,;:")


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

        nodes, fragment_nodes, skipped = self._extract_nodes(fragments)
        if not nodes:
            raise ProcessExtractionError(
                "No explicit process actions were found. Add actionable sentences, "
                "numbered steps, conditions, or acceptance criteria to the documentation."
            )
        nodes = self._inherit_context(nodes)
        edges = self._extract_edges(
            fragments,
            nodes,
            fragment_nodes,
            document_titles,
        )
        inferred_edges = sum(
            edge.reason.startswith("Low-confidence hypothesis:") for edge in edges
        )
        graph, provenance, node_metadata, transition_provenance = self._to_graph(
            nodes, edges
        )
        aggregation, aggregation_warning = self._aggregate(graph)
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
                    "Без LLM: L0 actions require linguistic or structural evidence. "
                    "Explicit transitions have high confidence; action-bearing "
                    "blocks in one procedural section may form labelled, "
                    "low-confidence sequence hypotheses. Plain sentence adjacency "
                    "is never presented as confirmed process flow. L1/L2 use the "
                    "same deterministic aggregation method as the LLM path."
                ),
            },
            "configuration": CONFIG,
            "process_title": self._process_title(document_titles),
            "warnings": warnings,
            "analysis": {
                "source_fragment_count": len(fragments),
                "extracted_node_count": len(nodes),
                "transition_count": len(edges),
                "explicit_transition_count": len(edges) - inferred_edges,
                "inferred_transition_count": inferred_edges,
                "skipped_fragment_count": skipped,
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
    def _aggregate(graph: ProcessGraph) -> tuple[AggregationRun, str | None]:
        levels = tuple(
            AggregationLevelConfig(
                q_min=item["q_min"],
                max_candidate_nodes=item["max_candidate_nodes"],
                radius=item["radius"],
                max_candidates=item["max_candidates"],
            )
            for item in CONFIG["levels"]
        )
        try:
            return (
                HierarchicalAggregator(
                    AggregationConfig(
                        weights=ScoreWeights(**CONFIG["weights"]),
                        levels=levels,
                    )
                ).run(graph),
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
                "were inferred from the order of separate action-bearing blocks "
                "inside a document; review them before use."
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

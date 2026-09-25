from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from process_hierarchy.aggregator import HierarchicalAggregator
from process_hierarchy.markdown_graph import MarkdownProcessGraphParser
from process_hierarchy.models import (
    AggregationConfig,
    AggregationLevelConfig,
    AggregationRun,
    ScoreWeights,
)

from .db_models import (
    ProcessModelResult,
    RunDocument,
    SegmentationResult,
    SegmentationRun,
)
from .hashing import sha256_json

PROCESS_MODEL_BUILDER_VERSION = "0.1.0"
DERIVATION_MODE = "control_baseline"
PROCESS_MODEL_CONFIG = {
    "schema_version": "1.0",
    "derivation_mode": DERIVATION_MODE,
    "weights": {"text": 0.2, "context": 0.4, "flow": 0.4},
    "levels": [
        {"target_level": 1, "q_min": 0.65},
        {"target_level": 2, "q_min": 0.50},
    ],
    "candidate_selection": "explicit_control_candidates",
    "branch_integrity": "required",
}
PROCESS_MODEL_CONFIG_SHA256 = sha256_json(PROCESS_MODEL_CONFIG)

SPACE_RE = re.compile(r"\s+")

CONTROL_DOCUMENTS = {
    "D1_access_recovery_support_policy.md",
    "D2_service_desk_access_incident_guide.md",
    "D3_identity_and_authentication_recovery_runbook.md",
    "D4_software_platform_recovery_runbook.md",
    "D5_recovery_verification_escalation_and_closure.md",
}

# These selectors provide deterministic provenance for the control baseline.
# They are deliberately separate from process extraction: a future LLM stage
# will replace this fixture for arbitrary uploaded corpora.
EVIDENCE_SELECTORS: dict[str, tuple[tuple[str, str], ...]] = {
    "v1": (("D2_service_desk_access_incident_guide.md", "A report may arrive"),),
    "v2": (
        (
            "D2_service_desk_access_incident_guide.md",
            "Search the service management system",
        ),
    ),
    "v3": (
        (
            "D2_service_desk_access_incident_guide.md",
            "Use the priority rules",
        ),
    ),
    "v4": (
        (
            "D2_service_desk_access_incident_guide.md",
            "Select the service category",
        ),
    ),
    "v5": (
        (
            "D2_service_desk_access_incident_guide.md",
            "When the evidence points to an account",
        ),
        (
            "D2_service_desk_access_incident_guide.md",
            "When the evidence points to a service",
        ),
    ),
    "v6": (
        (
            "D3_identity_and_authentication_recovery_runbook.md",
            "Review whether the account exists",
        ),
    ),
    "v7": (
        (
            "D3_identity_and_authentication_recovery_runbook.md",
            "Inspect the rules that apply",
        ),
    ),
    "v8": (
        (
            "D3_identity_and_authentication_recovery_runbook.md",
            "Use the external identity provider console",
        ),
    ),
    "v9": (
        (
            "D3_identity_and_authentication_recovery_runbook.md",
            "Apply the least disruptive action",
        ),
    ),
    "v10": (
        (
            "D4_software_platform_recovery_runbook.md",
            "Check the public and internal health endpoints",
        ),
    ),
    "v11": (
        (
            "D4_software_platform_recovery_runbook.md",
            "Review instance or replica health",
        ),
    ),
    "v12": (
        (
            "D4_software_platform_recovery_runbook.md",
            "Check the database, message broker",
        ),
    ),
    "v13": (
        (
            "D4_software_platform_recovery_runbook.md",
            "Choose the smallest controlled action",
        ),
    ),
    "v14": (
        (
            "D1_access_recovery_support_policy.md",
            "The Service Desk then verifies access",
        ),
        (
            "D5_recovery_verification_escalation_and_closure.md",
            "Ask the user to repeat the normal sign-in path",
        ),
    ),
    "v15": (
        (
            "D1_access_recovery_support_policy.md",
            "decides whether the incident has been resolved",
        ),
        (
            "D5_recovery_verification_escalation_and_closure.md",
            "Treat the incident as recovered",
        ),
    ),
    "v16": (
        (
            "D1_access_recovery_support_policy.md",
            "If normal access has not been restored",
        ),
        (
            "D5_recovery_verification_escalation_and_closure.md",
            "Create an escalation linked to the original incident",
        ),
    ),
    "v17": (
        (
            "D1_access_recovery_support_policy.md",
            "The user must be told what was restored",
        ),
        (
            "D5_recovery_verification_escalation_and_closure.md",
            "Tell the user whether access has been restored",
        ),
    ),
    "v18": (
        (
            "D1_access_recovery_support_policy.md",
            "The incident may be closed only after",
        ),
        (
            "D5_recovery_verification_escalation_and_closure.md",
            "Select a closure category",
        ),
    ),
}


class ProcessModelUnavailableError(ValueError):
    pass


@dataclass(frozen=True)
class ProcessModelSummary:
    id: str
    run_id: str
    level_count: int
    atomic_node_count: int
    cache_hit: bool
    payload: dict[str, Any]


def _normalise(value: str) -> str:
    return SPACE_RE.sub(" ", value).strip()


class ControlProcessModelBuilder:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        process_definition_path: Path,
    ) -> None:
        self.session_factory = session_factory
        self.process_definition_path = process_definition_path

    def build(self, run_id: str) -> ProcessModelSummary:
        with self.session_factory() as session:
            run = session.get(SegmentationRun, run_id)
            if run is None:
                raise KeyError(run_id)
            self._validate_control_corpus(session, run)

            cached = session.scalar(
                select(ProcessModelResult).where(
                    ProcessModelResult.run_id == run_id,
                    ProcessModelResult.builder_version == PROCESS_MODEL_BUILDER_VERSION,
                    ProcessModelResult.config_sha256 == PROCESS_MODEL_CONFIG_SHA256,
                )
            )
            if cached is not None:
                return self._summary(cached, cache_hit=True)

            payload = self._build_payload(session, run)
            result = ProcessModelResult(
                run_id=run.id,
                builder_version=PROCESS_MODEL_BUILDER_VERSION,
                config_sha256=PROCESS_MODEL_CONFIG_SHA256,
                derivation_mode=DERIVATION_MODE,
                payload=payload,
                level_count=3,
                atomic_node_count=len(payload["hierarchy"]["base_graph"]["nodes"]),
            )
            session.add(result)
            session.commit()
            return self._summary(result, cache_hit=False)

    def get(self, run_id: str) -> ProcessModelSummary | None:
        with self.session_factory() as session:
            result = session.scalar(
                select(ProcessModelResult)
                .where(ProcessModelResult.run_id == run_id)
                .order_by(ProcessModelResult.created_at.desc())
            )
            if result is None:
                return None
            return self._summary(result, cache_hit=True)

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

    @staticmethod
    def _validate_control_corpus(session: Session, run: SegmentationRun) -> None:
        paths = set(
            session.scalars(
                select(RunDocument.relative_path).where(
                    RunDocument.run_id == run.id,
                    RunDocument.status == "completed",
                )
            ).all()
        )
        basenames = {PurePosixPath(path).name for path in paths}
        if run.source_type != "example" or basenames != CONTROL_DOCUMENTS:
            raise ProcessModelUnavailableError(
                "The deterministic process hierarchy is available only for the "
                "built-in D1-D5 control corpus. LLM extraction for arbitrary "
                "uploaded documentation is not enabled."
            )

    def _build_payload(self, session: Session, run: SegmentationRun) -> dict[str, Any]:
        source = self.process_definition_path.read_text(encoding="utf-8")
        parser = MarkdownProcessGraphParser()
        graph = parser.parse(source)
        candidates = parser.parse_candidates(source, graph)
        aggregation = HierarchicalAggregator(
            AggregationConfig(
                weights=ScoreWeights(text=0.2, context=0.4, flow=0.4),
                levels=(
                    AggregationLevelConfig(q_min=0.65),
                    AggregationLevelConfig(q_min=0.50),
                ),
            )
        ).run(graph, candidates)
        provenance = self._resolve_provenance(session, run.id)
        hierarchy = aggregation.to_dict()

        return {
            "schema_version": "1.0",
            "source_run": {
                "id": run.id,
                "corpus_sha256": run.corpus_sha256,
            },
            "derivation": {
                "mode": DERIVATION_MODE,
                "universal_extraction": False,
                "builder_version": PROCESS_MODEL_BUILDER_VERSION,
                "config_sha256": PROCESS_MODEL_CONFIG_SHA256,
                "message": (
                    "The L0 topology and aggregation candidates are the "
                    "deterministic control baseline from the article. Source "
                    "evidence is resolved against the current D1-D5 segmentation. "
                    "No LLM inference is performed."
                ),
            },
            "configuration": PROCESS_MODEL_CONFIG,
            "summary": self._hierarchy_summary(aggregation),
            "provenance": provenance,
            "hierarchy": hierarchy,
        }

    @staticmethod
    def _hierarchy_summary(aggregation: AggregationRun) -> list[dict[str, int]]:
        levels = [aggregation.base_graph]
        levels.extend(item.graph for item in aggregation.levels)
        return [
            {
                "level": graph.level,
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
            }
            for graph in levels
        ]

    @staticmethod
    def _resolve_provenance(
        session: Session, run_id: str
    ) -> dict[str, list[dict[str, Any]]]:
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
        ).all()
        by_basename = {
            PurePosixPath(run_document.relative_path).name: (
                run_document,
                list(result.payload.get("fragments") or []),
            )
            for run_document, result in rows
        }

        resolved: dict[str, list[dict[str, Any]]] = {}
        missing: list[str] = []
        for node_id, selectors in EVIDENCE_SELECTORS.items():
            evidence: list[dict[str, Any]] = []
            for basename, needle in selectors:
                run_document, fragments = by_basename[basename]
                matches = [
                    fragment
                    for fragment in fragments
                    if fragment.get("status") == "active"
                    and needle.casefold()
                    in _normalise(str(fragment.get("text") or "")).casefold()
                ]
                match = min(
                    matches,
                    key=lambda fragment: len(
                        _normalise(str(fragment.get("text") or ""))
                    ),
                    default=None,
                )
                if match is None:
                    missing.append(f"{node_id}: {basename}: {needle}")
                    continue
                position = match.get("source_position") or {}
                evidence.append(
                    {
                        "document_path": run_document.relative_path,
                        "fragment_id": match.get("id"),
                        "fragment_type": match.get("fragment_type"),
                        "line_start": position.get("line_start"),
                        "line_end": position.get("line_end"),
                        "quote": _normalise(str(match.get("text") or "")),
                    }
                )
            resolved[node_id] = evidence

        if missing:
            raise ValueError(
                "Control provenance selectors did not match the segmented corpus: "
                + "; ".join(missing)
            )
        return resolved

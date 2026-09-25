from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .db_models import (
    DocumentationGraphResult,
    DocumentVersion,
    RunDocument,
    SegmentationResult,
    SegmentationRun,
)
from .hashing import sha256_json

GRAPH_BUILDER_VERSION = "0.1.0"
GRAPH_CONFIG = {
    "schema_version": "1.0",
    "levels": ["document", "section", "fragment"],
    "edge_types": ["contains", "document_order", "references"],
    "leaf_policy": "active_fragments_without_active_children",
    "reference_policy": "explicit_document_title_match",
}
GRAPH_CONFIG_SHA256 = sha256_json(GRAPH_CONFIG)

MARKDOWN_RE = re.compile(r"[*_`]+")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DocumentationGraphSummary:
    id: str
    run_id: str
    node_count: int
    edge_count: int
    cache_hit: bool
    payload: dict[str, Any]


def _stable_suffix(parts: list[str]) -> str:
    value = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:16]


def _clean_text(value: str) -> str:
    return SPACE_RE.sub(" ", MARKDOWN_RE.sub("", value)).strip()


def _fallback_title(relative_path: str) -> str:
    return PurePosixPath(relative_path).stem.replace("_", " ").replace("-", " ").strip()


def _fragment_order(fragment: dict[str, Any], index: int) -> tuple[int, int, int]:
    source = fragment.get("source_position") or {}
    return (
        int(source.get("line_start") or 0),
        int(source.get("char_start") or 0),
        index,
    )


def _leaf_fragments(fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active = [fragment for fragment in fragments if fragment.get("status") == "active"]
    active_parent_ids = {
        fragment.get("parent_fragment_id")
        for fragment in active
        if fragment.get("parent_fragment_id")
    }
    indexed = [
        (index, fragment)
        for index, fragment in enumerate(active)
        if fragment.get("fragment_type") != "heading"
        and fragment.get("id") not in active_parent_ids
    ]
    indexed.sort(key=lambda item: _fragment_order(item[1], item[0]))
    return [fragment for _, fragment in indexed]


def _document_title(fragments: list[dict[str, Any]], relative_path: str) -> str:
    headings = [
        fragment
        for fragment in fragments
        if fragment.get("fragment_type") == "heading"
    ]
    if headings:
        headings.sort(
            key=lambda fragment: (
                len(fragment.get("hierarchy_path") or []),
                int((fragment.get("source_position") or {}).get("line_start") or 0),
            )
        )
        title = _clean_text(str(headings[0].get("text") or ""))
        if title:
            return title
    return _fallback_title(relative_path)


class DocumentationGraphBuilder:
    """Build a provenance-preserving graph from persisted segmentation results.

    This builder intentionally makes no semantic process-flow claims.  It records
    containment, source order and explicit references so a later LLM stage can
    add action and dependency edges without losing their source evidence.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def build(self, run_id: str) -> DocumentationGraphSummary:
        with self.session_factory() as session:
            run = session.get(SegmentationRun, run_id)
            if run is None:
                raise KeyError(run_id)

            cached = session.scalar(
                select(DocumentationGraphResult).where(
                    DocumentationGraphResult.run_id == run_id,
                    DocumentationGraphResult.builder_version == GRAPH_BUILDER_VERSION,
                    DocumentationGraphResult.config_sha256 == GRAPH_CONFIG_SHA256,
                )
            )
            if cached is not None:
                return self._summary(cached, cache_hit=True)

            payload = self._build_payload(session, run)
            result = DocumentationGraphResult(
                run_id=run_id,
                builder_version=GRAPH_BUILDER_VERSION,
                config_sha256=GRAPH_CONFIG_SHA256,
                payload=payload,
                node_count=len(payload["nodes"]),
                edge_count=len(payload["edges"]),
            )
            session.add(result)
            session.commit()
            return self._summary(result, cache_hit=False)

    def get(self, run_id: str) -> DocumentationGraphSummary | None:
        with self.session_factory() as session:
            result = session.scalar(
                select(DocumentationGraphResult)
                .where(DocumentationGraphResult.run_id == run_id)
                .order_by(DocumentationGraphResult.created_at.desc())
            )
            if result is None:
                return None
            return self._summary(result, cache_hit=True)

    @staticmethod
    def _summary(
        result: DocumentationGraphResult, *, cache_hit: bool
    ) -> DocumentationGraphSummary:
        return DocumentationGraphSummary(
            id=result.id,
            run_id=result.run_id,
            node_count=result.node_count,
            edge_count=result.edge_count,
            cache_hit=cache_hit,
            payload=result.payload,
        )

    @staticmethod
    def _build_payload(
        session: Session, run: SegmentationRun
    ) -> dict[str, Any]:
        rows = session.execute(
            select(RunDocument, DocumentVersion, SegmentationResult)
            .join(DocumentVersion, DocumentVersion.id == RunDocument.document_version_id)
            .join(
                SegmentationResult,
                SegmentationResult.id == RunDocument.segmentation_result_id,
            )
            .where(
                RunDocument.run_id == run.id,
                RunDocument.status == "completed",
            )
            .order_by(RunDocument.position)
        ).all()

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        documents: list[dict[str, Any]] = []

        for run_document, version, result in rows:
            fragments = list(result.payload.get("fragments") or [])
            title = _document_title(fragments, run_document.relative_path)
            document_id = f"document:{run_document.id}"
            leaf_fragments = _leaf_fragments(fragments)

            document_node = {
                "id": document_id,
                "type": "document",
                "label": title,
                "parent_id": None,
                "depth": 0,
                "expandable": True,
                "source": {
                    "path": run_document.relative_path,
                    "sha256": version.content_sha256,
                },
                "metadata": {
                    "position": run_document.position,
                    "leaf_fragment_count": len(leaf_fragments),
                },
            }
            nodes.append(document_node)

            document = {
                "id": document_id,
                "title": title,
                "relative_path": run_document.relative_path,
                "fragments": leaf_fragments,
                "fragment_node_ids": {},
            }
            documents.append(document)

            section_ids: dict[tuple[str, ...], str] = {}
            section_first_line: dict[tuple[str, ...], int] = {}

            for fragment in fragments:
                hierarchy = [
                    _clean_text(str(item))
                    for item in fragment.get("hierarchy_path") or []
                    if _clean_text(str(item))
                ]
                if hierarchy and hierarchy[0].casefold() == title.casefold():
                    hierarchy = hierarchy[1:]
                line = int((fragment.get("source_position") or {}).get("line_start") or 0)
                for length in range(1, len(hierarchy) + 1):
                    path_tuple = tuple(hierarchy[:length])
                    previous = section_first_line.get(path_tuple)
                    if previous is None or line < previous:
                        section_first_line[path_tuple] = line

            for path_tuple, first_line in sorted(
                section_first_line.items(),
                key=lambda item: (item[1], len(item[0]), item[0]),
            ):
                section_id = (
                    f"section:{run_document.id}:"
                    f"{_stable_suffix([*path_tuple])}"
                )
                section_ids[path_tuple] = section_id
                parent_path = path_tuple[:-1]
                parent_id = section_ids.get(parent_path, document_id)
                nodes.append(
                    {
                        "id": section_id,
                        "type": "section",
                        "label": path_tuple[-1],
                        "parent_id": parent_id,
                        "depth": len(path_tuple),
                        "expandable": True,
                        "source": {
                            "path": run_document.relative_path,
                            "line_start": first_line,
                            "hierarchy_path": list(path_tuple),
                        },
                        "metadata": {},
                    }
                )
                edges.append(
                    {
                        "id": f"contains:{parent_id}:{section_id}",
                        "type": "contains",
                        "source": parent_id,
                        "target": section_id,
                        "confidence": 1.0,
                        "evidence": [],
                    }
                )

            previous_fragment_id: str | None = None
            for fragment in leaf_fragments:
                fragment_source_id = str(fragment.get("id"))
                fragment_id = f"fragment:{run_document.id}:{fragment_source_id}"
                hierarchy = [
                    _clean_text(str(item))
                    for item in fragment.get("hierarchy_path") or []
                    if _clean_text(str(item))
                ]
                if hierarchy and hierarchy[0].casefold() == title.casefold():
                    hierarchy = hierarchy[1:]
                parent_id = section_ids.get(tuple(hierarchy), document_id)
                source_position = fragment.get("source_position") or {}
                label = _clean_text(str(fragment.get("text") or ""))
                nodes.append(
                    {
                        "id": fragment_id,
                        "type": "fragment",
                        "label": label,
                        "parent_id": parent_id,
                        "depth": len(hierarchy) + 1,
                        "expandable": False,
                        "source": {
                            "path": run_document.relative_path,
                            "fragment_id": fragment_source_id,
                            "line_start": source_position.get("line_start"),
                            "line_end": source_position.get("line_end"),
                            "hierarchy_path": hierarchy,
                        },
                        "metadata": {
                            "fragment_type": fragment.get("fragment_type"),
                            "segmentation_method": fragment.get("segmentation_method"),
                            "segmentation_confidence": fragment.get(
                                "segmentation_confidence"
                            ),
                        },
                    }
                )
                document["fragment_node_ids"][fragment_source_id] = fragment_id
                edges.append(
                    {
                        "id": f"contains:{parent_id}:{fragment_id}",
                        "type": "contains",
                        "source": parent_id,
                        "target": fragment_id,
                        "confidence": 1.0,
                        "evidence": [],
                    }
                )
                if previous_fragment_id is not None:
                    edges.append(
                        {
                            "id": f"document-order:{previous_fragment_id}:{fragment_id}",
                            "type": "document_order",
                            "source": previous_fragment_id,
                            "target": fragment_id,
                            "confidence": 1.0,
                            "evidence": [],
                        }
                    )
                previous_fragment_id = fragment_id

        DocumentationGraphBuilder._append_reference_edges(documents, edges)
        edge_counts = Counter(edge["type"] for edge in edges)
        node_counts = Counter(node["type"] for node in nodes)

        return {
            "schema_version": "1.0",
            "source_run": {
                "id": run.id,
                "corpus_sha256": run.corpus_sha256,
            },
            "builder": {
                "version": GRAPH_BUILDER_VERSION,
                "config_sha256": GRAPH_CONFIG_SHA256,
                "config": GRAPH_CONFIG,
            },
            "semantic_analysis": {
                "status": "not_started",
                "message": (
                    "This graph represents document structure, source order and explicit "
                    "document references. It does not yet assert process-flow dependencies."
                ),
            },
            "summary": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "node_types": dict(sorted(node_counts.items())),
                "edge_types": dict(sorted(edge_counts.items())),
            },
            "nodes": nodes,
            "edges": edges,
        }

    @staticmethod
    def _append_reference_edges(
        documents: list[dict[str, Any]], edges: list[dict[str, Any]]
    ) -> None:
        for source in documents:
            for target in documents:
                if source["id"] == target["id"]:
                    continue
                target_title = target["title"]
                if len(target_title) < 8:
                    continue

                evidence: list[dict[str, Any]] = []
                for fragment in source["fragments"]:
                    text = _clean_text(str(fragment.get("text") or ""))
                    if target_title.casefold() not in text.casefold():
                        continue
                    fragment_source_id = str(fragment.get("id"))
                    evidence.append(
                        {
                            "node_id": source["fragment_node_ids"].get(
                                fragment_source_id
                            ),
                            "fragment_id": fragment_source_id,
                            "quote": text[:280],
                        }
                    )

                if evidence:
                    edges.append(
                        {
                            "id": f"references:{source['id']}:{target['id']}",
                            "type": "references",
                            "source": source["id"],
                            "target": target["id"],
                            "confidence": 1.0,
                            "evidence": evidence,
                        }
                    )

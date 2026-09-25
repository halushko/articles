from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from .models import CandidateDefinition, ProcessEdge, ProcessGraph, ProcessNode

SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    stripped = stripped.removeprefix("|").removesuffix("|")
    return [
        cell.replace(r"\|", "|").strip() for cell in re.split(r"(?<!\\)\|", stripped)
    ]


def _normalise_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _is_separator(row: list[str]) -> bool:
    return bool(row) and all(
        SEPARATOR_CELL_RE.match(cell.replace(" ", "")) for cell in row
    )


def _extract_tables(text: str) -> list[list[dict[str, str]]]:
    lines = text.splitlines()
    tables: list[list[dict[str, str]]] = []
    index = 0

    while index + 1 < len(lines):
        header = _split_row(lines[index])
        separator = _split_row(lines[index + 1])
        if (
            "|" not in lines[index]
            or not _is_separator(separator)
            or len(header) != len(separator)
        ):
            index += 1
            continue

        headers = [_normalise_header(cell) for cell in header]
        rows: list[dict[str, str]] = []
        index += 2
        while index < len(lines) and "|" in lines[index] and lines[index].strip():
            cells = _split_row(lines[index])
            if len(cells) != len(headers):
                break
            rows.append(dict(zip(headers, cells)))
            index += 1
        tables.append(rows)

    return tables


def _split_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(";") if item.strip())


class MarkdownProcessGraphParser:
    """Read an explicit process graph from Markdown tables.

    Required node columns: ``id``, ``operation``, ``role``, ``system``.
    Required edge columns: ``source``, ``target``. Optional columns are
    ``id``, ``type`` and ``condition``.
    """

    NODE_COLUMNS: ClassVar[set[str]] = {"id", "operation", "role", "system"}
    EDGE_COLUMNS: ClassVar[set[str]] = {"source", "target"}
    CANDIDATE_COLUMNS: ClassVar[set[str]] = {
        "candidate_id",
        "target_level",
        "members",
    }

    def parse_file(self, path: str | Path) -> ProcessGraph:
        source = Path(path)
        return self.parse(source.read_text(encoding="utf-8"))

    def parse(self, text: str) -> ProcessGraph:
        node_rows: list[dict[str, str]] | None = None
        edge_rows: list[dict[str, str]] | None = None

        for table in _extract_tables(text):
            if not table:
                continue
            columns = set(table[0])
            if self.NODE_COLUMNS <= columns:
                if node_rows is not None:
                    raise ValueError(
                        "Markdown contains more than one process-node table"
                    )
                node_rows = table
            elif self.EDGE_COLUMNS <= columns:
                if edge_rows is not None:
                    raise ValueError(
                        "Markdown contains more than one process-edge table"
                    )
                edge_rows = table

        if node_rows is None:
            raise ValueError(
                "Markdown must contain a node table with columns: id, operation, role, system"
            )
        if edge_rows is None:
            raise ValueError(
                "Markdown must contain an edge table with columns: source, target"
            )

        nodes: dict[str, ProcessNode] = {}
        for row in node_rows:
            node_id = row["id"].strip()
            if node_id in nodes:
                raise ValueError(f"Duplicate process node id: {node_id}")
            source_fragments = _split_values(row.get("source_fragment_ids", ""))
            nodes[node_id] = ProcessNode(
                id=node_id,
                operation=row["operation"].strip(),
                role=row["role"].strip(),
                system=row["system"].strip(),
                source_fragment_ids=source_fragments,
            )

        edges: list[ProcessEdge] = []
        seen_edge_ids: set[str] = set()
        for row_number, row in enumerate(edge_rows, start=1):
            edge_id = row.get("id", "").strip() or f"e{row_number}"
            if edge_id in seen_edge_ids:
                raise ValueError(f"Duplicate process edge id: {edge_id}")
            seen_edge_ids.add(edge_id)
            edges.append(
                ProcessEdge(
                    id=edge_id,
                    source=row["source"].strip(),
                    target=row["target"].strip(),
                    edge_type=row.get("type", "").strip() or "sequence",
                    condition=row.get("condition", "").strip() or None,
                )
            )

        return ProcessGraph(nodes=nodes, edges=tuple(edges), level=0)

    def parse_candidates(
        self,
        text: str,
        graph: ProcessGraph,
    ) -> tuple[CandidateDefinition, ...]:
        candidate_rows: list[dict[str, str]] | None = None
        for table in _extract_tables(text):
            if not table:
                continue
            columns = set(table[0])
            if self.CANDIDATE_COLUMNS <= columns:
                if candidate_rows is not None:
                    raise ValueError("Markdown contains more than one candidate table")
                candidate_rows = table

        if candidate_rows is None:
            return ()

        definitions: list[CandidateDefinition] = []
        seen_ids: set[str] = set()
        for row in candidate_rows:
            candidate_id = row["candidate_id"].strip()
            if candidate_id in seen_ids:
                raise ValueError(f"Duplicate candidate id: {candidate_id}")
            seen_ids.add(candidate_id)
            members = _split_values(row["members"])
            unknown = sorted(set(members) - set(graph.nodes))
            if unknown:
                raise ValueError(
                    f"Candidate {candidate_id!r} references unknown nodes: {', '.join(unknown)}"
                )
            definitions.append(
                CandidateDefinition(
                    id=candidate_id,
                    name=row.get("name", "").strip() or candidate_id,
                    target_level=int(row["target_level"].strip()),
                    atomic_node_ids=members,
                    purpose=row.get("purpose", "").strip().lower() or "aggregation",
                )
            )

        return tuple(definitions)

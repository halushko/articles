from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProcessNode:
    id: str
    operation: str
    role: str
    system: str
    member_ids: tuple[str, ...] = ()
    source_fragment_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Process node id must not be empty")
        if not self.operation.strip():
            raise ValueError(f"Process node {self.id!r} must have an operation name")
        if not self.role.strip():
            raise ValueError(f"Process node {self.id!r} must have a role")
        if not self.system.strip():
            raise ValueError(f"Process node {self.id!r} must have a system")

        if not self.member_ids:
            object.__setattr__(self, "member_ids", (self.id,))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["member_ids"] = list(self.member_ids)
        data["source_fragment_ids"] = list(self.source_fragment_ids)
        return data


@dataclass(frozen=True)
class ProcessEdge:
    id: str
    source: str
    target: str
    edge_type: str = "sequence"
    condition: str | None = None
    original_edge_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Process edge id must not be empty")
        if not self.source.strip() or not self.target.strip():
            raise ValueError(f"Process edge {self.id!r} must have source and target")
        if self.source == self.target:
            raise ValueError(f"Self-loop is not supported for edge {self.id!r}")
        if not self.original_edge_ids:
            object.__setattr__(self, "original_edge_ids", (self.id,))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["original_edge_ids"] = list(self.original_edge_ids)
        return data


@dataclass(frozen=True)
class ProcessGraph:
    nodes: dict[str, ProcessNode]
    edges: tuple[ProcessEdge, ...]
    level: int = 0

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("Process graph must contain at least one node")

        unknown: list[str] = []
        for edge in self.edges:
            if edge.source not in self.nodes:
                unknown.append(f"{edge.id}: source {edge.source}")
            if edge.target not in self.nodes:
                unknown.append(f"{edge.id}: target {edge.target}")
        if unknown:
            raise ValueError("Edges reference unknown nodes: " + ", ".join(unknown))

    def outgoing(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            result[edge.source].append(edge.target)
        return {key: tuple(sorted(values)) for key, values in result.items()}

    def incoming(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            result[edge.target].append(edge.source)
        return {key: tuple(sorted(values)) for key, values in result.items()}

    def undirected_neighbors(self) -> dict[str, set[str]]:
        result = {node_id: set() for node_id in self.nodes}
        for edge in self.edges:
            result[edge.source].add(edge.target)
            result[edge.target].add(edge.source)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "nodes": [self.nodes[node_id].to_dict() for node_id in sorted(self.nodes)],
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass(frozen=True)
class ScoreWeights:
    text: float = 0.2
    context: float = 0.4
    flow: float = 0.4

    def __post_init__(self) -> None:
        values = (self.text, self.context, self.flow)
        if any(value < 0 for value in values):
            raise ValueError("Score weights must be non-negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("Score weights must sum to 1")


@dataclass(frozen=True)
class AggregationLevelConfig:
    q_min: float
    max_candidate_nodes: int = 5
    radius: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.q_min <= 1:
            raise ValueError("q_min must be between 0 and 1")
        if self.max_candidate_nodes < 2:
            raise ValueError("max_candidate_nodes must be at least 2")
        if self.radius < 1:
            raise ValueError("radius must be at least 1")


@dataclass(frozen=True)
class AggregationConfig:
    levels: tuple[AggregationLevelConfig, ...]
    weights: ScoreWeights = field(default_factory=ScoreWeights)

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("At least one aggregation level must be configured")


@dataclass(frozen=True)
class CandidateDefinition:
    id: str
    name: str
    target_level: int
    atomic_node_ids: tuple[str, ...]
    purpose: str = "aggregation"

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Candidate id must not be empty")
        if self.target_level < 1:
            raise ValueError("Candidate target_level must be at least 1")
        if len(set(self.atomic_node_ids)) < 2:
            raise ValueError(f"Candidate {self.id!r} must contain at least two nodes")
        if self.purpose not in {"aggregation", "diagnostic"}:
            raise ValueError("Candidate purpose must be 'aggregation' or 'diagnostic'")


@dataclass(frozen=True)
class CandidateScore:
    visible_node_ids: tuple[str, ...]
    atomic_node_ids: tuple[str, ...]
    s_txt: float
    s_ctx: float
    s_flow: float
    q: float
    branch_integrity: bool
    rejection_reason: str | None = None
    candidate_id: str | None = None
    candidate_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AggregationLevelResult:
    source_level: int
    target_level: int
    graph: ProcessGraph
    accepted_candidates: tuple[CandidateScore, ...]
    rejected_candidates: tuple[CandidateScore, ...]
    mapping: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_level": self.source_level,
            "target_level": self.target_level,
            "graph": self.graph.to_dict(),
            "accepted_candidates": [
                item.to_dict() for item in self.accepted_candidates
            ],
            "rejected_candidates": [
                item.to_dict() for item in self.rejected_candidates
            ],
            "mapping": dict(sorted(self.mapping.items())),
        }


@dataclass(frozen=True)
class AggregationRun:
    base_graph: ProcessGraph
    levels: tuple[AggregationLevelResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_graph": self.base_graph.to_dict(),
            "levels": [level.to_dict() for level in self.levels],
        }

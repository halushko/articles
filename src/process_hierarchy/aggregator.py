from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace

from .branching import BranchIntegrityValidator
from .candidates import generate_connected_candidates
from .models import (
    AggregationConfig,
    AggregationLevelConfig,
    AggregationLevelResult,
    AggregationRun,
    CandidateDefinition,
    CandidateScore,
    ProcessEdge,
    ProcessGraph,
    ProcessNode,
)
from .scoring import CandidateScorer

MISSING_CONTEXT_VALUES = {
    "",
    "n/a",
    "none",
    "not recorded",
    "unknown",
    "unspecified",
}


def _dominant_or_mixed(values: list[str]) -> str:
    known = [
        value
        for value in values
        if value.strip().casefold() not in MISSING_CONTEXT_VALUES
    ]
    if not known:
        return "Unknown"
    distinct = sorted(set(known))
    if len(distinct) == 1:
        return distinct[0]
    dominant, _ = Counter(known).most_common(1)[0]
    return f"mixed (dominant: {dominant})"


class HierarchicalAggregator:
    def __init__(self, config: AggregationConfig) -> None:
        self.config = config

    def run(
        self,
        base_graph: ProcessGraph,
        candidate_definitions: tuple[CandidateDefinition, ...] = (),
    ) -> AggregationRun:
        if base_graph.level != 0:
            raise ValueError("Hierarchical aggregation must start from L0")

        scorer = CandidateScorer(base_graph, self.config.weights)
        branch_validator = BranchIntegrityValidator(base_graph)
        current_graph = base_graph
        results: list[AggregationLevelResult] = []

        for level_config in self.config.levels:
            result = self._aggregate_once(
                base_graph=base_graph,
                current_graph=current_graph,
                level_config=level_config,
                scorer=scorer,
                branch_validator=branch_validator,
                candidate_definitions=tuple(
                    item
                    for item in candidate_definitions
                    if item.target_level == current_graph.level + 1
                ),
            )
            if not result.accepted_candidates:
                break
            results.append(result)
            current_graph = result.graph

        return AggregationRun(base_graph=base_graph, levels=tuple(results))

    def _aggregate_once(
        self,
        *,
        base_graph: ProcessGraph,
        current_graph: ProcessGraph,
        level_config: AggregationLevelConfig,
        scorer: CandidateScorer,
        branch_validator: BranchIntegrityValidator,
        candidate_definitions: tuple[CandidateDefinition, ...],
    ) -> AggregationLevelResult:
        if candidate_definitions:
            candidate_items = self._defined_candidates(
                current_graph, candidate_definitions
            )
        else:
            candidate_items = [
                (visible_node_ids, None)
                for visible_node_ids in generate_connected_candidates(
                    current_graph,
                    max_candidate_nodes=level_config.max_candidate_nodes,
                    radius=level_config.radius,
                    max_candidates=level_config.max_candidates,
                )
            ]

        eligible: list[CandidateScore] = []
        rejected: list[CandidateScore] = []

        for visible_node_ids, definition in candidate_items:
            if definition is None:
                atomic_node_ids = self._atomic_ids(current_graph, visible_node_ids)
            else:
                atomic_node_ids = tuple(sorted(definition.atomic_node_ids))
            integrity_ok, integrity_reason = branch_validator.validate(
                set(atomic_node_ids)
            )
            score = scorer.score(
                visible_node_ids,
                atomic_node_ids,
                branch_integrity=integrity_ok,
                rejection_reason=integrity_reason,
                candidate_id=definition.id if definition else None,
                candidate_name=definition.name if definition else None,
                selection_basis=(
                    "explicit_source_structure"
                    if definition and definition.purpose == "structural"
                    else "score"
                ),
            )
            if definition and definition.purpose == "diagnostic":
                rejected.append(
                    score
                    if not integrity_ok
                    else replace(score, rejection_reason="diagnostic_only")
                )
            elif definition and not visible_node_ids:
                rejected.append(
                    replace(
                        score, rejection_reason="not_representable_at_current_level"
                    )
                )
            elif not integrity_ok:
                rejected.append(score)
            elif (
                not definition or definition.purpose != "structural"
            ) and score.q < level_config.q_min:
                rejected.append(replace(score, rejection_reason="below_q_min"))
            else:
                eligible.append(score)

        eligible.sort(
            key=lambda score: (
                -score.q,
                -score.s_flow,
                -len(score.visible_node_ids),
                score.visible_node_ids,
            )
        )

        accepted: list[CandidateScore] = []
        used_visible_nodes: set[str] = set()
        for score in eligible:
            visible_nodes = set(score.visible_node_ids)
            if visible_nodes & used_visible_nodes:
                rejected.append(
                    replace(score, rejection_reason="overlaps_selected_candidate")
                )
                continue
            accepted.append(score)
            used_visible_nodes.update(visible_nodes)

        next_graph, mapping = self._collapse_graph(
            base_graph=base_graph,
            current_graph=current_graph,
            accepted=accepted,
        )
        rejected.sort(
            key=lambda score: (score.visible_node_ids, score.rejection_reason or "")
        )

        return AggregationLevelResult(
            source_level=current_graph.level,
            target_level=current_graph.level + 1,
            graph=next_graph,
            accepted_candidates=tuple(accepted),
            rejected_candidates=tuple(rejected),
            mapping=mapping,
        )

    @staticmethod
    def _defined_candidates(
        graph: ProcessGraph,
        definitions: tuple[CandidateDefinition, ...],
    ) -> list[tuple[tuple[str, ...], CandidateDefinition]]:
        result: list[tuple[tuple[str, ...], CandidateDefinition]] = []
        for definition in definitions:
            requested = set(definition.atomic_node_ids)
            visible: list[str] = []
            representable = True
            covered: set[str] = set()

            for node_id, node in graph.nodes.items():
                members = set(node.member_ids)
                intersection = members & requested
                if not intersection:
                    continue
                if intersection != members:
                    representable = False
                    break
                visible.append(node_id)
                covered.update(members)

            if covered != requested:
                representable = False
            result.append((tuple(sorted(visible)) if representable else (), definition))
        return result

    @staticmethod
    def _atomic_ids(
        graph: ProcessGraph,
        visible_node_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        result: set[str] = set()
        for node_id in visible_node_ids:
            result.update(graph.nodes[node_id].member_ids)
        return tuple(sorted(result))

    def _collapse_graph(
        self,
        *,
        base_graph: ProcessGraph,
        current_graph: ProcessGraph,
        accepted: list[CandidateScore],
    ) -> tuple[ProcessGraph, dict[str, str]]:
        target_level = current_graph.level + 1
        mapping: dict[str, str] = {}
        target_nodes: dict[str, ProcessNode] = {}

        for index, candidate in enumerate(accepted, start=1):
            target_id = f"L{target_level}_A{index:03d}"
            for source_id in candidate.visible_node_ids:
                mapping[source_id] = target_id
            target_nodes[target_id] = self._make_aggregate_node(
                target_id=target_id,
                atomic_node_ids=candidate.atomic_node_ids,
                base_graph=base_graph,
                candidate_name=candidate.candidate_name,
            )

        singleton_index = 0
        for source_id in sorted(current_graph.nodes):
            if source_id in mapping:
                continue
            singleton_index += 1
            target_id = f"L{target_level}_S{singleton_index:03d}"
            mapping[source_id] = target_id
            source_node = current_graph.nodes[source_id]
            target_nodes[target_id] = ProcessNode(
                id=target_id,
                operation=source_node.operation,
                role=source_node.role,
                system=source_node.system,
                member_ids=source_node.member_ids,
                source_fragment_ids=source_node.source_fragment_ids,
            )

        grouped_edges: dict[tuple[str, str], list[ProcessEdge]] = defaultdict(list)
        for edge in current_graph.edges:
            source = mapping[edge.source]
            target = mapping[edge.target]
            if source == target:
                continue
            grouped_edges[(source, target)].append(edge)

        target_edges: list[ProcessEdge] = []
        for index, ((source, target), edges) in enumerate(
            sorted(grouped_edges.items()), start=1
        ):
            edge_types = sorted({edge.edge_type for edge in edges})
            conditions = sorted({edge.condition for edge in edges if edge.condition})
            original_ids = sorted(
                {
                    original_id
                    for edge in edges
                    for original_id in edge.original_edge_ids
                }
            )
            target_edges.append(
                ProcessEdge(
                    id=f"L{target_level}_E{index:03d}",
                    source=source,
                    target=target,
                    edge_type=edge_types[0] if len(edge_types) == 1 else "mixed",
                    condition=" | ".join(conditions) if conditions else None,
                    original_edge_ids=tuple(original_ids),
                )
            )

        return (
            ProcessGraph(
                nodes=target_nodes, edges=tuple(target_edges), level=target_level
            ),
            mapping,
        )

    @staticmethod
    def _make_aggregate_node(
        *,
        target_id: str,
        atomic_node_ids: tuple[str, ...],
        base_graph: ProcessGraph,
        candidate_name: str | None,
    ) -> ProcessNode:
        atomic_nodes = [base_graph.nodes[node_id] for node_id in atomic_node_ids]
        operations = [node.operation for node in atomic_nodes]
        roles = [node.role for node in atomic_nodes]
        systems = [node.system for node in atomic_nodes]
        source_fragments = sorted(
            {
                fragment_id
                for node in atomic_nodes
                for fragment_id in node.source_fragment_ids
            }
        )
        return ProcessNode(
            id=target_id,
            operation=candidate_name
            or HierarchicalAggregator._automatic_aggregate_name(operations),
            role=_dominant_or_mixed(roles),
            system=_dominant_or_mixed(systems),
            member_ids=atomic_node_ids,
            source_fragment_ids=tuple(source_fragments),
        )

    @staticmethod
    def _automatic_aggregate_name(operations: list[str]) -> str:
        if len(operations) == 2:
            return f"Stage: {operations[0]} → {operations[1]}"
        return f"Stage: {operations[0]} → {operations[-1]} ({len(operations)} actions)"

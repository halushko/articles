from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .models import ProcessGraph


@dataclass(frozen=True)
class BranchRegion:
    split: str
    join: str
    branches: tuple[frozenset[str], ...]


def _distances(start: str, outgoing: dict[str, tuple[str, ...]]) -> dict[str, int]:
    result = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for target in outgoing[current]:
            if target in result:
                continue
            result[target] = result[current] + 1
            queue.append(target)
    return result


def _ancestors(target: str, incoming: dict[str, tuple[str, ...]]) -> set[str]:
    result = {target}
    queue = deque([target])
    while queue:
        current = queue.popleft()
        for source in incoming[current]:
            if source in result:
                continue
            result.add(source)
            queue.append(source)
    return result


def _branch_nodes(
    start: str,
    join: str,
    outgoing: dict[str, tuple[str, ...]],
    join_ancestors: set[str],
) -> frozenset[str]:
    result: set[str] = set()
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == join or current in result or current not in join_ancestors:
            continue
        result.add(current)
        queue.extend(outgoing[current])
    return frozenset(result)


def detect_branch_regions(graph: ProcessGraph) -> tuple[BranchRegion, ...]:
    """Detect simple split-join regions in a directed process graph.

    For every node with multiple outgoing edges, the nearest common reachable
    node with multiple incoming edges is treated as its matching join.
    """

    outgoing = graph.outgoing()
    incoming = graph.incoming()
    regions: list[BranchRegion] = []

    for split in sorted(graph.nodes):
        successors = tuple(sorted(set(outgoing[split])))
        if len(successors) < 2:
            continue

        distances = [_distances(successor, outgoing) for successor in successors]
        common = set(distances[0])
        for item in distances[1:]:
            common &= item.keys()
        common = {
            node_id
            for node_id in common
            if node_id != split and len(set(incoming[node_id])) >= 2
        }
        if not common:
            continue

        join = min(
            common,
            key=lambda node_id: (
                max(item[node_id] for item in distances),
                sum(item[node_id] for item in distances),
                node_id,
            ),
        )
        join_ancestors = _ancestors(join, incoming)
        branches = tuple(
            _branch_nodes(successor, join, outgoing, join_ancestors)
            for successor in successors
        )
        if all(branches):
            regions.append(BranchRegion(split=split, join=join, branches=branches))

    return tuple(regions)


class BranchIntegrityValidator:
    def __init__(self, base_graph: ProcessGraph) -> None:
        self.regions = detect_branch_regions(base_graph)

    def validate(self, atomic_node_ids: set[str]) -> tuple[bool, str | None]:
        for region in self.regions:
            touched = [branch for branch in region.branches if branch & atomic_node_ids]
            if not touched:
                continue

            includes_split = region.split in atomic_node_ids
            includes_join = region.join in atomic_node_ids

            if len(touched) == 1:
                if includes_split or includes_join:
                    return (
                        False,
                        f"partial branch combined with gateway in {region.split}->{region.join}",
                    )
                continue

            if len(touched) != len(region.branches):
                return (
                    False,
                    f"only part of the alternatives is covered in {region.split}->{region.join}",
                )

            if any(not branch <= atomic_node_ids for branch in region.branches):
                return (
                    False,
                    (
                        "one or more alternatives are only partially covered in "
                        f"{region.split}->{region.join}"
                    ),
                )

            if includes_split != includes_join:
                return (
                    False,
                    f"split and join must be included together in {region.split}->{region.join}",
                )

        return True, None

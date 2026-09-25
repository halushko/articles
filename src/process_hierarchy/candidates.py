from __future__ import annotations

from collections import deque

from .models import ProcessGraph


def _nodes_within_radius(
    seed: str,
    neighbors: dict[str, set[str]],
    radius: int,
) -> set[str]:
    distances = {seed: 0}
    queue = deque([seed])

    while queue:
        current = queue.popleft()
        if distances[current] == radius:
            continue
        for neighbor in sorted(neighbors[current]):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)

    return set(distances)


def generate_connected_candidates(
    graph: ProcessGraph,
    *,
    max_candidate_nodes: int,
    radius: int,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate connected visible-node candidates inside local neighborhoods."""

    neighbors = graph.undirected_neighbors()
    candidates: set[frozenset[str]] = set()

    for seed in sorted(graph.nodes):
        allowed = _nodes_within_radius(seed, neighbors, radius)
        queue: deque[frozenset[str]] = deque([frozenset({seed})])
        visited: set[frozenset[str]] = {frozenset({seed})}

        while queue:
            current = queue.popleft()
            if len(current) >= 2:
                candidates.add(current)
            if len(current) == max_candidate_nodes:
                continue

            frontier: set[str] = set()
            for node_id in current:
                frontier.update(neighbors[node_id])
            frontier &= allowed
            frontier -= current

            for node_id in sorted(frontier):
                expanded = frozenset((*current, node_id))
                if expanded in visited:
                    continue
                visited.add(expanded)
                queue.append(expanded)

    ordered = sorted(candidates, key=lambda item: (len(item), tuple(sorted(item))))
    return tuple(tuple(sorted(item)) for item in ordered)

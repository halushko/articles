from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from .models import CandidateScore, ProcessGraph, ScoreWeights


def normalize_operation_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


MISSING_CONTEXT_VALUES = {
    "",
    "n/a",
    "none",
    "not recorded",
    "unknown",
    "unspecified",
}


def _dominant_share(values: list[str]) -> float:
    if not values:
        return 0.0
    known = [
        value
        for value in values
        if normalize_operation_name(value) not in MISSING_CONTEXT_VALUES
    ]
    if not known:
        return 0.0
    count = Counter(known).most_common(1)[0][1]
    # Missing values do not form a shared context and still reduce coverage.
    # One known role among five actions therefore contributes 1/5, not 1.0.
    return count / len(values)


class CandidateScorer:
    """Compute S_txt, S_ctx, S_flow and the integral score Q.

    Text and context components are calculated over atomic L0 nodes. Flow is
    also evaluated against the original graph. This keeps scores comparable
    when a candidate at a higher level contains previously aggregated nodes.
    """

    def __init__(self, base_graph: ProcessGraph, weights: ScoreWeights) -> None:
        if base_graph.level != 0:
            raise ValueError("CandidateScorer requires the L0 base graph")
        self.base_graph = base_graph
        self.weights = weights
        self._node_ids = tuple(sorted(base_graph.nodes))
        self._node_index = {
            node_id: index for index, node_id in enumerate(self._node_ids)
        }

        names = [
            normalize_operation_name(base_graph.nodes[node_id].operation)
            for node_id in self._node_ids
        ]
        self._vectors = self._build_tfidf_vectors(names)

    def score(
        self,
        visible_node_ids: tuple[str, ...],
        atomic_node_ids: tuple[str, ...],
        *,
        branch_integrity: bool,
        rejection_reason: str | None = None,
        candidate_id: str | None = None,
        candidate_name: str | None = None,
        selection_basis: str = "score",
    ) -> CandidateScore:
        atomic_node_ids = tuple(sorted(set(atomic_node_ids)))
        if len(atomic_node_ids) < 2:
            raise ValueError("A candidate must contain at least two atomic nodes")

        s_txt = self._text_similarity(atomic_node_ids)
        s_ctx = self._context_similarity(atomic_node_ids)
        s_flow = self._flow_coherence(set(atomic_node_ids))
        q = (
            self.weights.text * s_txt
            + self.weights.context * s_ctx
            + self.weights.flow * s_flow
        )

        return CandidateScore(
            visible_node_ids=tuple(sorted(visible_node_ids)),
            atomic_node_ids=atomic_node_ids,
            s_txt=s_txt,
            s_ctx=s_ctx,
            s_flow=s_flow,
            q=q,
            branch_integrity=branch_integrity,
            rejection_reason=rejection_reason,
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            selection_basis=selection_basis,
        )

    def _text_similarity(self, atomic_node_ids: tuple[str, ...]) -> float:
        indexes = [self._node_index[node_id] for node_id in atomic_node_ids]
        similarities: list[float] = []
        for left_position, left_index in enumerate(indexes):
            for right_index in indexes[left_position + 1 :]:
                left = self._vectors[left_index]
                right = self._vectors[right_index]
                if len(left) > len(right):
                    left, right = right, left
                similarities.append(
                    sum(value * right.get(term, 0.0) for term, value in left.items())
                )
        return sum(similarities) / len(similarities)

    @staticmethod
    def _build_tfidf_vectors(names: list[str]) -> list[dict[str, float]]:
        term_counts: list[Counter[str]] = []
        document_frequency: Counter[str] = Counter()

        for name in names:
            terms: list[str] = []
            for length in range(3, 6):
                terms.extend(
                    name[index : index + length]
                    for index in range(max(0, len(name) - length + 1))
                )
            if not terms:
                raise ValueError(
                    "Could not build character TF-IDF vectors. "
                    "Operation names must contain at least three characters."
                )
            counts = Counter(terms)
            term_counts.append(counts)
            document_frequency.update(counts.keys())

        document_count = len(names)
        idf = {
            term: math.log((1 + document_count) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }

        vectors: list[dict[str, float]] = []
        for counts in term_counts:
            raw = {term: count * idf[term] for term, count in counts.items()}
            norm = math.sqrt(sum(value * value for value in raw.values()))
            vectors.append({term: value / norm for term, value in raw.items()})
        return vectors

    def _context_similarity(self, atomic_node_ids: tuple[str, ...]) -> float:
        roles = [self.base_graph.nodes[node_id].role for node_id in atomic_node_ids]
        systems = [self.base_graph.nodes[node_id].system for node_id in atomic_node_ids]
        return (_dominant_share(roles) + _dominant_share(systems)) / 2

    def _flow_coherence(self, atomic_node_ids: set[str]) -> float:
        internal_edges = 0
        boundary_edges = 0

        for edge in self.base_graph.edges:
            source_inside = edge.source in atomic_node_ids
            target_inside = edge.target in atomic_node_ids
            if source_inside and target_inside:
                internal_edges += 1
            elif source_inside != target_inside:
                boundary_edges += 1

        denominator = 2 * internal_edges + boundary_edges
        if denominator == 0:
            return 0.0
        return 2 * internal_edges / denominator

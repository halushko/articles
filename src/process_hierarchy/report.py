from __future__ import annotations

from .models import AggregationRun, CandidateScore


def _candidate_row(score: CandidateScore, decision: str) -> str:
    candidate = score.candidate_id or ", ".join(score.visible_node_ids)
    members = ", ".join(score.atomic_node_ids)
    return (
        f"| {candidate} | {members} | {score.s_txt:.3f} | {score.s_ctx:.3f} | "
        f"{score.s_flow:.3f} | {score.q:.3f} | {decision} |"
    )


def render_markdown_report(run: AggregationRun) -> str:
    lines = [
        "# Hierarchical Aggregation Report",
        "",
        (
            f"The base graph contains **{len(run.base_graph.nodes)} nodes** and "
            f"**{len(run.base_graph.edges)} edges**."
        ),
        "",
    ]

    if not run.levels:
        lines.append("No candidate passed the configured aggregation threshold.")
        return "\n".join(lines) + "\n"

    for level in run.levels:
        lines.extend(
            [
                f"## L{level.source_level} → L{level.target_level}",
                "",
                "| Candidate | Atomic members | S_txt | S_ctx | S_flow | Q | Decision |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for score in level.accepted_candidates:
            lines.append(_candidate_row(score, "aggregated"))
        for score in level.rejected_candidates:
            if score.candidate_id is None:
                continue
            lines.append(_candidate_row(score, score.rejection_reason or "rejected"))

        lines.extend(
            [
                "",
                (
                    f"Result: **{len(level.graph.nodes)} nodes**, "
                    f"**{len(level.graph.edges)} edges**."
                ),
                "",
                "### Level mapping",
                "",
                "| Source node | Target node |",
                "|---|---|",
            ]
        )
        for source, target in sorted(level.mapping.items()):
            lines.append(f"| {source} | {target} |")
        lines.append("")

    return "\n".join(lines)

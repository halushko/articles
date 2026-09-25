from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregator import HierarchicalAggregator
from .markdown_graph import MarkdownProcessGraphParser
from .models import AggregationConfig, AggregationLevelConfig, ScoreWeights
from .report import render_markdown_report


def _parse_number_list(
    value: str,
    converter: type[float | int],
) -> list[float] | list[int]:
    try:
        return [converter(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid comma-separated value: {value}"
        ) from exc


def _expand(
    values: list[float] | list[int],
    length: int,
    name: str,
) -> list[float] | list[int]:
    if len(values) == 1:
        return values * length
    if len(values) != length:
        raise ValueError(
            f"{name} must contain one value or {length} comma-separated values"
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a process graph from Markdown and perform hierarchical aggregation."
        )
    )
    parser.add_argument(
        "path", type=Path, help="English Markdown process-document path"
    )
    parser.add_argument("--output", type=Path, help="Write JSON result to this path")
    parser.add_argument(
        "--output-format",
        choices=("json", "markdown"),
        default="json",
        help="Result representation",
    )
    parser.add_argument("--q-min", default="0.65,0.50", help="Thresholds per level")
    parser.add_argument(
        "--max-candidate-nodes",
        default="5",
        help="One value or values per level",
    )
    parser.add_argument("--radius", default="3", help="One value or values per level")
    parser.add_argument("--weight-text", type=float, default=0.2)
    parser.add_argument("--weight-context", type=float, default=0.4)
    parser.add_argument("--weight-flow", type=float, default=0.4)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    q_values = _parse_number_list(args.q_min, float)
    level_count = len(q_values)
    max_nodes = _expand(
        _parse_number_list(args.max_candidate_nodes, int),
        level_count,
        "--max-candidate-nodes",
    )
    radii = _expand(_parse_number_list(args.radius, int), level_count, "--radius")

    source_text = args.path.read_text(encoding="utf-8")
    markdown_parser = MarkdownProcessGraphParser()
    graph = markdown_parser.parse(source_text)
    candidate_definitions = markdown_parser.parse_candidates(source_text, graph)
    config = AggregationConfig(
        weights=ScoreWeights(
            text=args.weight_text,
            context=args.weight_context,
            flow=args.weight_flow,
        ),
        levels=tuple(
            AggregationLevelConfig(
                q_min=float(q_values[index]),
                max_candidate_nodes=int(max_nodes[index]),
                radius=int(radii[index]),
            )
            for index in range(level_count)
        ),
    )
    result = HierarchicalAggregator(config).run(graph, candidate_definitions)
    if args.output_format == "markdown":
        payload = render_markdown_report(result).rstrip("\n")
    else:
        payload = json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )

    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()

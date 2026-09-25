from __future__ import annotations

import argparse
import json
from pathlib import Path

from .segmenter import Segmenter


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment documentation artifact into Fragment objects.")
    parser.add_argument("path", help="Path to markdown/plain text file")
    parser.add_argument("--artifact-id", default=None, help="Artifact identifier")
    parser.add_argument("--format", choices=["markdown", "text"], default="markdown")
    parser.add_argument("--pretty", action="store_true", help="Pretty JSON output")

    args = parser.parse_args()

    path = Path(args.path)
    artifact_id = args.artifact_id or path.stem
    text = path.read_text(encoding="utf-8")

    segmenter = Segmenter(artifact_format=args.format)
    fragments = segmenter.segment(artifact_id=artifact_id, text=text)

    payload = [fragment.to_dict() for fragment in fragments]
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()

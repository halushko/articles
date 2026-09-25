from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def corpus_sha256(items: Iterable[Mapping[str, str]]) -> str:
    canonical = sorted(
        ({"path": item["path"], "sha256": item["sha256"]} for item in items),
        key=lambda item: item["path"],
    )
    return sha256_json(canonical)

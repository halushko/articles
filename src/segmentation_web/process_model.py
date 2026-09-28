from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProcessModelSummary:
    id: str
    run_id: str
    level_count: int
    atomic_node_count: int
    cache_hit: bool
    payload: dict[str, Any]

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class SourcePosition:
    line_start: int
    line_end: int
    char_start: int
    char_end: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class RawBlock:
    id: str
    block_type: str
    text: str
    hierarchy_path: list[str]
    parent_block_id: Optional[str]
    source_position: SourcePosition
    structural_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class Fragment:
    id: str
    artifact_id: str
    parent_block_id: Optional[str]
    parent_fragment_id: Optional[str]

    text: str
    normalized_text: str

    fragment_type: str
    hierarchy_path: list[str]

    source_position: SourcePosition

    segmentation_method: str
    segmentation_trigger: str
    segmentation_confidence: float

    structural_fields: dict[str, Any] = field(default_factory=dict)
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_position"] = self.source_position.to_dict()
        return data

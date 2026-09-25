from __future__ import annotations

from typing import Any, Optional

from .models import Fragment, SourcePosition
from .normalizer import normalize_fragment_text


class FragmentFactory:
    def __init__(self) -> None:
        self.counter = 0

    def next_id(self, prefix: str = "frag") -> str:
        self.counter += 1
        return f"{prefix}_{self.counter:05d}"

    def create(
        self,
        *,
        artifact_id: str,
        parent_block_id: Optional[str],
        parent_fragment_id: Optional[str],
        text: str,
        fragment_type: str,
        hierarchy_path: list[str],
        source_position: SourcePosition,
        segmentation_method: str,
        segmentation_trigger: str,
        segmentation_confidence: float,
        structural_fields: Optional[dict[str, Any]] = None,
        status: str = "active",
    ) -> Fragment:
        return Fragment(
            id=self.next_id(),
            artifact_id=artifact_id,
            parent_block_id=parent_block_id,
            parent_fragment_id=parent_fragment_id,
            text=text.strip(),
            normalized_text=normalize_fragment_text(text),
            fragment_type=fragment_type,
            hierarchy_path=list(hierarchy_path),
            source_position=source_position,
            segmentation_method=segmentation_method,
            segmentation_trigger=segmentation_trigger,
            segmentation_confidence=segmentation_confidence,
            structural_fields=structural_fields or {},
            status=status,
        )

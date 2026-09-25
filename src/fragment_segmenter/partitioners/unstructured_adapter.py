from __future__ import annotations

from typing import Any

from ..models import RawBlock, SourcePosition
from .base import DocumentPartitioner


class UnstructuredPartitioner(DocumentPartitioner):
    """Optional adapter.

    This adapter is intentionally not used as the default MVP path.
    It is useful as a future extension for formats such as DOCX/PDF/HTML,
    but exact line/char source positions are better controlled by the
    rule-based markdown/plain-text partitioners.
    """

    def __init__(self, artifact_format: str = "markdown") -> None:
        self.artifact_format = artifact_format

    def partition(self, artifact_id: str, text: str) -> list[RawBlock]:
        try:
            if self.artifact_format == "markdown":
                from unstructured.partition.md import partition_md
                elements = partition_md(text=text)
            elif self.artifact_format == "text":
                from unstructured.partition.text import partition_text
                elements = partition_text(text=text, max_partition=None)
            else:
                raise ValueError(f"Unsupported format for UnstructuredPartitioner: {self.artifact_format}")
        except ImportError as exc:
            raise RuntimeError(
                "UnstructuredPartitioner requires optional dependency: pip install -e '.[unstructured]'"
            ) from exc

        blocks: list[RawBlock] = []
        for index, element in enumerate(elements, start=1):
            element_type = element.__class__.__name__
            element_text = str(element).strip()
            if not element_text:
                continue

            metadata: dict[str, Any] = {}
            if getattr(element, "metadata", None):
                metadata = element.metadata.to_dict()

            block_type = self.map_element_type(element_type)

            blocks.append(
                RawBlock(
                    id=f"block_{index:05d}",
                    block_type=block_type,
                    text=element_text,
                    hierarchy_path=[],
                    parent_block_id=metadata.get("parent_id"),
                    source_position=SourcePosition(
                        line_start=-1,
                        line_end=-1,
                        char_start=-1,
                        char_end=-1,
                    ),
                    structural_fields={
                        "source_parser": "unstructured",
                        "unstructured_element_type": element_type,
                        "unstructured_element_id": getattr(element, "id", None),
                        "unstructured_metadata": metadata,
                    },
                )
            )

        return blocks

    @staticmethod
    def map_element_type(element_type: str) -> str:
        mapping = {
            "Title": "heading",
            "NarrativeText": "paragraph",
            "ListItem": "list_item",
            "UncategorizedText": "paragraph",
            "CodeSnippet": "code_block",
        }
        return mapping.get(element_type, "paragraph")

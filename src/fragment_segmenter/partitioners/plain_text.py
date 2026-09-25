from __future__ import annotations

from ..detectors import detect_api_operation
from ..models import RawBlock, SourcePosition
from ..normalizer import normalize_document_text
from .base import DocumentPartitioner


class PlainTextPartitioner(DocumentPartitioner):
    def partition(self, artifact_id: str, text: str) -> list[RawBlock]:
        normalized = normalize_document_text(text)
        lines = normalized.splitlines(keepends=True)

        blocks: list[RawBlock] = []
        paragraph_lines: list[str] = []
        paragraph_line_start = 1
        paragraph_char_start = 0

        char_pos = 0
        block_counter = 0

        def flush_paragraph(end_line: int, end_char: int) -> None:
            nonlocal block_counter, paragraph_lines, paragraph_line_start, paragraph_char_start
            paragraph_text = "".join(paragraph_lines).strip()
            if not paragraph_text:
                paragraph_lines = []
                return

            block_counter += 1
            blocks.append(
                RawBlock(
                    id=f"block_{block_counter:05d}",
                    block_type="paragraph",
                    text=paragraph_text,
                    hierarchy_path=[],
                    parent_block_id=None,
                    source_position=SourcePosition(
                        line_start=paragraph_line_start,
                        line_end=end_line,
                        char_start=paragraph_char_start,
                        char_end=end_char,
                    ),
                    structural_fields={"source_parser": "plain_text_rule"},
                )
            )
            paragraph_lines = []

        for line_index, line in enumerate(lines, start=1):
            if line.strip() == "":
                flush_paragraph(line_index - 1, char_pos)
                char_pos += len(line)
                paragraph_line_start = line_index + 1
                paragraph_char_start = char_pos
                continue


            if detect_api_operation(line.strip()):
                flush_paragraph(line_index - 1, char_pos)

                block_counter += 1
                blocks.append(
                    RawBlock(
                        id=f"block_{block_counter:05d}",
                        block_type="api_operation",
                        text=line.strip(),
                        hierarchy_path=[],
                        parent_block_id=None,
                        source_position=SourcePosition(
                            line_start=line_index,
                            line_end=line_index,
                            char_start=char_pos,
                            char_end=char_pos + len(line),
                        ),
                        structural_fields={
                            "source_parser": "plain_text_rule",
                            "detected_block_type": "api_operation",
                        },
                    )
                )
                char_pos += len(line)
                paragraph_line_start = line_index + 1
                paragraph_char_start = char_pos
                continue

            if not paragraph_lines:
                paragraph_line_start = line_index
                paragraph_char_start = char_pos

            paragraph_lines.append(line)
            char_pos += len(line)

        flush_paragraph(len(lines), char_pos)

        return blocks

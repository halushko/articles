from __future__ import annotations

from ..detectors import detect_api_operation, detect_heading, detect_list_item
from ..models import RawBlock, SourcePosition
from ..normalizer import normalize_document_text
from .base import DocumentPartitioner


class MarkdownRulePartitioner(DocumentPartitioner):
    def partition(self, artifact_id: str, text: str) -> list[RawBlock]:
        normalized = normalize_document_text(text)
        lines = normalized.splitlines(keepends=True)

        blocks: list[RawBlock] = []
        hierarchy_stack: list[tuple[int, str, str]] = []

        paragraph_lines: list[str] = []
        paragraph_line_start = 1
        paragraph_char_start = 0

        char_pos = 0
        block_counter = 0

        def current_hierarchy() -> list[str]:
            return [title for _, title, _ in hierarchy_stack]

        def current_parent_block_id() -> str | None:
            return hierarchy_stack[-1][2] if hierarchy_stack else None

        def next_block_id() -> str:
            nonlocal block_counter
            block_counter += 1
            return f"block_{block_counter:05d}"

        def flush_paragraph(end_line: int, end_char: int) -> None:
            nonlocal paragraph_lines, paragraph_line_start, paragraph_char_start
            paragraph_text = "".join(paragraph_lines).strip()
            if not paragraph_text:
                paragraph_lines = []
                return

            block_id = next_block_id()
            blocks.append(
                RawBlock(
                    id=block_id,
                    block_type="paragraph",
                    text=paragraph_text,
                    hierarchy_path=current_hierarchy(),
                    parent_block_id=current_parent_block_id(),
                    source_position=SourcePosition(
                        line_start=paragraph_line_start,
                        line_end=end_line,
                        char_start=paragraph_char_start,
                        char_end=end_char,
                    ),
                    structural_fields={"source_parser": "markdown_rule"},
                )
            )
            paragraph_lines = []

        for line_index, line in enumerate(lines, start=1):
            raw_line = line.rstrip("\n")
            stripped = raw_line.strip()

            heading = detect_heading(raw_line)
            list_item = detect_list_item(raw_line)

            if not stripped:
                flush_paragraph(line_index - 1, char_pos)
                char_pos += len(line)
                paragraph_line_start = line_index + 1
                paragraph_char_start = char_pos
                continue

            if heading:
                flush_paragraph(line_index - 1, char_pos)

                level = heading["level"]
                title = heading["title"]
                marker = heading["marker"]

                while hierarchy_stack and hierarchy_stack[-1][0] >= level:
                    hierarchy_stack.pop()

                block_id = next_block_id()
                hierarchy_path = [item[1] for item in hierarchy_stack] + [title]

                blocks.append(
                    RawBlock(
                        id=block_id,
                        block_type="heading",
                        text=title,
                        hierarchy_path=hierarchy_path,
                        parent_block_id=current_parent_block_id(),
                        source_position=SourcePosition(
                            line_start=line_index,
                            line_end=line_index,
                            char_start=char_pos,
                            char_end=char_pos + len(line),
                        ),
                        structural_fields={
                            "source_parser": "markdown_rule",
                            "heading_level": level,
                            "heading_marker": marker,
                        },
                    )
                )

                hierarchy_stack.append((level, title, block_id))
                char_pos += len(line)
                paragraph_line_start = line_index + 1
                paragraph_char_start = char_pos
                continue


            api_operation = detect_api_operation(raw_line)
            if api_operation:
                flush_paragraph(line_index - 1, char_pos)

                block_id = next_block_id()
                blocks.append(
                    RawBlock(
                        id=block_id,
                        block_type="api_operation",
                        text=stripped,
                        hierarchy_path=current_hierarchy(),
                        parent_block_id=current_parent_block_id(),
                        source_position=SourcePosition(
                            line_start=line_index,
                            line_end=line_index,
                            char_start=char_pos,
                            char_end=char_pos + len(line),
                        ),
                        structural_fields={
                            "source_parser": "markdown_rule",
                            "detected_block_type": "api_operation",
                        },
                    )
                )

                char_pos += len(line)
                paragraph_line_start = line_index + 1
                paragraph_char_start = char_pos
                continue

            if list_item:
                flush_paragraph(line_index - 1, char_pos)

                block_id = next_block_id()
                blocks.append(
                    RawBlock(
                        id=block_id,
                        block_type="list_item",
                        text=list_item["text"],
                        hierarchy_path=current_hierarchy(),
                        parent_block_id=current_parent_block_id(),
                        source_position=SourcePosition(
                            line_start=line_index,
                            line_end=line_index,
                            char_start=char_pos,
                            char_end=char_pos + len(line),
                        ),
                        structural_fields={
                            "source_parser": "markdown_rule",
                            "list_marker": list_item["marker"],
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

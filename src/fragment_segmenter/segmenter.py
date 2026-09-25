from __future__ import annotations

from typing import Iterable

from .detectors import (
    detect_acceptance_criterion,
    detect_api_operation,
    detect_user_story,
    is_acceptance_criteria_section,
)
from .fragment_factory import FragmentFactory
from .models import Fragment, RawBlock, SourcePosition
from .partitioners.markdown_rule import MarkdownRulePartitioner
from .partitioners.plain_text import PlainTextPartitioner
from .process_splitter import hard_process_split, soft_process_split
from .sentence_splitter import split_sentences


class Segmenter:
    def __init__(self, artifact_format: str = "markdown") -> None:
        self.artifact_format = artifact_format
        self.factory = FragmentFactory()

        if artifact_format == "markdown":
            self.partitioner = MarkdownRulePartitioner()
        elif artifact_format == "text":
            self.partitioner = PlainTextPartitioner()
        else:
            raise ValueError("artifact_format must be 'markdown' or 'text'")

    def segment(self, artifact_id: str, text: str) -> list[Fragment]:
        blocks = self.partitioner.partition(artifact_id, text)
        fragments: list[Fragment] = []

        for block in blocks:
            fragments.extend(self._fragments_from_block(artifact_id, block))

        return fragments

    def _fragments_from_block(self, artifact_id: str, block: RawBlock) -> list[Fragment]:
        if block.block_type == "heading":
            return [self._heading_fragment(artifact_id, block)]

        if block.block_type == "table_row":
            return [self._table_row_fragment(artifact_id, block)]

        api = detect_api_operation(block.text)
        if api:
            return self._api_operation_fragments(artifact_id, block, api)

        user_story = detect_user_story(block.text)
        if user_story:
            return [self._user_story_fragment(artifact_id, block, user_story)]

        inside_ac = is_acceptance_criteria_section(block.hierarchy_path)
        ac = detect_acceptance_criterion(block.text, inside_ac_section=inside_ac)
        if ac:
            return self._acceptance_criterion_fragments(artifact_id, block, ac)

        if block.block_type == "list_item":
            base_type = "list_item"
            method = "markdown_structure"
            trigger = "list_item"
        else:
            base_type = "paragraph"
            method = "plain_text_structure" if self.artifact_format == "text" else "markdown_structure"
            trigger = block.block_type

        base = self.factory.create(
            artifact_id=artifact_id,
            parent_block_id=block.parent_block_id,
            parent_fragment_id=None,
            text=block.text,
            fragment_type=base_type,
            hierarchy_path=block.hierarchy_path,
            source_position=block.source_position,
            segmentation_method=method,
            segmentation_trigger=trigger,
            segmentation_confidence=0.95,
            structural_fields=block.structural_fields,
        )

        children = self._sentence_and_process_fragments(
            artifact_id=artifact_id,
            block=block,
            parent_fragment=base,
            sentence_type="sentence",
        )

        return [base, *children]

    def _heading_fragment(self, artifact_id: str, block: RawBlock) -> Fragment:
        return self.factory.create(
            artifact_id=artifact_id,
            parent_block_id=block.parent_block_id,
            parent_fragment_id=None,
            text=block.text,
            fragment_type="heading",
            hierarchy_path=block.hierarchy_path,
            source_position=block.source_position,
            segmentation_method="markdown_structure",
            segmentation_trigger=block.structural_fields.get("heading_marker", "heading"),
            segmentation_confidence=1.0,
            structural_fields=block.structural_fields,
        )

    def _table_row_fragment(self, artifact_id: str, block: RawBlock) -> Fragment:
        is_header = bool(block.structural_fields.get("table_header"))
        return self.factory.create(
            artifact_id=artifact_id,
            parent_block_id=block.parent_block_id,
            parent_fragment_id=None,
            text=block.text,
            fragment_type="table_row",
            hierarchy_path=block.hierarchy_path,
            source_position=block.source_position,
            segmentation_method="markdown_structure",
            segmentation_trigger="table_header" if is_header else "table_row",
            segmentation_confidence=0.98,
            structural_fields=block.structural_fields,
        )

    def _user_story_fragment(self, artifact_id: str, block: RawBlock, user_story: dict) -> Fragment:
        structural_fields = {
            **block.structural_fields,
            **user_story,
            "pattern": "as_i_want_so_that",
        }
        return self.factory.create(
            artifact_id=artifact_id,
            parent_block_id=block.parent_block_id,
            parent_fragment_id=None,
            text=block.text,
            fragment_type="user_story",
            hierarchy_path=block.hierarchy_path,
            source_position=block.source_position,
            segmentation_method="user_story_detection",
            segmentation_trigger="as_i_want_so_that",
            segmentation_confidence=0.98,
            structural_fields=structural_fields,
        )

    def _acceptance_criterion_fragments(
        self,
        artifact_id: str,
        block: RawBlock,
        ac: dict,
    ) -> list[Fragment]:
        structural_fields = {
            **block.structural_fields,
            **ac,
        }
        ac_fragment = self.factory.create(
            artifact_id=artifact_id,
            parent_block_id=block.parent_block_id,
            parent_fragment_id=None,
            text=block.text,
            fragment_type="acceptance_criterion",
            hierarchy_path=block.hierarchy_path,
            source_position=block.source_position,
            segmentation_method="acceptance_criteria_detection",
            segmentation_trigger=ac.get("gherkin_keyword") or "acceptance_criterion",
            segmentation_confidence=0.96,
            structural_fields=structural_fields,
        )

        children = self._process_fragments_from_sentence_like(
            artifact_id=artifact_id,
            block=block,
            parent_fragment=ac_fragment,
            sentence_text=block.text,
            sentence_type="acceptance_criterion_sentence",
            ignore_initial_then=ac.get("gherkin_keyword", "").lower() == "then",
        )

        return [ac_fragment, *children]

    def _api_operation_fragments(
        self,
        artifact_id: str,
        block: RawBlock,
        api: dict,
    ) -> list[Fragment]:
        api_fragment = self.factory.create(
            artifact_id=artifact_id,
            parent_block_id=block.parent_block_id,
            parent_fragment_id=None,
            text=block.text,
            fragment_type="api_operation",
            hierarchy_path=block.hierarchy_path,
            source_position=block.source_position,
            segmentation_method="api_operation_detection",
            segmentation_trigger="http_method_path",
            segmentation_confidence=0.99,
            structural_fields={
                **block.structural_fields,
                **api,
            },
        )

        result = [api_fragment]
        description = api.get("description")
        if description:
            for sentence in split_sentences(description):
                child = self.factory.create(
                    artifact_id=artifact_id,
                    parent_block_id=block.id,
                    parent_fragment_id=api_fragment.id,
                    text=sentence,
                    fragment_type="api_description_sentence",
                    hierarchy_path=block.hierarchy_path,
                    source_position=block.source_position,
                    segmentation_method="sentence_segmentation",
                    segmentation_trigger="sentence_boundary",
                    segmentation_confidence=0.9,
                    structural_fields={"source": "api_description"},
                )
                result.append(child)
                result.extend(
                    self._process_fragments_from_sentence_like(
                        artifact_id=artifact_id,
                        block=block,
                        parent_fragment=child,
                        sentence_text=sentence,
                        sentence_type="api_description_sentence",
                    )
                )

        return result

    def _sentence_and_process_fragments(
        self,
        *,
        artifact_id: str,
        block: RawBlock,
        parent_fragment: Fragment,
        sentence_type: str,
    ) -> list[Fragment]:
        result: list[Fragment] = []
        sentences = split_sentences(block.text)

        if len(sentences) <= 1 and sentences and sentences[0] == block.text.strip():
            # Keep parent block fragment as the main logical fragment.
            result.extend(
                self._process_fragments_from_sentence_like(
                    artifact_id=artifact_id,
                    block=block,
                    parent_fragment=parent_fragment,
                    sentence_text=sentences[0],
                    sentence_type=sentence_type,
                )
            )
            return result

        for sentence in sentences:
            sentence_fragment = self.factory.create(
                artifact_id=artifact_id,
                parent_block_id=block.id,
                parent_fragment_id=parent_fragment.id,
                text=sentence,
                fragment_type=sentence_type,
                hierarchy_path=block.hierarchy_path,
                source_position=block.source_position,
                segmentation_method="sentence_segmentation",
                segmentation_trigger="sentence_boundary",
                segmentation_confidence=0.9,
                structural_fields={"source_fragment_type": parent_fragment.fragment_type},
            )
            result.append(sentence_fragment)
            result.extend(
                self._process_fragments_from_sentence_like(
                    artifact_id=artifact_id,
                    block=block,
                    parent_fragment=sentence_fragment,
                    sentence_text=sentence,
                    sentence_type=sentence_type,
                )
            )

        return result

    def _process_fragments_from_sentence_like(
        self,
        *,
        artifact_id: str,
        block: RawBlock,
        parent_fragment: Fragment,
        sentence_text: str,
        sentence_type: str,
        ignore_initial_then: bool = False,
    ) -> list[Fragment]:
        result: list[Fragment] = []

        hard = hard_process_split(sentence_text, ignore_initial_then=ignore_initial_then)
        active_targets: list[Fragment] = [parent_fragment]

        if hard:
            parent_fragment.status = "superseded"

            parts = hard["parts"]
            triggers = hard["triggers"]
            trigger = triggers[0] if triggers else "hard_process_split"

            process_steps: list[Fragment] = []
            for index, part in enumerate(parts, start=1):
                step = self.factory.create(
                    artifact_id=artifact_id,
                    parent_block_id=block.id,
                    parent_fragment_id=parent_fragment.id,
                    text=part,
                    fragment_type="process_step",
                    hierarchy_path=block.hierarchy_path,
                    source_position=block.source_position,
                    segmentation_method="hard_process_split",
                    segmentation_trigger=trigger,
                    segmentation_confidence=0.86,
                    structural_fields={
                        "step_index": index,
                        "source_sentence_type": sentence_type,
                    },
                )
                process_steps.append(step)

            result.extend(process_steps)
            active_targets = process_steps

        for target in active_targets:
            soft = soft_process_split(target.text)
            if not soft:
                continue

            result.append(
                self.factory.create(
                    artifact_id=artifact_id,
                    parent_block_id=block.id,
                    parent_fragment_id=target.id,
                    text=soft["condition_clause"],
                    fragment_type="condition_clause",
                    hierarchy_path=block.hierarchy_path,
                    source_position=block.source_position,
                    segmentation_method="soft_process_split",
                    segmentation_trigger=soft["trigger"],
                    segmentation_confidence=0.82,
                    structural_fields={"condition_trigger": soft["trigger"]},
                )
            )
            result.append(
                self.factory.create(
                    artifact_id=artifact_id,
                    parent_block_id=block.id,
                    parent_fragment_id=target.id,
                    text=soft["conditional_scope"],
                    fragment_type="conditional_scope",
                    hierarchy_path=block.hierarchy_path,
                    source_position=block.source_position,
                    segmentation_method="soft_process_split",
                    segmentation_trigger=soft["trigger"],
                    segmentation_confidence=0.82,
                    structural_fields={"condition_trigger": soft["trigger"]},
                )
            )

        return result

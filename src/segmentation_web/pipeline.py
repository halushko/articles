from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from fragment_segmenter.segmenter import Segmenter

from . import __version__
from .archive import SourceDocument
from .db_models import DocumentVersion, RunDocument, SegmentationResult, SegmentationRun
from .hashing import corpus_sha256, sha256_bytes, sha256_json

SEGMENTER_CONFIG = {
    "artifact_format": "markdown",
    "fragment_schema_version": "1.0",
}
CONFIG_SHA256 = sha256_json(SEGMENTER_CONFIG)


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    source_type: str
    status: str
    corpus_sha256: str
    document_count: int
    fragment_count: int
    cache_hits: int


class SegmentationPipeline:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def process(
        self, documents: list[SourceDocument], *, source_type: str
    ) -> RunSummary:
        document_hashes = [
            {
                "path": document.relative_path,
                "sha256": sha256_bytes(document.content),
            }
            for document in documents
        ]
        corpus_hash = corpus_sha256(document_hashes)

        with self.session_factory() as session:
            run = SegmentationRun(
                source_type=source_type,
                corpus_sha256=corpus_hash,
                status="processing",
                document_count=len(documents),
            )
            session.add(run)
            session.commit()

            fragment_count = 0
            cache_hits = 0
            failed_documents = 0

            for position, document in enumerate(documents, start=1):
                content_hash = sha256_bytes(document.content)
                version = self._get_or_create_document_version(
                    session,
                    content_sha256=content_hash,
                    size_bytes=len(document.content),
                )

                try:
                    result, cache_hit = self._get_or_create_segmentation_result(
                        session,
                        version=version,
                        text=document.text(),
                    )
                    session.add(
                        RunDocument(
                            run_id=run.id,
                            document_version_id=version.id,
                            segmentation_result_id=result.id,
                            relative_path=document.relative_path,
                            position=position,
                            status="completed",
                            cache_hit=cache_hit,
                        )
                    )
                    fragment_count += result.fragment_count
                    cache_hits += int(cache_hit)
                except Exception as exc:  # noqa: BLE001 - isolate failures per document
                    failed_documents += 1
                    session.add(
                        RunDocument(
                            run_id=run.id,
                            document_version_id=version.id,
                            segmentation_result_id=None,
                            relative_path=document.relative_path,
                            position=position,
                            status="failed",
                            cache_hit=False,
                            error=str(exc),
                        )
                    )
                session.commit()

            run.fragment_count = fragment_count
            run.completed_at = datetime.now(timezone.utc)
            run.status = "completed_with_errors" if failed_documents else "completed"
            if failed_documents:
                run.error = f"{failed_documents} document(s) could not be segmented"
            session.commit()

            return RunSummary(
                run_id=run.id,
                source_type=run.source_type,
                status=run.status,
                corpus_sha256=run.corpus_sha256,
                document_count=run.document_count,
                fragment_count=run.fragment_count,
                cache_hits=cache_hits,
            )

    @staticmethod
    def _get_or_create_document_version(
        session: Session,
        *,
        content_sha256: str,
        size_bytes: int,
    ) -> DocumentVersion:
        version = session.scalar(
            select(DocumentVersion).where(
                DocumentVersion.content_sha256 == content_sha256
            )
        )
        if version is not None:
            return version

        version = DocumentVersion(
            content_sha256=content_sha256,
            size_bytes=size_bytes,
            media_type="text/markdown",
        )
        session.add(version)
        session.flush()
        return version

    @staticmethod
    def _get_or_create_segmentation_result(
        session: Session,
        *,
        version: DocumentVersion,
        text: str,
    ) -> tuple[SegmentationResult, bool]:
        result = session.scalar(
            select(SegmentationResult).where(
                SegmentationResult.document_version_id == version.id,
                SegmentationResult.segmenter_version == __version__,
                SegmentationResult.config_sha256 == CONFIG_SHA256,
            )
        )
        if result is not None:
            return result, True

        artifact_id = f"sha256:{version.content_sha256}"
        fragments = Segmenter(artifact_format="markdown").segment(artifact_id, text)
        payload = {
            "schema_version": "1.0",
            "source": {
                "sha256": version.content_sha256,
                "size_bytes": version.size_bytes,
                "media_type": version.media_type,
            },
            "segmenter": {
                "version": __version__,
                "config_sha256": CONFIG_SHA256,
                "config": SEGMENTER_CONFIG,
            },
            "fragments": [fragment.to_dict() for fragment in fragments],
        }
        result = SegmentationResult(
            document_version_id=version.id,
            segmenter_version=__version__,
            config_sha256=CONFIG_SHA256,
            payload=payload,
            fragment_count=len(fragments),
        )
        session.add(result)
        session.flush()
        return result, False


def get_run_summary(session: Session, run_id: str) -> RunSummary | None:
    run = session.get(SegmentationRun, run_id)
    if run is None:
        return None
    cache_hits = session.scalars(
        select(RunDocument).where(
            RunDocument.run_id == run_id,
            RunDocument.cache_hit.is_(True),
        )
    ).all()
    return RunSummary(
        run_id=run.id,
        source_type=run.source_type,
        status=run.status,
        corpus_sha256=run.corpus_sha256,
        document_count=run.document_count,
        fragment_count=run.fragment_count,
        cache_hits=len(cache_hits),
    )

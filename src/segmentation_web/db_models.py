from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def new_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


JSON_PAYLOAD = JSON().with_variant(JSONB(), "postgresql")


class SegmentationRun(Base):
    __tablename__ = "segmentation_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    corpus_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="processing"
    )
    document_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fragment_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    content_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    media_type: Mapped[str] = mapped_column(
        String(100), nullable=False, default="text/markdown"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class SegmentationResult(Base):
    __tablename__ = "segmentation_results"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "segmenter_version",
            "config_sha256",
            name="uq_segmentation_result_cache_key",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    segmenter_version: Mapped[str] = mapped_column(String(50), nullable=False)
    config_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_PAYLOAD, nullable=False)
    fragment_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class RunDocument(Base):
    __tablename__ = "run_documents"
    __table_args__ = (
        UniqueConstraint("run_id", "relative_path", name="uq_run_document_path"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("segmentation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="RESTRICT"), nullable=False
    )
    segmentation_result_id: Mapped[str | None] = mapped_column(
        ForeignKey("segmentation_results.id", ondelete="SET NULL"), nullable=True
    )
    relative_path: Mapped[str] = mapped_column(String(500), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class DocumentationGraphResult(Base):
    __tablename__ = "documentation_graph_results"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "builder_version",
            "config_sha256",
            name="uq_documentation_graph_cache_key",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("segmentation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    builder_version: Mapped[str] = mapped_column(String(50), nullable=False)
    config_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_PAYLOAD, nullable=False)
    node_count: Mapped[int] = mapped_column(Integer, nullable=False)
    edge_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

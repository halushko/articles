"""Create segmentation run and result tables.

Revision ID: 20260925_01
Revises:
Create Date: 2026-09-25
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260925_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "segmentation_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_type", sa.String(length=20), nullable=False),
        sa.Column("corpus_sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("document_count", sa.Integer(), nullable=False),
        sa.Column("fragment_count", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_segmentation_runs_corpus_sha256",
        "segmentation_runs",
        ["corpus_sha256"],
    )

    op.create_table(
        "document_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("media_type", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_document_versions_content_sha256",
        "document_versions",
        ["content_sha256"],
        unique=True,
    )

    op.create_table(
        "segmentation_results",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("document_version_id", sa.String(length=36), nullable=False),
        sa.Column("segmenter_version", sa.String(length=50), nullable=False),
        sa.Column("config_sha256", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("fragment_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_version_id",
            "segmenter_version",
            "config_sha256",
            name="uq_segmentation_result_cache_key",
        ),
    )
    op.create_index(
        "ix_segmentation_results_document_version_id",
        "segmentation_results",
        ["document_version_id"],
    )

    op.create_table(
        "run_documents",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("document_version_id", sa.String(length=36), nullable=False),
        sa.Column("segmentation_result_id", sa.String(length=36), nullable=True),
        sa.Column("relative_path", sa.String(length=500), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("cache_hit", sa.Boolean(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["segmentation_runs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["segmentation_result_id"],
            ["segmentation_results.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "relative_path", name="uq_run_document_path"),
    )
    op.create_index("ix_run_documents_run_id", "run_documents", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_run_documents_run_id", table_name="run_documents")
    op.drop_table("run_documents")
    op.drop_index(
        "ix_segmentation_results_document_version_id",
        table_name="segmentation_results",
    )
    op.drop_table("segmentation_results")
    op.drop_index(
        "ix_document_versions_content_sha256",
        table_name="document_versions",
    )
    op.drop_table("document_versions")
    op.drop_index(
        "ix_segmentation_runs_corpus_sha256",
        table_name="segmentation_runs",
    )
    op.drop_table("segmentation_runs")

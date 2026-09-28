"""Cache grounded LLM process extractions by corpus.

Revision ID: 20260925_04
Revises: 20260925_03
Create Date: 2026-09-25
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260925_04"
down_revision = "20260925_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_extraction_results",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("corpus_sha256", sa.String(length=64), nullable=False),
        sa.Column("analyzer_version", sa.String(length=50), nullable=False),
        sa.Column("config_sha256", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model", sa.String(length=150), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "corpus_sha256",
            "analyzer_version",
            "config_sha256",
            name="uq_llm_extraction_cache_key",
        ),
    )
    op.create_index(
        "ix_llm_extraction_results_corpus_sha256",
        "llm_extraction_results",
        ["corpus_sha256"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_llm_extraction_results_corpus_sha256",
        table_name="llm_extraction_results",
    )
    op.drop_table("llm_extraction_results")

"""Store control process models and hierarchical aggregation results.

Revision ID: 20260925_03
Revises: 20260925_02
Create Date: 2026-09-25
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260925_03"
down_revision = "20260925_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "process_model_results",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("builder_version", sa.String(length=50), nullable=False),
        sa.Column("config_sha256", sa.String(length=64), nullable=False),
        sa.Column("derivation_mode", sa.String(length=50), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("level_count", sa.Integer(), nullable=False),
        sa.Column("atomic_node_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["segmentation_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id",
            "builder_version",
            "config_sha256",
            name="uq_process_model_cache_key",
        ),
    )
    op.create_index(
        "ix_process_model_results_run_id",
        "process_model_results",
        ["run_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_process_model_results_run_id",
        table_name="process_model_results",
    )
    op.drop_table("process_model_results")

"""Store versioned documentation graphs.

Revision ID: 20260925_02
Revises: 20260925_01
Create Date: 2026-09-25
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260925_02"
down_revision = "20260925_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documentation_graph_results",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("builder_version", sa.String(length=50), nullable=False),
        sa.Column("config_sha256", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("node_count", sa.Integer(), nullable=False),
        sa.Column("edge_count", sa.Integer(), nullable=False),
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
            name="uq_documentation_graph_cache_key",
        ),
    )
    op.create_index(
        "ix_documentation_graph_results_run_id",
        "documentation_graph_results",
        ["run_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_documentation_graph_results_run_id",
        table_name="documentation_graph_results",
    )
    op.drop_table("documentation_graph_results")

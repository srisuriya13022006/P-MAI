"""add temporal and conflict fields to memories

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d5e6f7a8b9c0"
down_revision: Union[str, Sequence[str], None] = "c4d5e6f7a8b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("memories") as batch_op:
        batch_op.add_column(
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true())
        )
        batch_op.add_column(
            sa.Column("superseded_by", sa.Integer(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("last_accessed_at", sa.DateTime(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("expires_at", sa.DateTime(), nullable=True)
        )
        batch_op.create_index("ix_memories_is_active", ["is_active"])
        batch_op.create_foreign_key(
            "fk_memories_superseded_by", "memories", ["superseded_by"], ["id"]
        )


def downgrade() -> None:
    with op.batch_alter_table("memories") as batch_op:
        batch_op.drop_constraint("fk_memories_superseded_by", type_="foreignkey")
        batch_op.drop_index("ix_memories_is_active")
        batch_op.drop_column("expires_at")
        batch_op.drop_column("last_accessed_at")
        batch_op.drop_column("superseded_by")
        batch_op.drop_column("is_active")

"""create memories table

Revision ID: a2b3c4d5e6f7
Revises: ce0bddb780da
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a2b3c4d5e6f7"
down_revision: Union[str, Sequence[str], None] = "ce0bddb780da"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memories",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(length=100), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("memory_type", sa.String(length=30), nullable=False),
        sa.Column("importance", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memories_user_id", "memories", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_memories_user_id", table_name="memories")
    op.drop_table("memories")
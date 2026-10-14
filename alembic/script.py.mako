<%!
import re
%>"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# Import thêm các thư viện đặc thù của project
import pgvector.sqlalchemy  # Nếu model có sử dụng Vector column
from src.nutrition_core.logging.logger import logger

# revision identifiers, used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: Union[str, Sequence[str], None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}

def upgrade() -> None:
    logger.info("Running upgrade for revision: %s", revision)
    ${upgrades if upgrades else "pass"}

def downgrade() -> None:
    logger.info("Running downgrade for revision: %s", revision)
    ${downgrades if downgrades else "pass"}
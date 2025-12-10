from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel

class ChatHistory(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, unique=True)
    flow_id: UUID = Field(foreign_key="flow.id", index=True)
    session_id: str = Field(index=True)
    sender: str
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
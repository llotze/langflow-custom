from fastapi import APIRouter, Depends
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from uuid import UUID
from typing import List

from langflow.services.database.models.chat_history.model import ChatHistory
from langflow.services.deps import get_session

router = APIRouter(prefix="/api/v1/chat-history", tags=["Chat History"])

@router.post("/", response_model=ChatHistory)
async def add_message(chat: ChatHistory, session: AsyncSession = Depends(get_session)):
    """Persist a chat history entry."""
    session.add(chat)
    await session.commit()
    await session.refresh(chat)
    return chat

@router.get("/", response_model=List[ChatHistory])
async def get_history(flow_id: UUID, session_id: str, session: AsyncSession = Depends(get_session)):
    """Return chat history for a flow/session ordered by timestamp."""
    statement = (
        select(ChatHistory)
        .where(ChatHistory.flow_id == flow_id, ChatHistory.session_id == session_id)
        .order_by(ChatHistory.timestamp)
    )
    result = await session.exec(statement)
    return result.all()
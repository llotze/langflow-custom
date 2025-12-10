from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from uuid import UUID
from typing import List
from pydantic import BaseModel

from langflow.services.database.models.chat_history.model import ChatHistory
from langflow.services.deps import get_session

router = APIRouter(prefix="/chat-history", tags=["Chat History"])

class ChatHistoryCreate(BaseModel):
    flow_id: UUID
    session_id: str
    sender: str
    message: str

@router.post("/", response_model=ChatHistory)
async def add_message(chat_data: ChatHistoryCreate, session: AsyncSession = Depends(get_session)):
    """Persist a chat history entry."""
    # Validate required fields
    if not chat_data.flow_id:
        raise HTTPException(
            status_code=400, 
            detail="flow_id is required and cannot be None"
        )
    if not chat_data.session_id or not chat_data.session_id.strip():
        raise HTTPException(
            status_code=400,
            detail="session_id is required and cannot be empty"
        )
    
    # Create ChatHistory instance
    chat = ChatHistory(
        flow_id=chat_data.flow_id,
        session_id=chat_data.session_id,
        sender=chat_data.sender,
        message=chat_data.message,
    )
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
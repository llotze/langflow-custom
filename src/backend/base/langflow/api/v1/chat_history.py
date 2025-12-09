from fastapi import APIRouter, Depends
from sqlmodel import Session, select
from uuid import UUID
from typing import List

from langflow.services.database.models.chat_history.model import ChatHistory
from langflow.services.database.session import get_session

router = APIRouter(prefix="/api/v1/chat-history", tags=["Chat History"])

@router.post("/", response_model=ChatHistory)
def add_message(chat: ChatHistory, session: Session = Depends(get_session)):
    session.add(chat)
    session.commit()
    session.refresh(chat)
    return chat

@router.get("/", response_model=List[ChatHistory])
def get_history(flow_id: UUID, session_id: str, session: Session = Depends(get_session)):
    statement = select(ChatHistory).where(
        ChatHistory.flow_id == flow_id,
        ChatHistory.session_id == session_id
    ).order_by(ChatHistory.timestamp)
    return session.exec(statement).all()
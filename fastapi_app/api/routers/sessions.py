from __future__ import annotations

from fastapi import APIRouter

from fastapi_app.schemas.sessions import SessionCreateResponse, SessionDeleteResponse
from fastapi_app.services.session_service import session_service

router = APIRouter()


@router.post("/sessions", response_model=SessionCreateResponse)
async def create_session() -> SessionCreateResponse:
    return session_service.create_session()


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str) -> SessionDeleteResponse:
    return session_service.delete_session(session_id)

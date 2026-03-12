from __future__ import annotations

from pydantic import BaseModel

from fastapi_app.schemas.sessions import SessionStage


class UploadResponse(BaseModel):
    success: bool
    session_id: str
    stage: SessionStage
    artifact: str
    row_count: int
    column_count: int


class SessionFileItem(BaseModel):
    name: str
    relative_path: str
    size_bytes: int


class SessionFilesResponse(BaseModel):
    success: bool
    session_id: str
    stage: SessionStage
    files: list[SessionFileItem]


class AttachmentsProcessResponse(BaseModel):
    success: bool
    session_id: str
    stage: SessionStage
    artifact: str
    row_count: int
    column_count: int


class CleanResponse(BaseModel):
    success: bool
    session_id: str
    stage: SessionStage
    artifact: str
    row_count: int
    column_count: int

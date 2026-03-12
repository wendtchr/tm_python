from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class SessionStage(StrEnum):
    INIT = "INIT"
    LOADED = "LOADED"
    ATTACHMENTS_PROCESSED = "ATTACHMENTS_PROCESSED"
    CLEANED = "CLEANED"
    MODELING_RUNNING = "MODELING_RUNNING"
    MODELED = "MODELED"
    ERROR = "ERROR"
    DELETED = "DELETED"


class SessionMetadata(BaseModel):
    session_id: str
    stage: SessionStage
    created_at: datetime
    updated_at: datetime
    base_dir: str
    temp_dir: str
    visualizations_dir: str
    reports_dir: str


class SessionCreateResponse(BaseModel):
    success: bool
    session_id: str
    stage: SessionStage
    session: SessionMetadata


class SessionDeleteResponse(BaseModel):
    success: bool
    session_id: str
    stage: SessionStage
    deleted: bool

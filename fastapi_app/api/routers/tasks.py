from __future__ import annotations

from fastapi import APIRouter

from fastapi_app.schemas.model import TaskStatusResponse
from fastapi_app.services.session_service import session_service
from fastapi_app.services.task_registry_service import task_registry_service

router = APIRouter()


@router.get("/sessions/{session_id}/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(session_id: str, task_id: str) -> TaskStatusResponse:
    session = session_service.get_session(session_id)
    task = task_registry_service.get_task(session_id=session_id, task_id=task_id)
    return TaskStatusResponse(
        success=True,
        session_id=session_id,
        stage=session.stage,
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        message=task.message,
        created_at=task.created_at,
        updated_at=task.updated_at,
        result=task.result,
        error=task.error,
    )

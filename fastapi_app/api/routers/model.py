from __future__ import annotations

from fastapi import APIRouter

from fastapi_app.schemas.model import ModelRunRequest, ModelRunResponse
from fastapi_app.schemas.sessions import SessionStage
from fastapi_app.services.modeling_service import modeling_service

router = APIRouter()


@router.post("/sessions/{session_id}/model/run", response_model=ModelRunResponse)
async def run_model(session_id: str, request: ModelRunRequest) -> ModelRunResponse:
    task_id = modeling_service.enqueue_model_run(session_id=session_id, request=request)
    return ModelRunResponse(
        success=True,
        session_id=session_id,
        stage=SessionStage.MODELING_RUNNING,
        task_id=task_id,
        status="queued",
    )

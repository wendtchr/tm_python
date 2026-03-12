from __future__ import annotations

from fastapi import APIRouter

from fastapi_app.schemas.model import (
    ModelResultsResponse,
    ModelRunRequest,
    ModelRunResponse,
    ModelSummaryResponse,
)
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


@router.get("/sessions/{session_id}/model/summary", response_model=ModelSummaryResponse)
async def get_model_summary(session_id: str) -> ModelSummaryResponse:
    summary = modeling_service.get_model_summary(session_id=session_id)
    return ModelSummaryResponse(success=True, **summary)


@router.get("/sessions/{session_id}/model/results", response_model=ModelResultsResponse)
async def get_model_results(session_id: str) -> ModelResultsResponse:
    results = modeling_service.get_model_results(session_id=session_id)
    return ModelResultsResponse(success=True, **results)

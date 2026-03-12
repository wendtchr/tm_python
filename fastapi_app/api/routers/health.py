from __future__ import annotations

from fastapi import APIRouter

from fastapi_app.schemas.common import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    return HealthResponse(status="ok", service="tm_python-fastapi", version="0.1.0")

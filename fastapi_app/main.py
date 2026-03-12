from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fastapi_app.api.routers import data, downloads, health, model, sessions, tasks, visualizations
from fastapi_app.core.errors import ApiError

app = FastAPI(title="tm_python FastAPI Parity API", version="0.1.0")

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(sessions.router, prefix="/api", tags=["sessions"])
app.include_router(data.router, prefix="/api", tags=["data"])
app.include_router(model.router, prefix="/api", tags=["model"])
app.include_router(tasks.router, prefix="/api", tags=["tasks"])
app.include_router(visualizations.router, prefix="/api", tags=["visualizations"])
app.include_router(downloads.router, prefix="/api", tags=["downloads"])


@app.exception_handler(ApiError)
async def handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "session_id": exc.session_id,
                "stage": exc.stage,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": exc.errors(),
                "session_id": None,
                "stage": None,
            }
        },
    )

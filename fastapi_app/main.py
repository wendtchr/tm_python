from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

from app_files.modules import config as shiny_config

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

FRONTEND_INDEX_PATH = Path(__file__).resolve().parent / "frontend" / "index.html"

MODEL_DEFAULTS = {
    "min_topic_size": 4,
    "ngram_min": 1,
    "ngram_max": 2,
    "top_n_words": 12,
    "umap_n_neighbors": 15,
    "umap_n_components": 5,
    "umap_min_dist": 0.1,
    "enable_chunking": False,
    "similarity_threshold": 0.75,
    "min_chunk_length": 200,
    "max_chunk_length": 2000,
}

SEED_TOPIC_DEFAULTS = shiny_config.TOPIC_MODELING["SEED_TOPICS"]["DEFAULT"]


@app.get("/", response_class=HTMLResponse)
async def parity_ui() -> HTMLResponse:
    content = FRONTEND_INDEX_PATH.read_text(encoding="utf-8")
    content = content.replace("__MODEL_DEFAULTS_JSON__", json.dumps(MODEL_DEFAULTS))
    content = content.replace("__SEED_TOPICS_JSON__", json.dumps(SEED_TOPIC_DEFAULTS))
    return HTMLResponse(content=content)


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
                "details": jsonable_encoder(exc.errors()),
                "session_id": None,
                "stage": None,
            }
        },
    )

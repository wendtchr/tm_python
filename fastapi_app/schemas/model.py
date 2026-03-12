from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from fastapi_app.schemas.sessions import SessionStage


class ModelRunRequest(BaseModel):
    min_topic_size: int = Field(default=4, ge=2)
    ngram_min: int = Field(default=1, ge=1, le=3)
    ngram_max: int = Field(default=2, ge=1, le=3)
    top_n_words: int = Field(default=12, ge=1, le=30)
    umap_n_neighbors: int = Field(default=15, ge=5)
    umap_n_components: int = Field(default=5, ge=2)
    umap_min_dist: float = Field(default=0.1, ge=0.0, le=1.0)
    enable_chunking: bool = Field(default=False)
    similarity_threshold: float = Field(default=0.75, ge=0.5, le=0.9)
    min_chunk_length: int = Field(default=200, ge=20, le=200)
    max_chunk_length: int = Field(default=2000, ge=500, le=5000)
    seed_topics: list[str] | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> "ModelRunRequest":
        if self.ngram_max < self.ngram_min:
            raise ValueError("ngram_max must be greater than or equal to ngram_min")
        if self.max_chunk_length < self.min_chunk_length:
            raise ValueError("max_chunk_length must be greater than or equal to min_chunk_length")
        return self


class ModelRunResponse(BaseModel):
    success: bool
    session_id: str
    stage: SessionStage
    task_id: str
    status: str


class TaskStatusResponse(BaseModel):
    success: bool
    session_id: str
    stage: SessionStage
    task_id: str
    status: str
    progress: float
    message: str
    created_at: datetime
    updated_at: datetime
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class ModelSummaryResponse(BaseModel):
    success: bool
    session_id: str
    stage: SessionStage
    artifact: str
    row_count: int
    topic_count: int
    columns: list[str]


class ModelResultsResponse(BaseModel):
    success: bool
    session_id: str
    stage: SessionStage
    artifact: str
    row_count: int
    columns: list[str]
    rows: list[dict[str, Any]]

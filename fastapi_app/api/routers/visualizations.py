from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from fastapi_app.services.artifact_registry_service import VisualizationKey, artifact_registry_service

router = APIRouter()


def _serve_visualization(session_id: str, key: VisualizationKey) -> FileResponse:
    path, media_type = artifact_registry_service.resolve_visualization(session_id=session_id, key=key)
    return FileResponse(path=path, media_type=media_type, filename=path.name)


@router.get("/sessions/{session_id}/visualizations/topic")
async def get_topic_visualization(session_id: str) -> FileResponse:
    return _serve_visualization(session_id=session_id, key=VisualizationKey.TOPIC)


@router.get("/sessions/{session_id}/visualizations/hierarchy")
async def get_hierarchy_visualization(session_id: str) -> FileResponse:
    return _serve_visualization(session_id=session_id, key=VisualizationKey.HIERARCHY)


@router.get("/sessions/{session_id}/visualizations/wordcloud")
async def get_wordcloud_visualization(session_id: str) -> FileResponse:
    return _serve_visualization(session_id=session_id, key=VisualizationKey.WORDCLOUD)


@router.get("/sessions/{session_id}/visualizations/word-scores")
async def get_word_scores_visualization(session_id: str) -> FileResponse:
    return _serve_visualization(session_id=session_id, key=VisualizationKey.WORD_SCORES)


@router.get("/sessions/{session_id}/visualizations/alignment")
async def get_alignment_visualization(session_id: str) -> FileResponse:
    return _serve_visualization(session_id=session_id, key=VisualizationKey.ALIGNMENT)

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from fastapi_app.services.artifact_registry_service import DownloadArtifactKey, artifact_registry_service

router = APIRouter()


@router.get("/sessions/{session_id}/downloads/{artifact_name}")
async def download_artifact(session_id: str, artifact_name: DownloadArtifactKey) -> FileResponse:
    path, media_type = artifact_registry_service.resolve_download(session_id=session_id, key=artifact_name)
    return FileResponse(path=path, media_type=media_type, filename=path.name)

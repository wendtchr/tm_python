from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi_app.core.errors import ApiError
from fastapi_app.core.settings import get_settings
from fastapi_app.schemas.sessions import SessionCreateResponse, SessionDeleteResponse, SessionMetadata, SessionStage

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class SessionService:
    def __init__(self) -> None:
        pass

    def _session_dir(self, session_id: str) -> Path:
        return get_settings().output_base_dir / session_id

    def _session_json_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session.json"

    def _validate_session_id(self, session_id: str) -> None:
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise ApiError(
                code="INVALID_SESSION_ID",
                message="Session ID contains invalid characters",
                status_code=400,
                details={"session_id": session_id},
                session_id=session_id,
                stage=SessionStage.ERROR,
            )

    def _write_session_json(self, metadata: SessionMetadata) -> None:
        payload = metadata.model_dump(mode="json")
        path = self._session_json_path(metadata.session_id)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_session(self, session_id: str) -> SessionMetadata:
        self._validate_session_id(session_id)
        path = self._session_json_path(session_id)
        if not path.exists():
            raise ApiError(
                code="SESSION_NOT_FOUND",
                message="Session not found",
                status_code=404,
                details={"session_id": session_id},
                session_id=session_id,
                stage=SessionStage.ERROR,
            )

        payload = json.loads(path.read_text(encoding="utf-8"))
        return SessionMetadata.model_validate(payload)

    def update_stage(self, session_id: str, stage: SessionStage) -> SessionMetadata:
        metadata = self.get_session(session_id)
        updated = metadata.model_copy(update={"stage": stage, "updated_at": datetime.now(timezone.utc)})
        self._write_session_json(updated)
        return updated

    def create_session(self) -> SessionCreateResponse:
        now = datetime.now(timezone.utc)
        session_id = f"{now.strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
        session_dir = self._session_dir(session_id)
        temp_dir = session_dir / "temp"
        viz_dir = session_dir / "visualizations"
        reports_dir = session_dir / "reports"

        session_dir.mkdir(parents=True, exist_ok=False)
        temp_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        metadata = SessionMetadata(
            session_id=session_id,
            stage=SessionStage.INIT,
            created_at=now,
            updated_at=now,
            base_dir=str(session_dir),
            temp_dir=str(temp_dir),
            visualizations_dir=str(viz_dir),
            reports_dir=str(reports_dir),
        )
        self._write_session_json(metadata)

        return SessionCreateResponse(
            success=True,
            session_id=session_id,
            stage=metadata.stage,
            session=metadata,
        )

    def delete_session(self, session_id: str) -> SessionDeleteResponse:
        self._validate_session_id(session_id)
        session_dir = self._session_dir(session_id)
        existed = session_dir.exists()
        if existed:
            shutil.rmtree(session_dir, ignore_errors=True)

        return SessionDeleteResponse(
            success=True,
            session_id=session_id,
            stage=SessionStage.DELETED,
            deleted=existed,
        )


session_service = SessionService()

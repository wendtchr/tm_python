from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

from fastapi_app.core.errors import ApiError
from fastapi_app.schemas.sessions import SessionStage


@dataclass
class TaskRecord:
    task_id: str
    session_id: str
    status: str
    progress: float
    message: str
    created_at: datetime
    updated_at: datetime
    result: dict | None = None
    error: dict | None = None


class TaskRegistryService:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._active_by_session: dict[str, str] = {}
        self._lock = Lock()

    def has_active_task(self, session_id: str) -> bool:
        with self._lock:
            task_id = self._active_by_session.get(session_id)
            if not task_id:
                return False
            task = self._tasks.get(task_id)
            return task is not None and task.status in {"queued", "running"}

    def create_task(self, session_id: str, message: str) -> TaskRecord:
        now = datetime.now(timezone.utc)
        record = TaskRecord(
            task_id=uuid4().hex,
            session_id=session_id,
            status="queued",
            progress=0.0,
            message=message,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._tasks[record.task_id] = record
            self._active_by_session[session_id] = record.task_id
        return record

    def get_task(self, session_id: str, task_id: str) -> TaskRecord:
        with self._lock:
            record = self._tasks.get(task_id)
        if record is None or record.session_id != session_id:
            raise ApiError(
                code="TASK_NOT_FOUND",
                message="Task not found",
                status_code=404,
                details={"task_id": task_id},
                session_id=session_id,
                stage=SessionStage.ERROR,
            )
        return record

    def mark_running(self, task_id: str, message: str) -> None:
        self.update_task(task_id=task_id, status="running", progress=10.0, message=message)

    def update_task(
        self,
        task_id: str,
        status: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        result: dict | None = None,
        error: dict | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return
            if status is not None:
                record.status = status
            if progress is not None:
                record.progress = progress
            if message is not None:
                record.message = message
            if result is not None:
                record.result = result
            if error is not None:
                record.error = error
            record.updated_at = now

    def mark_succeeded(self, task_id: str, result: dict) -> None:
        self.update_task(
            task_id=task_id,
            status="succeeded",
            progress=100.0,
            message="Modeling complete",
            result=result,
        )
        self._clear_active(task_id)

    def mark_failed(self, task_id: str, error: dict) -> None:
        self.update_task(
            task_id=task_id,
            status="failed",
            progress=100.0,
            message="Modeling failed",
            error=error,
        )
        self._clear_active(task_id)

    def _clear_active(self, task_id: str) -> None:
        with self._lock:
            record = self._tasks.get(task_id)
            if not record:
                return
            active_task = self._active_by_session.get(record.session_id)
            if active_task == task_id:
                del self._active_by_session[record.session_id]

    def reset(self) -> None:
        with self._lock:
            self._tasks.clear()
            self._active_by_session.clear()


task_registry_service = TaskRegistryService()

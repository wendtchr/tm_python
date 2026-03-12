from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app_files.modules import config
from app_files.modules.data_processing import (
    _get_output_filename,
    clean_data,
    process_attachments,
    read_csv_with_encoding,
)
from fastapi_app.core.errors import ApiError
from fastapi_app.schemas.data import (
    AttachmentsProcessResponse,
    CleanResponse,
    SessionFileItem,
    SessionFilesResponse,
    UploadResponse,
)
from fastapi_app.schemas.sessions import SessionStage
from fastapi_app.services.session_service import session_service

router = APIRouter()


@router.post("/sessions/{session_id}/upload", response_model=UploadResponse)
async def upload_data(session_id: str, file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise ApiError(
            code="UPLOAD_MISSING_FILE",
            message="No upload file was provided",
            status_code=400,
            details=None,
            session_id=session_id,
            stage=SessionStage.ERROR,
        )

    session = session_service.get_session(session_id)
    if session.stage == SessionStage.DELETED:
        raise ApiError(
            code="SESSION_DELETED",
            message="Session has been deleted",
            status_code=409,
            details={"session_id": session_id},
            session_id=session_id,
            stage=SessionStage.DELETED,
        )

    upload_bytes = await file.read()
    upload_name = Path(file.filename).name
    suffix = Path(upload_name).suffix or ".csv"
    staging_path = Path(session.temp_dir) / f"upload_{uuid.uuid4().hex}{suffix}"
    staging_path.write_bytes(upload_bytes)

    try:
        df = read_csv_with_encoding(str(staging_path))
        if df.empty:
            raise ValueError("Empty DataFrame")

        missing = [col for col in config.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        initial_path = _get_output_filename(Path(session.base_dir), "initial")
        df.to_csv(initial_path, index=False)
    except Exception as exc:
        raise ApiError(
            code="UPLOAD_PROCESSING_ERROR",
            message="Upload processing failed",
            status_code=400,
            details={"reason": str(exc)},
            session_id=session_id,
            stage=SessionStage.ERROR,
        ) from exc

    updated = session_service.update_stage(session_id, SessionStage.LOADED)
    return UploadResponse(
        success=True,
        session_id=session_id,
        stage=updated.stage,
        artifact="df_initial.csv",
        row_count=len(df),
        column_count=len(df.columns),
    )


@router.post(
    "/sessions/{session_id}/attachments/process",
    response_model=AttachmentsProcessResponse,
)
async def process_session_attachments(session_id: str) -> AttachmentsProcessResponse:
    session = session_service.get_session(session_id)
    if session.stage == SessionStage.DELETED:
        raise ApiError(
            code="SESSION_DELETED",
            message="Session has been deleted",
            status_code=409,
            details={"session_id": session_id},
            session_id=session_id,
            stage=SessionStage.DELETED,
        )

    base_dir = Path(session.base_dir)
    initial_path = _get_output_filename(base_dir, "initial")
    if not initial_path.exists():
        raise ApiError(
            code="DATA_NOT_LOADED",
            message="Initial dataset not found for attachment processing",
            status_code=409,
            details={"expected": initial_path.name},
            session_id=session_id,
            stage=session.stage,
        )

    try:
        df = read_csv_with_encoding(str(initial_path))
        attached_df = await process_attachments(df, output_dir=base_dir)
    except Exception as exc:
        raise ApiError(
            code="ATTACHMENTS_PROCESSING_ERROR",
            message="Attachment processing failed",
            status_code=400,
            details={"reason": str(exc)},
            session_id=session_id,
            stage=SessionStage.ERROR,
        ) from exc

    updated = session_service.update_stage(session_id, SessionStage.ATTACHMENTS_PROCESSED)
    return AttachmentsProcessResponse(
        success=True,
        session_id=session_id,
        stage=updated.stage,
        artifact="df_initial_attach.csv",
        row_count=len(attached_df),
        column_count=len(attached_df.columns),
    )


@router.post("/sessions/{session_id}/clean", response_model=CleanResponse)
async def clean_session_data(session_id: str) -> CleanResponse:
    session = session_service.get_session(session_id)
    if session.stage == SessionStage.DELETED:
        raise ApiError(
            code="SESSION_DELETED",
            message="Session has been deleted",
            status_code=409,
            details={"session_id": session_id},
            session_id=session_id,
            stage=SessionStage.DELETED,
        )

    base_dir = Path(session.base_dir)
    attach_path = _get_output_filename(base_dir, "attach")
    initial_path = _get_output_filename(base_dir, "initial")
    source_path = attach_path if attach_path.exists() else initial_path
    if not source_path.exists():
        raise ApiError(
            code="DATA_NOT_LOADED",
            message="No dataset available for cleaning",
            status_code=409,
            details={"expected_one_of": [initial_path.name, attach_path.name]},
            session_id=session_id,
            stage=session.stage,
        )

    try:
        df = read_csv_with_encoding(str(source_path))
        cleaned_df = await clean_data(df, output_dir=base_dir)
    except Exception as exc:
        raise ApiError(
            code="DATA_CLEANING_ERROR",
            message="Data cleaning failed",
            status_code=400,
            details={"reason": str(exc)},
            session_id=session_id,
            stage=SessionStage.ERROR,
        ) from exc

    updated = session_service.update_stage(session_id, SessionStage.CLEANED)
    return CleanResponse(
        success=True,
        session_id=session_id,
        stage=updated.stage,
        artifact="df_initial_attach_clean.csv",
        row_count=len(cleaned_df),
        column_count=len(cleaned_df.columns),
    )


@router.get("/sessions/{session_id}/files", response_model=SessionFilesResponse)
async def list_session_files(session_id: str) -> SessionFilesResponse:
    session = session_service.get_session(session_id)
    session_dir = Path(session.base_dir)

    files: list[SessionFileItem] = []
    for path in sorted(session_dir.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            SessionFileItem(
                name=path.name,
                relative_path=str(path.relative_to(session_dir)).replace("\\", "/"),
                size_bytes=path.stat().st_size,
            )
        )

    return SessionFilesResponse(
        success=True,
        session_id=session_id,
        stage=session.stage,
        files=files,
    )

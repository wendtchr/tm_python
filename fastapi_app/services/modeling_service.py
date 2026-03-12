from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from threading import Thread
from typing import Any

from app_files.modules import config as shiny_config
from app_files.modules.data_processing import _get_output_filename, read_csv_with_encoding

from fastapi_app.core.errors import ApiError
from fastapi_app.schemas.model import ModelRunRequest
from fastapi_app.schemas.sessions import SessionMetadata, SessionStage
from fastapi_app.services.session_service import session_service
from fastapi_app.services.task_registry_service import task_registry_service


class ModelingService:
    def enqueue_model_run(self, session_id: str, request: ModelRunRequest) -> str:
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

        source_path = self._resolve_source_path(Path(session.base_dir))
        if source_path is None:
            raise ApiError(
                code="DATA_NOT_LOADED",
                message="No dataset available for modeling",
                status_code=409,
                details={"expected_one_of": ["df_initial.csv", "df_initial_attach.csv", "df_initial_attach_clean.csv"]},
                session_id=session_id,
                stage=session.stage,
            )

        if task_registry_service.has_active_task(session_id):
            raise ApiError(
                code="MODEL_ALREADY_RUNNING",
                message="A modeling task is already running for this session",
                status_code=409,
                details={"session_id": session_id},
                session_id=session_id,
                stage=SessionStage.MODELING_RUNNING,
            )

        task = task_registry_service.create_task(session_id=session_id, message="Model run queued")
        session_service.update_stage(session_id, SessionStage.MODELING_RUNNING)
        Thread(
            target=self._run_modeling_task_sync,
            kwargs={
                "task_id": task.task_id,
                "session": session,
                "source_path": source_path,
                "request": request,
            },
            daemon=True,
        ).start()
        return task.task_id

    def _run_modeling_task_sync(
        self,
        task_id: str,
        session: SessionMetadata,
        source_path: Path,
        request: ModelRunRequest,
    ) -> None:
        asyncio.run(self._run_modeling_task(task_id=task_id, session=session, source_path=source_path, request=request))

    async def _run_modeling_task(
        self,
        task_id: str,
        session: SessionMetadata,
        source_path: Path,
        request: ModelRunRequest,
    ) -> None:
        task_registry_service.mark_running(task_id, "Modeling started")
        try:
            task_registry_service.update_task(task_id, progress=25.0, message="Preparing modeling inputs")
            result = await asyncio.to_thread(
                self._run_modeling_pipeline_sync,
                task_id,
                session,
                source_path,
                request,
            )
            session_service.update_stage(session.session_id, SessionStage.MODELED)
            task_registry_service.mark_succeeded(task_id=task_id, result=result)
        except Exception as exc:
            try:
                session_service.update_stage(session.session_id, SessionStage.ERROR)
            except Exception:
                pass
            task_registry_service.mark_failed(
                task_id=task_id,
                error={"message": str(exc)},
            )

    def _run_modeling_pipeline_sync(
        self,
        task_id: str,
        session: SessionMetadata,
        source_path: Path,
        request: ModelRunRequest,
    ) -> dict[str, Any]:
        return asyncio.run(self._run_modeling_pipeline(task_id, session, source_path, request))

    async def _run_modeling_pipeline(
        self,
        task_id: str,
        session: SessionMetadata,
        source_path: Path,
        request: ModelRunRequest,
    ) -> dict[str, Any]:
        from app_files.modules import topic_modeling

        base_dir = Path(session.base_dir)
        df = read_csv_with_encoding(str(source_path))
        if df.empty:
            raise ValueError("No dataset available for modeling")

        model_config = copy.deepcopy(shiny_config.TOPIC_MODELING)
        model_config["TOPIC"]["min_topic_size"] = request.min_topic_size
        model_config["NGRAM_RANGE"] = (request.ngram_min, request.ngram_max)
        model_config["TOP_N_WORDS"] = request.top_n_words
        model_config["UMAP"]["n_neighbors"] = request.umap_n_neighbors
        model_config["UMAP"]["min_dist"] = request.umap_min_dist
        model_config["UMAP"]["n_components"] = request.umap_n_components

        task_registry_service.update_task(task_id=task_id, progress=45.0)
        modeler = topic_modeling.TopicModeler(
            config_dict=model_config,
            seed_topics=request.seed_topics,
            num_topics="auto",
        )
        processed_df = await modeler.fit_transform_dataframe(df)
        task_registry_service.update_task(task_id=task_id, progress=75.0, message="Saving outputs")
        output = await topic_modeling.save_topic_modeling_outputs(model=modeler, df=processed_df, output_dir=base_dir)

        topics_path = base_dir / shiny_config.TOPIC_OUTPUT_CONFIG["DEFAULT_FILENAME"]
        return {
            "artifact": topics_path.name,
            "row_count": len(processed_df),
            "column_count": len(processed_df.columns),
            "visualizations": sorted([str(p.name) for p in output.get("visualizations", {}).values() if p.exists()]),
        }

    def get_model_summary(self, session_id: str) -> dict[str, Any]:
        session = session_service.get_session(session_id)
        topics_path, df = self._load_topics_dataframe(session.base_dir, session_id, session.stage)
        topic_count = int(df["Topic"].nunique()) if "Topic" in df.columns else 0
        return {
            "session_id": session_id,
            "stage": session.stage,
            "artifact": topics_path.name,
            "row_count": len(df),
            "topic_count": topic_count,
            "columns": [str(col) for col in df.columns],
        }

    def get_model_results(self, session_id: str) -> dict[str, Any]:
        session = session_service.get_session(session_id)
        topics_path, df = self._load_topics_dataframe(session.base_dir, session_id, session.stage)
        return {
            "session_id": session_id,
            "stage": session.stage,
            "artifact": topics_path.name,
            "row_count": len(df),
            "columns": [str(col) for col in df.columns],
            "rows": df.to_dict(orient="records"),
        }

    @staticmethod
    def _load_topics_dataframe(base_dir: str, session_id: str, stage: SessionStage) -> tuple[Path, Any]:
        topics_path = Path(base_dir) / shiny_config.TOPIC_OUTPUT_CONFIG["DEFAULT_FILENAME"]
        if not topics_path.exists():
            raise ApiError(
                code="MODEL_RESULTS_NOT_FOUND",
                message="Model results are not available for this session",
                status_code=404,
                details={"expected": topics_path.name},
                session_id=session_id,
                stage=stage,
            )
        df = read_csv_with_encoding(str(topics_path))
        return topics_path, df

    @staticmethod
    def _resolve_source_path(base_dir: Path) -> Path | None:
        cleaned = _get_output_filename(base_dir, "cleaned")
        attach = _get_output_filename(base_dir, "attach")
        initial = _get_output_filename(base_dir, "initial")
        for candidate in (cleaned, attach, initial):
            if candidate.exists():
                return candidate
        return None


modeling_service = ModelingService()

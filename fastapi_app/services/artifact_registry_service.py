from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from fastapi_app.core.errors import ApiError
from fastapi_app.schemas.sessions import SessionStage
from fastapi_app.services.session_service import session_service


class VisualizationKey(StrEnum):
    TOPIC = "topic"
    HIERARCHY = "hierarchy"
    WORDCLOUD = "wordcloud"
    WORD_SCORES = "word-scores"
    ALIGNMENT = "alignment"


class DownloadArtifactKey(StrEnum):
    DF_INITIAL = "df_initial"
    DF_INITIAL_ATTACH = "df_initial_attach"
    DF_INITIAL_ATTACH_CLEAN = "df_initial_attach_clean"
    DF_TOPICS = "df_topics"
    TOPIC_VISUALIZATION = "topic_visualization"
    TOPIC_HIERARCHY = "topic_hierarchy"
    TOPIC_WORDCLOUD = "topic_wordcloud"
    TOPIC_WORD_SCORES = "topic_word_scores"
    TOPIC_ALIGNMENT = "topic_alignment"
    TOPIC_COMPARISON = "topic_comparison"
    COMPARISON_REPORT = "comparison_report"
    TOPIC_REPORT = "topic_report"


_VISUALIZATION_CANDIDATES: dict[VisualizationKey, list[str]] = {
    VisualizationKey.TOPIC: [
        "visualizations/topic_distribution.html",
        "topic_distribution.html",
    ],
    VisualizationKey.HIERARCHY: [
        "visualizations/topic_hierarchy.html",
        "topic_hierarchy.html",
    ],
    VisualizationKey.WORDCLOUD: [
        "visualizations/topic_wordcloud.png",
        "topic_wordcloud.png",
    ],
    VisualizationKey.WORD_SCORES: [
        "visualizations/topic_word_scores.html",
        "topic_word_scores.html",
        "visualizations/word_scores.html",
        "word_scores.html",
    ],
    VisualizationKey.ALIGNMENT: [
        "topic_comparison/alignment_heatmap.html",
    ],
}

_DOWNLOAD_CANDIDATES: dict[DownloadArtifactKey, list[str]] = {
    DownloadArtifactKey.DF_INITIAL: ["df_initial.csv"],
    DownloadArtifactKey.DF_INITIAL_ATTACH: ["df_initial_attach.csv"],
    DownloadArtifactKey.DF_INITIAL_ATTACH_CLEAN: ["df_initial_attach_clean.csv"],
    DownloadArtifactKey.DF_TOPICS: ["df_topics.csv"],
    DownloadArtifactKey.TOPIC_VISUALIZATION: _VISUALIZATION_CANDIDATES[VisualizationKey.TOPIC],
    DownloadArtifactKey.TOPIC_HIERARCHY: _VISUALIZATION_CANDIDATES[VisualizationKey.HIERARCHY],
    DownloadArtifactKey.TOPIC_WORDCLOUD: _VISUALIZATION_CANDIDATES[VisualizationKey.WORDCLOUD],
    DownloadArtifactKey.TOPIC_WORD_SCORES: _VISUALIZATION_CANDIDATES[VisualizationKey.WORD_SCORES],
    DownloadArtifactKey.TOPIC_ALIGNMENT: _VISUALIZATION_CANDIDATES[VisualizationKey.ALIGNMENT],
    DownloadArtifactKey.TOPIC_COMPARISON: ["topic_comparison/topic_comparison.csv"],
    DownloadArtifactKey.COMPARISON_REPORT: ["topic_comparison/comparison_report.txt"],
    DownloadArtifactKey.TOPIC_REPORT: ["topic_report.html", "reports/topic_report.html"],
}

_MIME_BY_SUFFIX: dict[str, str] = {
    ".csv": "text/csv",
    ".html": "text/html; charset=utf-8",
    ".png": "image/png",
    ".txt": "text/plain; charset=utf-8",
}


class ArtifactRegistryService:
    def resolve_visualization(self, session_id: str, key: VisualizationKey) -> tuple[Path, str]:
        return self._resolve(session_id=session_id, candidates=_VISUALIZATION_CANDIDATES[key], key=key.value)

    def resolve_download(self, session_id: str, key: DownloadArtifactKey) -> tuple[Path, str]:
        return self._resolve(session_id=session_id, candidates=_DOWNLOAD_CANDIDATES[key], key=key.value)

    def _resolve(self, session_id: str, candidates: list[str], key: str) -> tuple[Path, str]:
        session = session_service.get_session(session_id)
        session_dir = Path(session.base_dir).resolve()

        for rel in candidates:
            path = (session_dir / rel).resolve()
            if not self._is_safe(path, session_dir):
                continue
            if path.exists() and path.is_file():
                return path, _MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")

        raise ApiError(
            code="ARTIFACT_NOT_FOUND",
            message="Requested artifact is not available for this session",
            status_code=404,
            details={"artifact_key": key, "candidates": candidates},
            session_id=session_id,
            stage=session.stage,
        )

    @staticmethod
    def _is_safe(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


artifact_registry_service = ArtifactRegistryService()

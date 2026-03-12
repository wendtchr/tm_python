from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from fastapi_app.main import app
from fastapi_app.services.modeling_service import modeling_service
from fastapi_app.services.task_registry_service import task_registry_service


def _create_session(client: TestClient) -> str:
    response = client.post("/api/sessions")
    assert response.status_code == 200
    return response.json()["session_id"]


def _create_loaded_session(client: TestClient) -> str:
    session_id = _create_session(client)
    payload = "Comment\nthis is a loaded record for modeling\n"
    response = client.post(
        f"/api/sessions/{session_id}/upload",
        files={"file": ("input.csv", payload, "text/csv")},
    )
    assert response.status_code == 200
    return session_id


def test_model_summary_and_results_contracts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    client = TestClient(app)
    session_id = _create_loaded_session(client)

    topics_path = tmp_path / session_id / "df_topics.csv"
    pd.DataFrame({"Topic": [0, 1], "Topic_Name": ["topic-0", "topic-1"], "Comment": ["a", "b"]}).to_csv(
        topics_path,
        index=False,
    )

    summary_response = client.get(f"/api/sessions/{session_id}/model/summary")
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert summary["success"] is True
    assert summary["session_id"] == session_id
    assert summary["artifact"] == "df_topics.csv"
    assert summary["row_count"] == 2
    assert summary["topic_count"] == 2
    assert "Topic" in summary["columns"]

    results_response = client.get(f"/api/sessions/{session_id}/model/results")
    assert results_response.status_code == 200
    results = results_response.json()
    assert results["success"] is True
    assert results["session_id"] == session_id
    assert results["artifact"] == "df_topics.csv"
    assert results["row_count"] == 2
    assert len(results["rows"]) == 2
    assert results["rows"][0]["Topic"] == 0


def test_model_summary_missing_results_returns_structured_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    client = TestClient(app)
    session_id = _create_loaded_session(client)

    response = client.get(f"/api/sessions/{session_id}/model/summary")
    assert response.status_code == 404
    body = response.json()["error"]
    assert body["code"] == "MODEL_RESULTS_NOT_FOUND"
    assert body["session_id"] == session_id


def test_model_run_validation_error_uses_structured_payload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    task_registry_service.reset()
    client = TestClient(app)
    session_id = _create_loaded_session(client)

    response = client.post(
        f"/api/sessions/{session_id}/model/run",
        json={"ngram_min": 3, "ngram_max": 1},
    )
    assert response.status_code == 422
    body = response.json()["error"]
    assert body["code"] == "VALIDATION_ERROR"
    assert body["session_id"] is None


def test_model_failure_transitions_session_to_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    task_registry_service.reset()
    client = TestClient(app)
    session_id = _create_loaded_session(client)

    def fail_pipeline(task_id: str, session, source_path: Path, request) -> dict:
        raise RuntimeError("boom")

    monkeypatch.setattr(modeling_service, "_run_modeling_pipeline_sync", fail_pipeline)

    enqueue_response = client.post(f"/api/sessions/{session_id}/model/run", json={})
    assert enqueue_response.status_code == 200
    task_id = enqueue_response.json()["task_id"]

    for _ in range(50):
        status_response = client.get(f"/api/sessions/{session_id}/tasks/{task_id}")
        assert status_response.status_code == 200
        status_body = status_response.json()
        if status_body["status"] == "failed":
            break
        time.sleep(0.02)
    assert status_body["status"] == "failed"

    session_json = json.loads((tmp_path / session_id / "session.json").read_text(encoding="utf-8"))
    assert session_json["stage"] == "ERROR"


def test_parallel_sessions_are_isolated_and_delete_cleans_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    client = TestClient(app)
    first_session = _create_session(client)
    second_session = _create_session(client)

    first_upload = client.post(
        f"/api/sessions/{first_session}/upload",
        files={"file": ("first.csv", "Comment\nfirst payload\n", "text/csv")},
    )
    second_upload = client.post(
        f"/api/sessions/{second_session}/upload",
        files={"file": ("second.csv", "Comment\nsecond payload\n", "text/csv")},
    )
    assert first_upload.status_code == 200
    assert second_upload.status_code == 200

    first_text = (tmp_path / first_session / "df_initial.csv").read_text(encoding="utf-8")
    second_text = (tmp_path / second_session / "df_initial.csv").read_text(encoding="utf-8")
    assert "first payload" in first_text
    assert "second payload" in second_text

    first_delete = client.delete(f"/api/sessions/{first_session}")
    assert first_delete.status_code == 200
    assert not (tmp_path / first_session).exists()
    assert (tmp_path / second_session).exists()

    second_delete = client.delete(f"/api/sessions/{second_session}")
    assert second_delete.status_code == 200
    assert not (tmp_path / second_session).exists()


def test_background_model_task_does_not_block_api(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    task_registry_service.reset()
    client = TestClient(app)
    session_id = _create_loaded_session(client)

    def slow_pipeline(task_id: str, session, source_path: Path, request) -> dict:
        time.sleep(0.2)
        pd.DataFrame({"Topic": [0], "Topic_Name": ["topic-0"]}).to_csv(
            Path(session.base_dir) / "df_topics.csv",
            index=False,
        )
        return {"artifact": "df_topics.csv", "row_count": 1, "column_count": 2, "visualizations": []}

    monkeypatch.setattr(modeling_service, "_run_modeling_pipeline_sync", slow_pipeline)

    enqueue_response = client.post(f"/api/sessions/{session_id}/model/run", json={})
    assert enqueue_response.status_code == 200

    health_response = client.get("/api/health")
    assert health_response.status_code == 200
    create_response = client.post("/api/sessions")
    assert create_response.status_code == 200

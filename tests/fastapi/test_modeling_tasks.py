from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from fastapi_app.main import app
from fastapi_app.services.modeling_service import modeling_service
from fastapi_app.services.task_registry_service import task_registry_service


def _create_loaded_session(client: TestClient) -> str:
    session_id = client.post("/api/sessions").json()["session_id"]
    payload = "Comment\nthis is a valid modeling input sentence with enough words\n"
    response = client.post(
        f"/api/sessions/{session_id}/upload",
        files={"file": ("input.csv", payload, "text/csv")},
    )
    assert response.status_code == 200
    return session_id


def test_model_run_enqueues_and_task_status_succeeds(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    task_registry_service.reset()
    client = TestClient(app)
    session_id = _create_loaded_session(client)

    def fake_pipeline(task_id: str, session, source_path: Path, request) -> dict:
        pd.DataFrame({"Topic": [0], "Topic_Name": ["topic-0"]}).to_csv(
            Path(session.base_dir) / "df_topics.csv",
            index=False,
        )
        return {"artifact": "df_topics.csv", "row_count": 1, "column_count": 2, "visualizations": []}

    monkeypatch.setattr(modeling_service, "_run_modeling_pipeline_sync", fake_pipeline)

    enqueue_response = client.post(f"/api/sessions/{session_id}/model/run", json={})
    assert enqueue_response.status_code == 200
    enqueue_body = enqueue_response.json()
    assert enqueue_body["stage"] == "MODELING_RUNNING"
    assert enqueue_body["status"] == "queued"
    task_id = enqueue_body["task_id"]

    final_status = None
    for _ in range(50):
        status_response = client.get(f"/api/sessions/{session_id}/tasks/{task_id}")
        assert status_response.status_code == 200
        final_status = status_response.json()
        if final_status["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.02)

    assert final_status is not None
    assert final_status["status"] == "succeeded"
    assert final_status["result"]["artifact"] == "df_topics.csv"

    session_json = json.loads((tmp_path / session_id / "session.json").read_text(encoding="utf-8"))
    assert session_json["stage"] == "MODELED"


def test_duplicate_model_run_while_active_returns_conflict(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    task_registry_service.reset()
    client = TestClient(app)
    session_id = _create_loaded_session(client)

    def slow_pipeline(task_id: str, session, source_path: Path, request) -> dict:
        time.sleep(0.2)
        return {"artifact": "df_topics.csv", "row_count": 1, "column_count": 2, "visualizations": []}

    monkeypatch.setattr(modeling_service, "_run_modeling_pipeline_sync", slow_pipeline)

    first_response = client.post(f"/api/sessions/{session_id}/model/run", json={})
    assert first_response.status_code == 200

    second_response = client.post(f"/api/sessions/{session_id}/model/run", json={})
    assert second_response.status_code == 409
    body = second_response.json()["error"]
    assert body["code"] == "MODEL_ALREADY_RUNNING"
    assert body["session_id"] == session_id


def test_task_status_unknown_task_returns_structured_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    task_registry_service.reset()
    client = TestClient(app)
    session_id = _create_loaded_session(client)

    response = client.get(f"/api/sessions/{session_id}/tasks/does-not-exist")
    assert response.status_code == 404
    body = response.json()["error"]
    assert body["code"] == "TASK_NOT_FOUND"
    assert body["session_id"] == session_id

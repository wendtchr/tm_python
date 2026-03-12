from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from fastapi_app.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "tm_python-fastapi"


def test_create_and_delete_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    client = TestClient(app)

    create_response = client.post("/api/sessions")
    assert create_response.status_code == 200
    create_body = create_response.json()

    session_id = create_body["session_id"]
    session_dir = tmp_path / session_id

    assert create_body["stage"] == "INIT"
    assert session_dir.exists()
    assert (session_dir / "temp").exists()
    assert (session_dir / "visualizations").exists()
    assert (session_dir / "reports").exists()
    assert (session_dir / "session.json").exists()

    session_json = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert session_json["session_id"] == session_id
    assert session_json["stage"] == "INIT"

    delete_response = client.delete(f"/api/sessions/{session_id}")
    assert delete_response.status_code == 200
    delete_body = delete_response.json()
    assert delete_body["stage"] == "DELETED"
    assert delete_body["deleted"] is True
    assert not session_dir.exists()


def test_delete_session_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    client = TestClient(app)

    response = client.delete("/api/sessions/unknown_session")
    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] is False
    assert body["stage"] == "DELETED"


def test_upload_sets_loaded_stage_and_creates_initial_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    client = TestClient(app)

    session_id = client.post("/api/sessions").json()["session_id"]
    payload = "Comment,Topic-Human\nhello world,seed-a\n"

    response = client.post(
        f"/api/sessions/{session_id}/upload",
        files={"file": ("input.csv", payload, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stage"] == "LOADED"
    assert body["artifact"] == "df_initial.csv"
    assert body["row_count"] == 1

    initial_path = tmp_path / session_id / "df_initial.csv"
    assert initial_path.exists()

    session_json = json.loads((tmp_path / session_id / "session.json").read_text(encoding="utf-8"))
    assert session_json["stage"] == "LOADED"


def test_upload_missing_required_comment_column_returns_structured_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    client = TestClient(app)

    session_id = client.post("/api/sessions").json()["session_id"]
    payload = "Text\nhello\n"

    response = client.post(
        f"/api/sessions/{session_id}/upload",
        files={"file": ("input.csv", payload, "text/csv")},
    )

    assert response.status_code == 400
    body = response.json()["error"]
    assert body["code"] == "UPLOAD_PROCESSING_ERROR"
    assert body["session_id"] == session_id
    assert "Missing required columns" in body["details"]["reason"]


def test_files_endpoint_lists_session_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    client = TestClient(app)

    session_id = client.post("/api/sessions").json()["session_id"]
    payload = "Comment\nhello world\n"
    upload_response = client.post(
        f"/api/sessions/{session_id}/upload",
        files={"file": ("input.csv", payload, "text/csv")},
    )
    assert upload_response.status_code == 200

    files_response = client.get(f"/api/sessions/{session_id}/files")
    assert files_response.status_code == 200
    body = files_response.json()
    assert body["stage"] == "LOADED"

    file_names = {item["name"] for item in body["files"]}
    assert "session.json" in file_names
    assert "df_initial.csv" in file_names


def test_attachments_process_sets_stage_and_writes_attach_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    client = TestClient(app)

    session_id = client.post("/api/sessions").json()["session_id"]
    payload = "Comment,Attachment Files\nhello world,\n"
    upload_response = client.post(
        f"/api/sessions/{session_id}/upload",
        files={"file": ("input.csv", payload, "text/csv")},
    )
    assert upload_response.status_code == 200

    attachments_response = client.post(f"/api/sessions/{session_id}/attachments/process")
    assert attachments_response.status_code == 200
    body = attachments_response.json()
    assert body["stage"] == "ATTACHMENTS_PROCESSED"
    assert body["artifact"] == "df_initial_attach.csv"

    attach_path = tmp_path / session_id / "df_initial_attach.csv"
    assert attach_path.exists()

    session_json = json.loads((tmp_path / session_id / "session.json").read_text(encoding="utf-8"))
    assert session_json["stage"] == "ATTACHMENTS_PROCESSED"


def test_clean_sets_stage_and_writes_clean_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    client = TestClient(app)

    session_id = client.post("/api/sessions").json()["session_id"]
    payload = "Comment,Attachment Files\nhello world,\n"
    upload_response = client.post(
        f"/api/sessions/{session_id}/upload",
        files={"file": ("input.csv", payload, "text/csv")},
    )
    assert upload_response.status_code == 200

    attachments_response = client.post(f"/api/sessions/{session_id}/attachments/process")
    assert attachments_response.status_code == 200

    clean_response = client.post(f"/api/sessions/{session_id}/clean")
    assert clean_response.status_code == 200
    body = clean_response.json()
    assert body["stage"] == "CLEANED"
    assert body["artifact"] == "df_initial_attach_clean.csv"

    cleaned_path = tmp_path / session_id / "df_initial_attach_clean.csv"
    assert cleaned_path.exists()

    session_json = json.loads((tmp_path / session_id / "session.json").read_text(encoding="utf-8"))
    assert session_json["stage"] == "CLEANED"

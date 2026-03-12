from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from fastapi_app.main import app


def _create_session(client: TestClient) -> str:
    response = client.post("/api/sessions")
    assert response.status_code == 200
    return response.json()["session_id"]


def test_visualization_topic_returns_registered_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    client = TestClient(app)
    session_id = _create_session(client)

    topic_path = tmp_path / session_id / "topic_distribution.html"
    topic_path.write_text("<html><body>topic</body></html>", encoding="utf-8")

    response = client.get(f"/api/sessions/{session_id}/visualizations/topic")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "topic" in response.text


def test_visualization_alignment_missing_returns_structured_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    client = TestClient(app)
    session_id = _create_session(client)

    response = client.get(f"/api/sessions/{session_id}/visualizations/alignment")
    assert response.status_code == 404
    body = response.json()["error"]
    assert body["code"] == "ARTIFACT_NOT_FOUND"
    assert body["session_id"] == session_id


def test_download_known_key_returns_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    client = TestClient(app)
    session_id = _create_session(client)

    topics_path = tmp_path / session_id / "df_topics.csv"
    topics_path.write_text("Topic,Topic_Name\n0,topic-0\n", encoding="utf-8")

    response = client.get(f"/api/sessions/{session_id}/downloads/df_topics")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "Topic,Topic_Name" in response.text


def test_download_unknown_key_returns_validation_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    client = TestClient(app)
    session_id = _create_session(client)

    response = client.get(f"/api/sessions/{session_id}/downloads/not-a-key")
    assert response.status_code == 422
    body = response.json()["error"]
    assert body["code"] == "VALIDATION_ERROR"


def test_download_topic_comparison_available_when_present(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TM_OUTPUT_BASE_DIR", str(tmp_path))
    client = TestClient(app)
    session_id = _create_session(client)

    comparison_dir = tmp_path / session_id / "topic_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = comparison_dir / "topic_comparison.csv"
    comparison_path.write_text("Topic,Human\n0,A\n", encoding="utf-8")

    response = client.get(f"/api/sessions/{session_id}/downloads/topic_comparison")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "Topic,Human" in response.text

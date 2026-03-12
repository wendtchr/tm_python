from __future__ import annotations

from fastapi.testclient import TestClient

from fastapi_app.main import app


def test_parity_ui_root_serves_html() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "tm_python FastAPI parity workflow" in response.text
    assert "Create Session" in response.text
    assert "Process Attachments" in response.text
    assert "Run Topic Modeling" in response.text


def test_parity_ui_includes_seed_topic_defaults() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "health hazards, respiratory illness" in response.text

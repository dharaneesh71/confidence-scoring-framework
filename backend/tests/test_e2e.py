"""
Task 21 — Full end-to-end pipeline test:
Chat → Feedback → Training Status → Deploy → History
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def register_and_login(client, email="e2e@test.com", password="e2epass"):
    client.post("/api/auth/register", json={"email": email, "password": password})
    res = client.post("/api/auth/login", data={"username": email, "password": password})
    data = res.json()
    return data.get("access_token", ""), data.get("role", "user")


def test_full_pipeline_chat_to_feedback(client):
    """Chat → get history_id → submit feedback."""
    token, _ = register_and_login(client, "pipeline@test.com", "pipepass")
    headers  = {"Authorization": f"Bearer {token}"}

    with patch("api.routes.endpoints.llama_service.generate_answer",
               return_value="NLP stands for Natural Language Processing."), \
         patch("api.routes.endpoints.llama_service.is_ready", return_value=True), \
         patch("api.routes.endpoints.chroma_service.search", return_value=[
             {"text": "NLP is a field of AI.", "source": "doc.pdf", "similarity_score": 0.9}
         ]):
        r1 = client.post("/api/query",
                         json={"question": "What is NLP?", "domain": "General"},
                         headers=headers)

    assert r1.status_code == 200
    body = r1.json()
    assert "answer" in body
    assert "confidence_score" in body
    assert "history_id" in body

    # Submit feedback
    r2 = client.post("/api/feedback",
                     json={"history_id": body["history_id"], "rating": 5, "comment": "Great!"},
                     headers=headers)
    assert r2.status_code == 200
    assert r2.json()["message"] == "Feedback received successfully"


def test_admin_retrain_deploy_history_flow(client):
    """Retrain trigger → training status → deploy model → model history."""
    token, _ = register_and_login(client, "admin2@test.com", "adminpass")
    headers  = {"Authorization": f"Bearer {token}"}

    # Training status always works
    r_status = client.get("/api/admin/training-status", headers=headers)
    assert r_status.status_code == 200

    # Deploy model with mocked swap
    with patch("api.routes.endpoints.llama_service.hot_swap_model", return_value=True):
        r_deploy = client.post(
            "/api/admin/deploy-model",
            json={"model_path": "meta-llama/Llama-3.2-1B-Instruct"},
            headers=headers
        )
    assert r_deploy.status_code == 200
    assert r_deploy.json()["status"] == "success"

    # Model history
    r_hist = client.get("/api/admin/model-history", headers=headers)
    assert r_hist.status_code == 200
    assert "history" in r_hist.json()


def test_session_creation_and_retrieval(client):
    """New session created on first query, retrievable afterwards."""
    token, _ = register_and_login(client, "session@test.com", "sesspass")
    headers  = {"Authorization": f"Bearer {token}"}

    with patch("api.routes.endpoints.llama_service.generate_answer",
               return_value="Test answer."), \
         patch("api.routes.endpoints.llama_service.is_ready", return_value=True), \
         patch("api.routes.endpoints.chroma_service.search", return_value=[]):
        r1 = client.post("/api/query",
                         json={"question": "Hello?"},
                         headers=headers)

    assert r1.status_code == 200
    session_id = r1.json()["session_id"]
    assert session_id is not None

    r2 = client.get(f"/api/session/{session_id}", headers=headers)
    assert r2.status_code == 200
    assert r2.json()["session_id"] == session_id

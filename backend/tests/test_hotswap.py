"""Integration tests for hot-swap and model history endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def get_admin_token(client):
    client.post("/api/auth/register", json={"email": "admin@test.com", "password": "test123"})
    res = client.post("/api/auth/login", data={"username": "admin@test.com", "password": "test123"})
    return res.json().get("access_token", "")


def test_model_history_empty(client):
    token = get_admin_token(client)
    res = client.get("/api/admin/model-history",
                     headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert "history" in res.json()


def test_deploy_model_endpoint_exists(client):
    token = get_admin_token(client)
    with patch("api.routes.endpoints.llama_service.hot_swap_model", return_value=True):
        res = client.post(
            "/api/admin/deploy-model",
            json={"model_path": "meta-llama/Llama-3.2-1B-Instruct"},
            headers={"Authorization": f"Bearer {token}"}
        )
    assert res.status_code == 200
    assert res.json()["status"] == "success"


def test_deploy_model_triggers_rollback_on_failure(client):
    token = get_admin_token(client)
    with patch("api.routes.endpoints.llama_service.hot_swap_model", return_value=False), \
         patch("api.routes.endpoints.llama_service.rollback_model", return_value=True):
        res = client.post(
            "/api/admin/deploy-model",
            json={"model_path": "bad/path", "backup_path": "meta-llama/Llama-3.2-1B-Instruct"},
            headers={"Authorization": f"Bearer {token}"}
        )
    assert res.status_code == 200
    assert res.json()["status"] == "rolled_back"
    assert res.json()["rollback_ok"] is True


def test_training_status_endpoint(client):
    token = get_admin_token(client)
    res = client.get("/api/admin/training-status",
                     headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert "status" in res.json()

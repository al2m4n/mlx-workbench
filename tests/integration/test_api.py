"""
Integration tests — run against a live backend.
Start the server first:  uvicorn main:app --port 8000
Then:                    pytest tests/integration/test_api.py -v
"""

import pytest
import httpx

BASE = "http://localhost:8000"


@pytest.fixture(scope="session")
def client():
    return httpx.Client(base_url=BASE, timeout=60)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready(client):
    r = client.get("/health/ready")
    assert r.status_code == 200


def test_info(client):
    r = client.get("/health/info")
    assert r.status_code == 200
    assert "python" in r.json()


def test_models_empty(client):
    r = client.get("/models")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_llm_load_and_chat(client):
    load_r = client.post("/llm/load", json={"model_id": "gpt2"})
    assert load_r.status_code == 200
    key = load_r.json()["model_key"]
    assert "llm::" in key

    chat_r = client.post("/llm/chat", json={
        "model_key": key,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
        "temperature": 0.5,
        "stream": False,
    })
    assert chat_r.status_code == 200
    body = chat_r.json()
    assert "text" in body
    assert "latency_ms" in body


def test_unload_model(client):
    load_r = client.post("/llm/load", json={"model_id": "gpt2-unload-test"})
    key = load_r.json()["model_key"]
    del_r = client.delete(f"/models/{key}")
    assert del_r.status_code == 200

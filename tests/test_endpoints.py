from fastapi.testclient import TestClient

from src.conversion.request_converter import _is_reasoning_model
from src.main import app

client = TestClient(app)


def test_list_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["has_more"] is False
    assert isinstance(data["data"], list)
    # Contains the 3 Claude aliases (one per tier) plus configured tier primaries
    assert len(data["data"]) >= 3
    for item in data["data"]:
        assert item["type"] == "model"
        assert item["object"] == "model"
        assert "display_name" in item
        assert "created_at" in item
        # display_name should not contain raw date suffixes like -20241022
        import re

        assert not re.search(r"-\d{8}", item["display_name"])
    model_ids = [m["id"] for m in data["data"]]
    assert "claude-3-5-sonnet-20241022" in model_ids


def test_api_hello_endpoint():
    res_get = client.get("/api/hello")
    assert res_get.status_code == 200
    assert res_get.json() == {"status": "ok"}

    res_head = client.head("/api/hello")
    assert res_head.status_code == 200


def test_is_reasoning_model():
    assert _is_reasoning_model("deepseek-r1") is True
    assert _is_reasoning_model("deepseek/deepseek-r1") is True
    assert _is_reasoning_model("moonshotai/kimi-k3") is True
    assert _is_reasoning_model("o1-mini") is True
    assert _is_reasoning_model("o3-mini") is True

    # Verify non-reasoning models with similar substrings do not falsely match
    assert _is_reasoning_model("gpt-4o") is False
    assert _is_reasoning_model("custom-provider1") is False
    assert _is_reasoning_model("server1") is False
    assert _is_reasoning_model("ver1") is False

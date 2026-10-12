"""Offline tests for the desktop-compat router (src/api/compat.py).

Covers the acceptance criteria from the spec: organization listing with
PROXY_ORG_ID override, non-expiring OAuth token echo, empty connector/
plugin listings, Anthropic-style 404s for unknown /v1|/api paths, default
404 outside those prefixes, and 401 parity with /v1/messages.
"""

import time

import pytest
from fastapi.testclient import TestClient

from src.api import compat
from src.main import app

client = TestClient(app)


def _override_org_id(monkeypatch, value):
    monkeypatch.setattr(compat, "ORG_ID", value)


def test_list_organizations_returns_org_id():
    response = client.get("/api/organizations")
    assert response.status_code == 200
    data = response.json()
    assert "organizations" in data
    org = data["organizations"][0]
    assert org["uuid"] == compat.ORG_ID
    assert compat.ORG_ID in (org["uuid"], org["id"])


def test_list_organizations_v1_alias():
    response = client.get("/v1/organizations")
    assert response.status_code == 200
    data = response.json()
    assert data["organizations"][0]["uuid"] == compat.ORG_ID


def test_organization_env_override(monkeypatch):
    _override_org_id(monkeypatch, "my-custom-org-id")
    response = client.get("/api/organizations")
    assert response.status_code == 200
    org = response.json()["organizations"][0]
    assert org["uuid"] == "my-custom-org-id"
    assert org["id"] == "my-custom-org-id"


def test_get_single_organization():
    response = client.get(f"/api/organizations/{compat.ORG_ID}")
    assert response.status_code == 200
    assert response.json()["uuid"] == compat.ORG_ID


def test_org_connectors_listing_empty():
    response = client.get(f"/api/organizations/{compat.ORG_ID}/connectors")
    assert response.status_code == 200
    data = response.json()
    assert data["connectors"] == []


def test_org_plugins_listing_empty():
    response = client.get(f"/api/organizations/{compat.ORG_ID}/plugins")
    assert response.status_code == 200
    data = response.json()
    assert data["plugins"] == []


def test_org_mcp_remote_servers_listing_empty():
    response = client.get(f"/api/organizations/{compat.ORG_ID}/mcp/remote_servers")
    assert response.status_code == 200
    data = response.json()
    assert data["remote_servers"] == []


def test_org_unknown_subresource_returns_empty_listing():
    response = client.get(f"/api/organizations/{compat.ORG_ID}/some/new/thing")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data.get("thing"), list)
    assert data["thing"] == []


def test_v1_me():
    response = client.get("/v1/me")
    assert response.status_code == 200
    data = response.json()
    assert data["organization_uuid"] == compat.ORG_ID


def test_oauth_token_echo_non_expiring():
    response = client.post("/v1/oauth/token", json={"grant_type": "refresh_token"})
    assert response.status_code == 200
    data = response.json()
    assert data["access_token"]
    assert data["token_type"] == "Bearer"
    assert data["expires_at"] > time.time() + 10 * 365 * 24 * 3600


def test_mcp_servers_listing_empty():
    response = client.get("/v1/mcp_servers")
    assert response.status_code == 200
    data = response.json()
    assert data["mcp_servers"] == []


def test_connectors_listing_empty():
    response = client.get("/api/connectors")
    assert response.status_code == 200
    assert response.json()["connectors"] == []


def test_plugins_listing_empty():
    response = client.get("/api/plugins")
    assert response.status_code == 200
    assert response.json()["plugins"] == []


def test_unknown_v1_path_anthropic_style_404():
    response = client.get("/v1/definitely-not-a-real-endpoint")
    assert response.status_code == 404
    data = response.json()
    assert data["type"] == "error"
    assert data["error"]["type"] == "not_found_error"
    assert "detail" not in data


def test_unknown_api_path_anthropic_style_404():
    response = client.get("/api/definitely-not-a-real-endpoint")
    assert response.status_code == 404
    data = response.json()
    assert data["type"] == "error"
    assert data["error"]["type"] == "not_found_error"


def test_unknown_v1_post_anthropic_style_404():
    response = client.post("/v1/definitely-not-a-real-endpoint", json={})
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found_error"


def test_random_path_outside_v1_api_keeps_default_404():
    response = client.get("/some/random/path")
    assert response.status_code == 404
    data = response.json()
    assert data == {"detail": "Not Found"}


def test_main_routes_still_take_precedence():
    # The compat catch-all must not shadow existing endpoints.
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 200
    assert client.get("/api/hello").status_code == 200


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/organizations"),
        ("get", "/v1/me"),
        ("post", "/v1/oauth/token"),
        ("get", "/v1/mcp_servers"),
        ("get", "/api/connectors"),
        ("get", "/api/plugins"),
    ],
)
def test_bad_api_key_rejected_with_401(method, path, monkeypatch):
    """When ANTHROPIC_API_KEY is set, compat routes reject bad keys like /v1/messages."""
    from src.core.config import config

    monkeypatch.setattr(config, "anthropic_api_key", "test-secret-key")
    requester = getattr(client, method)
    response = requester(path, headers={"x-api-key": "wrong-key"})
    assert response.status_code == 401


def test_good_api_key_accepted(monkeypatch):
    from src.core.config import config

    monkeypatch.setattr(config, "anthropic_api_key", "test-secret-key")
    response = client.get("/api/organizations", headers={"x-api-key": "test-secret-key"})
    assert response.status_code == 200

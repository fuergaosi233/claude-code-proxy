"""Anthropic compatibility endpoints for the Claude Code desktop app.

The desktop app (and some CLI code paths) probe a small set of Anthropic
control-plane endpoints that have nothing to do with message traffic:
account/organization info, an OAuth token-refresh shape, and
connector/plugin listings. When the proxy only implements /v1/messages,
those probes hit FastAPI's bare 404 and the desktop connectors/plugins
browse UI silently fails to open.

This router implements the minimal payloads those clients need:

- ``GET /api/organizations`` / ``GET /v1/organizations`` — a minimal
  single-organization body. The org id comes from the ``PROXY_ORG_ID``
  env var so clients that echo the org into sub-resource paths get a
  stable value back.
- ``GET /v1/me`` — minimal account info.
- ``POST /v1/oauth/token`` — echoes a non-expiring token so clients that
  try a refresh loop against the proxy don't break. No real OAuth.
- Connector/plugin listings (``/v1/mcp_servers``, ``/api/connectors``, and
  ``/api/organizations/{org}/...`` sub-resources) — empty-list bodies so
  the browse UI renders "no connectors/plugins" instead of erroring.

Anything else under ``/v1`` or ``/api`` falls through to a catch-all route
(registered last) that answers with a well-formed Anthropic error body and
logs the missed method+path at INFO, so users can report missing endpoints.
Paths outside /v1 and /api keep FastAPI's default behavior.
"""

import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse

from src.core.logging import logger

router = APIRouter()

# Stable fake organization id. Clients use it to build sub-resource paths
# like /api/organizations/<org>/plugins, so it must round-trip consistently.
ORG_ID = os.environ.get("PROXY_ORG_ID", "claude-code-proxy-org")

# Far-future epoch (2100-01-01T00:00:00Z) used for token expiry so clients
# that honor expires_at never attempt a refresh against the proxy.
_NON_EXPIRING_EXPIRES_AT = 4102444800

# Sub-resource listings keyed by suffix under /api/organizations/<org>/... .
# Each returns an empty listing so the desktop browse UI renders
# "no connectors/plugins" rather than erroring on a bare 404.
_ORG_LISTINGS = {
    "connectors": {"connectors": [], "has_more": False, "total_count": 0},
    "plugins": {"plugins": [], "has_more": False, "total_count": 0},
    "mcp_servers": {"mcp_servers": [], "has_more": False, "total_count": 0},
    "mcp/remote_servers": {
        "remote_servers": [],
        "has_more": False,
        "total_count": 0,
    },
}


def _org_payload() -> dict:
    """Minimal single-organization body the desktop app expects."""
    return {
        "uuid": ORG_ID,
        "id": ORG_ID,
        "name": "Claude Code Proxy",
        "data_retention": {"enabled": False},
        "role": "admin",
    }


def _me_payload() -> dict:
    """Minimal account-info body for /v1/me."""
    return {
        "id": ORG_ID,
        "email": "proxy@claude-code-proxy.local",
        "name": "Claude Code Proxy User",
        "organization_uuid": ORG_ID,
        "account": {"uuid": ORG_ID, "name": "Claude Code Proxy"},
    }


def _oauth_token_payload() -> dict:
    """Non-expiring token echo. No real OAuth — clients must not refresh."""
    return {
        "access_token": "claude-code-proxy-static-token",
        "token_type": "Bearer",
        "expires_in": _NON_EXPIRING_EXPIRES_AT,
        "expires_at": _NON_EXPIRING_EXPIRES_AT,
        "refresh_token": "claude-code-proxy-static-refresh-token",
    }


async def _compat_auth(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
) -> None:
    """Auth parity with /v1/messages: reuse the main router's validator.

    Delegates to endpoints.validate_api_key (which honors ANTHROPIC_API_KEY)
    so these compat routes reject bad keys with 401 exactly like /v1/messages.
    """
    from src.api.endpoints import validate_api_key

    await validate_api_key(x_api_key=x_api_key, authorization=authorization)


@router.get("/api/organizations")
@router.get("/v1/organizations")
async def list_organizations(_: None = Depends(_compat_auth)) -> dict:
    """Minimal single-organization listing the desktop app probes."""
    return {"organizations": [_org_payload()], "count": 1, "has_more": False}


@router.get("/api/organizations/{org_id}")
@router.get("/v1/organizations/{org_id}")
async def get_organization(org_id: str, _: None = Depends(_compat_auth)) -> dict:
    """Single-organization detail; always describes the proxy's fake org."""
    return _org_payload()


@router.get("/api/organizations/{org_id}/{resource_path:path}")
async def get_org_subresource(
    org_id: str, resource_path: str, _: None = Depends(_compat_auth)
) -> dict:
    """Empty listings for connector/plugin/mcp sub-resources.

    The desktop app builds /api/organizations/<org>/<resource> URLs for
    its Connectors and Plugins browse tabs. Unknown suffixes also get an
    empty listing (better than a hard 404 mid-UI), while the generic
    catch-all below still logs the path for reporting.
    """
    payload = _ORG_LISTINGS.get(resource_path.rstrip("/"))
    if payload is None:
        logger.info(
            "Compat org sub-resource not specifically stubbed: " "GET /api/organizations/%s/%s",
            org_id,
            resource_path,
        )
        resource_key = resource_path.rstrip("/").rsplit("/", 1)[-1]
        payload = {resource_key: [], "has_more": False, "total_count": 0}
    return payload


@router.get("/v1/me")
async def get_me(_: None = Depends(_compat_auth)) -> dict:
    """Minimal account info for clients probing /v1/me."""
    return _me_payload()


@router.post("/v1/oauth/token")
async def oauth_token(_: None = Depends(_compat_auth)) -> dict:
    """Echo a non-expiring token so refresh loops never break.

    There is no real OAuth server; the static token simply never expires.
    """
    return _oauth_token_payload()


@router.get("/v1/mcp_servers")
async def list_mcp_servers(_: None = Depends(_compat_auth)) -> dict:
    """Empty MCP/connector listing (Claude Code account-connectors probe)."""
    return {"mcp_servers": [], "has_more": False, "total_count": 0}


@router.get("/api/connectors")
async def list_connectors(_: None = Depends(_compat_auth)) -> dict:
    """Empty connector listing so the browse UI shows 'no connectors'."""
    return {"connectors": [], "has_more": False, "total_count": 0}


@router.get("/api/plugins")
async def list_plugins(_: None = Depends(_compat_auth)) -> dict:
    """Empty plugin listing so the browse UI shows 'no plugins'."""
    return {"plugins": [], "has_more": False, "total_count": 0}


@router.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def compat_catch_all(full_path: str, request: Request) -> JSONResponse:
    """Well-formed Anthropic-style 404 for unknown /v1 and /api paths.

    FastAPI's default 404 body ({"detail": "Not Found"}) breaks clients
    that expect Anthropic's error envelope. Anything unmatched under
    /v1 or /api gets {"type": "error", "error": {...}} instead, and the
    missed method+path is logged at INFO so users can report missing
    endpoints. Everything outside /v1 and /api keeps the default 404.
    """
    is_anthropic_shaped = (
        full_path == "v1"
        or full_path.startswith("v1/")
        or (full_path == "api" or full_path.startswith("api/"))
    )
    if is_anthropic_shaped:
        logger.info(
            "Unimplemented Anthropic path hit compat catch-all: %s /%s",
            request.method,
            full_path,
        )
        return JSONResponse(
            status_code=404,
            content={
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": (
                        f"Endpoint not found: {request.method} /{full_path}. "
                        "This proxy implements /v1/messages and a minimal set of "
                        "Anthropic compatibility endpoints; this path is not among them."
                    ),
                },
            },
        )
    # Outside /v1 and /api: preserve FastAPI's default 404 behavior.
    return JSONResponse(
        status_code=404,
        content={"detail": "Not Found"},
    )

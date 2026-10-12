# Spec: Desktop app connectors/plugins compatibility (job 001)

## Goal

When Claude Code (CLI or desktop app) points at this proxy, the desktop app cannot browse or add
connectors and plugins. The proxy only implements `/v1/messages`, `/v1/messages/count_tokens`,
`/v1/models`, `/health`, `/test-connection`, `/api/hello`, and `/`; every other Anthropic API path
the client probes (account/organization info, OAuth token shapes, connector/plugin listing) hits a
bare FastAPI 404 and the UI feature silently fails. Fix it generically: implement the minimal set of
Anthropic compatibility endpoints Claude Code actually needs, plus a catch-all layer so any other
unknown Anthropic-shaped path returns a well-formed Anthropic-style response instead of an HTML 404.

## Background

- All routes live in `src/api/endpoints.py` on one `APIRouter`; `src/main.py` mounts it on the
  FastAPI app. FastAPI's default 404 returns JSON `{"detail": "Not Found"}`, but more importantly the
  *paths themselves* are missing — the desktop app expects account, organization, and (behind a
  feature flag) connector/plugin endpoints to exist.
- Community proxies (LiteLLM's Anthropic-compatible UI, `claude-code-proxy` forks) solved this by
  returning minimal valid payloads for `/api/organizations` / `/v1/me`-style account endpoints and a
  generic stub for everything else, rather than implementing real marketplaces.

## Approach

1. **Investigate first** (builder): run the proxy, point Claude Code desktop at it
   (`ANTHROPIC_BASE_URL=http://localhost:8082`), watch `logs/proxy.log` + access logs for 404s from
   the app, and cross-check against web research on which Anthropic API surface the desktop client
   calls (Anthropic docs, LiteLLM/claude-code-proxy issues). Record the observed path list in the PR
   description. Do not merge web-research claims without a probe confirming the desktop app calls them.
2. **New module `src/api/compat.py`** with its own `APIRouter`, mounted in `src/main.py` after the
   main router (explicit compatibility routes take precedence; a catch-all route registered last
   covers the rest). Keeps `endpoints.py` focused on message traffic; mirrors the existing
   router-per-concern pattern. Reuse the existing `validate_api_key` dependency for auth parity.
3. **Explicit compatibility routes** (per-path handling where a real payload is required):
   - `GET /api/organizations` (and `/v1/organizations` if research confirms the desktop app uses it)
     — a minimal single-organization JSON body with a stable fake org UUID from a new env var
     `PROXY_ORG_ID` (default `claude-code-proxy-org`).
   - `GET /v1/me`-style account info if probing shows the app requests it — minimal account JSON.
   - `POST /v1/oauth/token` / token-refresh shape — echo back a non-expiring token so clients that
     try a refresh loop don't break (no real OAuth).
   - Connector/plugin listing endpoints surfaced by the investigation — return empty-list bodies
     (`{"connectors": [], ...}`) so the browse UI renders "no connectors" instead of erroring.
4. **Catch-all route** (`/{full_path:path}` for GET/POST, registered last): any other unmatched
   path under `/v1/...` or `/api/...` returns HTTP 404 with a well-formed Anthropic error body
   (`{"type": "error", "error": {"type": "not_found_error", "message": ...}}`) rather than FastAPI's
   default, and logs the unmatched method+path at INFO so users can report missing endpoints.
   Everything outside `/v1` and `/api` keeps the current default behavior.
5. **Live-verification script** `scripts/test_desktop_compat.py` (httpx against a running proxy on
   :8082, same style as `scripts/test_cancellation.py`; not part of pytest) probing each new
   endpoint and printing a checklist.
6. **Docs**: update README.md (feature bullet + "Environment Variables" section for
   `PROXY_ORG_ID`), QUICKSTART.md (one line: desktop app now works), `.env.example`
   (`PROXY_ORG_ID` commented entry).

## Acceptance criteria

1. `uv run pytest tests/` passes; the offline subset
   (`tests/test_endpoints.py`, `tests/test_effort_mapper.py`, `tests/test_context_manager.py`)
   includes new offline tests (via `fastapi.testclient.TestClient`) asserting:
   a. `GET /api/organizations` returns 200 with JSON containing an `org UUID` key and the
      `PROXY_ORG_ID` value (env override respected).
   b. `POST /v1/oauth/token` returns 200 with a JSON body containing a non-empty token field and
      an expiry far in the future.
   c. The connector/plugin listing route returns 200 with an empty-list payload.
   d. `GET /v1/definitely-not-a-real-endpoint` returns 404 with JSON body
      `{"type": "error", "error": {"type": "not_found_error", ...}}` (not FastAPI's
      `{"detail": "Not Found"}`), and `GET /some/random/path` outside `/v1|/api` is untouched.
   e. When `ANTHROPIC_API_KEY` is set, the new routes reject bad keys with 401 (same as `/v1/messages`).
2. `uv run black src/ && uv run isort src/` produces no diff.
3. With the proxy running, `python scripts/test_desktop_compat.py` exits 0 and prints per-endpoint
   PASS lines; running the Claude Code desktop app against `http://localhost:8082` no longer breaks
   the connectors/plugins browse UI — it opens and shows an empty list (verified manually by the
   user; the script is the automatable proxy-side half).
4. README.md, QUICKSTART.md, and `.env.example` document the new endpoints and `PROXY_ORG_ID`.

## Out of scope

- No real OAuth server or token issuance — the refresh endpoint echoes a static token.
- No actual plugin/connector marketplace backend — listing endpoints return empty lists; adding a
  connector through the UI will not persist anything.
- No change to `/v1/messages`, conversion logic, model mapping, or effort handling.
- No CORS/auth changes beyond reusing the existing `validate_api_key` dependency.

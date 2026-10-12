#!/usr/bin/env python3
"""Live verification for the desktop connectors/plugins compat endpoints.

Probes each new compatibility route on a running proxy (default
http://localhost:8082) and prints a per-endpoint PASS/FAIL checklist.
Not part of the pytest suite — this needs a live server, like
scripts/test_cancellation.py.

Usage:
    python scripts/test_desktop_compat.py [base_url]
"""

import sys

import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8082"

failures = []


def check(name, response, predicates):
    """Run predicates against an httpx response; record PASS/FAIL."""
    try:
        body = response.json()
    except Exception:
        body = None
    problems = []
    for label, ok in predicates:
        if not ok(response, body):
            problems.append(label)
    if problems:
        failures.append((name, problems, response.status_code, body))
        print(f"FAIL  {name} (HTTP {response.status_code}) — {', '.join(problems)}")
    else:
        print(f"PASS  {name} (HTTP {response.status_code})")


def main():
    print(f"Probing desktop-compat endpoints against {BASE_URL}")
    print("=" * 60)

    with httpx.Client(timeout=10) as client:
        # 0. Server must be up.
        health = client.get(f"{BASE_URL}/health")
        if health.status_code != 200:
            print(f"FAIL  server health check (HTTP {health.status_code}) — is the proxy running?")
            print("      Start it with: uv run claude-code-proxy (or ./start.sh)")
            sys.exit(1)
        print(f"PASS  server is running (HTTP {health.status_code})")

        # 1. Organization listing.
        check(
            "GET /api/organizations",
            client.get(f"{BASE_URL}/api/organizations"),
            [
                ("status 200", lambda r, b: r.status_code == 200),
                (
                    "org uuid present",
                    lambda r, b: isinstance(b, dict)
                    and b.get("organizations")
                    and b["organizations"][0].get("uuid"),
                ),
            ],
        )
        check(
            "GET /v1/organizations",
            client.get(f"{BASE_URL}/v1/organizations"),
            [
                ("status 200", lambda r, b: r.status_code == 200),
                (
                    "org uuid present",
                    lambda r, b: isinstance(b, dict)
                    and b.get("organizations")
                    and b["organizations"][0].get("uuid"),
                ),
            ],
        )

        # 2. Account info.
        check(
            "GET /v1/me",
            client.get(f"{BASE_URL}/v1/me"),
            [
                ("status 200", lambda r, b: r.status_code == 200),
                ("has id", lambda r, b: isinstance(b, dict) and b.get("id")),
            ],
        )

        # 3. OAuth token echo (non-expiring).
        check(
            "POST /v1/oauth/token",
            client.post(f"{BASE_URL}/v1/oauth/token", json={"grant_type": "refresh_token"}),
            [
                ("status 200", lambda r, b: r.status_code == 200),
                ("non-empty access_token", lambda r, b: b.get("access_token")),
                (
                    "expiry far in future",
                    lambda r, b: b.get("expires_at", 0) > 4102444800 - 10 * 365 * 24 * 3600,
                ),
            ],
        )

        # 4. Connector/plugin listings render as empty lists.
        check(
            "GET /v1/mcp_servers",
            client.get(f"{BASE_URL}/v1/mcp_servers"),
            [
                ("status 200", lambda r, b: r.status_code == 200),
                ("empty mcp_servers list", lambda r, b: b.get("mcp_servers") == []),
            ],
        )
        check(
            "GET /api/connectors",
            client.get(f"{BASE_URL}/api/connectors"),
            [
                ("status 200", lambda r, b: r.status_code == 200),
                ("empty connectors list", lambda r, b: b.get("connectors") == []),
            ],
        )
        check(
            "GET /api/plugins",
            client.get(f"{BASE_URL}/api/plugins"),
            [
                ("status 200", lambda r, b: r.status_code == 200),
                ("empty plugins list", lambda r, b: b.get("plugins") == []),
            ],
        )

        # Grab the org id for sub-resource probes.
        orgs = client.get(f"{BASE_URL}/api/organizations").json()
        org_id = orgs["organizations"][0]["uuid"]
        for sub in ("connectors", "plugins", "mcp/remote_servers"):
            check(
                f"GET /api/organizations/{{org}}/{sub}",
                client.get(f"{BASE_URL}/api/organizations/{org_id}/{sub}"),
                [
                    ("status 200", lambda r, b: r.status_code == 200),
                    (
                        "empty listing",
                        lambda r, b: isinstance(b, dict)
                        and isinstance(b.get(sub.rsplit("/", 1)[-1]), list)
                        and b[sub.rsplit("/", 1)[-1]] == [],
                    ),
                ],
            )

        # 5. Unknown /v1 and /api paths get Anthropic-style 404 bodies.
        check(
            "GET /v1/definitely-not-a-real-endpoint",
            client.get(f"{BASE_URL}/v1/definitely-not-a-real-endpoint"),
            [
                ("status 404", lambda r, b: r.status_code == 404),
                (
                    "anthropic error body",
                    lambda r, b: b == b
                    and b.get("type") == "error"
                    and b.get("error", {}).get("type") == "not_found_error",
                ),
                (
                    "not FastAPI default",
                    lambda r, b: b != {"detail": "Not Found"},
                ),
            ],
        )
        check(
            "GET /api/definitely-not-a-real-endpoint",
            client.get(f"{BASE_URL}/api/definitely-not-a-real-endpoint"),
            [
                ("status 404", lambda r, b: r.status_code == 404),
                (
                    "anthropic error body",
                    lambda r, b: b.get("type") == "error"
                    and b.get("error", {}).get("type") == "not_found_error",
                ),
            ],
        )
        check(
            "GET /some/random/path keeps default 404",
            client.get(f"{BASE_URL}/some/random/path"),
            [("status 404", lambda r, b: r.status_code == 404)],
        )

        # 6. Existing routes unaffected.
        check(
            "GET /v1/models still works",
            client.get(f"{BASE_URL}/v1/models"),
            [("status 200", lambda r, b: r.status_code == 200)],
        )
        check(
            "GET /api/hello still works",
            client.get(f"{BASE_URL}/api/hello"),
            [("status 200", lambda r, b: r.status_code == 200)],
        )

    print("=" * 60)
    if failures:
        print(f"{len(failures)} check(s) FAILED")
        sys.exit(1)
    print("All desktop-compat checks passed.")


if __name__ == "__main__":
    main()

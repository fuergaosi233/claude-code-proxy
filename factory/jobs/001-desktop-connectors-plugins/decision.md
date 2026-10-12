# Decision: job 001-desktop-connectors-plugins

APPROVE

## Rationale

- **Acceptance criteria met.** I independently ran the offline suite
  (`uv run pytest tests/test_effort_mapper.py tests/test_endpoints.py
  tests/test_context_manager.py tests/test_compat_endpoints.py`) — 61 passed —
  and `black --check src/` / `isort --check-only src/` are clean. The new
  tests in `tests/test_compat_endpoints.py` genuinely assert every spec item:
  PROXY_ORG_ID override (a), non-expiring token (b), empty listings (c),
  Anthropic-style 404 inside `/v1|/api` with default 404 outside (d), and
  401-on-bad-key parity with `/v1/messages` (e). Docs are synced: README
  feature bullet + `PROXY_ORG_ID` env entry, QUICKSTART paragraph, commented
  `.env.example` line.
- **PASS verdicts spot-checked and deserved.** I read the full branch diff.
  The many touched files outside the new feature are formatting-only
  (verified via `git diff -w`: pure black/isort reflow, no logic changes —
  including the odd-looking `test_main.py` hunks, which are black fixing
  misindented literals). `src/api/compat.py` matches what the reviews describe:
  catch-all registered last in a router mounted after the main router, so
  `/v1/messages`, `/health`, `/v1/models`, `/docs` etc. keep precedence
  (also confirmed live by the api review on a scratch :8799 server); all
  explicit routes reuse `validate_api_key` via `_compat_auth`; the catch-all
  serves only 404s, logs only method+path, and exposes no data.
- **The stub design is safe to merge on its own.** No real OAuth, no secrets,
  no new dependencies, no changes to message traffic/conversion/effort
  handling. The failure mode of an incomplete endpoint set is benign: the app
  gets a well-formed 404 (logged at INFO) exactly as before, never a worse
  outcome than pre-change. This makes the disclosed deviation (endpoint list
  from community knowledge, not a live desktop probe) low-risk.
- **Both disclosed deviations are acceptable.** `PROXY_ORG_ID` read at module
  import is consistent with how the rest of the codebase reads env
  (module-level constants, `load_dotenv` at `src/__init__.py` before compat
  import) and is covered by a monkeypatch test. The missing live probe
  deviates from the spec's approach step 1, but the catch-all's INFO logging
  surfaces any still-missing paths on the first real desktop run, and the
  spec's criterion 3 assigns the manual desktop-UI check to the user anyway.

## Notes for the human (not blockers)

- The manual half of acceptance criterion 3 (opening the desktop app's
  connectors/plugins UI against `http://localhost:8082` and seeing empty
  lists) is still outstanding by design — do that check on first use; the
  catch-all's INFO log lines ("Unimplemented Anthropic path hit compat
  catch-all") are the place to look for gaps.
- The user's planned pass-through job (forwarding connector endpoints to
  api.anthropic.com) supersedes these stubs; the stub layer is a safe interim
  state and nothing here constrains that follow-up.
- Minor code-review nits (deferred import in `_compat_auth`, duplicated
  empty-listing literals, the `b == b` leftover in
  `scripts/test_desktop_compat.py:156`) are cosmetic and can ride along with
  the follow-up job.

# Refined Prompt — Job 002: `/v1/models` cache TTL

Raw feature text: "Add a `/v1/models` cache TTL so model discovery doesn't hit upstream on every request"

## Finding that reframes this job (read first)

The premise of the raw text does not match the current code. Two different things are
called "models endpoints" in this repo:

1. **`GET /v1/models` (the proxy's own route)** — `src/api/endpoints.py:310-362`.
   This is **entirely static**: it returns three hardcoded Claude alias entries
   (one per tier) and never contacts the upstream. There is no per-request upstream
   hit to cache here.
2. **Upstream model discovery** — `src/core/provider_discovery.py`
   (`run_discovery_and_patch`), launched once in the background at server startup
   (`src/main.py:24-41`). It queries the upstream `{base_url}/models`, extracts
   context-window/output limits, and already persists them to a local cache file
   (`model_contexts_cache.json`, resolved relative to CWD) so that subsequent
   startups skip the upstream call when every configured model's limits are known.

So "doesn't hit upstream on every request" is already true today. The **actual gap**
the raw text points at: the discovery cache has **no freshness/TTL mechanism**.
Limits are fetched once and then trusted forever — across restarts (cache file)
and within a long-running process (discovery runs only at startup). If the provider
changes a model's context window, or a new model id is added to `BIG/MIDDLE/SMALL_MODEL`
mid-session, the proxy keeps stale limits until the cache file is manually deleted
and the server restarted.

This job is therefore interpreted as: **add an optional TTL to the upstream
model-discovery cache, so limits are periodically refreshed instead of being fetched
once and frozen.**

## Assumptions

- **Confirmed (from code):** `GET /v1/models` in `src/api/endpoints.py` is static and
  does not call upstream. No work is needed on that route for this job.
- **Confirmed (from code):** Upstream discovery runs once at startup via
  `run_discovery_and_patch` (`src/main.py`), backed by `model_contexts_cache.json`
  on disk and `context_manager._DISCOVERED_LIMITS` in memory.
- **Confirmed (from code):** `context_manager.get_model_limits()` precedence is
  `MODEL_CONTEXT_LIMITS` env overrides > runtime-discovered limits > built-in table.
  Discovery results must never override explicit `MODEL_CONTEXT_LIMITS` entries;
  the implementation must preserve this.
- **Confirmed (from code):** Only the first provider in the rotation is probed today
  (`src/main.py`: "We only probe the first provider in the rotation for limits").
  This job should not change that; a TTL refresh refreshes the same single source.
- **Confirmed (from project profile):** All configuration via env vars, documented in
  both `.env.example` and README "Environment Variables".
- **To-confirm:** The reframing above (TTL for the *discovery cache*, not the static
  `/v1/models` route) is what the user wants — see Open Questions Q1.
- **To-confirm:** Default TTL value and whether the feature is off by default
  (see Q2).
- **To-confirm:** Whether the cache file should move out of CWD to a stable location
  (e.g. project-root `logs/` or a `cache/` dir) as part of this job, and whether it
  is gitignored (see Q3).

## Open questions for the user

Only Q1 actually gates the design; Q2/Q3 are cheap to answer alongside it.

1. **Scope confirmation:** The proxy's `GET /v1/models` is static and already makes no
   upstream calls. Is the goal (as this prompt assumes) to add a TTL/periodic refresh
   to the upstream *model-limit discovery* cache (`provider_discovery.py`), rather than
   to change anything about the `/v1/models` response itself?
2. **Default TTL:** Should refresh be off by default (`MODEL_DISCOVERY_TTL=0` =
   current behavior, discovery only at startup), or should there be a nonzero
   default (e.g. 24 h)? Proposed: off by default; opt-in via env var.
3. **Cache file location:** `model_contexts_cache.json` is currently written to the
   process CWD. Move it to a fixed path (project root, like `logs/.last_model`) and
   ensure it is gitignored, or leave location as-is?

## Enhanced prompt

*(One feature — not split. The static `/v1/models` route needs no change; if the user
answers Q1 with "actually expose real upstream models in /v1/models", that is a
different backlog item and this prompt should be re-refined.)*

Add an optional TTL to the upstream model-limit discovery cache so stale context
limits are refreshed instead of frozen, gated entirely by a new env var.

### Behavior

1. New env var `MODEL_DISCOVERY_TTL` (seconds). `0` or unset = current behavior
   (discovery only at startup, cache file reused indefinitely). Positive value =
   the in-memory discovery results are considered stale after that many seconds.
2. Freshness metadata: record when the upstream discovery last succeeded (a
   `"_fetched_at"` epoch field inside the JSON cache file, or equivalent). The
   startup path and any refresh path both consult it.
3. Refresh mechanism (pick one, smallest that satisfies the goal): a lightweight
   background loop in `src/main.py` startup (alongside the existing
   `start_health_checker`) that re-runs `run_discovery_and_patch` for the first
   provider when the cache is older than `MODEL_DISCOVERY_TTL`. Refresh must be
   best-effort: on failure, log at WARNING, keep the existing cached limits, and
   retry on the next TTL boundary (no tight retry loop).
4. Precedence guarantee: refreshed discovered limits still never shadow explicit
   `MODEL_CONTEXT_LIMITS` env overrides (`get_model_limits()` order is env >
   discovered > builtin — keep it that way).
5. No changes to the `GET /v1/models` route response.

### Files expected to change

- `src/core/provider_discovery.py` — timestamped cache entries / stale check.
- `src/main.py` — optional background refresh loop.
- `src/core/config.py` — read `MODEL_DISCOVERY_TTL` from env.
- `.env.example` + README.md "Environment Variables" (+ QUICKSTART.md if it lists
  discovery behavior) — document the new var.
- `.gitignore` — cache file, if Q3 says move/pin it.

### Conventions

- Env-var-only config, never hardcoded; document in `.env.example` and README.
- Offline tests must pass without a live proxy:
  `uv run pytest tests/test_effort_mapper.py tests/test_endpoints.py tests/test_context_manager.py`
- `uv run black src/ && uv run isort src/` (line-length 100).
- Conventional commit ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Testable goals

1. With `MODEL_DISCOVERY_TTL` unset/`0`, behavior is byte-identical to today:
   discovery runs once at startup; no background refresh task is scheduled.
2. With `MODEL_DISCOVERY_TTL=60` and a cache older than 60 s, a refresh is
   attempted; with a cache younger than 60 s, it is skipped (assert from logs or by
   injecting a fake clock / monkeypatched `discover_provider_limits` call counter).
3. A failed refresh (upstream unreachable) leaves the previously cached limits
   intact and active — `get_model_limits()` still returns the old values.
4. `MODEL_CONTEXT_LIMITS` overrides still win over discovered limits after a
   refresh (existing precedence test/behavior unchanged).
5. Offline test subset passes: `uv run pytest tests/test_effort_mapper.py
   tests/test_endpoints.py tests/test_context_manager.py`.
6. New env var appears in both `.env.example` and README "Environment Variables".

## Edge cases

- **Upstream down at refresh time:** keep old limits; no exception escapes the
  background loop (the loop must never crash the server).
- **TTL=0 / unset / negative / non-numeric values:** treat as disabled; negative or
  garbage must not crash startup.
- **Clock skew / system suspend:** stale check uses monotonic-or-epoch timestamps
  stored in the cache; a wildly future `_fetched_at` should not permanently block
  refresh (clamp or ignore implausible values).
- **Concurrent access:** the background refresh writes the JSON cache while request
  threads read `_DISCOVERED_LIMITS`; write atomically (temp file + `os.replace`,
  same pattern as `_write_last_model`) and treat the in-memory dict update as
  replace-per-key, never a destructive swap that could race a reader.
- **Manual cache file edits:** the cache file is user-editable today (documented as
  a fallback injection point); a refresh must not clobber manually added entries for
  models the upstream still doesn't report — merge, don't overwrite the whole dict.
- **Multi-key providers:** only provider[0] is probed (existing limitation, keep);
  TTL applies to that one source. Note this in the env-var doc comment.
- **Cache file in CWD:** if the server is launched from different directories, the
  cache silently forks (Q3). Whatever the answer, the refresh logic must not create
  duplicate discovery storms across stale copies.
- **First-run with no cache:** `MODEL_DISCOVERY_TTL` set but cache file absent —
  normal startup discovery happens immediately, then TTL cadence applies.

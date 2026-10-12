"""
Provider model discovery: query the upstream /v1/models endpoint at startup
and extract real context-window limits so context_manager uses accurate numbers
instead of relying solely on the built-in lookup table.

Supported providers (auto-detected by inspecting the response fields):
  - NVIDIA NIM  : model.max_context_length  /  model.context_length
  - Together AI : model.context_length  /  model.max_context_length
  - OpenAI      : model.context_window  (not always present in /v1/models)
  - Generic     : any of the above field names, plus common variants

The discovered limits are written into context_manager._DISCOVERED_LIMITS
(a dict keyed by model id) which get_model_limits() checks first.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

_log = logging.getLogger("provider_discovery")


async def discover_provider_limits(
    base_url: str,
    api_key: str,
    timeout: float = 10.0,
    custom_headers: Optional[dict] = None,
) -> dict[str, tuple[int, int]]:
    """Query {base_url}/models and return {model_id: (ctx_tokens, max_output)}.

    Returns an empty dict if the request fails or returns no useful data.
    This is best-effort — failures are logged at WARNING, never raised.
    """
    url = base_url.rstrip("/") + "/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    if custom_headers:
        headers.update(custom_headers)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        _log.warning("Provider model discovery failed (%s): %s", url, exc)
        return {}

    models_raw = data if isinstance(data, list) else data.get("data", [])
    if not isinstance(models_raw, list):
        _log.warning("Unexpected /v1/models response shape from %s", url)
        return {}

    discovered: dict[str, tuple[int, int]] = {}
    for m in models_raw:
        if not isinstance(m, dict):
            continue
        model_id: Optional[str] = m.get("id")
        if not model_id:
            continue

        ctx = _extract_context_length(m)
        out = _extract_max_output(m, ctx)
        if ctx:
            discovered[model_id] = (ctx, out)
            _log.debug("Discovered [%s] ctx=%d out=%d", model_id, ctx, out)

    _log.info(
        "Provider discovery complete: %d model(s) with known context limits from %s",
        len(discovered),
        url,
    )
    return discovered


def _extract_context_length(m: dict) -> int:
    """Try every known field name for context window size."""
    # Ordered by reliability / prevalence
    candidates = [
        "max_context_length",  # NVIDIA NIM primary field
        "context_length",  # Together AI, some NIM models
        "context_window",  # OpenAI extended metadata
        "max_tokens",  # generic
        "total_tokens",  # some older APIs
    ]
    for field in candidates:
        val = m.get(field)
        if val and isinstance(val, int) and val > 0:
            return val

    # Nested: model_card / metadata sub-dicts (NIM sometimes nests them)
    for wrapper in ("model_card", "metadata", "capabilities"):
        sub = m.get(wrapper)
        if isinstance(sub, dict):
            for field in candidates:
                val = sub.get(field)
                if val and isinstance(val, int) and val > 0:
                    return val

    return 0


def _extract_max_output(m: dict, ctx: int) -> int:
    """Try to find a max-output field; fall back to 25% of context."""
    candidates = [
        "max_output_tokens",
        "max_completion_tokens",
        "max_output",
    ]
    for field in candidates:
        val = m.get(field)
        if val and isinstance(val, int) and val > 0:
            return val
    for wrapper in ("model_card", "metadata", "capabilities"):
        sub = m.get(wrapper)
        if isinstance(sub, dict):
            for field in candidates:
                val = sub.get(field)
                if val and isinstance(val, int) and val > 0:
                    return val

    # Fallback: assume 25% of context for output
    return max(4096, ctx // 4)


async def run_discovery_and_patch(
    base_url: str,
    api_key: str,
    custom_headers: Optional[dict] = None,
) -> None:
    """Discover limits, utilizing a local cache file to avoid unnecessary API calls."""
    import json
    import os

    from src.core import context_manager
    from src.core.config import config

    cache_file = "model_contexts_cache.json"
    cached_limits: dict[str, tuple[int, int]] = {}

    # 1. Load from local cache if it exists
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                raw_cache = json.load(f)
                for k, v in raw_cache.items():
                    if isinstance(v, list) and len(v) == 2:
                        cached_limits[k] = (v[0], v[1])
            _log.info(
                "Loaded %d model limits from local cache (%s)", len(cached_limits), cache_file
            )
        except Exception as exc:
            _log.warning("Failed to load local context cache: %s", exc)

    # Put cached limits into the active mapping immediately
    if cached_limits:
        context_manager._DISCOVERED_LIMITS.update(cached_limits)

    # 2. Check if we actually need to make an API call
    # We only call the API if a specifically configured model is totally unknown
    configured_models = set(config.big_models + config.middle_models + config.small_models)

    missing_models = []
    for model in configured_models:
        model_lower = (model or "").lower()
        # Is it in env overrides?
        if model_lower in context_manager._ENV_OVERRIDES or any(
            k in model_lower for k in context_manager._ENV_OVERRIDES
        ):
            continue
        # Is it in the cache we just loaded?
        if model in cached_limits or any(k.lower() in model_lower for k in cached_limits):
            continue

        # We purposely DO NOT check _BUILTIN_LIMITS here. We want to fetch the real
        # accurate limits from the provider rather than settling for our hardcoded guesses.

        missing_models.append(model)

    if not missing_models:
        _log.info("All configured models have known context limits. Skipping upstream discovery.")
        return

    _log.info(
        "Missing context limits for configured models: %s. Initiating upstream discovery.",
        missing_models,
    )

    # 3. Make the API call to discover limits
    new_limits = await discover_provider_limits(
        base_url=base_url,
        api_key=api_key,
        custom_headers=custom_headers,
    )

    # 4. Process all missing models (fallback if API didn't provide metadata)
    added_to_cache = False

    # Update live mapping with any new discoveries first
    if new_limits:
        context_manager._DISCOVERED_LIMITS.update(new_limits)
        cached_limits.update(new_limits)
        added_to_cache = True

    # For any configured model that NIM/OpenAI didn't provide context metadata for,
    # we inject a fallback so the user can manually edit it in the JSON file.
    for model in missing_models:
        if model not in cached_limits:
            # First try our built-in table
            model_lower = (model or "").lower()
            found = False
            for marker, ctx, out in context_manager._BUILTIN_LIMITS:
                if marker in model_lower:
                    cached_limits[model] = (ctx, out)
                    found = True
                    break

            # Absolute fallback
            if not found:
                cached_limits[model] = (32_000, 4_096)

            context_manager._DISCOVERED_LIMITS[model] = cached_limits[model]
            added_to_cache = True

    if added_to_cache:
        # 5. Save back to disk
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached_limits, f, indent=2)
            _log.info("Saved %d total model limits to %s", len(cached_limits), cache_file)
        except Exception as exc:
            _log.warning("Failed to save local context cache: %s", exc)

import asyncio
import itertools
import logging
import os
import sys
import threading
import time

_log = logging.getLogger("config")


# Configuration
class Config:
    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Add Anthropic API key for client validation
        raw_anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if raw_anthropic_key:
            raw_anthropic_key = raw_anthropic_key.strip().strip("'\"")
        self.anthropic_api_key = raw_anthropic_key if raw_anthropic_key else None
        if not self.anthropic_api_key:
            _log.info("ANTHROPIC_API_KEY not set or empty. Client API key validation is disabled.")

        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.azure_api_version = os.environ.get("AZURE_API_VERSION")  # For Azure OpenAI
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8082"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "4096"))
        self.min_tokens_limit = int(os.environ.get("MIN_TOKENS_LIMIT", "100"))

        # --- Multi‑provider support via OPENAI_API_KEY (comma‑separated) ---
        # If OPENAI_API_KEY contains commas, treat each as a separate key sharing the same base URL.
        if self.openai_api_key and "," in self.openai_api_key:
            keys = [k.strip() for k in self.openai_api_key.split(",") if k.strip()]
            self.providers = [{"api_key": k, "base_url": self.openai_base_url} for k in keys]
        else:
            # Fallback to single provider configuration
            self.providers = [{"api_key": self.openai_api_key, "base_url": self.openai_base_url}]

        # Round‑robin iterator over providers
        self._provider_cycle = itertools.cycle(self.providers)

        # Usage tracking for RPM limits
        self._provider_usage = {
            p["api_key"]: {"count": 0, "window_start": time.time()} for p in self.providers
        }
        self._provider_penalties = {}  # api_key -> penalty_end_time
        self._rpm_limit = int(os.environ.get("RPM_LIMIT", "40"))  # RPM limit per key

        # Connection settings
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "90"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))

        # Optional default for reasoning effort. This is a DEFAULT, not an
        # override: when the client (Claude Code) sends reasoning_effort via
        # /effort, that per-request value always wins and this is ignored.
        # Accepts low|medium|high|max|xhigh (xhigh normalizes to max).
        # Validated lazily by the effort mapper so a bad value degrades to the
        # built-in default instead of crashing the proxy at startup.
        self.reasoning_effort_override = os.environ.get("REASONING_EFFORT")

        # Optional per-tier effort defaults (BIG_MODEL_EFFORT, etc.). When set,
        # these take precedence over the global REASONING_EFFORT for that tier
        # (but still lose to an explicit /effort). Each is independent: setting
        # only SMALL_MODEL_EFFORT leaves big/middle on the global default.
        self.big_model_effort = os.environ.get("BIG_MODEL_EFFORT")
        self.middle_model_effort = os.environ.get("MIDDLE_MODEL_EFFORT")
        self.small_model_effort = os.environ.get("SMALL_MODEL_EFFORT")

        # --- Multi-model support with fallback (comma-separated) ---
        # Each tier can have a list of models separated by commas.
        # The first model is the primary; subsequent ones are fallbacks.
        # When the primary model encounters errors (429, 503, model not found),
        # the proxy automatically switches to the next fallback.
        # A background health checker periodically tests the primary and
        # switches back when it's healthy again.
        raw_big = os.environ.get("BIG_MODEL", "gpt-4o")
        raw_middle = os.environ.get("MIDDLE_MODEL", raw_big)
        raw_small = os.environ.get("SMALL_MODEL", "gpt-4o-mini")

        self.big_models = [m.strip() for m in raw_big.split(",") if m.strip()]
        self.middle_models = [m.strip() for m in raw_middle.split(",") if m.strip()]
        self.small_models = [m.strip() for m in raw_small.split(",") if m.strip()]

        # Active model index for each tier (0 = primary, 1+ = fallback)
        self._active_big_idx = 0
        self._active_middle_idx = 0
        self._active_small_idx = 0

        # Model health tracking: model_name -> {"failed_at": timestamp, "consecutive_failures": count}
        self._model_failures: dict = {}
        self._model_failure_threshold = int(os.environ.get("MODEL_FAILURE_THRESHOLD", "3"))
        self._model_health_check_interval = int(os.environ.get("MODEL_HEALTH_CHECK_INTERVAL", "60"))

        # --- Probe-based primary restoration ---
        # Restoring the primary model from a fallback is gated on a real probe
        # request succeeding against the primary. This eliminates the flip-flop /
        # thrashing behavior that occurred when the primary was still unhealthy
        # but the loop restored it blindly after a fixed interval.
        self._probe_timeout = int(os.environ.get("MODEL_PROBE_TIMEOUT", "10"))
        self._probe_max_tokens = int(os.environ.get("MODEL_PROBE_MAX_TOKENS", "1"))
        # After a failed restore attempt, wait this many health-check intervals
        # before retrying. Doubles on each consecutive failure, capped at this value.
        self._restore_backoff_max_intervals = int(os.environ.get("MODEL_RESTORE_BACKOFF_MAX", "8"))
        # Track consecutive probe failures per tier so we can back off gracefully
        # for tiers whose primary remains unhealthy.
        self._restore_probe_failures: dict = {}

        # Easy accessors for single-model compatibility (updated dynamically)
        self.big_model = self.big_models[0]
        self.middle_model = self.middle_models[0]
        self.small_model = self.small_models[0]

        # Lock for thread-safe model index updates
        self._model_lock = threading.Lock()

        # Background health checker task (started after event loop is running)
        self._health_check_task = None

    def get_effort_default(self, tier: str):
        """Resolve the env-configured default effort for a model tier.

        Precedence: tier-specific (BIG/MIDDLE/SMALL_MODEL_EFFORT) > global
        REASONING_EFFORT. Returns None when nothing is configured; the effort
        mapper then uses its built-in default ("high"). An explicit /effort
        from the client still overrides whatever this returns.
        """
        per_tier = {
            "big": self.big_model_effort,
            "middle": self.middle_model_effort,
            "small": self.small_model_effort,
        }.get((tier or "").lower())
        return per_tier or self.reasoning_effort_override

    def validate_api_key(self):
        """Basic API key validation"""
        if not self.openai_api_key:
            return False
        # Basic format check for OpenAI API keys
        if not self.openai_api_key.startswith("sk-"):
            return False
        return True

    def validate_client_api_key(self, client_api_key):
        """Validate client's Anthropic API key"""
        # If no ANTHROPIC_API_KEY is set in environment, skip validation
        if not self.anthropic_api_key:
            return True

        # Check if the client's API key matches the expected value
        return client_api_key == self.anthropic_api_key

    def get_custom_headers(self):
        """Get custom headers from environment variables"""
        custom_headers = {}

        # Get all environment variables
        env_vars = dict(os.environ)

        # Find CUSTOM_HEADER_* environment variables
        for env_key, env_value in env_vars.items():
            if env_key.startswith("CUSTOM_HEADER_"):
                # Convert CUSTOM_HEADER_KEY to Header-Key
                # Remove 'CUSTOM_HEADER_' prefix and convert to header format
                header_name = env_key[14:]  # Remove 'CUSTOM_HEADER_' prefix

                if header_name:  # Make sure it's not empty
                    # Convert underscores to hyphens for HTTP header format
                    header_name = header_name.replace("_", "-")
                    custom_headers[header_name] = env_value

        return custom_headers

    def report_rate_limit(self, api_key: str):
        """Mark a provider as rate-limited and set a cooldown penalty."""
        penalty_duration = 60  # 60 seconds cooldown
        self._provider_penalties[api_key] = time.time() + penalty_duration
        _log.warning(
            "Provider %s... reported rate limit. Penalizing for %ds.", api_key[:8], penalty_duration
        )

    def get_next_provider(self):
        """Return the next provider configuration respecting RPM limits.
        Each provider can be used up to 40 times per minute. If a provider has
        exhausted its quota or is under penalty, the method skips it and checks the next one. If all
        providers are exhausted, it raises a RuntimeError indicating the need
        to wait before making more requests.
        """
        # Clean up usage windows that have expired
        now = time.time()
        for api_key, usage in self._provider_usage.items():
            if now - usage["window_start"] >= 60:
                usage["count"] = 0
                usage["window_start"] = now
        # Attempt to find a provider with remaining quota
        for _ in range(len(self.providers)):
            provider = next(self._provider_cycle)
            api_key = provider["api_key"]
            usage = self._provider_usage[api_key]

            # Skip if under penalty
            if api_key in self._provider_penalties:
                if now < self._provider_penalties[api_key]:
                    continue
                else:
                    del self._provider_penalties[api_key]

            if usage["count"] < self._rpm_limit:
                # Increment usage for this selection
                usage["count"] += 1
                return provider
        # All providers exhausted
        raise RuntimeError(
            "All API keys have reached their RPM limit or are under penalty. Wait before retrying."
        )

    async def get_next_client_async(self):
        """Create an OpenAIClient for the next provider (async, non-blocking).
        If all API keys have reached their RPM limit, returns None so the caller
        can handle backpressure gracefully (e.g., return 429/503 to the client).
        """
        # Lazy import to avoid circular dependency
        from src.core.client import OpenAIClient

        try:
            prov = self.get_next_provider()
        except RuntimeError:
            return None  # Signal backpressure instead of blocking

        custom_headers = self.get_custom_headers()
        return OpenAIClient(
            api_key=prov["api_key"],
            base_url=prov["base_url"],
            timeout=self.request_timeout,
            api_version=self.azure_api_version,
            custom_headers=custom_headers,
        )

    def get_next_client(self):
        """Synchronous wrapper: obtain a client, with brief blocking if keys exhausted.
        For async endpoints, prefer get_next_client_async().
        """
        # Lazy import to avoid circular dependency
        from src.core.client import OpenAIClient

        try:
            prov = self.get_next_provider()
        except RuntimeError:
            # All keys exhausted – calculate shortest wait and sleep briefly
            now = time.time()
            wait_times = []
            for usage in self._provider_usage.values():
                elapsed = now - usage["window_start"]
                remaining = max(0, 60 - elapsed)
                wait_times.append(remaining)
            wait = min(wait_times) if wait_times else 60
            _log.warning(
                "All API keys exhausted. Waiting %ds (of %ds needed) before retrying.",
                int(min(wait, 2.0)),
                int(wait),
            )
            time.sleep(min(wait, 2.0))  # Cap at 2s to avoid hanging
            prov = self.get_next_provider()
        custom_headers = self.get_custom_headers()
        return OpenAIClient(
            api_key=prov["api_key"],
            base_url=prov["base_url"],
            timeout=self.request_timeout,
            api_version=self.azure_api_version,
            custom_headers=custom_headers,
        )

    # --- Model fallback / hot-swap ---

    def _get_model_list(self, tier: str) -> tuple:
        """Get the model list and index attr name for a tier."""
        if tier == "big":
            return self.big_models, "_active_big_idx"
        elif tier == "middle":
            return self.middle_models, "_active_middle_idx"
        else:
            return self.small_models, "_active_small_idx"

    def get_active_model(self, tier: str) -> str:
        """Get the currently active model for a tier."""
        models, idx_attr = self._get_model_list(tier)
        idx = getattr(self, idx_attr)
        return models[idx] if idx < len(models) else models[0]

    def report_model_failure(self, model_name: str, tier: str):
        """Report a model failure. If threshold reached, switch to next fallback."""
        now = time.time()
        with self._model_lock:
            models, idx_attr = self._get_model_list(tier)
            current_idx = getattr(self, idx_attr)

            if current_idx >= len(models):
                return  # No more fallbacks available

            current_model = models[current_idx]
            if current_model != model_name:
                return  # Stale report for a non-active model

            entry = self._model_failures.get(
                model_name, {"failed_at": now, "consecutive_failures": 0}
            )
            entry["consecutive_failures"] += 1
            entry["failed_at"] = now
            self._model_failures[model_name] = entry

            if entry["consecutive_failures"] >= self._model_failure_threshold:
                next_idx = current_idx + 1
                if next_idx < len(models):
                    setattr(self, idx_attr, next_idx)
                    new_model = models[next_idx]
                    old_model = model_name
                    # Update easy accessor
                    if tier == "big":
                        self.big_model = new_model
                    elif tier == "middle":
                        self.middle_model = new_model
                    else:
                        self.small_model = new_model
                    _log.warning(
                        "🔄 MODEL SWITCH [%s] %s → %s (after %d failures)",
                        tier.upper(),
                        old_model,
                        new_model,
                        entry["consecutive_failures"],
                    )
                    from src.core.logging import log_active_models

                    log_active_models(_log)
                else:
                    _log.warning("All %s tier models exhausted. No fallback available.", tier)

    def report_model_success(self, model_name: str, tier: str):
        """Report a successful model call, resetting failure counters."""
        with self._model_lock:
            if model_name in self._model_failures:
                del self._model_failures[model_name]

    async def _probe_model(self, model_name: str) -> bool:
        """Issue a tiny completion to verify whether a model is healthy.
        Returns True on success, False on any error / timeout. Uses the next
        available provider key (separate from RPM accounting of caller) so a
        probe cannot starve real traffic beyond a single bounded call.
        """
        # Lazy import to avoid circular dependency
        from src.core.client import OpenAIClient

        try:
            client = self.get_next_client()
        except RuntimeError:
            # No provider available right now; treat as probe failure.
            return False

        probe_request = {
            "model": model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": self._probe_max_tokens,
        }

        try:
            await asyncio.wait_for(
                client.create_chat_completion(probe_request, model_tier="unknown"),
                timeout=self._probe_timeout,
            )
            return True
        except asyncio.TimeoutError:
            _log.debug("Probe timeout for '%s' (%ds)", model_name, self._probe_timeout)
            return False
        except Exception as e:
            _log.debug("Probe failed for '%s': %s", model_name, str(e)[:120])
            return False

    async def check_and_restore_primary_async(self):
        """Probe each tier's primary model before flipping back from a fallback.
        Only restores when the probe succeeds. On probe failure, backs off
        subsequent restore attempts (exponential, capped).
        """
        for tier in ["big", "middle", "small"]:
            models, idx_attr = self._get_model_list(tier)
            # Read current index without holding the lock during the (slow) probe
            current_idx = getattr(self, idx_attr)
            if not (current_idx > 0 and current_idx < len(models)):
                # On primary (or exhausted) -> clear any stale backoff state
                self._restore_probe_failures.pop(tier, None)
                continue

            primary = models[0]
            # Determine whether we should attempt a probe now, given backoff
            consecutive_failures = self._restore_probe_failures.get(tier, 0)
            backoff_intervals = min(2**consecutive_failures, self._restore_backoff_max_intervals)
            failure_entry = self._model_failures.get(primary)
            if failure_entry:
                elapsed = time.time() - failure_entry["failed_at"]
                if elapsed < self._model_health_check_interval * backoff_intervals:
                    continue  # Too soon to retry
            # If we are on a fallback but have no failure_entry (shouldn't happen
            # normally), probe immediately.

            healthy = await self._probe_model(primary)
            if healthy:
                _log.info(
                    "Probe succeeded: restoring primary %s model '%s' after %d probe failure(s)",
                    tier,
                    primary,
                    consecutive_failures,
                )
                with self._model_lock:
                    # Re-check the index under the lock; another caller may have
                    # mutated it concurrently. Only flip if still on a fallback.
                    now_idx = getattr(self, idx_attr)
                    if now_idx > 0 and now_idx < len(models):
                        setattr(self, idx_attr, 0)
                        if tier == "big":
                            self.big_model = primary
                        elif tier == "middle":
                            self.middle_model = primary
                        else:
                            self.small_model = primary
                        # Clear failure tracking so one real failure re-penalizes
                        # from scratch.
                        self._model_failures.pop(primary, None)
                # Reset backoff for this tier
                self._restore_probe_failures.pop(tier, None)
            else:
                # Probe failed -> increment backoff counter, refresh failed_at
                # so the next attempt waits another full backoff window.
                self._restore_probe_failures[tier] = consecutive_failures + 1
                _log.info(
                    "Probe '%s' still unhealthy (%s tier); backing off for %d intervals",
                    primary,
                    tier,
                    min(
                        2 ** self._restore_probe_failures[tier], self._restore_backoff_max_intervals
                    ),
                )
                with self._model_lock:
                    self._model_failures[primary] = {
                        "failed_at": time.time(),
                        "consecutive_failures": (
                            failure_entry.get("consecutive_failures", 0)
                            if failure_entry
                            else self._model_failure_threshold
                        ),
                    }

    async def _run_health_check_loop(self):
        """Background task: periodically attempt to probe-restore primary models."""
        while True:
            await asyncio.sleep(self._model_health_check_interval)
            try:
                await self.check_and_restore_primary_async()
            except Exception as e:
                _log.error("Health check error: %s", e)

    def start_health_checker(self):
        """Start the background model health checker (must be called with a running event loop)."""
        if self._health_check_task is None:
            try:
                loop = asyncio.get_running_loop()
                self._health_check_task = loop.create_task(self._run_health_check_loop())
                _log.info(
                    "Model health checker started (interval: %ds).",
                    self._model_health_check_interval,
                )
                # Log initial active models
                from src.core.logging import log_active_models

                log_active_models(_log)
            except RuntimeError:
                _log.warning("Cannot start health checker: no running event loop.")


try:
    config = Config()
    _log.info("Configuration loaded: API_KEY=%s, BASE_URL='%s'", "*" * 20, config.openai_base_url)
    _log.info("Big models: %s", config.big_models)
    _log.info("Middle models: %s", config.middle_models)
    _log.info("Small models: %s", config.small_models)
except Exception as e:
    _log.critical("Configuration Error: %s", e)
    sys.exit(1)

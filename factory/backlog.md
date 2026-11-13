# Factory Backlog

Features queued for `/factory next`. One per line, checked off when run.

- [x] Add a `/v1/models` cache TTL so model discovery doesn't hit upstream on every request
- [ ] Health-check endpoint reports per-key RPM and cooldown status
- [ ] Optional per-request token-usage logging (env-var gated) for cost tracking
- [ ] Pass-through mode: forward desktop-app connector/plugin/org endpoints to api.anthropic.com instead of stubbing them (no restrictions on connectors; needs user's real Anthropic auth to work)

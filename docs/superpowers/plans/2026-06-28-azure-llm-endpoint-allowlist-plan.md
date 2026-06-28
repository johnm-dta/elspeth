# Web-authored Azure LLM endpoint allowlist — implementation plan

**Date:** 2026-06-28
**Status:** Ready to implement
**Design (source of truth):** `docs/superpowers/specs/2026-06-28-azure-llm-endpoint-allowlist-design.md`
**Sibling precedent (mirror EXACTLY, different provider field):** commit `ba5a24aad`
(OpenRouter `base_url` canonical-pin gate).

## 0. What this builds (one paragraph)

A web author can compose an `llm` node with `provider="azure"` whose `endpoint` is
an author-controlled free string, while its `api_key` resolves **server-side**
(`AZURE_API_KEY` ships in `WebSettings.server_secret_allowlist` by default). That
asymmetry lets an author point the server's Azure bearer at an attacker-controlled
or internal host — a credential-egress / SSRF vector. We close it with an
operator-tunable host **allowlist**, default `("*.openai.azure.com",)`, enforced as
a new blocking validation check `azure_llm_endpoint_policy` (declared position #11,
immediately after `llm_base_url_policy` #10) inside `validate_pipeline` — the single
fail-closed chokepoint already audited for the OpenRouter gate. This mirrors the
OpenRouter gate's exact structure across five source files plus tests.

The gate matches hosts dot-anchored: `*.openai.azure.com` matches
`myresource.openai.azure.com` but **not** `evil-openai.azure.com` (no dot before
`openai`) and **not** the apex `openai.azure.com`. The helper requires HTTPS itself
(parse with `urlsplit`; non-https or hostless → fail closed). An empty allowlist
`()` denies all web-authored Azure LLM endpoints (the "disable web Azure LLM"
posture, no code change).

## 1. Files touched

| # | File | Change |
|---|------|--------|
| A | `src/elspeth/web/config.py` | `WebSettings.allowed_azure_llm_endpoints` field + `field_validator` |
| B | `src/elspeth/web/provider_config_policy.py` | `AZURE_LLM_ENDPOINT_POLICY_ERROR` + `web_azure_llm_endpoint_policy_error` + `_host_matches_any` + imports |
| C | `src/elspeth/web/execution/schemas.py` | register `azure_llm_endpoint_policy` (Literal + `Final` const + tuple insertion at #11) |
| D | `src/elspeth/web/execution/protocol.py` | extend `ValidationSettings` Protocol with `allowed_azure_llm_endpoints` |
| E | `src/elspeth/web/execution/validation.py` | schemas import + policy import + alias + the per-node gate loop after `llm_base_url_policy` |
| F | `tests/unit/web/execution/test_validation.py` | helper unit tests, integration tests, count 16→17, cascade extension |
| G | `tests/unit/web/test_config.py` | config-validator tests |

All edited source files (A–E) sit under the web tier governed by
`config/cicd/enforce_tier_model/web.yaml`, but today only `config.py`
(`_allow_insecure_test_keys`) and `schemas.py` (`RunEvent._resolve_data_from_event_type`)
carry *recorded* fingerprints there, and those anchors are unrelated to the regions
this plan edits — so the staleness risk is low. Treat the §6 tier-model check as
VERIFY-then-rotate: run it, and only rotate if an AST shift actually staled a
recorded fingerprint.

## 2. Implementation order (TDD — mirror how the OpenRouter gate landed)

Land in this order so failing tests precede implementation and the formatter's
F401 import-strip gotcha is avoided:

1. **F (tests, helper + config)** — write the failing helper unit tests
   (`TestWebAzureLlmEndpointPolicyHelper`) and the config-validator tests (G). They
   fail at import (`web_azure_llm_endpoint_policy_error` / field do not exist yet).
2. **A (config field + validator)** — make G pass.
3. **B (policy helper)** — make the helper unit tests pass. **Edit ordering matters
   (F401 gotcha):** add the *using code* (the new functions that reference `urlsplit`
   and `Sequence`) **first**, then update the import lines — see §3.B.
4. **C (schemas registration)** — add the check name (Literal + const + tuple).
5. **D (protocol)** — extend `ValidationSettings`.
6. **E (validation gate loop)** — wire the gate; mirror the `llm_base_url_policy`
   loop exactly.
7. **F (integration + maintenance)** — write `TestValidatePipelineAzureLlmEndpointPolicy`,
   bump the all-checks-passed count 16→17, extend the skipped-after-failure cascade.
8. Run the full verification set (§6).

---

## 3. Concrete edits

### A. `src/elspeth/web/config.py`

**A1 — add the field.** Anchor: the `server_secret_allowlist` tuple field (currently
lines 161–167). Insert the new field immediately **after** its closing `)` and
**before** `orphan_run_max_age_seconds` (line 168):

```python
    server_secret_allowlist: tuple[str, ...] = (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_API_KEY",
        "AZURE_CONTENT_SAFETY_KEY",
    )
    # Web-authored Azure OpenAI LLM nodes carry an author-controlled `endpoint`
    # while their `api_key` resolves server-side (AZURE_API_KEY is allowlisted
    # above). This host allowlist is the egress gate: a web-authored Azure LLM
    # endpoint is permitted only when its host matches one of these patterns.
    # Each entry is a HOST PATTERN, not a URL: either a "*.suffix" dot-anchored
    # subdomain wildcard or an exact dotted hostname. The non-empty default
    # permits any genuine Azure OpenAI resource and blocks arbitrary/internal
    # hosts; operators tighten to their resource host or extend for a private
    # OpenAI-compatible gateway. An empty tuple () denies all web-authored
    # Azure LLM endpoints. Mirrors server_secret_allowlist's tuple+validator.
    allowed_azure_llm_endpoints: tuple[str, ...] = ("*.openai.azure.com",)
    orphan_run_max_age_seconds: int = Field(default=3600, ge=60)
```

**A2 — add the validator.** Anchor: immediately **after** `_validate_server_secret_allowlist`
(currently lines 416–423) and **before** the `_validate_auth_fields` model_validator
(line 425). Mirror `_validate_server_secret_allowlist` in shape (a `@field_validator`
that lowercase-normalises and rejects malformed entries):

```python
    @field_validator("allowed_azure_llm_endpoints")
    @classmethod
    def _validate_allowed_azure_llm_endpoints(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """Normalise + validate web-authored Azure LLM endpoint host patterns.

        Entries are hostname patterns, not URLs. Each is either a ``*.suffix``
        dot-anchored subdomain wildcard or an exact dotted hostname. Reject any
        entry that carries a scheme, a path, a port, embedded credentials, or
        whitespace, so an operator cannot accidentally allowlist a URL or a
        credentialed string. A leading ``*.`` is permitted; the remainder must
        be a valid dotted hostname. Mirrors ``_validate_server_secret_allowlist``.
        An empty tuple is accepted (the deny-all posture).
        """

        def _is_dns_label(label: str) -> bool:
            # ASCII DNS label: 1–63 chars of [a-z0-9-], not edge-hyphenated.
            if not label or len(label) > 63:
                return False
            if label[0] == "-" or label[-1] == "-":
                return False
            return all((ch.isascii() and ch.isalnum()) or ch == "-" for ch in label)

        validated: list[str] = []
        for raw in v:
            entry = raw.lower()
            if not entry:
                raise ValueError("allowed_azure_llm_endpoints entries must not be empty")
            if any(ch.isspace() for ch in entry):
                raise ValueError(f"allowed_azure_llm_endpoints entry must not contain whitespace: {raw!r}")
            if "://" in entry:
                raise ValueError(f"allowed_azure_llm_endpoints entry must be a host, not a URL (scheme found): {raw!r}")
            if "/" in entry:
                raise ValueError(f"allowed_azure_llm_endpoints entry must be a host, not a path: {raw!r}")
            if "@" in entry:
                raise ValueError(f"allowed_azure_llm_endpoints entry must not embed credentials: {raw!r}")
            if ":" in entry:
                raise ValueError(f"allowed_azure_llm_endpoints entry must not include a port: {raw!r}")
            host = entry[2:] if entry.startswith("*.") else entry
            labels = host.split(".")
            if len(labels) < 2 or not all(_is_dns_label(label) for label in labels):
                raise ValueError(
                    f"allowed_azure_llm_endpoints entry must be a valid dotted hostname "
                    f"(optionally '*.'-prefixed): {raw!r}"
                )
            validated.append(entry)
        return tuple(validated)
```

Notes:
- **No new module-level import.** The label check is a nested helper to keep AST
  churn localised (avoids the wide `import re` Module.body shift that cascades
  tier-model fingerprints — see §6). The field + validator both land *after*
  config.py's only fingerprinted suppression (`_allow_insecure_test_keys`, line 33),
  so its fp should be unaffected; still run the tier-model check.
- `:` rejection runs after the `://` check, so a leftover `:` is a port.
- `*.` itself (no remainder) → `host=""` → `labels=[""]` → `len < 2` → rejected.

### B. `src/elspeth/web/provider_config_policy.py`

This file is the sibling of `web_llm_base_url_policy_error`. The helper requires
HTTPS itself (parse with `urlsplit`; non-https or hostless → error), then does a
dot-anchored host match.

**Edit ordering (F401 gotcha — the OpenRouter change hit this).** The PostToolUse
ruff autofix strips a freshly added import whose first use has not yet landed. So:

**B1 — add the using code FIRST.** Anchor: end of file, immediately **after**
`web_llm_base_url_policy_error` (which ends at line 144/145, returning
`LLM_BASE_URL_POLICY_ERROR`). Append the constant and both functions. They reference
`Sequence`, `urlsplit`, `Mapping`, `Any`, `Final` — `Mapping`, `Any`, `Final` are
already imported; `Sequence`, `urlsplit` are added in B2:

```python
AZURE_LLM_ENDPOINT_POLICY_ERROR: Final[str] = (
    "Web-authored Azure OpenAI LLM nodes may only target an operator-allowlisted "
    "endpoint host. The api_key resolves server-side, so an arbitrary endpoint would "
    "direct the server-held bearer to an author-chosen destination (a credential-"
    "egress / SSRF path). Use an allowlisted Azure OpenAI host (default: any "
    "*.openai.azure.com resource), or have the operator add the host to "
    "WebSettings.allowed_azure_llm_endpoints; private gateways are an operator-"
    "controlled runtime concern."
)


def _host_matches_any(host: str, entries: Sequence[str]) -> bool:
    """Return whether ``host`` matches any allowlist entry.

    ``*.base.domain`` entries match dot-anchored subdomains (``x.base.domain``
    but NOT ``base.domain`` nor ``evilbase.domain``). Other entries match the
    host exactly. Entries are re-stripped/lowercased defensively even though the
    config validator already normalises them.
    """
    for entry in entries:
        e = entry.strip().lower()
        if e.startswith("*."):
            base = e[2:]
            if base and host.endswith("." + base):  # dot-anchored subdomain
                return True
        elif host == e:
            return True
    return False


def web_azure_llm_endpoint_policy_error(
    plugin: str | None,
    options: Mapping[str, Any],
    allowed_endpoints: Sequence[str],
) -> str | None:
    """Reject web-authored Azure LLM endpoints whose host is not allowlisted.

    The Azure provider sends the server-resolved api_key to the author-chosen
    ``endpoint``. In a web-authored pipeline the author cannot read that
    server-scoped credential, so an arbitrary endpoint is a credential-egress /
    SSRF vector. Only the ``llm`` plugin with ``provider == "azure"`` is in
    scope. This gate requires HTTPS itself (rather than leaning on the later
    config-construction gate) so the failure is attributed here with a clear
    message; a non-https or hostless endpoint fails closed.
    """
    if plugin != "llm" or options.get("provider") != "azure":
        return None
    endpoint = options.get("endpoint")
    if not isinstance(endpoint, str):
        # Missing/non-str endpoint is rejected at config construction (pydantic);
        # this network-policy gate only adjudicates author-chosen string hosts.
        return None
    parsed = urlsplit(endpoint.strip())
    if parsed.scheme != "https" or not parsed.hostname:
        return AZURE_LLM_ENDPOINT_POLICY_ERROR
    if _host_matches_any(parsed.hostname.lower(), allowed_endpoints):
        return None
    return AZURE_LLM_ENDPOINT_POLICY_ERROR
```

**B2 — add the imports SECOND** (after B1's use-site exists, so ruff keeps them).
Anchor: the current import block at the top of the file:

```python
from collections.abc import Mapping
from typing import Any, Final
```

Change to:

```python
from collections.abc import Mapping, Sequence
from typing import Any, Final
from urllib.parse import urlsplit
```

(Place `from urllib.parse import urlsplit` in the stdlib import group; ruff isort
will keep `collections.abc` and `typing` ordering. The `pydantic` / `elspeth`
imports below are unchanged.) After this edit, re-confirm both `Sequence` and
`urlsplit` survived the PostToolUse ruff pass (the memory note: "edit the use-site
FIRST then the import, verify it survived").

### C. `src/elspeth/web/execution/schemas.py`

Three insertions keep the `ValidationCheckName` Literal and
`VALIDATION_BLOCKING_CHECK_NAMES` tuple in lock-step (the existing `frozenset`
assertion at lines 100–101 enforces it).

**C1 — Literal member.** Anchor: in `ValidationCheckName` (lines 33–53), the line
`    "llm_base_url_policy",` (line 43). Insert immediately **after** it:

```python
    "llm_base_url_policy",
    "azure_llm_endpoint_policy",
    "settings_load",
```

**C2 — Final constant.** Anchor: the line
`CHECK_LLM_BASE_URL_POLICY: Final[ValidationCheckName] = "llm_base_url_policy"` (line
65). Insert immediately **after** it:

```python
CHECK_LLM_BASE_URL_POLICY: Final[ValidationCheckName] = "llm_base_url_policy"
CHECK_AZURE_LLM_ENDPOINT_POLICY: Final[ValidationCheckName] = "azure_llm_endpoint_policy"
CHECK_SETTINGS: Final[ValidationCheckName] = "settings_load"
```

**C3 — blocking-tuple insertion (position #11).** Anchor: in
`VALIDATION_BLOCKING_CHECK_NAMES` (lines 76–95), the line `    CHECK_LLM_BASE_URL_POLICY,`
(line 86). Insert immediately **after** it (so the new check sits between
`llm_base_url_policy` #10 and `settings_load`):

```python
    CHECK_LLM_BASE_URL_POLICY,
    CHECK_AZURE_LLM_ENDPOINT_POLICY,
    CHECK_SETTINGS,
```

No change to the `frozenset(VALIDATION_CHECK_NAMES) != VALIDATION_CHECK_NAME_VALUES`
assertion — it now validates the new member automatically.

### D. `src/elspeth/web/execution/protocol.py`

Extend the `ValidationSettings` Protocol (currently exposes only `data_dir`,
lines 16–24). Anchor: the `data_dir` property:

```python
class ValidationSettings(Protocol):
    """Settings needed by direct runtime preflight validation.

    Structurally satisfied by WebSettings (data_dir is a Path, and
    allowed_azure_llm_endpoints is a tuple[str, ...]) and any test stub that
    exposes the same attributes.
    """

    @property
    def data_dir(self) -> Path: ...

    @property
    def allowed_azure_llm_endpoints(self) -> tuple[str, ...]: ...
```

(Update the docstring's "Structurally satisfied by WebSettings (data_dir is a Path)…"
line to mention the new attribute, as shown.)

### E. `src/elspeth/web/execution/validation.py`

**E1 — schemas import.** Anchor: the `from elspeth.web.execution.schemas import (`
block (lines 80–107). Insert `CHECK_AZURE_LLM_ENDPOINT_POLICY,` alphabetically as the
first name (before `CHECK_BATCH_TRANSFORM_OPTIONS,`):

```python
from elspeth.web.execution.schemas import (
    CHECK_AZURE_LLM_ENDPOINT_POLICY,
    CHECK_BATCH_TRANSFORM_OPTIONS,
    CHECK_BLOB_INLINE_REFS,
    ...
```

**E2 — provider_config_policy import.** Anchor: the
`from elspeth.web.provider_config_policy import (` block (lines 115–119). Insert
`web_azure_llm_endpoint_policy_error,` alphabetically first:

```python
from elspeth.web.provider_config_policy import (
    web_azure_llm_endpoint_policy_error,
    web_llm_base_url_policy_error,
    web_llm_retry_budget_policy_error,
    web_rag_provider_config_policy_error,
)
```

**E3 — `_CHECK_*` alias.** Anchor: the line
`_CHECK_LLM_BASE_URL_POLICY = CHECK_LLM_BASE_URL_POLICY` (line 132). Insert
immediately **after** it (mirrors declared order: azure #11 follows base_url #10):

```python
_CHECK_LLM_BASE_URL_POLICY = CHECK_LLM_BASE_URL_POLICY
_CHECK_AZURE_LLM_ENDPOINT_POLICY = CHECK_AZURE_LLM_ENDPOINT_POLICY
_CHECK_SETTINGS = CHECK_SETTINGS
```

**E4 — the gate loop.** Anchor: immediately **after** the `llm_base_url_policy`
loop's trailing passed-check append (ends at line 1556, the `checks.append(...
name=_CHECK_LLM_BASE_URL_POLICY, passed=True ...)` block) and **before** the
`# Step 3: Settings loading` comment (line 1558). Mirror the base_url loop EXACTLY,
swapping the helper, check name, detail/message prose, and error_code, and reading the
allowlist directly off settings. Full block to insert:

```python
    # azure_llm_endpoint_policy (#11) — web-authored Azure OpenAI LLM nodes may
    # only target an operator-allowlisted endpoint host. The api_key resolves
    # server-side (AZURE_API_KEY is allowlisted by default), so an arbitrary
    # author-chosen endpoint would direct the server-held bearer to a
    # destination the author picked — a credential-egress / SSRF vector. Unlike
    # OpenRouter there is no single canonical endpoint, so the policy is an
    # operator host allowlist (default any *.openai.azure.com resource). Runs
    # after llm_base_url_policy (#10) to match its declared position. The
    # allowlist is read directly off settings (settings.data_dir is read the
    # same way) — the ValidationSettings Protocol declares the attribute, so a
    # conforming settings object always provides it; a getattr-with-default
    # would be a dead implicit default that the null-correctness doctrine
    # forbids. Mirrors the managed_identity / web_scrape network policies.
    allowed_azure_llm_endpoints = settings.allowed_azure_llm_endpoints
    for node in state.nodes:
        if node.node_type != "transform":
            continue
        azure_llm_endpoint_policy_error = web_azure_llm_endpoint_policy_error(
            node.plugin, node.options, allowed_azure_llm_endpoints
        )
        if azure_llm_endpoint_policy_error is not None:
            checks.append(
                ValidationCheck(
                    name=_CHECK_AZURE_LLM_ENDPOINT_POLICY,
                    passed=False,
                    detail=f"Transform '{node.id}' targets a non-allowlisted Azure LLM endpoint",
                    affected_nodes=(node.id,),
                    outcome_code=None,
                )
            )
            _append_skipped_checks(checks, _CHECK_AZURE_LLM_ENDPOINT_POLICY)
            return ValidationResult(
                is_valid=False,
                checks=checks,
                errors=[
                    ValidationError(
                        component_id=node.id,
                        component_type="transform",
                        message=azure_llm_endpoint_policy_error,
                        suggestion="Use an allowlisted Azure OpenAI host (default: any *.openai.azure.com resource), or have the operator add the host to WebSettings.allowed_azure_llm_endpoints.",
                        error_code="azure_llm_endpoint_not_allowed",
                    ),
                ],
                readiness=_blocked_readiness(
                    code=_CHECK_AZURE_LLM_ENDPOINT_POLICY,
                    detail=f"transform {node.id} targets a non-allowlisted Azure LLM endpoint",
                    component_id=node.id,
                    component_type="transform",
                ),
                semantic_contracts=serialize_semantic_contracts(semantic_contracts),
            )
    checks.append(
        ValidationCheck(
            name=_CHECK_AZURE_LLM_ENDPOINT_POLICY,
            passed=True,
            detail="No non-allowlisted web-authored Azure LLM endpoint",
            affected_nodes=(),
            outcome_code=None,
        )
    )
```

`_ALL_CHECKS = list(VALIDATION_BLOCKING_CHECK_NAMES)` (line 198) automatically picks
up the new member, so `_skipped_checks` / `_append_skipped_checks` cascade correctly
with no further change.

---

## 4. Tests

### F. `tests/unit/web/execution/test_validation.py`

Existing helpers reused: `_make_state`, `_make_node`, `_make_output`, `_make_settings`
(default WebSettings → `allowed_azure_llm_endpoints == ("*.openai.azure.com",)`),
`_check`, the `MagicMock(spec=YamlGenerator)` + `mock_load.side_effect =
ValueError("settings stop")` pattern that lets a node reach a pre-settings gate.

**Gotcha (the OpenRouter work hit this):** an `llm` node with a `model` set stages an
`llm_model_choice` interpretation review (gate #6), which fires before this gate.
Construct Azure test nodes with `provider`/`endpoint` but **without** `model` (and
without `queries`/`base_url`) so they pass gates #1–#10 and reach the azure gate (#11).

**F1 — helper unit tests.** Insert a new class after
`TestValidatePipelineLlmBaseUrlPolicy` (ends ~line 564), before
`TestValidatePipelineBatchTransformOptions` (line 567):

```python
class TestWebAzureLlmEndpointPolicyHelper:
    """Unit coverage for the web-authored Azure LLM endpoint policy helper."""

    DEFAULT = ("*.openai.azure.com",)

    def test_non_llm_plugin_is_ignored(self) -> None:
        from elspeth.web.provider_config_policy import web_azure_llm_endpoint_policy_error

        opts = {"provider": "azure", "endpoint": "https://evil.example.com"}
        assert web_azure_llm_endpoint_policy_error("web_scrape", opts, self.DEFAULT) is None

    def test_non_azure_provider_is_ignored(self) -> None:
        from elspeth.web.provider_config_policy import web_azure_llm_endpoint_policy_error

        opts = {"provider": "openrouter", "endpoint": "https://evil.example.com"}
        assert web_azure_llm_endpoint_policy_error("llm", opts, self.DEFAULT) is None

    def test_default_allows_genuine_azure_resource_host(self) -> None:
        from elspeth.web.provider_config_policy import web_azure_llm_endpoint_policy_error

        opts = {"provider": "azure", "endpoint": "https://myresource.openai.azure.com"}
        assert web_azure_llm_endpoint_policy_error("llm", opts, self.DEFAULT) is None

    def test_default_blocks_arbitrary_and_private_and_dotanchor_and_apex_and_http(self) -> None:
        from elspeth.web.provider_config_policy import web_azure_llm_endpoint_policy_error

        for endpoint in (
            "https://evil.example.com",
            "https://10.0.0.5",
            "https://evil-openai.azure.com",      # dot-anchor: no dot before openai
            "https://openai.azure.com",           # apex: wildcard does not match apex
            "http://myresource.openai.azure.com",  # non-https fails closed
        ):
            opts = {"provider": "azure", "endpoint": endpoint}
            assert web_azure_llm_endpoint_policy_error("llm", opts, self.DEFAULT) is not None, endpoint

    def test_operator_added_exact_host_allowed(self) -> None:
        from elspeth.web.provider_config_policy import web_azure_llm_endpoint_policy_error

        allow = ("myorg-aoai.openai.azure.com",)
        opts = {"provider": "azure", "endpoint": "https://myorg-aoai.openai.azure.com"}
        assert web_azure_llm_endpoint_policy_error("llm", opts, allow) is None

    def test_operator_added_wildcard_pattern_allows_matching_subdomain(self) -> None:
        from elspeth.web.provider_config_policy import web_azure_llm_endpoint_policy_error

        allow = ("*.gw.example.com",)
        opts = {"provider": "azure", "endpoint": "https://llm.gw.example.com"}
        assert web_azure_llm_endpoint_policy_error("llm", opts, allow) is None

    def test_empty_allowlist_blocks_every_azure_endpoint(self) -> None:
        from elspeth.web.provider_config_policy import web_azure_llm_endpoint_policy_error

        opts = {"provider": "azure", "endpoint": "https://myresource.openai.azure.com"}
        assert web_azure_llm_endpoint_policy_error("llm", opts, ()) is not None

    def test_missing_or_non_str_endpoint_is_ignored(self) -> None:
        from elspeth.web.provider_config_policy import web_azure_llm_endpoint_policy_error

        assert web_azure_llm_endpoint_policy_error("llm", {"provider": "azure"}, self.DEFAULT) is None
        assert web_azure_llm_endpoint_policy_error("llm", {"provider": "azure", "endpoint": 123}, self.DEFAULT) is None
```

**F2 — integration tests through `validate_pipeline`.** Insert directly after F1's
class:

```python
class TestValidatePipelineAzureLlmEndpointPolicy:
    """Web-authored Azure LLM nodes may only target an allowlisted host."""

    @staticmethod
    def _azure_options(endpoint: str) -> dict[str, object]:
        # provider/endpoint only — NO `model` (would stage interpretation_review
        # gate #6, which fires before this gate #11) and NO base_url/queries.
        return {"provider": "azure", "endpoint": endpoint}

    def test_disallowed_endpoint_rejected_before_yaml_generation(self) -> None:
        state = _make_state(
            nodes=(_make_node(plugin="llm", options=self._azure_options("https://evil.example.com")),),
            outputs=(_make_output(name="results"),),
        )
        settings = _make_settings()  # default allowlist = ("*.openai.azure.com",)
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv\n  options: {}\n"

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("settings stop")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "azure_llm_endpoint_policy").passed is False
        assert result.errors[0].component_id == "test_node"
        assert result.errors[0].error_code == "azure_llm_endpoint_not_allowed"
        # Do NOT assert generate_yaml was not called: the azure gate (#11) fires
        # AFTER Step 2 YAML generation (it sits between llm_base_url_policy #10 and
        # the Step 3 settings load), so generate_yaml has already run by the time
        # this gate returns is_valid=False. Mirror the sibling base_url failing
        # test (test_loopback_base_url_rejected_before_yaml_generation), which omits
        # any generate_yaml assertion for exactly this reason.

    def test_allowlisted_endpoint_passes_gate_and_reaches_yaml_generation(self) -> None:
        state = _make_state(
            nodes=(_make_node(plugin="llm", options=self._azure_options("https://myresource.openai.azure.com")),),
            outputs=(_make_output(name="results"),),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv\n  options: {}\n"

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("settings stop")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert _check(result, "azure_llm_endpoint_policy").passed is True
        assert all(error.error_code != "azure_llm_endpoint_not_allowed" for error in result.errors)
        mock_yaml_gen.generate_yaml.assert_called_once_with(state)
```

**F3 — bump the all-checks-passed count 16 → 17.** Anchor:
`test_valid_pipeline_returns_all_checks_passed` (line 1680). Change:

```python
        assert len(result.checks) == 16
```
to
```python
        assert len(result.checks) == 17
```

And add an explicit assertion alongside the existing `_check(...)` asserts (after the
`llm_base_url_policy` one at line 1686):

```python
        assert _check(result, "llm_base_url_policy").passed is True
        assert _check(result, "azure_llm_endpoint_policy").passed is True
```

**F4 — extend the skipped-after-failure cascade.** Anchor:
`test_web_scrape_failure_skips_later_managed_identity_and_llm_retry_checks` (the tuple
at line 422). Change:

```python
        for later_check in ("managed_identity_policy", "llm_retry_budget_policy", "llm_base_url_policy"):
```
to
```python
        for later_check in (
            "managed_identity_policy",
            "llm_retry_budget_policy",
            "llm_base_url_policy",
            "azure_llm_endpoint_policy",
        ):
```

(`azure_llm_endpoint_policy` is declared #11, after `web_scrape_network_policy` #2, so
when web_scrape fails it must be recorded skipped with
`CHECK_OUTCOME_SKIPPED_AFTER_FAILURE`.)

### G. `tests/unit/web/test_config.py`

Insert a new class after `TestServerSecretAllowlistValidation` (ends line 642),
before `TestAuthFieldValidationContinued` (line 644). Each `WebSettings(...)` call
needs the required kwargs (`composer_max_composition_turns`,
`composer_max_discovery_turns`, `composer_timeout_seconds`,
`composer_rate_limit_per_minute`, `shareable_link_signing_key=b"\x00" * 32`) — copy
from the sibling tests. `ValidationError` is already imported in this module.

```python
class TestAllowedAzureLlmEndpointsValidation:
    """Tests for allowed_azure_llm_endpoints field validation."""

    _REQUIRED = dict(
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )

    def test_default_is_azure_openai_wildcard(self) -> None:
        settings = WebSettings(**self._REQUIRED)
        assert settings.allowed_azure_llm_endpoints == ("*.openai.azure.com",)

    def test_accepts_wildcard_and_bare_hostnames_and_lowercases(self) -> None:
        settings = WebSettings(
            allowed_azure_llm_endpoints=("*.openai.azure.com", "MyOrg-AOAI.openai.azure.com", "llm-gw.internal.example.com"),
            **self._REQUIRED,
        )
        assert settings.allowed_azure_llm_endpoints == (
            "*.openai.azure.com",
            "myorg-aoai.openai.azure.com",
            "llm-gw.internal.example.com",
        )

    def test_accepts_empty_tuple_deny_all(self) -> None:
        settings = WebSettings(allowed_azure_llm_endpoints=(), **self._REQUIRED)
        assert settings.allowed_azure_llm_endpoints == ()

    @pytest.mark.parametrize(
        "bad",
        [
            "https://myresource.openai.azure.com",   # scheme/URL
            "myresource.openai.azure.com/path",       # path
            "myresource.openai.azure.com:443",        # port
            "user@myresource.openai.azure.com",       # embedded credentials
            "my resource.openai.azure.com",           # whitespace
            "",                                        # empty
            "localhost",                              # single-label (not dotted)
            "*.",                                     # wildcard with no host
        ],
    )
    def test_rejects_malformed_entries(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            WebSettings(allowed_azure_llm_endpoints=(bad,), **self._REQUIRED)
```

(If `pytest` is not already imported in `test_config.py`, it is — the existing
`TestServerSecretAllowlistValidation` uses `pytest.raises`.)

---

## 5. Out of scope (from the design — do NOT do these)

- No change to OpenRouter (left pinned to its canonical endpoint by `ba5a24aad`).
- No change to the plugin config-layer Azure `endpoint` validator (stays
  HTTPS-required, shared with CLI). The web execution boundary is the trust boundary.
- No change to Azure RAG/Search managed-identity handling (already gated by
  `managed_identity_policy`).
- No new enforcement proof: the gate lives inside `validate_pipeline`, the single
  fail-closed chokepoint already audited for the OpenRouter gate (it runs before
  `create_run` in `execute()`, and is the body of composer `_runtime_preflight` /
  `_cached_runtime_preflight`, `preview_pipeline`, and `/validate`).

---

## 6. Verification commands (run all; all must pass before "done")

From the repo root `/home/john/elspeth` (use the main `.venv`, Python 3.13).

**Targeted + broad pytest** (validation + schemas modules, config, then a broad web
unit pass — plain selection is CI-equivalent; do NOT pass `-o addopts=""`):

```bash
pytest tests/unit/web/execution/test_validation.py tests/unit/web/execution/test_schemas.py tests/unit/web/test_config.py -q
pytest tests/unit/web/ -q
```

Expect: the new helper/integration/config tests pass; the bumped count
(`len(result.checks) == 17`) passes; the extended cascade passes; the schemas
import-time `frozenset` assertion holds (it loads on import).

**ruff** (changed source files):

```bash
ruff check src/elspeth/web/config.py src/elspeth/web/provider_config_policy.py \
  src/elspeth/web/execution/schemas.py src/elspeth/web/execution/protocol.py \
  src/elspeth/web/execution/validation.py
```

After editing `provider_config_policy.py`, re-confirm `Sequence` and `urlsplit`
survived the PostToolUse ruff autofix (F401 strip gotcha — §3.B).

**mypy** (changed files):

```bash
mypy src/elspeth/web/config.py src/elspeth/web/provider_config_policy.py \
  src/elspeth/web/execution/schemas.py src/elspeth/web/execution/protocol.py \
  src/elspeth/web/execution/validation.py
```

VERIFY (do not pre-assert clean): `web_azure_llm_endpoint_policy_error` is fully
typed and the `ValidationSettings` Protocol now declares
`allowed_azure_llm_endpoints` so the direct `settings.allowed_azure_llm_endpoints`
read type-checks. This is also where the Protocol fan-out surfaces — any other type
checked as `ValidationSettings` now needs the attribute; WebSettings provides it.
Run mypy and resolve anything it reports rather than assuming a clean pass.

**wardline** (trust-boundary gate; takes one directory path):

```bash
wardline scan src/elspeth/web --fail-on ERROR
```

Expect exit 0. The new code reads author-controlled `options`/`endpoint` (Tier-3) at
a validation boundary and fails closed; if wardline flags the `endpoint` taint into
`urlsplit`, fix at the boundary (the gate already rejects non-https/hostless and
non-allowlisted), not the sink — and follow the `wardline-gate` skill's
explain→fix→rescan loop before any waiver.

**tier-model** (config.py + web/execution files are tracked in
`config/cicd/enforce_tier_model/web.yaml`):

```bash
python -m elspeth_lints.core.cli check --files \
  src/elspeth/web/config.py src/elspeth/web/provider_config_policy.py \
  src/elspeth/web/execution/schemas.py src/elspeth/web/execution/protocol.py \
  src/elspeth/web/execution/validation.py
```

If fingerprints stale (AST shifted by the inserted code), rotate per the documented
tool: `python -m elspeth_lints.core.cli rotate` (all-or-nothing; restore stale-first;
co-land the updated `web.yaml` with the source change). The plan deliberately avoids
a new top-level `import` in `config.py` to keep that cascade localised.

---

## 7. Risks

- **Default `*.openai.azure.com` still permits an attacker-owned Azure tenant.**
  Accepted residual (per design). `AZURE_API_KEY` is resource-scoped — it will not
  authenticate against another resource — so the impact is exposure of the key value
  to that endpoint's logs, not cross-tenant use. Operators who care tighten the
  allowlist to their exact resource host (an exact-hostname entry); an empty `()`
  disables web Azure LLM entirely. No code change needed for either posture.
- **`ValidationSettings` Protocol change.** Adding
  `allowed_azure_llm_endpoints: tuple[str, ...]` to the Protocol means any object
  type-checked against it must expose the attribute. WebSettings does. The runtime
  gate reads it directly (`settings.allowed_azure_llm_endpoints`) — the Protocol makes
  the attribute part of the contract, so a conforming settings object always provides
  it (the existing `settings.data_dir` reads work the same way). Tests build a real
  `WebSettings` via `_make_settings()`, which carries the field default; the Protocol
  obligation is a mypy/structural concern, not
  a runtime break. The allowlist is only *iterated* when an actual `llm`+`azure` node
  is present, so non-Azure test fixtures never touch it.
- **Behaviour change for any deployment relying on a non-Azure web LLM endpoint.**
  None known. A web-authored Azure LLM node pointed at a non-`*.openai.azure.com` host
  now fails validation with `error_code="azure_llm_endpoint_not_allowed"` instead of
  passing. This is the intended fail-closed posture; operators restore prior behaviour
  for a legitimate private gateway by adding its exact host to
  `WebSettings.allowed_azure_llm_endpoints`. OpenRouter, OpenAI, Anthropic, and Azure
  RAG/Search paths are untouched.
- **Tier-model fingerprint churn (config.py + web/execution).** Inserting a field, a
  validator, Literal/const/tuple members, imports, and the gate loop shifts AST nodes;
  fingerprints may stale. Mitigated by avoiding a new module-level import in config.py
  and by the rotate step in §6. Co-land the regenerated `web.yaml` with the source
  change (do not split across commits).
- **Interpretation-review ordering trap in tests.** Azure test nodes must omit
  `model` (and `queries`/`base_url`) or gate #6 (or #9/#10) fires first and the
  azure-gate assertions never run. Captured in F1/F2.

---

## 8. Review notes (resolved findings — 2026-06-28 review cycle)

- **[BLOCKING — resolved] F2 failing test must not assert `generate_yaml` was
  never called.** The original draft included
  `mock_yaml_gen.generate_yaml.assert_not_called()` with the comment
  "Gate fired before YAML generation." Both were factually wrong: `generate_yaml`
  runs in Step 2 (validation.py:1338) and the azure gate (#11) is inserted *after*
  `llm_base_url_policy` (#10) and *before* the Step 3 settings load (validation.py:1558),
  so by the time the gate returns `is_valid=False`, `generate_yaml` has already run
  and the mock has recorded the call — the assertion would have failed
  unconditionally. **Fix applied:** the assertion and comment are removed and replaced
  by a NOTE that mirrors the sibling failing tests
  (`test_loopback_base_url_rejected_before_yaml_generation` /
  `test_arbitrary_https_base_url_rejected_before_yaml_generation`, test_validation.py:498/516),
  both of which omit any `generate_yaml` assertion in the failure path. The remaining
  F2 failure assertions (`is_valid is False`, `azure_llm_endpoint_policy` failed,
  `component_id == "test_node"`, `error_code == "azure_llm_endpoint_not_allowed"`)
  are correct and unchanged. The passing-path test keeps
  `mock_yaml_gen.generate_yaml.assert_called_once_with(state)`.
- **[DECLINED — finding 2's alternative remediation] Do not replace the dropped
  assertion with `assert_called_once_with(state)` in the failure-path test.** That
  form is also correct but couples the test to `materialized_state is state` for the
  Azure fixture (the real call is `generate_yaml(materialized_state)`), and it diverges
  from the two sibling failing tests, which omit the assertion entirely. Dropping is
  strictly safer and is the exact mirror. The positive `assert_called_once_with(state)`
  belongs (and stays) only on the passing-path test, matching the sibling.
- **[DECLINED — nit + finding 1's optional rename] Keep the method name
  `test_disallowed_endpoint_rejected_before_yaml_generation`.** The gate technically
  fires *after* YAML generation and *before* settings load, so
  `..._before_settings_load` would be more literally accurate. But finding 3 (blocking)
  says do not rename, finding 1 makes the rename explicitly optional, and the nit is
  non-blocking. The plan's core mandate is "mirror the OpenRouter gate EXACTLY"; the
  sibling carries the same imprecise `..._before_yaml_generation` name, so renaming only
  *this* test would introduce a one-off naming divergence for marginal gain. The name is
  kept for sibling consistency; the imprecision is acknowledged here.
- **[MINOR — resolved] C1 anchor line number corrected** from "(line 42)" to
  "(line 43)": in schemas.py line 42 is `"llm_retry_budget_policy"` and line 43 is
  `"llm_base_url_policy"`. The Edit matches on text, so no behavioural change.

# Web-authored Azure LLM endpoint allowlist — design

**Date:** 2026-06-28
**Status:** Approved (design); implementation pending
**Related:** commit `ba5a24aad` (OpenRouter `base_url` canonical-pin gate — the sibling
fix this mirrors). Background security model: `feedback_web_provider_url_egress_gate`
(agent memory).

## Problem

A web-authored pipeline can include an `llm` node with `provider="azure"`. Unlike
the OpenRouter recipes (which refuse Azure), the composer explicitly directs authors
to compose Azure nodes directly. Such a node carries:

- `endpoint` (required) — a free string, validated only by
  `validate_credential_safe_https_url(..., field_name="endpoint")`, which accepts
  **any HTTPS host** (no loopback exception, but no host allowlist either).
- `api_key` (required) — in the web path this resolves **server-side**.
  `WebSecretService.resolve()` falls through user scope to `ServerSecretStore`, and
  `WebSettings.server_secret_allowlist` ships with `AZURE_API_KEY` by default. So a
  web author can reference a deployment-held bearer they cannot read.

Because `endpoint` is author-controlled while `api_key` is a server credential, a
web author can direct the server's Azure bearer at an attacker-controlled host
(`https://evil.example.com`) or an internal/private host (`https://10.0.0.5`,
metadata endpoints) — a credential-egress / SSRF vector. This is the same trust
asymmetry the OpenRouter gate closed, on a different provider field.

Unlike OpenRouter there is **no canonical endpoint to pin to** — every Azure
resource has its own `https://<resource>.openai.azure.com` — so the fix is an
operator-tunable host allowlist rather than an equality check.

Note: Azure **LLM** has no `use_managed_identity` option (that exists only on Azure
**RAG/Search**, already gated by `managed_identity_policy`). The only Azure-LLM
egress vector is `endpoint` + server `api_key`.

## Chosen policy

Operator host allowlist with a **non-empty default** of `*.openai.azure.com`:

- Default permits any genuine Azure OpenAI resource host; blocks arbitrary non-Azure
  hosts and private/internal hosts.
- Operators tighten (to their specific resource host) or extend (a private
  OpenAI-compatible gateway host) by editing the allowlist.
- An **empty** allowlist denies all web-authored Azure LLM endpoints — the
  "disable web Azure LLM entirely" posture, available without code change.

This mirrors the established "web gets the safe default; the dangerous capability is
operator-controlled" precedents (`managed_identity_policy`,
`web_scrape_network_policy`, `server_secret_allowlist`).

**Accepted residual:** with the `*.openai.azure.com` default a web author could still
target an *attacker-owned* Azure OpenAI tenant. `AZURE_API_KEY` is resource-scoped
(it will not authenticate against another resource), so the impact is exposure of the
key value to that endpoint's logs, not cross-tenant use. Operators who care tighten
the allowlist to their exact resource host.

## Components

### 1. Config surface — `src/elspeth/web/config.py`

Add to `WebSettings`, mirroring `server_secret_allowlist`:

```python
allowed_azure_llm_endpoints: tuple[str, ...] = ("*.openai.azure.com",)
```

- **Entries are hostname patterns, not URLs.** Each entry is either:
  - a `*.base.domain` suffix pattern (dot-anchored subdomain match), or
  - an exact hostname (`myorg-aoai.openai.azure.com`, or a private gateway host
    `llm-gw.internal.example.com`).
- A `field_validator("allowed_azure_llm_endpoints")` normalises each entry to
  lowercase and rejects malformed entries: empty, containing a scheme (`://`), a
  path/`/`, a port (`:`), embedded credentials (`@`), or whitespace. A `*.` prefix is
  permitted; the remainder must be a valid dotted hostname. (URLs are rejected so
  operators cannot accidentally allowlist a path-bearing or credentialed string.)

### 2. Policy helper — `src/elspeth/web/provider_config_policy.py`

Sibling of `web_llm_base_url_policy_error`:

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

def web_azure_llm_endpoint_policy_error(
    plugin: str | None,
    options: Mapping[str, Any],
    allowed_endpoints: Sequence[str],
) -> str | None:
    if plugin != "llm" or options.get("provider") != "azure":
        return None
    endpoint = options.get("endpoint")
    if not isinstance(endpoint, str):
        # Missing/non-str endpoint is rejected at config construction (pydantic);
        # this network-policy gate only adjudicates author-chosen string hosts.
        return None
    parsed = urlsplit(endpoint.strip())
    # Require HTTPS here rather than leaning on the later config-construction gate
    # (#12), so this check is self-sufficient and the failure is attributed to the
    # endpoint policy with a clear message. A non-https or hostless endpoint fails
    # closed.
    if parsed.scheme != "https" or not parsed.hostname:
        return AZURE_LLM_ENDPOINT_POLICY_ERROR
    if _host_matches_any(parsed.hostname.lower(), allowed_endpoints):
        return None
    return AZURE_LLM_ENDPOINT_POLICY_ERROR
```

`_host_matches_any(host, entries)`:

```python
def _host_matches_any(host: str, entries: Sequence[str]) -> bool:
    for entry in entries:
        e = entry.strip().lower()
        if e.startswith("*."):
            base = e[2:]
            if base and host.endswith("." + base):  # dot-anchored subdomain
                return True
        elif host == e:
            return True
    return False
```

Dot-anchoring guarantees `*.openai.azure.com` matches `myresource.openai.azure.com`
but **not** `evil-openai.azure.com` (no dot before `openai`) and not the apex
`openai.azure.com`. The helper requires HTTPS itself (rather than relying on the
later config-construction gate at #12) so the check is self-sufficient and a non-https
or hostless endpoint fails closed with an endpoint-policy attribution.

### 3. Gate wiring — `src/elspeth/web/execution/`

- `schemas.py`: register a new blocking check `azure_llm_endpoint_policy`
  (Literal member + `CHECK_AZURE_LLM_ENDPOINT_POLICY` constant + insertion into
  `VALIDATION_BLOCKING_CHECK_NAMES` immediately after `llm_base_url_policy`, i.e.
  position #11). The existing `frozenset` self-consistency assertion keeps the
  Literal and tuple in lock-step.
- `protocol.py`: extend the `ValidationSettings` Protocol with
  `allowed_azure_llm_endpoints: tuple[str, ...]` (WebSettings already satisfies it).
- `validation.py`: import the helper + alias the check constant; add a per-node loop
  immediately after the `llm_base_url_policy` loop, mirroring its structure exactly
  (failed `ValidationCheck` + `_append_skipped_checks` + early `ValidationResult(
  is_valid=False, ...)` with `error_code="azure_llm_endpoint_not_allowed"`, else a
  passed check). Read the allowlist defensively:
  `getattr(settings, "allowed_azure_llm_endpoints", ("*.openai.azure.com",))` so
  minimal/stub settings objects in tests still resolve a safe default.

### 4. Enforcement coverage

No new enforcement proof is required: the gate is a blocking check inside
`validate_pipeline`, which is the single fail-closed chokepoint already audited for
the OpenRouter gate — it runs before `create_run` in `execute()`, and is the literal
body of composer `_runtime_preflight` / `_cached_runtime_preflight`,
`preview_pipeline`, and `/validate`. The engine's live provider preflight
(`transform.py:1565`) only runs inside an engine run, which `execute()` gates.

## Testing

`tests/unit/web/` — mirror the OpenRouter gate's test set:

- **Helper unit tests** (`web_azure_llm_endpoint_policy_error`):
  - non-`llm` plugin → ignored; `provider != "azure"` → ignored.
  - default allowlist: `https://myresource.openai.azure.com` allowed.
  - blocked: `https://evil.example.com`, `https://10.0.0.5`,
    `https://evil-openai.azure.com` (dot-anchor), apex `https://openai.azure.com`,
    and `http://myresource.openai.azure.com` (non-https fails closed).
  - operator-added exact host allowed; operator-added `*.gw.example.com` pattern
    allowed for a matching subdomain.
  - **empty allowlist** `()` → every Azure endpoint blocked.
- **Config validator tests**: `allowed_azure_llm_endpoints` rejects URL/scheme/
  port/path/credential entries; accepts `*.suffix` and bare hostnames.
- **Integration tests** through `validate_pipeline`: an Azure LLM node with a
  disallowed endpoint yields `is_valid=False`, check `azure_llm_endpoint_policy`
  failed, `error_code="azure_llm_endpoint_not_allowed"`; an allowlisted endpoint
  passes the gate and reaches YAML generation.
- **Maintenance**: bump the "all checks passed" count (16 → 17) and extend the
  skipped-after-failure cascade assertion to include `azure_llm_endpoint_policy`.

## Out of scope

- **OpenRouter is left as-is** (pinned to its canonical endpoint in `ba5a24aad`).
  Not folded into this allowlist mechanism — the two providers have different shapes
  (one canonical endpoint vs per-resource hosts) and conflating them adds config
  surface without benefit. If a future deployment needs custom OpenRouter gateways
  from the web, that is a separate, explicitly-scoped change.
- No change to the plugin config-layer Azure `endpoint` validator (it stays
  HTTPS-required, shared with CLI). The web execution boundary is the trust boundary.
- No change to Azure RAG/Search managed-identity handling (already gated).

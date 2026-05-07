# Audit Provenance Boundary Enforcement — Design Spec

**Status:** Implemented
**Bug:** `elspeth-7527dacf08`
**Date:** 2026-03-21
**Reviewed by:** Architecture Critic, Systems Thinker, Quality Engineer, Python Expert (2026-03-21)

## Problem

External-call transforms (LLM, WebScrape) put audit provenance metadata directly into pipeline rows. This data exists solely for forensic reconstruction and has no downstream operational use. It belongs in the audit trail (Landscape `node_states.success_reason`), not in pipeline data that flows to sinks.

### Violations

1. **Redundancy** — provenance data exists in both the row and the audit trail.
2. **Divergent hashes** — WebScrape computes blob hashes differently from AuditedHTTPClient (`f"GET {url}".encode()` vs `canonical_json(HTTPCallRequest.to_dict()).encode()`). The row hash and the audit hash for the same request don't match. An auditor correlating `fetch_request_hash` in the row with `request_hash` in the calls table will find they diverge.
3. **Dual PayloadStore wiring** — WebScrape accesses PayloadStore directly from PluginContext to store blobs for row hashes, creating a second wiring path alongside the recorder path. The PluginContext path was broken until recently (orchestrator forgot to pass `payload_store`).
4. **Row bloat** — LLM transforms add 6 audit fields per row (multiplied by query count in multi-query mode). WebScrape adds 3.

### Root Cause

No enforced boundary between pipeline data (operational values that downstream transforms and sinks consume) and audit provenance (forensic metadata for `elspeth explain` reconstruction). Transforms defaulted to putting everything in the row because it was the easiest path.

## Principle

**Pipeline rows carry operational data. Audit provenance lives in the Landscape.**

The test for whether a field belongs in the row:

> Would a pipeline operator or downstream transform ever make a decision based on this value?

- `fetch_status = 403` → a gate might route forbidden pages to review → **row**
- `template_hash = abc123` → forensic reconstruction only → **audit trail**
- `rag_score = 0.85` → a gate might filter low-relevance rows → **row**
- `lookup_source = /path/to/config.yaml` → config provenance → **audit trail**

Provenance data goes into `success_reason["metadata"]` on `TransformResult.success()`, which the Landscape persists in `node_states.success_reason_json`. Retrievable via `elspeth explain`.

**Error-path provenance:** When a transform returns `TransformResult.error()`, no `success_reason` is produced. Error-path provenance is already covered by the `calls` table — `AuditedHTTPClient` records every call (success and error) via `recorder.record_call()`, and the `state_id` links the call record to the node_state. No additional provenance linkage is needed for error cases.

**Hash representations in the system:** After this migration, two hash types remain: (1) PayloadStore SHA-256 hashes (content-addressable blob references, used in `Call.request_ref`/`Call.response_ref` and `success_reason["metadata"]`), and (2) `stable_hash()` values used in telemetry events (ephemeral, not forensic). The divergent third representation (`f"GET {url}".encode()`) is eliminated.

## Field Inventory

### LLM Transforms — 6 Fields Move to Audit Trail

The code already labels these correctly as `LLM_AUDIT_SUFFIXES` in `plugins/transforms/llm/__init__.py`.

**Remove from pipeline rows:**

| Field | Content | Why audit-only |
|-------|---------|----------------|
| `{prefix}_template_hash` | SHA-256 of prompt template | Forensic — which template produced this output |
| `{prefix}_variables_hash` | SHA-256 of rendered variables | Forensic — what data was sent to the LLM |
| `{prefix}_template_source` | Config file path of template | Forensic — config provenance |
| `{prefix}_lookup_hash` | SHA-256 of lookup data | Forensic — reference data fingerprint |
| `{prefix}_lookup_source` | Config file path of lookup data | Forensic — config provenance |
| `{prefix}_system_prompt_source` | Config file path of system prompt | Forensic — config provenance |

**Keep in pipeline rows (operational):**

| Field | Content | Why operational |
|-------|---------|----------------|
| `{prefix}_usage` | Token usage dict | Downstream budgeting, cost routing |
| `{prefix}_model` | Model identifier | Multi-model routing, filtering |

**Error-path `template_hash` in error reasons:** The `template_rendering_failed` error reason currently includes `template_hash` in the `TransformResult.error()` reason dict. This is a forensic field inside an error reason — it stays in `result.reason` because error reasons are already part of the audit trail (stored in `node_states.error_reason_json`). No change needed for error paths.

**Migration approach:** Split `populate_llm_metadata_fields()` into two functions:

```python
def populate_llm_operational_fields(
    output: dict[str, object],
    field_prefix: str,
    *,
    usage: TokenUsage | None,
    model: str | None,
) -> None:
    """Populate operational metadata into the output row (stays in pipeline data)."""
    output[f"{field_prefix}_usage"] = usage.to_dict() if usage is not None else None
    output[f"{field_prefix}_model"] = model


def build_llm_audit_metadata(
    field_prefix: str,
    *,
    template_hash: str,
    variables_hash: str,
    template_source: str | None,
    lookup_hash: str | None,
    lookup_source: str | None,
    system_prompt_source: str | None,
) -> dict[str, object]:
    """Build audit provenance dict for inclusion in success_reason["metadata"].

    Does NOT write to the output row — audit provenance lives in the Landscape only.
    """
    return {
        f"{field_prefix}_template_hash": template_hash,
        f"{field_prefix}_variables_hash": variables_hash,
        f"{field_prefix}_template_source": template_source,
        f"{field_prefix}_lookup_hash": lookup_hash,
        f"{field_prefix}_lookup_source": lookup_source,
        f"{field_prefix}_system_prompt_source": system_prompt_source,
    }
```

Remove `get_llm_audit_fields()` from `__all__` — it becomes an internal helper referenced only by `build_llm_audit_metadata()`. The public API is `build_llm_audit_metadata()` (returns a dict) and `populate_llm_operational_fields()` (mutates the output row).

Call sites in `transform.py`, `multi_query.py`, `openrouter_batch.py`, and `azure_batch.py` each call both functions: operational fields go into the output row, audit metadata dict gets merged into `success_reason["metadata"]` via `{**existing_metadata, **audit_metadata}`.

### Multi-Query `_QuerySuccess` Extension

Multi-query transforms return `_QuerySuccess(fields=...)` from `_execute_one_query()`. The `_QuerySuccess` dataclass has no slot for audit metadata. Extend it:

```python
@dataclass(frozen=True, slots=True)
class _QuerySuccess:
    fields: dict[str, Any]
    audit_metadata: dict[str, object]  # built by build_llm_audit_metadata()
```

In `_execute_sequential`/`_execute_parallel`, collect per-query `audit_metadata` dicts from all `_QuerySuccess` results and merge into the final `success_reason["metadata"]`.

### WebScrape Transform — 3 Fields Move, 3 Stay

**Remove from pipeline rows:**

| Field | Content | Why audit-only |
|-------|---------|----------------|
| `fetch_request_hash` | Blob hash of HTTP request | Forensic — what was requested |
| `fetch_response_raw_hash` | Blob hash of raw response bytes | Forensic — what came back |
| `fetch_response_processed_hash` | Blob hash of post-extraction content | Forensic — what the transform produced |

**Keep in pipeline rows (operational):**

| Field | Content | Why operational |
|-------|---------|----------------|
| `fetch_status` | HTTP status code | Gate routing (403 → review, 200 → proceed) |
| `fetch_url_final` | Final URL after redirects | Deduplication, operator visibility |
| `fetch_url_final_ip` | Resolved IP address | SSRF forensics, operational visibility |

**Fix: `fetch_url_final_ip` is a misnomer.** The current code sets `output["fetch_url_final_ip"] = str(response.url)`, which is the post-redirect URL, not the resolved IP. Fix: use `safe_request.resolved_ip` from the SSRF validation. This is the IP that was DNS-pinned for the initial request. For redirects, each hop resolves independently via `_follow_redirects_safe()`, but the initial `safe_request.resolved_ip` is what the field name promises. If the value is `None` on any path, crash — it's a bug in the SSRF validation.

**Migration:** WebScrape stops calling `payload_store.store()` directly. The request/response blob hashes come from the `Call` object returned by `recorder.record_call()` (currently discarded — see "Surface Call return" section). The processed-content blob is stored via `recorder.store_payload()` (new method). All three hashes go into `success_reason["metadata"]`.

### RAG Transform — No Changes

All 4 fields (`rag_context`, `rag_score`, `rag_count`, `rag_sources`) are operational data consumed by downstream LLM transforms. No audit provenance in rows.

## `SchemaConfig.audit_fields` Cleanup

`SchemaConfig` has an `audit_fields` tuple (`contracts/schema.py:317`) currently populated with LLM audit field names in the `_output_schema_config` construction at `transform.py:976` and `transform.py:1021`. The field's documented purpose: "Fields that exist in output but are NOT part of the stability contract."

After this change, audit fields no longer exist in output rows. `SchemaConfig.audit_fields` becomes a lie — it claims fields are in the output when they aren't.

**Resolution:** Remove audit field names from `SchemaConfig.audit_fields` at all construction sites. Pass `audit_fields=None` (or omit it) since there are no longer any unstable-but-present fields in the output. The `audit_fields` attribute on `SchemaConfig` stays for potential future use (other transforms may have legitimately unstable output fields), but LLM transforms stop populating it.

**Output schema builders must also be updated:** `_build_augmented_output_schema()` (`__init__.py:207-222`) and `_build_multi_query_output_schema()` (`__init__.py:270`) both call `get_llm_audit_fields()` to add audit field names to the Pydantic output schema. After this change, audit fields are no longer in output rows, so `get_llm_audit_fields()` must be removed from both schema builder functions. Leaving them would declare phantom fields that never arrive — in `fixed` mode, schema validation would expect fields that don't exist.

Affected sites:
- `plugins/transforms/llm/__init__.py:207-222` — `_build_augmented_output_schema()`, remove `get_llm_audit_fields()` call
- `plugins/transforms/llm/__init__.py:270` — `_build_multi_query_output_schema()`, same
- `plugins/transforms/llm/transform.py` lines 976, 1021 — `SchemaConfig(audit_fields=...)` construction
- `plugins/transforms/llm/multi_query.py` line 232-235 — iterates `LLM_AUDIT_SUFFIXES` to build `declared_output_fields`

## PayloadStore Architecture Fix

### Remove `payload_store` from PluginContext

With hash fields out of rows, no transform needs direct PayloadStore access. `ctx.payload_store` is accessed only in `WebScrapeTransform.on_start()` (`web_scrape.py:262`) — no other plugin reads it. Remove:

- `payload_store: PayloadStore | None` from `PluginContext` dataclass (`contracts/plugin_context.py:84`)
- `payload_store` reference from `PluginContext` docstring (`contracts/plugin_context.py:66-68`)
- `payload_store` property from `LifecycleContext` protocol (`contracts/contexts.py:199`)
- `payload_store=payload_store` from orchestrator's `PluginContext` construction (`engine/orchestrator/core.py:1489`)

### Surface `Call` Return from AuditedHTTPClient

**IMPORTANT: `get_ssrf_safe()` does NOT use `_record_and_emit()`.** It calls `self._recorder.record_call()` directly inline at lines 608-617. The `Call` must be captured from this inline call, not from `_record_and_emit()`.

Changes:

1. **`get_ssrf_safe()` success path** (line 608-617): Capture the `Call` return from the inline `recorder.record_call()` call. Change return type from `tuple[httpx.Response, str]` to `tuple[httpx.Response, str, Call]`.

2. **`get_ssrf_safe()` error path** (line 661-672): Also calls `recorder.record_call()` inline. The `Call` return is discarded — the error path re-raises, so no return value is produced. Explicitly discard with `_ = self._recorder.record_call(...)` for clarity.

3. **`_record_and_emit()`** (line 191-258): Independently change to return `Call` instead of `None`. This benefits `_execute_request()` callers (`post()`/`get()`) if they ever need call hashes. Currently `_execute_request()` ignores the return value — this is acceptable.

4. **`WebScrapeTransform._fetch_url()`** (line 366): Wraps `get_ssrf_safe()`. Change return type from `tuple[httpx.Response, str]` to `tuple[httpx.Response, str, Call]` to surface the `Call` to `process()`.

**Replay/verifier compatibility:** Confirmed non-issue. Neither `replayer.py` nor `verifier.py` implements `get_ssrf_safe()`. They use separate `replay()`/`verify()` APIs. No changes needed.

### Add `recorder.store_payload()`

New method on `LandscapeRecorder`:

```python
def store_payload(self, content: bytes, *, purpose: str) -> str:
    """Store a transform-produced artifact in the payload store.

    For blobs that have no corresponding external call record — e.g.,
    post-extraction processed content. The purpose label is a code-level
    documentation convention — it is not persisted or emitted to telemetry.
    It exists solely to force callers to name what they're storing at the
    call site, making the intent visible in code review.

    Args:
        content: Raw bytes to store.
        purpose: Semantic label (e.g., "processed_content", "extracted_markdown").
            Not persisted — call-site documentation only.

    Returns:
        SHA-256 hex digest of stored content.

    Raises:
        FrameworkBugError: If recorder was constructed without a payload_store.
    """
    if self._payload_store is None:
        raise FrameworkBugError(
            f"store_payload(purpose={purpose!r}) called but recorder has no "
            f"payload_store. Orchestrator must configure LandscapeRecorder with "
            f"a payload_store when transforms that produce processed content "
            f"blobs are in the pipeline."
        )
    return self._payload_store.store(content)
```

**Note:** The `_prepare_call_payloads()` method in `ExecutionRepository` already handles blob storage for `record_call()` payloads (lines 540-547). `store_payload()` is for blobs that don't correspond to an external call — only the processed-content case currently.

## Enforcement

### 1. AGENTS.md Boundary Documentation

Create `src/elspeth/plugins/transforms/AGENTS.md` documenting the pipeline-data vs audit-provenance boundary. This file is read by both human reviewers and AI coding assistants (Claude Code, Codex, etc.) when working in the transforms directory.

Contents:
- The decision test ("would a pipeline operator make a decision based on this?")
- Examples of each category
- Where audit provenance goes (`success_reason["metadata"]`)
- Where operational data goes (output row via `declared_output_fields`)
- The scope constraint on `recorder.store_payload()`: if it appears in more than 2-3 transforms, the Shifting the Burden archetype is reforming and the design needs revisiting

### 2. CI Naming Convention Check

Extend the existing tier model enforcer (or add a lightweight check): fields matching `*_hash`, `*_source`, `*_ref` patterns in `declared_output_fields` trigger a warning. Not a hard block — there could be legitimate operational hashes (e.g., `content_fingerprint`) — but a flag that forces justification via allowlist entry.

### 3. Self-Documenting Audit Suffix Constants

Keep `LLM_AUDIT_SUFFIXES` as the pattern. WebScrape should define its own `WEBSCRAPE_AUDIT_FIELDS` constant. These tuples are the single source of truth for "which fields are audit-only" per transform, making the boundary visible in the code.

## Testing Strategy

### Unit Tests

For each transform with moved fields:
- Assert audit fields are **absent** from `result.row.to_dict()`
- Assert audit fields are **present** in `result.success_reason["metadata"]`
- Assert operational fields remain in the output row
- Assert `declared_output_fields` no longer contains audit field names
- Assert `_output_schema_config.audit_fields` is `None` or empty for LLM transforms

### New Unit Tests

- `recorder.store_payload()` — returns 64-char SHA-256 hex string, content is retrievable, `FrameworkBugError` raised when `payload_store` is `None`
- `AuditedHTTPClient.get_ssrf_safe()` — verify `Call.request_ref` and `Call.response_ref` are non-None after a successful fetch
- `build_llm_audit_metadata()` — verify return dict contains all 6 audit field names with correct prefix

### Integration Test

End-to-end pipeline with ChaosLLM and WebScrape (via ChaosWeb):
- Verify output CSV contains only operational fields
- Query Landscape `node_states` to confirm provenance metadata is persisted in `success_reason_json`
- Verify `elspeth explain` retrieves the provenance for a given row

### Update Affected Test Assertions

Existing tests that assert audit fields in output rows or `declared_output_fields` must be updated. Two categories of assertion changes:

**Category 1 — Row content assertions:** Tests asserting `"template_hash" in result.row` must change to assert the field is in `result.success_reason["metadata"]` instead.

**Category 2 — Schema contract assertions:** Tests asserting audit field names in `transform.declared_output_fields` must remove those assertions entirely (the fields are no longer declared output fields).

### Example Pipeline Updates

- `chaosweb/settings.yaml` — remove hash fields from sink schemas
- `chaosllm_sentiment/` — verify output doesn't change (audit fields weren't in fixed schemas)
- Verify all example pipelines still pass

## Affected Files

### Source Changes

| File | Change |
|------|--------|
| `plugins/transforms/llm/__init__.py` | Split `populate_llm_metadata_fields()` into `populate_llm_operational_fields()` + `build_llm_audit_metadata()`. Remove `get_llm_audit_fields()` from `__all__`. Remove audit suffixes from `declared_output_fields` computation. Remove `get_llm_audit_fields()` from `_build_augmented_output_schema()` (~line 209) and `_build_multi_query_output_schema()` (~line 270). |
| `plugins/transforms/llm/transform.py` | Call both new functions. Merge audit dict into `success_reason["metadata"]`. Remove audit fields from `SchemaConfig(audit_fields=...)` at lines 976, 1021. |
| `plugins/transforms/llm/multi_query.py` | Remove `LLM_AUDIT_SUFFIXES` iteration from `declared_output_fields` build (line 232-235). Extend `_QuerySuccess` with `audit_metadata` field. Collect per-query audit dicts and merge into final `success_reason["metadata"]`. |
| `plugins/transforms/llm/openrouter_batch.py` | Same pattern as transform.py |
| `plugins/transforms/llm/azure_batch.py` | Same pattern as transform.py |
| `plugins/transforms/web_scrape.py` | Remove direct `payload_store` access. Use `Call` return from `_fetch_url()` for request/response hashes (`Call.request_ref`, `Call.response_ref`). Store processed content via `recorder.store_payload()`. Put all 3 hashes in `success_reason["metadata"]`. Update `_fetch_url()` return type to `tuple[Response, str, Call]`. Fix `fetch_url_final_ip` to use `safe_request.resolved_ip`. Define `WEBSCRAPE_AUDIT_FIELDS` constant. |
| `plugins/infrastructure/clients/http.py` | `get_ssrf_safe()` captures `Call` from inline `record_call()` and returns `tuple[Response, str, Call]`. Error path discards `Call` explicitly (`_ = ...`). `_record_and_emit()` independently changed to return `Call`. |
| `core/landscape/recorder.py` | Add `store_payload(content, *, purpose)` method with `FrameworkBugError` guard for None `_payload_store`. |
| `contracts/plugin_context.py` | Remove `payload_store` field and update docstring. |
| `contracts/contexts.py` | Remove `payload_store` property from `LifecycleContext` protocol. |
| `engine/orchestrator/core.py` | Remove `payload_store=` from `PluginContext` construction. |

### Test Changes

| File | Change |
|------|--------|
| `tests/unit/plugins/transforms/test_web_scrape.py` | Assert hashes in success_reason, not row. Remove `payload_store.exists()` mock calls. Remove hash fields from `declared_output_fields` assertions. |
| `tests/unit/plugins/transforms/test_web_scrape_security.py` | Remove hash field assertions from rows and `declared_output_fields`. |
| `tests/unit/contracts/transform_contracts/test_web_scrape_contract.py` | Remove `fetch_request_hash`, `fetch_response_raw_hash`, `fetch_response_processed_hash` from output field contract assertions (~lines 210-212, 234-236). |
| `tests/unit/plugins/llm/test_transform.py` | Assert audit suffixes in success_reason, not row. Remove from `declared_output_fields` assertions (line 1163). Assert `_output_schema_config.audit_fields` is None. |
| `tests/unit/plugins/llm/test_multi_query.py` | Remove audit field assertions from `declared_output_fields` (line 162). Assert audit metadata in success_reason per query. |
| `tests/unit/plugins/llm/test_azure_multi_query.py` | Remove `declared_output_fields` assertions for audit fields (lines 886-887). Assert audit metadata in success_reason. |
| `tests/unit/plugins/llm/test_openrouter.py` | Remove `template_hash in result.row` assertions (lines 363-364, 689). Assert in success_reason. Error-path `template_hash` in `result.reason` (line 389) stays — error reasons are audit trail. |
| `tests/unit/plugins/llm/test_openrouter_multi_query.py` | Remove `template_hash in output` assertions (lines 728-729). Assert in success_reason. |
| `tests/unit/plugins/llm/test_openrouter_batch.py` | Remove audit field assertions from rows (lines 561, 797). Assert in success_reason. |
| `tests/unit/plugins/llm/test_azure_batch.py` | Remove `declared_output_fields` assertion for `template_hash` (lines 1692-1696). Assert in success_reason. |
| `tests/unit/plugins/llm/test_azure.py` | Remove `template_hash in result.row` assertions (lines 328, 577). Assert in success_reason. |
| `tests/unit/plugins/llm/test_llm_success_reason.py` | Add assertions for the 6 audit field names in `success_reason["metadata"]`. This is the natural home for provenance assertions. |
| `tests/unit/contracts/test_schema_config.py` | Add assertion that LLM transforms produce `audit_fields=None` after the change. |

### Documentation and Diagnostics

| File | Change |
|------|--------|
| `src/elspeth/plugins/transforms/AGENTS.md` | New — boundary rule documentation for human reviewers and AI assistants |
| `examples/chaosweb/settings.yaml` | Remove hash fields from sink schemas |
| `tui/` and/or `mcp/` | Follow-up task: verify `elspeth explain` reads provenance from `success_reason_json` and displays it. If it currently reads from the row, update it. |

## Implementation Notes

### Multi-Query LLM Complexity

Multi-query LLM transforms produce audit fields per query (e.g., `category_template_hash`, `sentiment_template_hash`). The `LLM_AUDIT_SUFFIXES` tuple defines which suffixes are audit-only. The migration must:

1. Extend `_QuerySuccess` with `audit_metadata: dict[str, object]` field
2. Call `build_llm_audit_metadata()` inside `_execute_one_query()` and store in `_QuerySuccess.audit_metadata`
3. In `_execute_sequential`/`_execute_parallel`, collect all per-query `audit_metadata` dicts and merge into `success_reason["metadata"]`
4. Remove `LLM_AUDIT_SUFFIXES` iteration from `declared_output_fields` build — only `MULTI_QUERY_GUARANTEED_SUFFIXES` (`_usage`, `_model`) contribute

### `declared_output_fields` Shrinkage

Removing audit fields from rows means `declared_output_fields` shrinks. This affects:
- `_output_schema_config` (guaranteed_fields will be smaller)
- Output schema construction (fewer fields declared)
- DAG edge validation (downstream `required_input_fields` must not reference removed fields)

Verify no downstream transform or sink has `required_input_fields` referencing the removed audit fields. If any do, that's a configuration bug (depending on audit data as pipeline input) and should be fixed in the pipeline config, not preserved.

### Backward Compatibility

None. CLAUDE.md: "WE HAVE NO USERS YET. Deferring breaking changes until we do is the opposite of what we want." Delete the fields from rows completely. No deprecation warnings, no compatibility shims.

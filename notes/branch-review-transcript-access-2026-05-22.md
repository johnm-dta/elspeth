# Branch review: `fix/test-audit-transcript-access`

**Reviewer:** Claude Opus 4.7 (1M)
**Date:** 2026-05-22
**Commit reviewed:** `8568b3bbc Audit log transcript access flags` (Wed May 20 15:35:31 2026 +1000)
**Position:** 1 commit ahead of RC5.2 (`14017bac2`), 42 commits behind RC5.2.
**Files touched:** `src/elspeth/web/sessions/routes/messages.py` (+15 / -1) and `tests/unit/web/sessions/test_record_audit_grade_view.py` (+77 / -7 net +70).

---

## 1. What the change does

### Plain-language description

The `GET /api/sessions/{session_id}/messages` handler already writes an `audit_access_log` row (real Landscape-tier audit table, sync, crash-on-failure) whenever the caller requests **tool-row** disclosure (`include_tool_rows=true`). The handler also accepts two other audit-grade disclosure flags — `include_llm_audit` (exposes LLM-call sidecar messages) and `include_raw_content` (exposes the pre-synthesis assistant prose hidden by the empty-state synthesizer). Before this commit, those two flags were honoured by the response shaper but **did not trigger an audit-access-log write**. After this commit, any one (or more) of the three flags triggers the audit write.

A small helper is introduced for the gate predicate:

```python
def _requests_audit_grade_messages_view(
    *,
    include_tool_rows: bool,
    include_llm_audit: bool,
    include_raw_content: bool,
) -> bool:
    return include_tool_rows or include_llm_audit or include_raw_content
```

and the call site changes from `if include_tool_rows:` to a call to that helper. The query-args allowlist (`AUDIT_GRADE_VIEW_QUERY_ARG_ALLOWLIST` in `src/elspeth/web/sessions/protocol.py`) already contains all three flags plus `limit`/`offset`, so no allowlist change is needed.

### Verbatim key block (post-change)

```python
if _requests_audit_grade_messages_view(
    include_tool_rows=include_tool_rows,
    include_llm_audit=include_llm_audit,
    include_raw_content=include_raw_content,
):
    audit_query_args = {key: value for key, value in request.query_params.items() if key in AUDIT_GRADE_VIEW_QUERY_ARG_ALLOWLIST}
    await service.record_audit_grade_view_async(
        session_id=str(session.id),
        requesting_principal=user.user_id,
        request_path=request.url.path,
        query_args=audit_query_args,
        ip_address=request.client.host if request.client else None,
    )
```

**Confidence: high.** Evidence: `git show 8568b3bbc`, `git show RC5.2:src/elspeth/web/sessions/routes/messages.py` (lines 810–855), `git show RC5.2:src/elspeth/web/sessions/protocol.py` (lines 90–106 — allowlist already covers all three flags), `git show RC5.2:src/elspeth/web/sessions/service.py` (lines 2937–2995 — `record_audit_grade_view[_async]` writes a row to `audit_access_log_table`, raises `AuditAccessLogWriteError` on SQL failure, increments `audit_access_log_write_failed_total` telemetry on failure and `audit_grade_view_total` on success).

### Why the previous gate was wrong

The previous gate (`if include_tool_rows`) had a **disclosure-audit gap**. A caller could exfiltrate raw LLM-call sidecars (`include_llm_audit=true`) or pre-synthesis prose (`include_raw_content=true`) without leaving an `audit_access_log` trail — those are the two disclosure flags whose entire reason for existing is audit-grade access. The patch closes that gap by making the audit-log write track **any** audit-grade disclosure, not only the tool-row variant.

This is consistent with the docstring above the handler, which describes `include_raw_content` as "Eval/diagnosis tooling enables it to verify whether the model converged on useful output that the synthesizer hid." That is audit-grade access by intent; the handler now treats it as such mechanically. **Confidence: high.**

---

## 2. Correctness against CLAUDE.md principles

### Trust tiers (high confidence)

The audit write is into the **Tier 1 / Landscape** audit table `audit_access_log`. The write is sync, the engine.begin() block raises `AuditAccessLogWriteError` on SQL failure (no swallowing), the call site `await`s it before the response is shaped. This is the canonical "audit primacy — fires first, sync, crash-on-failure" pattern from CLAUDE.md. The added gate is a Boolean OR over three plain booleans pulled from FastAPI `Query(False)` defaults — they are pre-coerced by FastAPI from query params, so the gate operates on already-typed values. No coercion or `.get()` on dicts. **Pass.**

### Audit primacy (high confidence)

The branch *strengthens* audit primacy. Before, two of three audit-grade disclosure flags bypassed the audit write entirely; now all three trigger it. The mechanism is the existing `record_audit_grade_view_async` path, which writes to the audit table first, increments telemetry on failure, and propagates the exception to the FastAPI error layer (the existing test `test_endpoint_fails_closed_when_audit_access_log_write_fails` proves response.status_code == 500 and `error_type == audit_access_log_write_failed`). **Correct primacy: audit → telemetry → logger; logger is not used in the changed path at all.** **Pass.**

### Offensive programming (high confidence)

The added helper takes named keyword-only booleans and returns a Boolean. The site of use passes literal kwargs from the typed FastAPI handler signature. No `.get()`, `getattr(_, _, default)`, `hasattr`, `isinstance` defensive guard, or try/except suppression introduced. The one pre-existing `request.client.host if request.client else None` is untouched and is at the HTTP-protocol boundary (Tier 3 trust boundary — `request.client` is genuinely optional in the FastAPI/Starlette protocol, not a defensive guard against a bug). **Pass.**

### Layer compliance (high confidence)

`src/elspeth/web/sessions/routes/messages.py` is L3 (web/application). It imports from `._helpers` (sibling L3) which re-exports symbols originating in L3 (`protocol`, `service`). No upward import is introduced. **Pass.**

### No legacy code policy (high confidence)

The old branch `if include_tool_rows:` is **replaced** by the helper call — no parallel old/new path, no feature flag, no deprecation wrapper. Single source of truth. **Pass.**

### Helper naming nit (medium confidence, low severity)

The helper is named `_requests_audit_grade_messages_view`. The word order is mildly awkward (it scans as "[underscore] requests audit-grade messages view" rather than the more natural "is_audit_grade_messages_view_request" or "wants_audit_grade_view"). This is a style nit only; the name is unambiguous in context and the leading underscore correctly marks it module-private. **Not a blocker.**

---

## 3. Test adequacy

### What the test additions cover

1. `test_endpoint_emits_audit_log_when_include_llm_audit_true_without_tool_rows` — exercises `?include_llm_audit=true` alone, asserts one audit row with the expected `query_args`. **Covers the previously-uncovered LLM-audit-only path.**
2. `test_endpoint_emits_audit_log_when_include_raw_content_true_without_tool_rows` — exercises `?include_raw_content=true` alone, asserts (a) one audit row with the expected `query_args` and (b) the response body actually includes the raw-content payload (`provider final prose` for the assistant message). **Covers the previously-uncovered raw-content-only path AND verifies the response actually carried the audit-grade payload, not just the audit write.**
3. `test_endpoint_records_combined_audit_grade_query_args` — exercises all three flags + an unallowlisted `api_key=secret` + `limit=25`; asserts exactly one audit row, with the unallowlisted `api_key` correctly stripped and `limit` retained. **Covers the allowlist-filter discipline under combined flags.**
4. `test_endpoint_fails_closed_when_audit_access_log_write_fails_for_non_tool_audit_views` (parametrised over `include_llm_audit=true` and `include_raw_content=true`) — injects a write failure, asserts 500 response, `error_type == audit_access_log_write_failed`, no `messages` in body, telemetry counter incremented. **Covers fail-closed semantics for both newly-audited paths.**

### Helper change

`_seed_user_assistant_tool_rows` gains an `assistant_raw_content: str | None = None` parameter so the raw-content test can seed real prose. Default `None` preserves existing call sites' behaviour; the existing tool-rows test still passes through `raw_content: None`. **Backwards-compatible within the test module, no behavioural drift.**

### Untested branches

- The existing test for `include_tool_rows=true` alone is not modified — that path is still covered by `test_endpoint_emits_audit_log_when_include_tool_rows_true`.
- The "no flag" (default) negative test is unchanged.
- The helper itself is exercised through the endpoint tests; there is no unit test for `_requests_audit_grade_messages_view` in isolation, but as a 3-arg `or` it is trivial and exhaustively exercised via the endpoint cases.

### Test-name vs assertion alignment (high confidence)

Test names accurately describe assertions. No misleading names. **Pass.**

**Overall test verdict: adequate and well-targeted.** The new tests directly mirror the structure of the existing audit-grade tests and close every behavioural gap the production change opens.

---

## 4. Multi-token collision assessment

### What multi-token touches in `messages.py`

`git diff RC5.2..feat/multi-source-token-scheduler -- src/elspeth/web/sessions/routes/messages.py` (full diff, 12 lines):

```python
@@ -640,6 +640,7 @@ def register_message_routes(router: APIRouter) -> None:
                     _transition_state_d = _transition_state.to_dict()
                     _transition_state_data = CompositionStateData(
                         source=_transition_state_d["source"],
+                        sources=_transition_state_d["sources"],
                         nodes=_transition_state_d["nodes"],
                         edges=_transition_state_d["edges"],
                         outputs=_transition_state_d["outputs"],
```

That is the **entirety** of multi-token's `messages.py` change — a single `+sources=_transition_state_d["sources"],` line inside a `CompositionStateData` constructor at line ~640, which lives inside the `POST .../compose` flow (the composer SSE/compose endpoint), nowhere near the `GET .../messages` handler that begins around line ~810 in the post-change file.

### Mechanical merge

`git merge-tree $(git merge-base RC5.2 fix/test-audit-transcript-access) RC5.2 fix/test-audit-transcript-access` reports `merged` for both files with no conflict markers — clean apply on top of RC5.2.

`git merge-tree $(git merge-base feat/multi-source-token-scheduler fix/test-audit-transcript-access) feat/multi-source-token-scheduler fix/test-audit-transcript-access` also reports `merged` — **clean merge against multi-token as well**, contrary to the operator's apprehension. The user-facing GET-messages handler and the composer's `CompositionStateData` constructor are in disjoint regions of the file.

### Semantic collision check

- Multi-token does **not** modify `_helpers.py` audit-grade machinery (`AUDIT_GRADE_VIEW_QUERY_ARG_ALLOWLIST`, etc.) — verified via `git diff RC5.2..feat/multi-source-token-scheduler -- src/elspeth/web/sessions/routes/_helpers.py | grep -E "AUDIT_GRADE|record_audit_grade|include_tool_rows|include_llm_audit|include_raw_content"` (no output).
- Multi-token does **not** modify `service.py` audit-grade machinery (`record_audit_grade_view[_async]`, `audit_access_log_table`) — verified the same way against `service.py` (no output).
- The `GET .../messages` handler signature (`include_tool_rows`, `include_llm_audit`, `include_raw_content` Query parameters) is untouched by multi-token.
- The disclosure surface this branch audit-logs (tool rows, LLM audit sidecars, raw content) is **not removed or relocated** by multi-token — multi-token's scope is sources/scheduler, not message-disclosure semantics.

**Verdict: no collision. The audit gap this branch closes is independent of multi-token's rework and persists after multi-token lands. The branch can land before or after multi-token without rework.** **Confidence: high.**

---

## 5. Landing recommendation

**`LAND_AS_IS`.**

### Justification

1. The bug being fixed is a real disclosure-audit gap: two of three audit-grade response-shaping flags were not generating audit-access-log rows. This is a Tier-1 audit primacy defect, exactly the class of bug ELSPETH's auditability standard treats as load-bearing.
2. The fix is minimal, correct, and conforms to every relevant CLAUDE.md principle (audit primacy, no defensive programming, layer compliance, no legacy shims).
3. Test coverage closes every behavioural gap opened by the production change (positive case for each new flag, combined-flags allowlist behaviour, fail-closed semantics parametrised over both new flags).
4. The branch merges cleanly on top of current RC5.2 and on top of multi-token. The 42-commit RC5.2 drift since the branch tip does not touch the same handler.
5. The "fix/test-audit-*" branch-naming convention is misleading — this branch actually fixes production code, not just tests — but the commit message ("Audit log transcript access flags") is accurate. No code change required for landing on those grounds; consider a commit-message touch-up if doing an in-flight rebase, but not a blocker.

### Edits that would be *nice* but are not required

- Rename helper `_requests_audit_grade_messages_view` → `_is_audit_grade_view_request` (style nit, not a correctness issue).
- Optional unit test for the helper in isolation (trivial; current endpoint tests already exercise every combination).

Neither rises to `LAND_WITH_EDITS`. The branch as-is meets the project's auditability bar.

---

## 6. Risks if landed on RC5.2 today

1. **Audit volume increase (medium confidence, low severity).** Every legitimate eval/diagnosis caller now writes an extra row per request when they pass `include_llm_audit` or `include_raw_content` (previously silent). Eval pipelines that hammer the endpoint will see proportionally more `audit_access_log` rows. This is *correct* behaviour — the auditability standard demands it — but if any operator was relying on the audit table size as a proxy for tool-row-disclosure volume, that proxy is now contaminated. Mitigation: writer_principal is pinned to `audit_grade_view`, so query consumers can already segment, but `query_args` is now the field of truth for which flags were requested.

2. **Fail-closed behaviour now extends to two more flags (low severity but operator-visible).** Previously, an injected `audit_access_log` write failure would only fail-close requests carrying `include_tool_rows=true`; requests carrying only `include_llm_audit=true` or `include_raw_content=true` would succeed even with a broken audit writer. After this change, the eval harness will see 500s on those flag-only requests if the audit DB is wedged. This is *correct* (matches CLAUDE.md "audit fires first, crash-on-failure"), but represents a behavioural change visible to existing eval tooling. If any eval harness opportunistically retries on 500, this is harmless; if any harness treats 500 as a hard error and abandons the run, the wedged-audit-DB blast radius is now larger.

3. **Branch naming misleads triage (low severity, social risk).** "fix/test-audit-transcript-access" reads as a test-only branch; reviewers may skim it. The commit body is accurate; PR description should call out the +15 production-line change explicitly so reviewers don't approve on the wrong mental model.

None of these are landing-blockers. Risk #1 is the intended consequence of the fix; risks #2 and #3 are operator-facing communication concerns.

---

## Confidence summary

| Claim | Confidence | Evidence |
|---|---|---|
| Change closes a real disclosure-audit gap | High | RC5.2 baseline shows `if include_tool_rows:` gating the audit write; allowlist already covers all three flags |
| `record_audit_grade_view_async` is Tier-1 Landscape audit (sync, crash-on-failure) | High | `src/elspeth/web/sessions/service.py:2937-2995` |
| No CLAUDE.md violation (audit primacy, offensive programming, layers, no legacy) | High | Direct reading of the diff against principles |
| Tests cover every new branch of the production change | High | Diff contains positive case per flag, combined-flags case, fail-closed parametrised over both new flags |
| Clean merge against current RC5.2 | High | `git merge-tree` reports `merged` with no conflict markers |
| Clean merge against `feat/multi-source-token-scheduler` | High | `git merge-tree` against multi-token reports `merged`; multi-token's only `messages.py` edit is at line 641, disjoint from the GET-messages handler at line 830+ |
| Multi-token does not relocate or remove the audit-grade disclosure surface | High | `grep -E "AUDIT_GRADE\|record_audit_grade\|include_tool_rows\|include_llm_audit\|include_raw_content"` in multi-token diffs of `_helpers.py` and `service.py` returns nothing |
| Helper naming nit is style-only, not blocking | Medium | Subjective; project style guide doesn't dictate predicate-naming form |

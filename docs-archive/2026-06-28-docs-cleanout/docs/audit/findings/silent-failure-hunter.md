# Silent-Failure Hunter — R4/R6 Suppression Audit

**Date:** 2026-05-19
**Auditor:** error-handling-auditor (independent review)
**Scope:** 6 selected R4/R6 suppressions from `config/cicd/enforce_tier_model/*.yaml`
**Method:** Read live code at cited fingerprints; verify against author-claimed `reason` / `safety`.

---

## Per-Entry Verdicts

### Entry 1 — `plugins/infrastructure/clients/http.py` :: `AuditedHTTPClient._emit_telemetry_after_audit`

**Key:** `plugins/infrastructure/clients/http.py:R4:AuditedHTTPClient:_emit_telemetry_after_audit:fp=588df931d0f0d1f1`
**Owner:** `architecture` | **Expires:** 2026-06-07 (near-expiry)
**Verdict:** **LOAD-BEARING**
**Confidence:** HIGH

**Code excerpt** (`src/elspeth/plugins/infrastructure/clients/http.py:255-288`):

```python
try:
    self._telemetry_emit(ExternalCallCompleted(...))
except contract_errors.TIER_1_ERRORS:
    raise  # System bugs and audit integrity violations must crash
except (TypeError, AttributeError, KeyError, NameError):
    raise  # Programming errors must crash
except Exception as tel_err:
    logger.warning(
        "telemetry_emit_failed",
        error=str(tel_err),
        error_type=type(tel_err).__name__,
        run_id=self._run_id,
        state_id=self._telemetry_state_id(),
        operation_id=self._telemetry_operation_id(),
        call_type=call_type_label,
        exc_info=True,
    )
```

**Analysis:**
- Tier-1 errors **re-raised first** (audit integrity preserved).
- Programmer errors (`TypeError`, `AttributeError`, `KeyError`, `NameError`) **explicitly re-raised**.
- Residual `Exception` branch is the documented telemetry-best-effort tier per CLAUDE.md primacy order (audit committed first; telemetry is async/ephemeral).
- Records the swallow at WARN with `run_id`, `state_id`, `operation_id`, `call_type`, error class, and full traceback (`exc_info=True`) — sufficient for an auditor to correlate to the post-audit moment and reconstruct what failed.
- This is the *canonical correct shape* for "audit-fires-first, telemetry-best-effort". Reason matches code precisely.

**Recommendation:** **Renew with extended TTL** (this is the reference exemplar for the pattern — propose making it permanent or 12-month renewal).

---

### Entry 2 — `contracts/contract_builder.py` :: `ContractBuilder.process_first_row`

**Key:** `contracts/contract_builder.py:R6:ContractBuilder:process_first_row:fp=4d3f0bdcf6fdd56f`
**Owner:** `bugfix` | **Expires:** 2026-06-15 (near-expiry)
**Verdict:** **LOAD-BEARING DIFFERENTLY**
**Confidence:** HIGH

**Code excerpt** (`src/elspeth/contracts/contract_builder.py:106-119`):

```python
try:
    python_type = normalize_type_for_contract(value)
except TypeError:
    python_type = object
```

**Analysis:**
- The catch IS narrow: `except TypeError` from a single function call. R6 fires because TypeError is swallowed without re-raise — but R6's purpose is to detect *swallowed programmer errors*, and this TypeError is **a documented protocol signal** from `normalize_type_for_contract` (see `src/elspeth/contracts/type_normalization.py:103-109` — explicit `raise TypeError(...)` for "unsupported type for schema contract").
- Critically: `normalize_type_for_contract` is called with `value` (Tier-3 external row data). The TypeError signals "this value's type cannot be represented in a schema contract" — that's a data shape signal, not a programmer bug.
- The fallback to `object` is **not silent**: the locked `FieldContract` is persisted with `python_type=object, source="inferred"`. An auditor inspecting the contract sees exactly which fields fell back. The audit trail does capture the inference outcome.
- However, the reason ("TypeError from normalize_type_for_contract for unsupported types (dict, list, Decimal) — maps to object") is **terse and underspecifies the safety argument**. It doesn't state that the persisted `FieldContract` is the audit record, doesn't note that the bare `TypeError` could in principle also match an unrelated `TypeError` raised inside `normalize_type_for_contract`'s lazy numpy/pandas imports (e.g., NumPy library bug), and doesn't note that this is one of three sibling sites (contract_builder, contract_propagation, type_normalization callers) implementing the same pattern.

**Drift assessment:** The code is correct. The reason text doesn't capture the audit-recording argument that makes this safe. A reviewer reading only the reason would conclude "this is silent fallback" — they wouldn't see the `FieldContract` persistence side-channel.

**Recommendation:** **Renew with corrected reason.** Proposed reason:

> `normalize_type_for_contract(value)` raises `TypeError` as a documented protocol signal when a Tier-3 row value's runtime type is outside ALLOWED_CONTRACT_TYPES (dict, list, Decimal, np.bytes_, custom classes). Fallback to `object` is recorded in the persisted FieldContract as `python_type=object, source="inferred"`, so an auditor can identify exactly which fields hit the fallback via the locked schema contract. Programmer errors (e.g. AttributeError from a corrupted FieldInfo) propagate normally.

**Residual risk:** A `TypeError` raised inside the lazy `import numpy as np` block (e.g. broken numpy install) would also be caught and mapped to `object`. Low likelihood but non-zero; consider narrowing further by inlining the protocol-signal as a sentinel return value rather than an exception — but this is a contracts-package refactor, not a 1-month bugfix.

---

### Entry 3 — `core/security/secret_loader.py` :: `CompositeSecretLoader.get_secret`

**Key:** `core/security/secret_loader.py:R6:CompositeSecretLoader:get_secret:fp=cb4d8fc57f9917d4`
**Owner:** `bugfix`
**Verdict:** **LOAD-BEARING**
**Confidence:** HIGH

**Code excerpt** (`src/elspeth/core/security/secret_loader.py:311-317`):

```python
for backend in self._backends:
    try:
        return backend.get_secret(name)
    except SecretNotFoundError:
        continue

raise SecretNotFoundError(f"Secret '{name}' not found in any backend")
```

**Analysis:**
- The catch is narrow to **a single domain-specific exception type** (`SecretNotFoundError`). This is **textbook iterator-of-providers fallback** — not a broad catch at all.
- Why R6 even fires here is mildly surprising — the rule should arguably not flag domain-specific exceptions used to drive control flow in a documented protocol. The fingerprint exists because the AST visitor treats *any* `except` without `raise` inside the handler as R6.
- After exhausting all backends, raises a synthesized `SecretNotFoundError` with the secret name — caller can act on it. No silent failure.
- Reason ("Intentional fallback pattern - CompositeSecretLoader tries loaders in priority order") matches code exactly.

**Recommendation:** **Renew indefinitely.** This is a model fallback-iterator pattern; the suppression should be permanent.

---

### Entry 4 — `contracts/data.py` :: `_get_allow_inf_nan`

**Key:** `contracts/data.py:R6:_get_allow_inf_nan:fp=a073efa3e28410f3`
**Owner:** `bugfix`
**Verdict:** **LOAD-BEARING**
**Confidence:** HIGH

**Code excerpt** (`src/elspeth/contracts/data.py:152-161`):

```python
for item in field.metadata:
    try:
        item_vars = vars(item)
    except TypeError:
        # Metadata items like plain strings don't have __dict__
        continue
    if "allow_inf_nan" in item_vars:
        value: bool = item_vars["allow_inf_nan"]
        return value
return None
```

**Analysis:**
- The catch is **the narrowest possible**: `vars()` raises `TypeError` (and only `TypeError`) when its argument has no `__dict__`. This is a CPython interpreter-level guarantee documented in the stdlib.
- The function is a **read-only query helper** asking "does this Pydantic metadata item carry an `allow_inf_nan` flag?". Items without `__dict__` (plain strings, ints, `_PydanticGeneralMetadata` subclasses without slots) cannot carry the flag by definition. `continue` to the next metadata item is the *only correct behaviour*.
- No state mutation, no fabricated value. Returns `None` (= "no constraint expressed") if no item carries the flag — a faithful absence signal.
- Reason matches code exactly.

**Recommendation:** **Renew indefinitely.**

---

### Entry 5 — `contracts/contract_propagation.py` :: `_infer_new_field_contract`

**Key:** `contracts/contract_propagation.py:R6:_infer_new_field_contract:fp=7af526cf8ac3b4be`
**Owner:** `bugfix`
**Verdict:** **LOAD-BEARING DIFFERENTLY** (same as Entry 2)
**Confidence:** HIGH

**Code excerpt** (`src/elspeth/contracts/contract_propagation.py:25-28`):

```python
try:
    python_type = normalize_type_for_contract(value)
except TypeError:
    python_type = object
```

**Analysis:**
- Identical pattern to Entry 2 — same `normalize_type_for_contract` documented-TypeError protocol signal, same fallback to `object`, same audit-recording side channel (the `FieldContract` returned by this helper is what gets persisted into the propagated `SchemaContract`).
- Reason ("TypeError from normalize_type_for_contract for unsupported transform-created field values is intentionally mapped to object by the shared inference helper.") is **stronger than Entry 2's reason** because it names "shared inference helper" and acknowledges the cross-site duplication. Better but still doesn't state the audit-recording side channel.
- Same residual concern: a TypeError from lazy numpy/pandas imports inside `normalize_type_for_contract` would be caught.

**Recommendation:** **Renew with the same corrected reason proposed for Entry 2.** Consider filing a follow-up issue to refactor `normalize_type_for_contract` to return a `Result[type, UnsupportedTypeSignal]` instead of raising — would eliminate both sites' need for an R6 exemption.

---

### Entry 6 — `contracts/url.py` :: `SanitizedDatabaseUrl.from_raw_url`

**Key:** `contracts/url.py:R6:SanitizedDatabaseUrl:from_raw_url:fp=a8a9139e2ff09e37`
**Owner:** `architecture`
**Verdict:** **LOAD-BEARING**
**Confidence:** HIGH

**Code excerpt** (`src/elspeth/contracts/url.py:165-183`):

```python
fingerprint: str | None = None
try:
    get_fingerprint_key()
    have_key = True
except ValueError:
    have_key = False

if have_key:
    fingerprint = secret_fingerprint(unquote(parsed.password))
elif fail_if_no_key:
    raise SecretFingerprintError(
        "Database URL contains a password but ELSPETH_FINGERPRINT_KEY "
        "is not set. ..."
    )
# else: dev mode - just remove password without fingerprint
```

**Analysis:**
- The catch is narrow to `ValueError` — and verified at the source: `get_fingerprint_key()` (`src/elspeth/contracts/security.py:46-65`) raises `ValueError` **and only `ValueError`** when `ELSPETH_FINGERPRINT_KEY` is unset. No other path through that function can produce a `ValueError`.
- The handler **does not silently swallow**: it sets a flag (`have_key = False`) and immediately routes to one of two explicit downstream branches — production raises `SecretFingerprintError` with full guidance, dev mode strips the password without a fingerprint. Both branches are documented behaviour.
- This is a probe-then-branch pattern, equivalent to (and clearer than) checking `os.environ.get("ELSPETH_FINGERPRINT_KEY") is None` directly — except it goes through the proper accessor so future fingerprint-key sources (Azure Key Vault loader) are honoured.
- Reason matches code exactly.

**Recommendation:** **Renew indefinitely.**

---

## Aggregate Signal

| # | Owner | Verdict | Renew? |
|---|-------|---------|--------|
| 1 | architecture | LOAD-BEARING | Yes, extend TTL |
| 2 | bugfix | LOAD-BEARING DIFFERENTLY | Yes, with corrected reason |
| 3 | bugfix | LOAD-BEARING | Yes, indefinitely |
| 4 | bugfix | LOAD-BEARING | Yes, indefinitely |
| 5 | bugfix | LOAD-BEARING DIFFERENTLY | Yes, with corrected reason |
| 6 | architecture | LOAD-BEARING | Yes, indefinitely |

**6 of 6** suppressions are load-bearing. **Zero silent-failure risks identified**, zero outright-fixable suppressions in this sample.

### Pattern observations

1. **No silent failures.** Every suppression in this sample either (a) records the swallowed condition in the audit trail (Entry 1: structured WARN with run/state/op/call context; Entries 2,5: persisted FieldContract; Entry 6: explicit two-branch routing) or (b) is a read-only query helper that returns a faithful "absence" signal (Entries 3,4).

2. **Owner-tag does NOT predict drift.** `bugfix`-owned entries are no more drifted than `architecture`-owned. Entry 1 (architecture) has the most rigorous code; Entry 3 (bugfix) is also a clean indefinitely-renewable pattern. The two `LOAD-BEARING DIFFERENTLY` verdicts (Entries 2 and 5, both `bugfix`) share a single underlying cause — both wrap the same helper function with terse reason text. This is one drifted reason replicated across sites, not a generalised owner-tag-correlated quality problem.

3. **The recurring pattern worth addressing.** Entries 2 and 5 both catch `TypeError` from `normalize_type_for_contract` to map unsupported types to `object`. This works correctly, but using exceptions for protocol signalling is fragile (any internal `TypeError` from numpy/pandas lazy imports would also be caught). A follow-up refactor to make `normalize_type_for_contract` return a sentinel (`UNSUPPORTED_TYPE_SENTINEL` or a `Result` type) would eliminate two R6 exemptions and is the only structural improvement this audit surfaces.

4. **R6 false-positive shape.** Entries 3 (domain-exception fallback iterator), 4 (read-only metadata probe), and 6 (probe-then-branch with explicit routing) are patterns where R6 fires structurally but no silent failure exists. Worth considering whether the rule could grow more precise predicates — e.g., "narrow catch followed by explicit branching on a sentinel flag" — but that's a static-analysis enhancement, not an audit finding.

### Recommendation on the two near-expiry entries (1 and 2)

- **Entry 1 (expires 2026-06-07):** **Renew, extend TTL to 12 months or mark permanent.** This is the project's canonical reference for the audit-fires-first / telemetry-best-effort pattern. Re-evaluating it every 30 days produces no signal — the code is the textbook implementation.

- **Entry 2 (expires 2026-06-15):** **Renew with corrected reason** (text above). Optionally file a follow-up issue to refactor `normalize_type_for_contract` to a sentinel-returning API, which would let Entries 2 and 5 both be deleted entirely.

---

## Risk and confidence per SME protocol

- **Confidence:** HIGH on all 6 verdicts. Each was verified against live code at the cited line ranges; cross-references (`normalize_type_for_contract`, `get_fingerprint_key`) were read directly rather than inferred.
- **Residual risks:**
  - Entries 2 and 5: a `TypeError` from inside numpy/pandas lazy imports could be caught and mapped to `object` rather than crashing. Realistic likelihood is low (broken numpy install would fail much earlier) but non-zero. Documented above.
  - Entry 1: the `exc_info=True` log captures the traceback but the WARN-level emission depends on the project's structlog routing actually persisting WARN to a durable channel during a production incident. If WARN is dropped (e.g. log-level filter set to ERROR in a deployment), the swallow becomes silent. This is a deployment-config concern, not a code defect.
- **Out of scope:** This audit covered 6 of 144 R4/R6 suppressions. The 0% drift rate in this sample is encouraging but is not statistically representative of the full corpus — `architecture`-owned and `bugfix`-owned entries were deliberately selected for high-suspicion review, so a clean result here is more reassuring than a clean random sample would be, but does not generalise to "all 144 are clean".

---

## Files cited

- `/home/john/elspeth/src/elspeth/plugins/infrastructure/clients/http.py` (lines 239-288)
- `/home/john/elspeth/src/elspeth/contracts/contract_builder.py` (lines 48-139)
- `/home/john/elspeth/src/elspeth/contracts/type_normalization.py` (lines 40-110)
- `/home/john/elspeth/src/elspeth/core/security/secret_loader.py` (lines 282-317)
- `/home/john/elspeth/src/elspeth/contracts/data.py` (lines 141-161)
- `/home/john/elspeth/src/elspeth/contracts/contract_propagation.py` (lines 17-46)
- `/home/john/elspeth/src/elspeth/contracts/url.py` (lines 134-215)
- `/home/john/elspeth/src/elspeth/contracts/security.py` (lines 46-65)
- `/home/john/elspeth/scripts/cicd/enforce_tier_model.py` (lines 188-207, rule definitions)

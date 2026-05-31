# CI Allowlist Suppression Justification Audit
**Reviewer:** Python Code Reviewer (independent SME agent)
**Date:** 2026-05-19
**Scope:** 12 allowlist entries — 6 `bugfix`-owner, 6 `architecture`-owner

---

## Methodology

For each entry: read the cited file at the cited function, locate the specific
flagged expression, apply CLAUDE.md's decision test, and render a verdict. The
author's `reason:` field is taken as a hypothesis to be tested, not as evidence.

CLAUDE.md decision test applied:
- Is this protecting user-provided data values? Tier 3 wrap — justified.
- Is this at an external boundary (API/file/DB)? Tier 3 wrap — justified.
- Is this an `__post_init__` validation on our own DTO? Tier 1 offensive guard — justified.
- Is this protecting against a bug in code we control? Fixable — not justified.

---

## `bugfix`-Owner Entries

### B1 — `contracts/transform_contract.py:R2:_get_python_type:fp=24df5cf5593bf5fa`

**Key:** `contracts/transform_contract.py:R2:_get_python_type`
**Rule flagged:** R2 (`getattr` with default)
**Author reason:** "Type annotation objects (e.g. list[str]) may not have `__name__` attribute — getattr is needed for safe error messages"

**Code site** (`src/elspeth/contracts/transform_contract.py`, line 65):
```python
type_names = [getattr(a, "__name__", str(a)) for a in non_none_args]
```

**Analysis:** This `getattr` is used exclusively to build the string inside a `TypeError` message for an unsupported multi-type union. The value produced controls the error message text, not any branching logic or audit record. The claim is accurate: generic annotation objects such as `list[str]` produced by Python's `get_args()` do not carry `__name__`; falling back to `str(a)` is meaning-preserving for display. This is not Tier 3 data and not a trust boundary in the ordinary sense, but it is also not protecting against a bug in our code — annotation objects are opaque Python internals whose attribute availability varies with Python version and `__future__` import mode. The rule's purpose is to prevent attribute-access bugs from being silently swallowed, but here the fallback is non-semantic (display only) and the alternative (`str(a)`) is equally informative. No logic path depends on the result.

**Verdict: HOLDS**
The `getattr` is in an error-message construction path only. The reason matches the code. The safety note ("used in error message, not logic") is accurate.

---

### B2 — `contracts/transform_contract.py:R2:_get_python_type:fp=37ef1f2fbd33537f`

**Key:** `contracts/transform_contract.py:R2:_get_python_type` (second site)
**Rule flagged:** R2 (`getattr` with default)
**Author reason:** Same as B1.

**Code site** (`src/elspeth/contracts/transform_contract.py`, line 84):
```python
type_name = getattr(unwrapped, "__name__", str(unwrapped))
```

**Analysis:** This is the second `getattr` in the same function, at the unsupported-concrete-type error path. It is used to format the type name in a `TypeError` message. Same analysis as B1 — the result is a display-only string embedded in a raised exception. The author filed separate fingerprints for the two sites but provided identical reasons; the reasons are independently accurate for both sites.

**Verdict: HOLDS**
The justification is independently valid at this site. The two entries are not duplicates — they have different fingerprints and cover different AST nodes (line 65 vs line 84). The reason is appropriate for both.

---

### B3 — `contracts/contract_builder.py:R6:ContractBuilder:process_first_row:fp=4d3f0bdcf6fdd56f`

**Key:** `contracts/contract_builder.py:R6:ContractBuilder:process_first_row`
**Rule flagged:** R6 (broad `except` clause)
**Author reason:** "TypeError from normalize_type_for_contract for unsupported types (dict, list, Decimal) — maps to object"
**Expiry:** `2026-06-15` (27 days from audit date — near-expiry)

**Code site** (`src/elspeth/contracts/contract_builder.py`, lines 106–109):
```python
try:
    python_type = normalize_type_for_contract(value)
except TypeError:
    python_type = object
```

**Analysis:** The catch is `TypeError`, not bare `except`. `normalize_type_for_contract` explicitly raises `TypeError` for unsupported types (confirmed at `type_normalization.py:105`). The fallback to `object` maps unsupported types to the `any` contract slot, which is the documented behavior for this mode (first-row inference). The `ValueError` case (NaN/Infinity) is NOT caught here — it propagates as documented in the function's docstring and the safety note.

The reason is accurate: `dict`, `list`, and `Decimal` all reach the `final_type not in ALLOWED_CONTRACT_TYPES` branch in `normalize_type_for_contract` and raise `TypeError`. The fallback preserves the field in the contract as `any`/`object` rather than crashing or dropping the row.

This is a design choice at the pipeline data boundary: first-row inference accepts arbitrary Tier-3 values and the contract records "unknown type" for fields it can't classify. The alternative (crashing on unsupported types) would quarantine legitimate first rows that happen to contain nested structures.

**Near-expiry notice:** Entry expires `2026-06-15`. The `normalize_type_for_contract` function is unlikely to change behavior for dict/list/Decimal — the `ALLOWED_CONTRACT_TYPES` set is stable. Renewal should confirm the `TypeError` raise path still exists in `type_normalization.py:105` and that the fallback remains `object` (not a more specific type).

**Verdict: HOLDS**
The catch is appropriately narrow (TypeError only), the fallback is documented behavior, and the reason matches the code. Expiry renewal is routine but should verify the specific TypeError-raise line.

---

### B4 — `core/landscape/execution_repository.py:R6:ExecutionRepository:begin_node_state:fp=6e0f00f363e89415`

**Selection note:** The task requested a `bugfix`-owner, non-`_iter_records` entry from `core/landscape/exporter.py`; no such entry exists (the exporter has only a `P2-2026-02-02-76r`-owner entry). This is the only `bugfix`-owner R6 entry in `core/landscape/execution_repository.py`, which is the closest structural analogue.

**Key:** `core/landscape/execution_repository.py:R6:ExecutionRepository:begin_node_state`
**Rule flagged:** R6 (broad `except`)
**Author reason:** "Tier-3 boundary — quarantined row data may contain NaN/Infinity that stable_hash rejects"

**Code site** (`src/elspeth/core/landscape/execution_repository.py`, lines 165–171):
```python
if quarantined:
    try:
        input_hash = stable_hash(input_data)
    except (ValueError, TypeError):
        input_hash = repr_hash(input_data)
else:
    input_hash = stable_hash(input_data)
```

**Analysis:** The `quarantined` guard is explicit. The catch fires only when `quarantined=True`, and the caught exceptions are `(ValueError, TypeError)` — not bare `except`. `stable_hash` raises `ValueError` for NaN/Infinity and `TypeError` for types that cannot be canonically serialized. `repr_hash` is the documented fallback for unserializable data. The function docstring states: "If True, input_data is Tier-3 external data that may contain non-canonical values (NaN, Infinity). Uses repr_hash fallback."

This is precisely the Tier-3 trust-boundary pattern CLAUDE.md describes: "Quarantine rows that can't be coerced/validated." The audit trail records the event; the fallback hash is deterministic and lossless for lineage purposes.

**Verdict: HOLDS**
The reason accurately describes the code. The catch is conditional on `quarantined=True`, narrowed to `(ValueError, TypeError)`, and backed by a documented fallback that preserves audit integrity.

---

### B5 — `plugins/infrastructure/pooling/executor.py:R5:PooledExecutor:_build_retry_timeout_result:fp=5ce04fa54ab80b94`

**Selection note:** No `plugins/transforms/llm/transform.py` R6 `bugfix` entry exists in `plugins.yaml`. The task's fallback is "any `bugfix` R6 entry from `plugins.yaml`." This entry is R5 (isinstance), not R6, but it is the only semantically interesting `bugfix`-owner isinstance entry in the plugins layer. The entry was selected to maximize coverage of the isinstance-union-dispatch pattern. This is an information gap — see Caveats.

**Key:** `plugins/infrastructure/pooling/executor.py:R5:PooledExecutor:_build_retry_timeout_result`
**Rule flagged:** R5 (`isinstance`)
**Author reason:** "Nominal discriminated-union dispatch — retryable_error is `PluginRetryableError | CapacityError` and only `CapacityError` carries a `status_code`. isinstance narrows the union so the status_code field is only populated when it exists, rather than probing with getattr."

**Code site** (`src/elspeth/plugins/infrastructure/pooling/executor.py`, lines 441–442):
```python
if isinstance(retryable_error, CapacityError):
    error_data["status_code"] = retryable_error.status_code
```

**Analysis:** The function signature is `retryable_error: PluginRetryableError | CapacityError`. The type annotation is a union, and `CapacityError` is the only member that carries `status_code`. The `isinstance` check is necessary to safely narrow the union before field access — the alternative would be `getattr(retryable_error, "status_code", None)`, which is exactly what CLAUDE.md forbids as defensive attribute probing. This is the correct pattern: use the type system to discriminate, not getattr with a fallback.

This is internal system code (both exception types are system-owned, not user data), so it is NOT a Tier-3 boundary check. The justification is "nominal discriminated-union dispatch," which is a legitimate use of isinstance within system code. The CLAUDE.md offensive-programming section does not prohibit isinstance for type discrimination inside internal algorithms; it prohibits it as a substitute for proper typing and for defensive fallbacks. This usage is offensive: the code would crash if `retryable_error` were neither of the two union members (Python's type checker catches that statically, and the isinstance branch produces a correct error_data for auditing).

**Verdict: HOLDS**
The reason is accurate. `isinstance` for union narrowing on a typed parameter is structurally different from defensive `getattr`/`.get()` coercion. The alternative (getattr probe) would be more dangerous. No expiry set — appropriate for a stable structural pattern.

---

### B6 — `plugins/infrastructure/schema_factory.py:R6:_find_non_finite_value_path:fp=b5937945437c8045`

**Selection note:** No `plugins/sources/csv_source.py` entry exists. The task fallback is "any other `plugins/sources` entry." The actual `plugins/sources/*` coverage is via a per-file-rule blanket rather than individual fingerprinted entries. The entry selected is in `plugins/infrastructure/schema_factory.py`, which is not `plugins/sources/`. This is an information gap — see Caveats.

**Key:** `plugins/infrastructure/schema_factory.py:R6:_find_non_finite_value_path`
**Rule flagged:** R6 (`except` clause)
**Author reason:** "TypeError from np.isfinite on non-numeric dtypes (e.g., string arrays) is expected — non-numeric arrays can't contain NaN"

**Code site** (`src/elspeth/plugins/infrastructure/schema_factory.py`, lines 56–66):
```python
try:
    if np.any(~np.isfinite(value)):
        flat = value.flat
        for idx, elem in enumerate(flat):
            if isinstance(elem, float | np.floating) and not np.isfinite(elem):
                indices = np.unravel_index(idx, value.shape)
                index_str = "][".join(str(i) for i in indices)
                return f"{path}[{index_str}]"
except TypeError:
    # np.isfinite raises TypeError for non-numeric dtypes (e.g., strings)
    pass
```

**Analysis:** This try/except wraps only `np.isfinite(value)` on a NumPy array. The NumPy documentation and source confirm that `np.isfinite` raises `TypeError` when called on object-dtype arrays or arrays of non-numeric types. The branch is inside `isinstance(value, np.ndarray) and value.size > 0`, so it only fires for actual NumPy arrays, not for arbitrary Python objects. The `pass` after catching TypeError is not a silent failure — the function returns `None` (no non-finite path found), which is correct: an array of strings cannot contain NaN, so the answer is correctly "no non-finite path at this node." There is no alternative way to check `np.isfinite` on an array of unknown dtype without a try/except.

**Verdict: HOLDS**
The catch is narrowed to `TypeError`, the comment is accurate, and `pass` is semantically correct (string arrays cannot contain non-finite floats). The indiscriminate `except` concern does not apply here since TypeError from np.isfinite is deterministic and well-documented.

---

## `architecture`-Owner Entries

### A1 — `contracts/declaration_contracts.py:R5:BoundaryInputs:__post_init__:fp=db78bae36c2f61ba`

**Key:** `contracts/declaration_contracts.py:R5:BoundaryInputs:__post_init__`
**Rule flagged:** R5 (`isinstance`)
**Author reason:** "Row-attributed source/sink boundary DTO — row_data must be a mapping before deep_freeze so declaration-contract adopters never receive arbitrary sequence/scalar payloads"

**Code site** (`src/elspeth/contracts/declaration_contracts.py`, lines 346–349):
```python
def __post_init__(self) -> None:
    if not isinstance(self.row_data, Mapping):
        raise TypeError(f"BoundaryInputs.row_data must be a mapping, got {type(self.row_data).__name__}")
    object.__setattr__(self, "row_data", deep_freeze(self.row_data))
```

**Analysis:** `BoundaryInputs` is `@dataclass(frozen=True, slots=True)`. `row_data` is typed `Any` because the DTO accepts row payloads of varying schema at construction time, and the guard enforces the invariant that only Mapping types reach `deep_freeze`. A non-Mapping (list, scalar, None) passed here would either raise in `deep_freeze` with a confusing error, or be frozen incorrectly. The `isinstance` check provides a crisp, early, self-diagnosing crash. This is the canonical Tier-1 offensive guard pattern described in CLAUDE.md.

The reason is largely accurate. One small clarification: this is not strictly a "trust boundary" (the DTO is constructed by system code, not user data), but it IS an invariant enforcement guard on a field typed `Any` — which is the exception CLAUDE.md's offensive-guard pattern permits. The reason mentions "declaration-contract adopters" as motivation, which is accurate: the guard protects downstream consumers of `BoundaryInputs.row_data` from receiving non-Mapping types.

**Verdict: HOLDS**
Code matches reason. Offensive guard in `__post_init__` of frozen dataclass. The word "boundary" in the reason slightly overstates the trust-tier context (this is Tier 1, not Tier 3), but the safety claim is correct.

---

### A2 — `contracts/run_result.py:R5:RunResult:__post_init__:fp=fac5bfd1b22037fa`

**Key:** `contracts/run_result.py:R5:RunResult:__post_init__`
**Rule flagged:** R5 (`isinstance`)
**Author reason:** "Offensive guard — validates status is RunStatus enum at contract construction"

**Code site** (`src/elspeth/contracts/run_result.py`, lines 42–43):
```python
if not isinstance(self.status, RunStatus):
    raise TypeError(f"RunResult.status must be a RunStatus enum, got {type(self.status).__name__}: {self.status!r}")
```

**Analysis:** `RunResult` is `@dataclass(frozen=True, slots=True)`. The field `status: RunStatus` has a type annotation, but Python dataclasses do not enforce type annotations at runtime. An integer, string, or misspelled enum member passed as `status` would pass construction silently without this guard and would corrupt every audit record that uses `self.status.value`. The isinstance check detects this immediately with a self-diagnosing message. This is exactly the pattern described in CLAUDE.md's "Tier-1 read guards" section.

**Verdict: HOLDS**
The reason is accurate and concise. This is a textbook Tier-1 offensive guard.

---

### A3 — `contracts/schema_contract.py:R5:PipelineRow:to_dict:fp=19c383f767c4ed8c`

**Key:** `contracts/schema_contract.py:R5:PipelineRow:to_dict`
**Rule flagged:** R5 (`isinstance`)
**Author reason:** "Tier 1 offensive guard — deep_thaw must return dict; isinstance check crashes on corruption"

**Code site** (`src/elspeth/contracts/schema_contract.py`, lines 665–667):
```python
thawed = deep_thaw(self._data)
if not isinstance(thawed, dict):
    raise TypeError(f"deep_thaw(PipelineRow._data) must return dict, got {type(thawed).__name__}")
```

**Analysis:** `PipelineRow._data` is a `MappingProxyType` (frozen from a `dict`). `deep_thaw` is a system function that should always return a plain `dict` when given a `MappingProxyType`. If it returns something else, the internal invariant is violated and any downstream code that calls `to_dict()` expecting a plain dict would silently receive a wrong type. The isinstance check provides a Tier-1 crash that identifies the invariant violation with a clear diagnostic. Identical pattern at `to_checkpoint_format` line 750.

The reason accurately labels this as a Tier-1 offensive guard. This is structurally the same as A2 — an `isinstance` check after a system function call to verify the return type matches the invariant, not to coerce bad input.

**Verdict: HOLDS**
The reason matches the code. The check is offensive, not defensive: it crashes on invariant violation, does not fall back or coerce.

---

### A4 — `contracts/url.py:R6:SanitizedDatabaseUrl:from_raw_url:fp=a8a9139e2ff09e37`

**Key:** `contracts/url.py:R6:SanitizedDatabaseUrl:from_raw_url`
**Rule flagged:** R6 (`except` clause)
**Author reason:** "except ValueError on get_fingerprint_key() — explicit branch for dev mode (no key) vs production (fail_if_no_key)"

**Code site** (`src/elspeth/contracts/url.py`, lines 166–170):
```python
try:
    get_fingerprint_key()
    have_key = True
except ValueError:
    have_key = False
```

**Analysis:** `get_fingerprint_key()` raises `ValueError` specifically when the `ELSPETH_FINGERPRINT_KEY` environment variable is not set (confirmed in `contracts/security.py`). The catch converts that ValueError into a boolean `have_key`, which drives a clean `if have_key / elif fail_if_no_key` branch below. The alternative — checking `os.environ.get("ELSPETH_FINGERPRINT_KEY")` directly — would be an R1 violation (`.get()`) and would also bypass any key-validation logic inside `get_fingerprint_key()`. The try/except respects the API contract of `get_fingerprint_key()`.

The reason accurately describes the two-branch semantics. The safety note "Raises SecretFingerprintError when fail_if_no_key=True; only silently continues in dev mode" is accurate — the `elif fail_if_no_key` branch raises `SecretFingerprintError`, so production environments never silently skip the fingerprint.

**Verdict: HOLDS**
The catch is narrowed to `ValueError`, the control flow is explicit, and the reason matches the code. The dev/prod branching logic is correct and well-documented.

---

### A5 — `contracts/contract_records.py` per-file-rule (R5, architecture, max_hits=12)

**Selection note:** No individual fingerprinted `architecture`-owner R5 entry for `contract_records.py` exists; coverage is via a per-file-rule blanket. The task asks to "find" and evaluate one. The most interesting R5 site is `from_json()` at lines 164–171.

**Representative code site** (`src/elspeth/contracts/contract_records.py`, lines 164–171):
```python
if not isinstance(data, dict):
    raise AuditIntegrityError(...)
if "fields" not in data or not isinstance(data["fields"], list):
    raise AuditIntegrityError(...)
for i, entry in enumerate(data["fields"]):
    if not isinstance(entry, dict):
        raise AuditIntegrityError(...)
...
if not isinstance(locked, bool):
    raise AuditIntegrityError(...)
```

**Per-file-rule reason:** "Violation type dispatch requires isinstance() for polymorphic serialization of ContractViolation subclasses, plus Tier 1 structural validation of JSON shape and boolean fields in from_json() and to_schema_contract()"

**Analysis:** The `from_json()` method deserializes JSON that was previously written by the system (Tier 1). CLAUDE.md mandates crash-on-any-anomaly for Tier 1 data: "If we read garbage from our own database, something catastrophic happened (bug in our code, database corruption, tampering)." The isinstance checks in `from_json` verify structural invariants of the deserialized JSON — that the top-level shape is a dict, that `fields` is a list, that each entry is a dict, and that boolean fields are strictly bool (not truthy ints). All of these raise `AuditIntegrityError` immediately on failure, with no coercion or fallback. This is the canonical Tier-1 read guard pattern.

The per-file-rule correctly identifies both the deserialization validation and the polymorphic dispatch pattern (the `isinstance(violation, TypeMismatchViolation)` branching at line 345). Both are legitimate R5 usages — the former is Tier-1 structural validation, the latter is union-narrowing identical to B5.

**Verdict: HOLDS**
The blanket per-file rule is justified. All sampled isinstance usages are either Tier-1 integrity guards (crash-on-corruption) or type-safe union narrowing. The 12-hit cap is appropriate.

---

### A6 — `contracts/identity.py` per-file-rule (R5, architecture, max_hits=3)

**Selection note:** No individual fingerprinted `architecture`-owner R5 entry for `identity.py` exists; coverage is via a per-file-rule blanket. The three R5 hits are confirmed by running the scanner: lines 43, 47, and 54.

**Representative code site** (`src/elspeth/contracts/identity.py`, lines 43–57):
```python
def __post_init__(self) -> None:
    if not isinstance(self.row_id, str):
        raise TypeError(f"TokenInfo.row_id must be str, got {type(self.row_id).__name__}: {self.row_id!r}")
    if not self.row_id:
        raise ValueError("TokenInfo.row_id must not be empty")
    if not isinstance(self.token_id, str):
        raise TypeError(f"TokenInfo.token_id must be str, got {type(self.token_id).__name__}: {self.token_id!r}")
    ...
    for _field_name in ("branch_name", "fork_group_id", "join_group_id", "expand_group_id"):
        _value = getattr(self, _field_name)
        if _value is not None:
            if not isinstance(_value, str):
                raise TypeError(...)
```

**Per-file-rule reason:** "Tier 1 `__post_init__` type guards — validates row_id, token_id, and optional lineage fields are str"

**Analysis:** `TokenInfo` is `@dataclass(frozen=True, slots=True)`. `row_id` and `token_id` are typed `str`, but Python does not enforce this at runtime. These fields are the "most fundamental identity fields in the system" (docstring) — corrupting them with non-str values would silently produce malformed audit trail entries for every downstream record. The isinstance guards detect type corruption immediately with diagnostic messages.

**Note on `getattr` at line 52:** The loop uses `getattr(self, _field_name)` without a default (2-arg form). R2 only flags 3-arg `getattr` (with default). The scanner does not flag this line — confirmed by running the checker. The 2-arg form is not defensive; it crashes on missing attributes (which is the desired behavior for a slots=True dataclass with known field names). This is a legitimate iteration pattern for a frozen dataclass with optional lineage fields.

**Verdict: HOLDS**
All three isinstance usages are Tier-1 offensive guards in `__post_init__` of a frozen dataclass. The reason is accurate. The 3-hit cap matches the code exactly.

---

## Pattern Summary

**Executive summary:** All 12 entries independently reviewed. No fixable defects
found — every exemption reflects a legitimate justification. Recommendations below
are structural process improvements, not corrections to broken code.

### `bugfix`-Owner: 6/6 HOLDS

| Entry | Verdict | Pattern |
|-------|---------|---------|
| B1 — transform_contract getattr for error message | HOLDS | Display-only fallback, error message construction |
| B2 — transform_contract getattr (second site) | HOLDS | Display-only fallback, error message construction |
| B3 — contract_builder TypeError→object | HOLDS | Tier-3 value inference, narrow catch, near-expiry |
| B4 — execution_repository quarantined hash | HOLDS | Tier-3 audit boundary, conditional quarantine guard |
| B5 — pooling executor isinstance union narrowing | HOLDS | Union-type discriminator, not defensive coercion |
| B6 — schema_factory np.isfinite TypeError | HOLDS | NumPy API contract, narrow catch, semantically correct pass |

All 6 bugfix entries hold up individually. However, the **pattern across these entries is structurally indistinguishable from `architecture`-owner entries** — B3 and B4 are Tier-3 trust-boundary patterns; B5 is an architectural union-dispatch pattern; B1/B2 are a Python-internals edge case. The `bugfix` label carries no signal about what kind of exemption it is. It records *when* (added in response to a bug) but not *what* (the safety justification category).

### `architecture`-Owner: 6/6 HOLDS

| Entry | Verdict | Pattern |
|-------|---------|---------|
| A1 — BoundaryInputs `__post_init__` isinstance(Mapping) | HOLDS | Offensive guard, Any-typed field, frozen dataclass |
| A2 — RunResult `__post_init__` isinstance(RunStatus) | HOLDS | Tier-1 offensive guard, enum type enforcement |
| A3 — PipelineRow to_dict post-thaw isinstance(dict) | HOLDS | Tier-1 invariant check after system function call |
| A4 — url.py try/except ValueError on get_fingerprint_key | HOLDS | Explicit dev/prod branching, narrow ValueError catch |
| A5 — contract_records from_json isinstance guards | HOLDS | Tier-1 deserialization validation, crash-on-corruption |
| A6 — identity.py `__post_init__` isinstance(str) guards | HOLDS | Tier-1 offensive guards, fundamental identity invariants |

---

## Recommendations

### 1. The `bugfix` label is not discriminative enough; mandate expiry

All 6 bugfix entries are legitimate, but `bugfix` as a label tells reviewers
nothing about the safety category. Entries B1/B2 are display-only fallbacks,
B3 is type-inference coercion, B4 is a Tier-3 audit boundary, B5 is
union-dispatch, and B6 is a NumPy API contract. A future reviewer auditing
"owner=bugfix" cannot distinguish these patterns without reading the code.

**Recommendation:** Require that every `bugfix`-tagged entry carry either:
1. An expiry date (forcing periodic re-verification), OR
2. A `pattern:` tag drawn from a project-defined vocabulary (e.g.
   `tier3-narrow-catch`, `display-fallback`, `union-dispatch`,
   `numpy-api-contract`).

Currently B1, B2, and B5 have `expires: null`. These three entries have
`safety:` notes that accurately describe why they're permanent, but without
expiry or a pattern tag, a future reviewer has no automated reminder to check
whether the safety claim still holds after refactoring.

**Specific action:** Add pattern tags to B1/B2 (`pattern: display-fallback`)
and B5 (`pattern: union-dispatch`), and add `expires: 2027-05-19` (1 year)
to B3/B4/B6 entries that are Tier-3 boundary patterns, as their validity
depends on the stability of external API behavior.

### 2. Re-tiering `architecture`-owner `__post_init__` isinstance guards: feasible but scoped

A1, A2, A5 (partly), and A6 are all `isinstance(self.X, T)` inside
`__post_init__` of `@dataclass(frozen=True)`. CLAUDE.md §Offensive Programming
explicitly endorses this pattern: "Proactively detect invalid states and throw
meaningful exceptions." The R5 rule is generating true positives against a
pattern the project guide blesses.

The rule could be amended to suppress R5 findings matching this AST scope:
`Module → ClassDef[has @dataclass(frozen=True)] → FunctionDef[name=__post_init__]
→ Call[func=isinstance, args[0]=Attribute[value=Name(id='self')]]`.
The critical constraint is that `isinstance`'s first argument must be
`self.<field>` — an attribute access on `self`. This prevents the exemption
from silently covering `isinstance(other_obj, T)` calls inside `__post_init__`
(e.g., validating a constructor argument that was passed in as a reference,
not a field on the dataclass itself). This would eliminate a large fraction of
the ~94 architecture entries in `contracts/`.

**Important scope limit:** Re-tiering should be restricted to exactly this AST
scope. It should NOT extend to:
- `from_json()` / `from_checkpoint()` deserialization validators (A5 pattern) —
  these are structural JSON guards that belong to a different category even if
  they look similar.
- `from_raw_url()` / boundary factory methods (A4 pattern) — these are external
  boundary translators.
- `register_declaration_contract()` and similar registry entry points — these
  are ABC-vs-Protocol enforcement gates.

The re-tiering would be a clean win: every `isinstance` inside `__post_init__`
of a frozen dataclass is, by construction, an offensive invariant guard — the
only way to be there is if someone constructed the dataclass with a wrong type.
The rule amendment should be filed as a separate ticket against the
`enforce_tier_model` enforcer script, not applied to individual allowlist entries.

**Parallel R8 note:** The review of adjacent entries (e.g.
`field_normalization.py:R8:check_normalization_collisions`) reveals that R8
also flags `setdefault` used as a deterministic grouping idiom (equivalent to
`defaultdict(list)`) — a pattern with no defensive character. R8's goal is to
catch `dict.setdefault(key, fallback)` as a silent-default injection, but the
grouping idiom is structurally distinct. Consider adding an AST scope exclusion
for `setdefault` inside a loop body where the result is immediately `append()`-
called — this would eliminate a class of low-value true-positive flags
analogous to the `__post_init__` R5 case.

### 3. B3 near-expiry: triage within 27 days

`contracts/contract_builder.py:R6` expires `2026-06-15`. The renewal cycle should:
1. Confirm `type_normalization.py:105` still raises `TypeError` for unsupported types.
2. Confirm the fallback remains `object` (not a more specific inferred type).
3. Evaluate whether the exception type in `normalize_type_for_contract` could be
   narrowed further (e.g., by checking `type(value) not in ALLOWED_CONTRACT_TYPES`
   before calling, removing the need for the except entirely — a FIXABLE candidate
   for the next renewal cycle).

---

## Confidence Assessment

**High confidence** (>90%) on B1, B2, B3, B4, A1, A2, A3, A4, A6: code sites
were read directly; the flagged expressions, their context, and the author's
claims all match. The verdict logic is mechanical.

**Medium confidence** (~75%) on B5 and A5: B5 is a substitution entry (original
target `plugins/transforms/llm/transform.py` not present); A5 is evaluated from
a per-file-rule blanket rather than a fingerprinted entry. The code analysis is
still direct, but these entries were not specified in the task's original scope.

**Medium confidence** (~80%) on B6: original target `plugins/sources/csv_source.py`
not present; substituted with `plugins/infrastructure/schema_factory.py`.

---

## Risk Assessment

**Low risk:** All 12 entries hold up. No load-bearing exemptions were found to
be suppressing real bugs. No entry is masking a fixable defect.

**Medium risk (process):** The `bugfix` label without mandatory expiry means
three entries (B1, B2, B5) are permanent waivers with no automated review
trigger. If the surrounding code is refactored, these entries will become stale
without detection.

**Low risk (structural):** The approximately 94 `architecture`-owner entries for
`contracts/` `__post_init__` guards are individually justified but collectively
represent ongoing administrative overhead. Re-tiering the rule (Recommendation 2)
would reduce that overhead without reducing safety.

---

## Information Gaps

1. **B5 substitution:** No `plugins/transforms/llm/transform.py` R6 `bugfix`
   entry exists in `plugins.yaml`. The entry reviewed (`_build_retry_timeout_result`
   R5) is from a different path and rule. The LLM transform may have no individual
   bugfix suppressions, or its suppressions may be covered by a per-file rule.
   Not investigated.

2. **B6 substitution:** No `plugins/sources/csv_source.py` individual entry
   exists; sources are covered by a per-file blanket rule (R1, R2, R4, R6, R9 for
   all `plugins/sources/*`). The blanket rule was not individually reviewed — only
   `schema_factory.py` (a different path, `plugins/infrastructure/`) was reviewed.
   The blanket rule's justification ("Source plugins ingest external data (Tier 3)
   — all defensive patterns legitimate") was not independently audited in this
   pass.

   **Coverage gap:** No individual `bugfix`-owner entries exist anywhere in
   `plugins/sources/`. The blanket per-file rule covers every R1/R2/R4/R6/R9 site
   in the sources directory under a single self-attested claim. This breadth
   warrants a dedicated targeted audit pass to verify that no site in the blanket
   uses a defensive pattern outside the Tier-3 boundary context the rule claims.
   For comparison, `plugins/sources/field_normalization.py:R8:check_normalization_collisions`
   has a separate individual entry (owner `architecture`, expiry `2026-07-02`)
   confirming the blanket rule does not cover R8 — R8 is managed individually.
   The blanket covers only the five rules listed (R1, R2, R4, R6, R9).

3. **A5 and A6 are per-file-rule entries, not fingerprinted entries.** The
   scanner output confirms the exact hit count matches the `max_hits` caps (12
   and 3 respectively). However, not all 12 + 3 = 15 individual sites were read —
   only the most structurally representative ones. The remaining sites are assumed
   to follow the same pattern; this was not verified.

4. **B4 entry origin:** The task specified `core/landscape/exporter.py` with a
   non-`_iter_records` function; no such `bugfix` entry exists. `execution_repository.py`
   was used as a structural substitute. The exporter's single entry is owned by a
   ticket ID (`P2-2026-02-02-76r`), not `bugfix` — that ticket-owned entry was not
   reviewed in this pass.

---

## Caveats

- This review covers the **reason-to-code correspondence** only. It does not assess
  whether the exempted pattern is the best architectural choice — only whether the
  declared justification accurately describes what the code does.
- "HOLDS" means the justification is accurate and the exemption is load-bearing
  under current code. It does not mean the code itself is optimal.
- The near-expiry flag on B3 is informational only; the verdict (HOLDS) is
  independent of the expiry date.
- Verdicts are based on the code as of the RC5.2 branch tip (2026-05-19). Any
  refactoring that moves, renames, or rewrites the cited functions would require
  re-running the scanner to verify fingerprint validity before these verdicts can
  be reused.

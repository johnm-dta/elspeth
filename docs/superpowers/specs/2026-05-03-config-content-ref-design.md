# Audited Content Injection — Widened `blob_ref` Design

**Epic:** [elspeth-fdebcaa79a](filigree:elspeth-fdebcaa79a) — Audited content injection: widen `blob_ref` to support `inline_content` mode for plugin config fields (config-content-ref capability)
**Parent release:** [elspeth-d0ef7cbf54](filigree:elspeth-d0ef7cbf54) — RC 5
**Date:** 2026-05-03
**Status:** Proposed (revision 1 — translates the 2026-05-03 five-reviewer panel synthesis into a phase-decomposed spec, applies advisor reframe on phase ordering and bug-verification scope)
**Branch:** RC5-UX (or successor)
**Tier-artifact match:** **M-tier** change delivered as a six-phase plan. The epic's 10 high-level steps collapse into six independently reviewable, independently testable, independently deployable PRs. P1 is a markdown PR (ADR + VAL data); P2 / P2b / P3 / P4 / P5 carry production code. P2b is a shape-preserving refactor that retires architectural gap [`elspeth-be405bac87`](filigree:elspeth-be405bac87) (canonical DB-form ↔ YAML-form adapter for `composition_state` walks) before P3's walker extension lands. Each PR is one phase against the parent epic. The frontend recovery surface (SecretsPanel-equivalent UI for inline refs) is **not** a phase of this epic — it is filed as a separate Filigree follow-up at P5 close (F-1 in the overview), so the RC5 ship gate does not depend on UI polish.

**Path conventions.** Throughout this spec, `web/...`, `core/...`, `contracts/...`, and `engine/...` are shorthand for `src/elspeth/web/...`, `src/elspeth/core/...`, `src/elspeth/contracts/...`, and `src/elspeth/engine/...` respectively. Full paths appear the first time a file is cited per major section; later references in the same section may use the shorthand. Test paths are written full from repo root (`tests/...`).

**Phase ordering pivot.** An earlier draft proposed validation parity in P3 followed by runtime preflight in P4. The five-reviewer panel's Risk table ranks validation parity at "High — Shape-1-class footgun." Shipping composer-green / runtime-red as a deliberate intermediate state would synthesise the exact divergence Shapes 1 and 8 in `tests/integration/pipeline/test_composer_runtime_agreement.py` exist to close. Revision 1 inverts the ordering: **runtime preflight + audit + lifecycle pinning land first** (P3, fail-closed; no path emits refs), **validation parity + composer L3 widening land second** (P4, with Shape 9 sub-pin in the agreement suite). Composer-emitted refs are mechanically impossible to author until P4 has merged.

---

## 1. Goals and Non-Goals

### 1.1 Goal

Operators of ELSPETH pipelines can place long-form content (LLM system prompts, SQL queries, regex libraries, JSON templates, public certs, allowlists/denylists) into any plugin config field by reference, with the content resolved at runtime from the existing blob store and recorded in the Tier 1 audit trail. Specifically:

1. The same `blob_ref` shape that today binds a source's data file (`{blob_ref: ID}`) is widened with a `mode` discriminator to also late-bind raw content into arbitrary string-typed plugin fields (`{blob_ref: ID, mode: inline_content, sha256: HASH, encoding?: ENC}`).
2. The composer pins the content hash at submit time. The runtime fetches by ID, verifies the hash, fails closed on mismatch, and writes a Tier 1 audit row recording `(field_path, blob_id, content_hash, byte_length, mime_type, encoding)` *before* the resolved bytes flow into plugin instantiation.
3. The blob retention machinery (`blob_run_links`) extends to cover any `blob_ref` discovered during config tree-walk at submit time, not just source-data references. This closes the gap where a transform-options-referenced blob could be GC'd while a pipeline still references it.
4. The composer's chat-driven LLM authors widened-`blob_ref` markers without ever seeing the underlying bytes. The composer surfaces only the ref-shape metadata (`{blob_id, mime_type, size_bytes, content_hash, filename}`) to the LLM.

### 1.2 Non-Goals

- **Non-blob-backed late-binding.** No env-binding, no upstream-node-output binding, no plugin-instance refs. Closed by the ADR's late-binding rule (§5.5).
- **Changes to `secret_ref` resolver semantics.** Secrets stay sensitive — fingerprinting, redaction, audit isolation, never-echo-on-error. Per the panel verdict, extending the secrets system to cover non-sensitive content is a category error.
- **Backwards compatibility for the original-proposal `{blob_content_ref: ...}` shape.** Never shipped, no migration needed.
- **General refactor of `core/secrets.py`.** The secret-ref resolver is the design precedent; this spec adds a sibling resolver, not a unified walk.
- **Composer-side preview rendering of resolved content.** Out of scope for this epic; if dashboard scoring requires it, it will be filed as a follow-up against the epic that ships the SecretsPanel-equivalent UI for blob refs (§13).

### 1.3 In-Scope Plugin Field Surface

Any string-typed plugin option in any of: source `options`, transform `options`, sink `options`, gate `options`, aggregation `options`. The resolver tree-walk is structurally identical to `resolve_secret_refs` (`src/elspeth/core/secrets.py:62-88`); the field surface is "anywhere a string can appear in a plugin config." There is no closed list of "fields that may be inlined" — the closure rule is on *who decides to use the late-binding shape*, not on which fields it can target.

### 1.4 Quantified NFRs

| NFR | Target | Verification | Source |
|---|---|---|---|
| Per-ref content size (upper) | `256 KiB` — covers the largest sampled internal prompt-like artifact (`203,370 B`) with headroom while bounding each resolved value | Unit + integration coverage of `BlobContentResolutionError(oversized=...)`; numeric value is recorded in ADR-025 with cited dataset | M1 |
| Per-ref content size (lower) | Soft warning below `256 B` for composer-authored refs; no hard validation reject solely for being small, because measured config examples include legitimate short public content | Unit coverage that the composer tool emits the warning and validation still accepts small refs that otherwise satisfy the marker contract | M1 |
| Aggregate per-config bytes cap | `1 MiB` — allows four near-cap refs or many smaller refs while bounding total preflight bytes under the existing 30-second `_call_async` timeout budget | Unit test pinning the aggregate-budget arithmetic; integration test with N refs at the per-ref cap demonstrating preflight completes within 30s on CI infra | M1, S2 |
| Preflight resolution latency (sanity bound) | p95 ≤ 1.5s with N ≤ 8 refs and per-ref content at the per-ref cap, measured in CI on standard infra | `tests/integration/web/test_blob_inline_preflight_latency.py` (new) — order-of-magnitude bound, not a tight budget | M1 |
| Hash-pin determinism (round-trip) | The audit-row `content_hash` re-derived from the blob's stored bytes at run-time equals the value composer pinned at submit time, byte-for-byte | Property test against `BlobServiceImpl.read_blob_content`'s integrity verification + new resolver round-trip | C1, S6 |
| Lifecycle pinning coverage | Every ref discovered during config tree-walk at submit time creates exactly one `blob_run_links` row with `direction='input'`. Zero orphan-blob GC events in property-test campaign. | New unit test pinning the link-creation walk; property test sweeping ref distributions across source/transform/sink/gate/aggregation | H5, S5 |
| Composer→runtime hash agreement | `0` events expected: any non-zero count is an audit incident, not an SLO budget | OTel counter `composer.blob_inline.hash_mismatch_total` (composer-pinned hash != runtime-fetched hash); SLO threshold = 0 | C1 |
| Audit-row write failure (Tier-1) | `0` events expected: a resolved ref must produce an audit row before bytes reach plugin instantiation | OTel counter `composer.blob_inline.audit_row_tier1_violation_total`; SLO threshold = 0 | C1, S5 |
| Validation parity violation rate | `0` events expected: `validate_pipeline` accepts a config iff runtime preflight resolves it to the same set of `BlobContentResolutionError` codes (modulo `hash_mismatch` which is a runtime-only concern) | Shape 9 in `tests/integration/pipeline/test_composer_runtime_agreement.py` with three sub-pins (§9) | H5 |

The cap values are now fixed by ADR-025 and the 2026-05-24 VAL evidence in `docs/composer/evidence/blob-inline-content-cap-2026-05-24.md`. The local persisted composition state is small (p99 `649 B`), but the prompt-like repository corpus includes artifacts up to `203,370 B`; that evidence rejects a guessed `64 KiB` cap while still keeping each ref bounded.

---

## 2. Context — What Already Exists

### 2.1 The two existing ref forms

ELSPETH has two production ref forms today, both implementing late-binding semantics over different trust tiers:

| Ref form | Where authored | Where resolved | Trust transition | Audit handling |
|---|---|---|---|---|
| `{secret_ref: NAME}` | Anywhere in plugin config | `core/secrets.py::resolve_secret_refs` (`src/elspeth/core/secrets.py:62-88`) | Tier 3 → Tier 2 (resolver crashes on missing) | HMAC fingerprint only; never echo value on error; redacted from logs |
| Exact `${NAME}` | Anywhere in plugin config | Same resolver, branch at `_is_secret_env_ref` (`core/secrets.py:100-110`) | Tier 3 → Tier 2 | Same |
| `{blob_ref: ID, path: ...}` | **Source `options` only** (composer rejects elsewhere — see `web/composer/tools.py:1483` `_WEB_ONLY_SOURCE_KEYS = frozenset({"blob_ref"})`) | `web/execution/service.py:410-486` (preflight UUID parse + ownership check + `link_blob_to_run` with `direction='input'`) | Bound at **path level**: the source plugin reads the blob's storage file the same way it would read any local file. The blob bytes themselves are not loaded into config. | `blob_run_links` row created at run start; lifecycle pinning prevents GC during run |

The blob form is structurally fused to source-plugin path binding. Three load-bearing sites encode that fusion:

- `web/composer/tools.py:1483` — `_WEB_ONLY_SOURCE_KEYS = frozenset({"blob_ref"})` filters `blob_ref` out of source pre-validation (because it's web-layer state, not engine config).
- `web/composer/tools.py:1567-1574` — `_execute_set_source` rejects `blob_ref` in `options` to force callers through `set_source_from_blob`, which canonicalises `path` to the blob's `storage_path`.
- `web/blobs/service.py:85-116` — `_source_references_blob` walks `composition_states.source.options` looking for a `blob_ref` (canonical) or matching `path`/`file` (fallback). The walker is hardcoded to source.

### 2.2 The blob service primitives this spec relies on

- `BlobServiceImpl.read_blob_content(blob_id) -> bytes` (`web/blobs/service.py:515-566`) — async; lifecycle guard (`status == 'ready'`); integrity verification (re-hash bytes, compare via `hmac.compare_digest`); raises `BlobIntegrityError` / `BlobContentMissingError` / `BlobStateError` / `BlobNotFoundError`.
- `BlobServiceImpl.get_blob(blob_id) -> BlobRecord` (`web/blobs/service.py:421-432`) — async; metadata only (no bytes, no integrity verification beyond the `_row_to_record` Tier-1 read guards). Returns `BlobRecord` with `content_hash`, `size_bytes`, `mime_type`, `status`.
- `BlobServiceImpl.link_blob_to_run(blob_id, run_id, direction)` (`web/blobs/service.py:568-594`) — async; cross-session-ownership guard via `_assert_blob_run_same_session`; inserts into `blob_run_links` with `direction in {'input', 'output'}` (`web/blobs/protocol.py:39`).
- `content_hash(data: bytes) -> str` (`web/blobs/service.py:51-63`) — SHA-256 hex, 64 lowercase chars, **the canonical hash form** validated by `_validate_finalize_hash` and used at every Tier-1 site.

### 2.3 The secret-ref resolver pattern this spec mirrors

`core/secrets.py` is the architectural precedent. The shape:

- Pure-sync tree-walk (`_walk` at `core/secrets.py:118-159`) over a `deepcopy` of the config dict.
- Recognition by mapping shape (`{"secret_ref": "NAME"}`, exactly one key) — see `_is_secret_ref` at `core/secrets.py:91-97`.
- Single error class (`SecretResolutionError(missing: list[str])` at `core/secrets.py:53-59`) — batched: every missing ref is collected, raised once at end of walk.
- Returns `(resolved_config, list[ResolvedSecret])` where `ResolvedSecret` (`contracts/secrets.py:126-147`) carries `(name, value, scope, fingerprint)` for downstream audit.

The runtime call site is `web/execution/service.py:711-733` (sync, in a thread worker). The validate site is `web/execution/validation.py:331-369` (sync, in a `_call_async`-fed thread worker).

### 2.4 The validation parity precedent

`web/composer/tools.py:1448-1480` strips secret-ref markers from the merged config before Pydantic validation, then filters Pydantic's "field required" errors for the stripped fields, on the rationale that a `secret_ref`'d field IS provisioned (the user wired it via `wire_secret_ref`), just deferred to execution time. This is the pattern Phase 1.1 (issue `elspeth-72d1dccd44`, commit `3b7ca22b`) established and that Shape 1 of the agreement suite (`tests/integration/pipeline/test_composer_runtime_agreement.py:2153-2333`) pins.

The widened-`blob_ref` design uses the same strip-before-validate pattern with a separate marker-recognition function so that the secret-ref strip remains untouched.

---

## 3. The Reframe — Why Widen `blob_ref`, Not Add a Sibling

### 3.1 Panel verdict (2026-05-03)

The original epic proposed `{blob_content_ref: BLOB_ID}` as a third sibling ref form alongside `{secret_ref: ...}` and `{blob_ref: ...}`. Five reviewers (leverage-analyst, solution-design-reviewer, python-code-reviewer, quality-assurance-analyst, architecture-critic) concurred on rejecting that shape and reframing as **widened `blob_ref` with mode discriminator + sha256 pin**:

```yaml
# Source (path-binding semantics — unchanged)
sources:
  - plugin: csv
    options:
      blob_ref: 7c3a-...
      mode: bind_source         # default for sources; canonicalises path to storage_path
      path: <set by set_source_from_blob>

# Transform / sink / gate (NEW — content-binding semantics)
transforms:
  - name: classify
    plugin: llm
    options:
      system_prompt:
        blob_ref: 7c3a-...-e91f
        mode: inline_content
        sha256: <pinned at submit>     # CRITICAL — closes auditability gap
        encoding: utf-8                # optional; default utf-8; closed set
      api_key: { secret_ref: OPENROUTER_KEY }
```

### 3.2 Why widen, not add a sibling

The architecture-critic finding (H1, ranked CRITICAL in their assessment) is the load-bearing argument. Adding `blob_content_ref` as a sibling would produce two parallel systems sharing only the blob *table* — the blob system was originally built as "named external content with run-link audit," then prematurely narrowed to source binding. Adding a sibling re-enacts the same narrowing at twice the surface area.

Both architecture-critic and leverage-analyst diagnosed the systems-thinking archetype: **Shifting the Burden**. Each new ref form (`secret_ref`, `blob_ref`, would-be `blob_content_ref`) treats a symptom while the underlying defect — config has no first-class late-binding model — is never named. Widening is the leverage-analyst's "Level 6" intervention (information flows / unified registry) at the smallest possible surface.

The closure rule, captured in the ADR (P1), is: **no new ref forms without ADR amendment**. Future late-binding needs (env, upstream-node-output, plugin-instance refs) widen the existing model or are rejected.

### 3.3 Why NOT extend the secrets system

Rejected as category error. Secrets discipline (fingerprinting, redaction, audit isolation, never-echo-on-error) exists *because* values are sensitive. Selectively undoing it for "non-sensitive named values" produces operator-hostile surprises later: "wait, why did our public regex pattern get redacted in the audit?" All five reviewers concurred. The blob system already has the right primitives for this work — content addressing, lifecycle, run-link audit — without the sensitivity discipline that doesn't apply.

---

## 4. Architecture Decision Record

The following decision rows are the design closures the spec commits to. The ADR (P1) re-asserts these with full rationale, rejected alternatives, and dataset-grounded numbers for the M1 caps.

| Decision | Choice | Rationale | Source |
|---|---|---|---|
| Marker shape | `{blob_ref: ID, mode: <bind_source\|inline_content>, sha256: HASH?, encoding: ENC?}` (mode discriminator + optional hash + optional encoding) | Identical recognition shape to `{secret_ref: NAME}` (single mapping key recognises the marker); discriminated unions add JSON-schema noise and break symmetry | S3 |
| Hash pinning | `sha256` field is **required when `mode: inline_content`**, **forbidden when `mode: bind_source`** (the latter binds a path, not content; the source plugin re-reads bytes per row, integrity verified by the existing `read_blob_content` integrity guard) | C1 — closes audit-fraud window between composer pinning and runtime resolution | C1 |
| Composer pins hash | At submit time, the composer reads the blob (`BlobServiceImpl.get_blob` for metadata; `read_blob_content` only when LLM-emitted refs lack a hash, which the composer auto-fills from `BlobRecord.content_hash` rather than re-hashing bytes) and writes the hash into the marker before persisting `composition_states` | Composer has the read access. Auto-fill from `BlobRecord.content_hash` avoids byte transfers on the composer→runtime hop. | C1, H4 |
| Hash mismatch | Runtime fails closed with `BlobIntegrityError` (existing exception, `web/blobs/protocol.py:228-248`); audit row is **not** written; `composer.blob_inline.hash_mismatch_total` increments | A run cannot proceed against bytes whose hash disagrees with what the composer pinned | C1 |
| Encoding | Strict UTF-8 by default + explicit `encoding:` field as escape hatch (closed set: `utf-8`, `utf-8-sig`, `utf-16`, `latin-1`). Resolution decode failure → `BlobContentResolutionError(undecodable=...)`. | M2 reconciliation: strict default + explicit override = discipline is explicit, not silent | M2 |
| Resolver shape | Sibling resolver, NOT unified walk. Three-function split: `_discover_blob_content_refs` (sync) + `async _fetch_blob_contents` (async, gathered via `asyncio.gather`) + `_substitute_blob_content_refs` (sync) | H3 — closes async/sync boundary defect; different sync/async profile + different audit shape from secrets | S2, H3 |
| Resolver location | New module `core/blobs_inline.py` at L1 (sibling to `core/secrets.py`); imports L0 only | Matches secret-resolver layout; layer-import enforcer permits L1 imports | S2 |
| L0 contract | New module `contracts/blobs_inline.py` carrying `ResolvedBlobContent` frozen dataclass; **`AllowedMimeType` and the closed encoding set move from L3 (`web/blobs/protocol.py:40`) to L0** per CLAUDE.md cross-layer resolution rule (move down before extracting) | S1 — L0 is the only layer L1 may import from | S1 |
| Validation parity | `web/composer/tools.py::_prevalidate_plugin_options` extends its strip-before-validate set to recognise the widened-`blob_ref` marker on **any** plugin field (transform/sink/gate/aggregation), with a separate `blob_inline_ref_keys` set so the secret-ref strip remains untouched | Mirrors Phase 1.1 / Shape 1 pattern; isolates the new logic | H5 |
| Runtime preflight | Resolver wired into `_run_pipeline` immediately **after** `resolve_secret_refs` (`web/execution/service.py:711-733`); audit row written **before** bytes flow into plugin instantiation; lifecycle-pinning `link_blob_to_run` calls fire **before** the resolver dereferences the IDs | Audit-primacy: every Tier-1 record exists before the data it describes is used | C1, S5 |
| Lifecycle pinning | Reuse `direction='input'` in `blob_run_links`. Distinguish source-data reads from config-content reads via `field_path` in the new `blob_inline_resolutions` audit table, not via link direction. | Avoids CHECK-constraint migration. Verification gate at the head of P3 confirms no existing query filters `blob_run_links` by direction in a load-bearing way (§7.3). | S5 |
| Audit row | New table `blob_inline_resolutions` with composite PK `(run_id, field_path, blob_id, attempt)`. Schema: `(run_id, attempt, field_path, blob_id, content_hash, byte_length, mime_type, encoding, resolved_at)`. Same-run retries (e.g., resume) increment `attempt`. | Audit shape distinct from `blob_run_links` (lifecycle) and `secret_resolutions` (sensitivity); attempt key handles `elspeth resume` semantics | S5, H5 |
| Tier-1 escape hatch | `BlobIntegrityError`, `BlobContentMissingError`, `AuditIntegrityError` propagate immediately and uncaught from the resolver. They do **not** join the batched `BlobContentResolutionError`. | These are Tier-1 anomalies — silent suppression here is the audit-fraud pattern | S4 |
| Composer LLM visibility | The composer surfaces only `{blob_id, mime_type, size_bytes, content_hash, filename}` to the LLM. Bytes are opaque. Composer **refuses** any LLM-emitted ref that carries inline content, raw bytes, or a manually-typed `sha256` that disagrees with `BlobRecord.content_hash`. | H4 — defines the trust boundary as ADR-grade; debugging story is "the LLM saw N bytes hashed X, the runtime resolved bytes hashed X" | H4 |
| `set_source` rejection scope | `set_source` continues to reject `mode: bind_source` (the path-canonicalising semantics that `set_source_from_blob` enforces), but **accepts `mode: inline_content`** in any `options` field (no `path` constraint exists for content-binding mode). | The cross-cutting recognition of the widened marker happens in the pre-validation strip; `set_source` only owns the source-specific path constraint, which doesn't apply to inline-content. | Advisor reframe |
| Frontend recovery surface | Filed as a separate follow-up issue (TBD ID) explicitly **outside** this epic. RC5 ships with API-grade lineage; SecretsPanel-equivalent UI for blob refs ships when dashboard polish is independently scheduled. | Per advisor reframe — "maybe out of scope" is the operator-hostile shape; commit one way | Advisor reframe |
| `elspeth resume` semantics | On resume, the resolver re-fetches by ID, re-verifies the pinned hash. Hash mismatch → fail closed (same as first run). Audit row records `attempt` ≥ 2 in `blob_inline_resolutions`. | Same fail-closed posture as first run; attempt counter makes resume visible in audit | H5 |

---

## 5. Wire Shape — Authoritative Grammar

### 5.1 The widened marker

```yaml
# Marker grammar (informal):
#   {blob_ref: <UUID>, mode: bind_source, path: <storage_path>}   # source-only; path canonicalised by set_source_from_blob
#   {blob_ref: <UUID>, mode: inline_content, sha256: <64-hex>}   # any plugin option field; default encoding=utf-8
#   {blob_ref: <UUID>, mode: inline_content, sha256: <64-hex>, encoding: <enc>}  # explicit encoding override
```

**Authoritative recognition rule:** A mapping is the widened `blob_ref` marker iff:

1. It contains the `blob_ref` key.
2. `blob_ref` is a string parseable as `UUID`.
3. The `mode` key is **present and required** (`mode in {'bind_source', 'inline_content'}`). A mapping with a `blob_ref` key but no `mode` key is **not** the widened marker; it is a malformed config that the resolver rejects with a structural error at YAML load. Per CLAUDE.md no-legacy-code policy and the project's DB-migration policy (operator deletes the dev/staging DB), there is no mode-less compat shim — P3 also updates `set_source_from_blob` to emit `mode: bind_source` explicitly so every persisted marker is round-trippable through the recognition function.
4. Every other key is in the closed set `{mode, path, sha256, encoding}`.
5. If `mode == 'bind_source'`: `path` MAY be present, `sha256` MUST NOT be present, `encoding` MUST NOT be present.
6. If `mode == 'inline_content'`: `sha256` MUST be present (64-char lowercase hex), `path` MUST NOT be present, `encoding` is optional (closed set).

The recognition function is `is_widened_blob_ref(value: Any) -> WidenedBlobRefShape | None` in `contracts/blobs_inline.py` (P2). Returns a structured shape with the parsed mode rather than `None|str` so callers branch on mode, not on stringly-typed checks.

### 5.2 Encoding closed set

```python
# contracts/blobs_inline.py
ContentEncoding = Literal["utf-8", "utf-8-sig", "utf-16", "latin-1"]
ALLOWED_CONTENT_ENCODINGS: frozenset[str] = frozenset(get_args(ContentEncoding))
```

The set is closed at the L0 layer. Adding a member is a single-site edit (the get_args derivation enforces lockstep — same anti-drift mechanism `web/blobs/protocol.py` uses for `BlobStatus`, `BlobRunLinkDirection`, etc.). The default is `utf-8`.

### 5.3 Authorship paths

| Surface | API | What it does |
|---|---|---|
| Composer LLM discovery | New MCP tool `list_composer_blobs(session_id) -> list[BlobInlineDescriptor]` (P5) where `BlobInlineDescriptor = {blob_id: UUID, mime_type: AllowedMimeType, size_bytes: int, content_hash: str, filename: str}` | The LLM cannot author a ref for a blob it does not know about. This tool is the load-bearing site for the H4 trust boundary — it is the one place the LLM learns blob identifiers, and its return shape is **exactly** the H4 visibility shape (no bytes, no preview, no source description text that could leak intent). |
| Composer LLM authorship | New MCP tool `wire_blob_inline_ref(field_path: str, blob_id: UUID, encoding: str = 'utf-8') -> ToolResult` (P5) | LLM emits the call with a `blob_id` obtained from `list_composer_blobs`; composer reads `BlobRecord.content_hash` for the blob; composer constructs the marker with the pinned hash and writes it into `composition_states` at `field_path`. |
| Composer dashboard UI | Filed as follow-up issue — outside this epic | SecretsPanel-equivalent renders existing markers with mode + hash + size; click-through to upload/swap. |
| Direct YAML | Operator hand-edits a YAML file with the widened marker; `elspeth validate --settings ...` resolves and verifies | Same resolver wiring; same audit row; same fail-closed semantics. |

---

## 6. Resolver Shape — Three-Function Split

### 6.1 Sync/async boundary

The async/sync defect (H3) is the load-bearing argument for the three-function split. `BlobServiceImpl.read_blob_content` is async; both resolver call sites (`_run_pipeline` at `web/execution/service.py:636` and `validate_pipeline` at `web/execution/validation.py:208`) are sync `def` functions running in worker threads. Calling `asyncio.run()` from a worker thread that already has a loop raises `RuntimeError`.

The fix: split the resolver into three functions with disjoint sync/async profiles, then call the async middle function via the existing `_call_async` bridge (`web/execution/service.py:245-264`).

### 6.2 The three functions

```python
# core/blobs_inline.py (L1 — imports L0 only)

def _discover_blob_content_refs(config: dict[str, Any]) -> list[BlobInlineRef]:
    """Pure-sync tree walk. Returns a list of (field_path, blob_id, sha256, encoding)
    tuples — no I/O. Tree-walks the same way `_walk` in `core/secrets.py:118-159`
    does, but recognises the widened `blob_ref` marker via `is_widened_blob_ref`
    from `contracts/blobs_inline.py` and only collects the `inline_content` mode.

    The `field_path` for each ref is canonical (identity-anchored, see §8.1):
      - source.options.<key>[.<sub-key>...]
      - node:<node_id>.options.<key>[.<sub-key>...]
      - output:<sink_name>.options.<key>[.<sub-key>...]
    The discoverer enumerates `state.source.options`, `[node.options for node
    in state.nodes]` (covering transforms/gates/aggregations/coalesce, since
    `CompositionState.nodes` mixes all four — see `web/composer/state.py:1348`
    for the class and `state.py:1364` for the `nodes:` field),
    and `[output.options for output in state.outputs]`. The runtime path
    operates on the YAML-serialised dict; the equivalent canonical paths are
    derived during walk by tracking the structural ancestor identifier on
    descent (mirrors how `web/execution/validation.py:331-369` enumerates the
    same surface for secret refs).
    """

async def _fetch_blob_contents(
    blob_service: BlobServiceProtocol,
    refs: list[BlobInlineRef],
) -> dict[BlobInlineRef, bytes]:
    """Async fetch. Uses `asyncio.gather` (or `TaskGroup` on 3.11+) to issue
    `read_blob_content` calls in parallel. Raises `BlobIntegrityError`,
    `BlobContentMissingError`, `BlobNotFoundError`, `BlobStateError`
    immediately and uncaught (Tier-1 escape hatch — these never join the
    batched error). Returns a dict keyed by ref for the sync-side substituter.
    """

def _substitute_blob_content_refs(
    config: dict[str, Any],
    fetched: dict[BlobInlineRef, bytes],
    *,
    refs: list[BlobInlineRef],
) -> tuple[dict[str, Any], list[ResolvedBlobContent]]:
    """Pure-sync. Re-walks the config, substitutes each marker with the
    decoded string per the ref's encoding, and emits a list of
    `ResolvedBlobContent` records for audit. Hash verification happens
    here — the `sha256` from the marker is compared against the
    composer-pinned hash (which equals `BlobRecord.content_hash` if
    composer authored the marker correctly). Any mismatch raises
    `BlobIntegrityError` (Tier-1 escape, NOT batched).

    Decode errors collect into `BlobContentResolutionError(undecodable=...)`.
    """
```

### 6.3 Composition site

```python
# web/execution/service.py::_run_pipeline (replaces / extends line ~733)

# Resolve secret refs first (existing — unchanged)
resolved_dict, secret_resolutions = resolve_secret_refs(
    config_dict, self._secret_service, user_id, env_ref_names=env_ref_names,
)

# Discover content refs (sync) — uses the same dict the secret resolver
# returned (deepcopy semantics already applied)
inline_refs = _discover_blob_content_refs(resolved_dict)

# Lifecycle pinning — link every UNIQUE ref'd blob to this run BEFORE
# fetching bytes. Audit row exists before the resolver mutates config.
#
# Dedupe by blob_id: blob_run_links has UNIQUE(blob_id, run_id, direction)
# (see web/sessions/models.py:240, uq_blob_run_link). A single config can
# reference one blob from multiple field_paths (e.g. the same prompt
# template inlined into two transforms). Per-field_path lifecycle is
# captured in blob_inline_resolutions; blob_run_links is the per-blob
# retention pin and tolerates exactly one row per (blob, run, direction).
#
# A single _call_async(asyncio.gather(...)) covers all link inserts in one
# worker→loop→worker round-trip; a per-ref loop would burn N round-trips
# for no gain.
unique_blob_ids = sorted({ref.blob_id for ref in inline_refs})

# `_call_async(coro, ...)` invokes `asyncio.run_coroutine_threadsafe` with
# its first argument; `asyncio.gather(...)` returns a Future, not a
# coroutine — wrap each batched I/O in a thin `async def` helper so the
# value passed to `_call_async` is a coroutine.
async def _link_all() -> None:
    await asyncio.gather(*[
        self._blob_service.link_blob_to_run(
            blob_id=blob_id, run_id=run_uuid, direction="input",
        ) for blob_id in unique_blob_ids
    ])
self._call_async(_link_all())

# Fetch bytes (async). `_fetch_blob_contents` is itself an `async def`
# (returns a coroutine), so it can be passed directly to `_call_async`.
fetched = self._call_async(_fetch_blob_contents(self._blob_service, inline_refs))

# Substitute (sync) — verifies hash, decodes, raises on Tier-1 anomaly
resolved_dict, blob_resolutions = _substitute_blob_content_refs(
    resolved_dict, fetched, refs=inline_refs,
)

# Write Tier-1 audit rows BEFORE plugin instantiation reads any of these bytes
self._call_async(self._session_service.record_blob_inline_resolutions(
    run_id=run_uuid, resolutions=blob_resolutions,
))

# … existing settings load + plugin instantiation continues here …
```

### 6.4 Validate path divergence

`validate_pipeline` runs without a `run_id` — there is no run row to link to. The validate-side helper:

```python
# core/blobs_inline.py

def _validate_blob_content_refs(
    config: dict[str, Any],
    blob_service: BlobServiceProtocol,
    user_id: str,
) -> list[BlobInlineValidationViolation]:
    """Validate-side parity. Discovers refs, fetches metadata only
    (`get_blob`, NOT `read_blob_content`), and checks:
      - blob exists and `status == 'ready'`
      - blob `content_hash` matches the marker's `sha256`
      - blob `size_bytes` is within the per-ref + aggregate caps
      - the marker's `mime_type` (derived from `BlobRecord.mime_type`) is
        compatible with the encoding (validation is metadata-only — the
        actual decode happens at runtime)
    Returns a list of violations; never raises (the caller assembles them
    into the ValidationResult shape consumed by /validate).
    """
```

The validate path **does not** create `blob_run_links` rows (no run exists), **does not** write to `blob_inline_resolutions` (no run exists), and **does not** read blob bytes. Composer-pinned hash equality is checked against `BlobRecord.content_hash` (metadata), which is sufficient because the runtime path runs the same equality check against the actual bytes via `BlobServiceImpl.read_blob_content`'s integrity guard.

### 6.5 Error taxonomy

```python
# contracts/blobs_inline.py

class BlobContentResolutionError(Exception):
    """Batched error mirroring SecretResolutionError. Collects every
    operationally recoverable failure across the tree-walk so the operator
    sees them in one shot.

    Tier-1 anomalies (BlobIntegrityError, BlobContentMissingError,
    AuditIntegrityError) are NOT included — they propagate immediately
    and uncaught.
    """
    def __init__(
        self,
        *,
        missing: list[str],                                   # field_path -> blob_id not in DB
        oversized: list[tuple[str, int, int]],                # (field_path, actual, cap)
        undecodable: list[tuple[str, str]],                   # (field_path, encoding)
        not_ready: list[tuple[str, str]],                     # (field_path, status) — status != "ready"
        cross_session: list[str],                             # field_path -> blob owned by different session
        malformed: list[tuple[str, str]],                     # (field_path, reason) — has `blob_ref` key but missing `mode` or other shape violation per §5.1
    ) -> None: ...
```

The error class lives in L0 (`contracts/blobs_inline.py`) so the resolver and the validate-path helper both raise/return values built from it without crossing layer boundaries.

**HTTP status mapping** (matches the inline `web/blobs/protocol.py` convention where each error type documents its expected status):

| Exception | HTTP | Surfaced as |
|---|---|---|
| `BlobContentResolutionError` (batched: missing / oversized / undecodable / not_ready / cross_session) | 422 | Structured `ValidationResult(is_valid=False)` with one `ValidationError` per case in the batch |
| `BlobIntegrityError` (Tier-1 escape) | 500 | Uncaught; existing exception handler chain |
| `BlobContentMissingError` (Tier-1 escape) | 500 | Uncaught; existing exception handler chain |
| `BlobNotFoundError` raised post-validate (the validate-side helper would have caught a missing blob; runtime mismatch indicates the blob was deleted between submit and run) | 500 | Uncaught — runtime-only inconsistency, not a validation-recoverable case |
| `AuditIntegrityError` from `record_blob_inline_resolutions` | 500 | Uncaught — Tier-1 audit-write failure |

---

## 7. Validation Parity — Strip-Before-Validate

### 7.1 The pattern

`web/composer/tools.py:1448-1480` already implements strip-before-validate for `secret_ref`. The widened `blob_ref` extends the same site:

```python
# Existing block (unchanged):
secret_ref_keys: set[str] = set()
for key, value in list(merged.items()):
    if isinstance(value, Mapping) and len(value) == 1 and "secret_ref" in value and isinstance(value["secret_ref"], str):
        secret_ref_keys.add(key)
        del merged[key]

# NEW block (P4):
blob_inline_ref_keys: set[str] = set()
for key, value in list(merged.items()):
    shape = is_widened_blob_ref(value)
    if shape is not None and shape.mode == "inline_content":
        blob_inline_ref_keys.add(key)
        del merged[key]
```

The Pydantic-error filter at lines 1474-1479 extends to drop errors whose `loc[0]` is in `secret_ref_keys | blob_inline_ref_keys`. Two separate sets so the filter logic stays auditable; no shared state with the secret-ref path.

### 7.2 Composer rejection rules

The composer enforces these at marker-construction time (the new MCP tool `wire_blob_inline_ref` in P5 is the only sanctioned authorship path; direct YAML edits go through `validate_pipeline`):

1. **No LLM-emitted bytes.** `wire_blob_inline_ref` accepts `(field_path, blob_id, encoding)` — never raw content. The hash is auto-filled from `BlobRecord.content_hash`. (H4)
2. **No LLM-disagreeing hash.** If the LLM hand-types a `sha256`, the composer compares against `BlobRecord.content_hash` and rejects on mismatch. (H4)
3. **No `mode: bind_source` in non-source fields.** `wire_blob_inline_ref` always emits `mode: inline_content`. The set_source path continues through `set_source_from_blob`.
4. **Blob must be `ready`.** `BlobRecord.status == 'ready'` checked at marker-construction time. Pending/error blobs are rejected with a structured error.
5. **MIME-encoding compatibility.** `BlobRecord.mime_type` must be compatible with the requested encoding (binary MIME types reject text encoding). This is a structural reject, not a coercion attempt.

### 7.3 Direction reuse + unique-constraint verification gate (head of P3)

The S5 reuse-of-`direction='input'` claim depends on:

1. No existing query filtering `blob_run_links.direction` in a load-bearing way that segregates source-data links from config-content links.
2. The unique constraint `UniqueConstraint("blob_id", "run_id", "direction")` (`web/sessions/models.py:240`, `uq_blob_run_link`) tolerating the dedupe-by-blob-id strategy in §6.3 — i.e., exactly one row per `(blob, run, direction='input')` regardless of how many `field_path`s reference that blob.

The verification step at the head of P3 is two checks:

```bash
# Direction-segregation check — fails P3 if any non-test query filters
# direction in a way that segregates source-data links from config-content
# links.
grep -nR --include='*.py' "blob_run_links\b" src/elspeth/ | grep -E '\.direction|direction\b'

# Unique-constraint check — confirms uq_blob_run_link covers (blob_id,
# run_id, direction). The §6.3 dedupe strategy assumes the unique
# constraint is exactly this shape; a constraint that adds more columns
# would let duplicate (blob, run, input) rows slip in.
grep -n "uq_blob_run_link\|UniqueConstraint.*blob_run_links" src/elspeth/web/sessions/models.py
```

Expected findings:

- **Direction-segregation:** `_assert_blob_run_same_session`, `link_blob_to_run` (write-side), `finalize_run_output_blobs` (filters `direction == 'output'`, **doesn't conflict — input rows don't appear in output finalization**), `_row_to_link_record` (read-side guard). No segregation that excludes config-content reads.
- **Unique constraint:** `UniqueConstraint("blob_id", "run_id", "direction", name="uq_blob_run_link")` at `web/sessions/models.py:240` is the exact shape the dedupe strategy assumes.

If a new query is found that segregates by direction in a way that would make config-content reads invisible to existing source-data tooling, S5 collapses and P3 introduces a new column (`link_kind` or similar) instead of reusing direction. If the unique constraint shape diverges from `(blob_id, run_id, direction)`, the dedupe strategy is invalid and P3 either tightens the constraint or moves to a SAVEPOINT-and-IGNORE-on-conflict pattern at the link site. The verification steps are checked off **before any P3 implementation begins**.

---

## 8. Lifecycle Pinning + Audit Row

### 8.1 Schema

```sql
-- New table (P3)
CREATE TABLE blob_inline_resolutions (
    run_id           TEXT NOT NULL,
    attempt          INTEGER NOT NULL DEFAULT 1,
    field_path       TEXT NOT NULL,         -- canonical, identity-anchored (see field_path format below)
    blob_id          TEXT NOT NULL,
    content_hash     TEXT NOT NULL,         -- 64-char lowercase SHA-256 hex (matches blobs.content_hash form)
    byte_length      INTEGER NOT NULL,
    mime_type        TEXT NOT NULL,
    encoding         TEXT NOT NULL,
    resolved_at      TIMESTAMP NOT NULL,
    PRIMARY KEY (run_id, field_path, blob_id, attempt),
    FOREIGN KEY (run_id) REFERENCES runs (id) ON DELETE CASCADE,
    FOREIGN KEY (blob_id) REFERENCES blobs (id),
    CONSTRAINT ck_blob_inline_resolutions_hash_format
        CHECK (length(content_hash) = 64),
    CONSTRAINT ck_blob_inline_resolutions_encoding
        CHECK (encoding IN ('utf-8', 'utf-8-sig', 'utf-16', 'latin-1')),
    CONSTRAINT ck_blob_inline_resolutions_field_path
        -- Identity-anchored prefixes only. Disallows positional list-index
        -- paths (transforms[2].options.X) which become wrong on insert/delete
        -- of upstream nodes — defeats the purpose of an audit row that must
        -- survive composer state mutation across resume.
        CHECK (
            field_path LIKE 'source.options.%'
            OR field_path LIKE 'node:%.options.%'
            OR field_path LIKE 'output:%.options.%'
        )
);

CREATE INDEX ix_blob_inline_resolutions_blob_id ON blob_inline_resolutions(blob_id);
```

**`field_path` canonical format.** The audit row exists to support cross-resume querying ("did this run resolve the same blob into the same field that the prior attempt did?"); list-index encodings (`transforms[2].options.system_prompt`) become wrong the moment a new transform is inserted at an earlier position. Per ADR-005 (declarative DAG wiring), node IDs and sink names are stable identifiers — the canonical encoding anchors to those:

| Surface | Canonical prefix | Example |
|---|---|---|
| Source | `source.options.<key>[.<sub>...]` | `source.options.system_prompt` |
| Transform/gate/aggregation/coalesce (any node in `state.nodes`) | `node:<node_id>.options.<key>[.<sub>...]` | `node:classify_with_llm.options.system_prompt` |
| Sink | `output:<sink_name>.options.<key>[.<sub>...]` | `output:writeback.options.body_template` |

The discoverer in §6.2 emits these forms. The CHECK constraint enforces the prefix at the DB layer; the L0 contract carries a regex validator on `BlobInlineRef.field_path` so violations crash at construction (offensive programming).

**No Alembic migration.** Per the project's `db_migration_policy` (operator deletes and recreates the dev/staging DB — see CLAUDE.md and the "DB migration = delete the old DB" memory), the new table is added to the SQLAlchemy schema definitions in `web/sessions/models.py` and the operator drops the dev DB during P3 deployment. Pre-RC5 there are no production deployments to migrate.

### 8.2 Audit row write site

In `_run_pipeline` (after `_substitute_blob_content_refs` returns):

```python
# Audit primacy: write the Tier-1 record BEFORE plugin instantiation reads
# any resolved bytes. If the audit write fails, the run fails — bytes never
# flow into the engine.
self._call_async(self._session_service.record_blob_inline_resolutions(
    run_id=run_uuid,
    resolutions=blob_resolutions,
    attempt=run_attempt,
))
```

`record_blob_inline_resolutions` is a new method on `SessionsService` that writes the rows in a single transaction, raising `AuditIntegrityError` on any DB-level failure. Tier-1 escape: any failure here propagates uncaught.

### 8.2.1 Pre-link gap closure (extending `_source_references_blob`)

`BlobServiceImpl.delete_blob` carries two active-run guards (`web/blobs/service.py:467-502`): the explicit `blob_run_links` join (covers blobs already linked to an active run) and the pre-link window walker `_source_references_blob` (covers the gap between `create_run` and `link_blob_to_run`, walking `composition_states.source.options` for `blob_ref`/`path`/`file` matches).

The pre-link walker is hardcoded to source.options. Without extending it, a blob referenced ONLY from a transform/sink option (no source binding) can be deleted in the window between `create_run` and the new resolver's `link_blob_to_run` calls in `_run_pipeline` — exactly the silent-data-loss pattern the existing source-only walker was added to close.

**Resolution (P3, Task 5b):** Replace `_source_references_blob` with a wider `_composition_references_blob` that walks source.options, transforms/gates/aggregations/coalesce[].options, and outputs.<sink>.options for any widened-blob_ref marker (any mode) carrying the matching `blob_id`. Existing `path`/`file` legacy match preserved for set_source-without-set_source_from_blob compatibility (no behavioural change for the existing source-only path). Tier-1 read guards on `composition_states` shape preserved.

The walker semantically subsumes the existing source-only walker, so the call site at `web/blobs/service.py:501` updates to pass the full `composition_states` row instead of `composition_states.source` alone.

**Adapter (P2b prerequisite):** the walker is YAML-form-shaped (it traverses `transforms`, `gates`, `aggregations`, `coalesce`, `outputs.<sink>` keys), but `composition_states` rows are persisted with a flat `nodes` JSON column — there is no `transforms` / `gates` / `aggregations` / `coalesce` key on a raw DB row. P2b (`2026-05-03-config-content-ref-phase-2b-state-adapter.md`) extracts `generate_pipeline_dict(state) -> dict` from `yaml_generator.generate_yaml`; the P3 walker consumes `generate_pipeline_dict(state_from_record(record))` so DB-form callers obtain the YAML-form dict via a single canonical adapter. Closing P2b retires `elspeth-be405bac87` and removes the per-implementer DB-vs-YAML-shape caveat that earlier drafts of this plan carried.

### 8.3 Resume semantics

`elspeth resume <run_id>` re-runs the resolver. The same `field_path` + `blob_id` keys produce a new row with `attempt = max(prior attempts) + 1`. The audit trail then carries the full attempt history for that run, queryable as:

```sql
SELECT * FROM blob_inline_resolutions
WHERE run_id = ? AND field_path = ? AND blob_id = ?
ORDER BY attempt;
```

If between attempts the underlying blob's bytes changed (replacement, GC-and-recreate), the new attempt's hash will differ from the marker's pinned `sha256` and the run fails closed via `BlobIntegrityError`. The audit row is **not** written when the hash check fails — the failed-run record exists in `runs.error`, and the lifecycle-pinning `blob_run_links` row already created at attempt start carries the link.

---

## 9. Test Plan + Shape 9

### 9.1 Test surfaces

| Surface | Location | Purpose |
|---|---|---|
| Unit | `tests/unit/contracts/test_blobs_inline.py` | `is_widened_blob_ref` recognition; `ResolvedBlobContent` validators; `BlobContentResolutionError` shape |
| Unit | `tests/unit/core/test_blobs_inline.py` | Tree-walk parity with `_walk` from `core/secrets.py`; three-function split (sync→async→sync); decode failure paths |
| Unit | `tests/unit/web/blobs/test_inline_validation.py` | `_validate_blob_content_refs` returns structured violations; never raises on operational errors; raises only on Tier-1 anomalies |
| Unit | `tests/unit/web/composer/test_wire_blob_inline_ref.py` | Composer rejection rules (LLM-emitted bytes, hash disagreement, non-ready blob, MIME-encoding mismatch); auto-fill from `BlobRecord.content_hash` |
| Unit | `tests/unit/web/execution/test_blob_inline_audit.py` | Audit-row primacy: bytes never reach plugin instantiation if audit write fails; resume `attempt` increments |
| Integration | `tests/integration/web/test_blob_inline_round_trip.py` | End-to-end: composer authors → /validate green → /run → audit row matches; hash determinism via the existing `BlobServiceImpl.read_blob_content` integrity guard |
| Integration | `tests/integration/web/test_blob_inline_lifecycle_pinning.py` | A blob ref'd in a transform option cannot be GC'd while the run is `pending`/`running` (mirrors the existing source-data `BlobActiveRunError` semantics) |
| Integration | `tests/integration/web/test_blob_inline_preflight_latency.py` | NFR sanity bound: N=8 refs at the per-ref cap resolve within p95 ≤ 1.5s |
| Agreement | `tests/integration/pipeline/test_composer_runtime_agreement.py` (Shape 9, P4) | See §9.2 |

### 9.2 Shape 9 — agreement-suite registry entry

Shape 9 entry in the module docstring (added in P4):

```text
* Shape 9 — Phase 5 widened blob_ref / inline_content config-content-ref
  capability. Closes ``elspeth-fdebcaa79a``. Pinned by
  ``TestComposerRuntimeBlobInlineAgreement`` with three sub-pins:

  * Sub-pin A — composer ``/validate`` recognises the widened
    ``blob_ref`` marker on transform/sink/gate/aggregation options
    and returns ``BlobContentResolutionError`` cases as structured
    ``ValidationResult(is_valid=False)`` (NOT a 500). Bug verification:
    revert the ``blob_inline_ref_keys`` extension at
    ``web/composer/tools.py::_prevalidate_plugin_options``; observe
    Pydantic ``ValidationError`` propagating as composer_plugin_error.

  * Sub-pin B — runtime preflight raises
    ``BlobContentResolutionError`` / ``BlobIntegrityError`` /
    ``BlobNotFoundError`` BEFORE the first row reaches plugin
    instantiation. Bug verification: revert the resolver wiring in
    ``web/execution/service.py::_run_pipeline`` (the
    ``_call_async(_fetch_blob_contents(...))`` call); observe the
    pipeline crash on first-row plugin call with no audit record of
    the missing/oversized/mismatched ref.

  * Sub-pin C — audit row determinism. The ``content_hash`` recorded
    in ``blob_inline_resolutions`` for a run equals the hash
    re-derived from the blob's stored bytes via
    ``BlobServiceImpl.read_blob_content`` in the same run. Bug
    verification: revert the audit-write site in
    ``web/execution/service.py::_run_pipeline`` (the
    ``record_blob_inline_resolutions`` call); observe that an audit
    query of ``blob_inline_resolutions`` returns no row for a
    completed run that resolved a ref.
```

### 9.3 Bug-verification protocol per fix site

Three load-bearing fix sites, each with its own bug-verification at the same PR boundary (per the agreement-suite docstring's protocol at lines 78-90):

| Fix site | Phase | Production line reverted | Expected failure observed |
|---|---|---|---|
| Runtime resolver wiring | P3 | `web/execution/service.py::_run_pipeline` — the `_call_async(_fetch_blob_contents(...))` call | First-row plugin crash with no audit row in `blob_inline_resolutions` |
| Audit-write primacy | P3 | `web/execution/service.py::_run_pipeline` — the `record_blob_inline_resolutions` call | Run reports success but audit-table is empty for any resolved ref |
| Composer parity strip | P4 | `web/composer/tools.py::_prevalidate_plugin_options` — the `blob_inline_ref_keys` block (lines added in P4) | `/validate` returns `composer_plugin_error` 500 instead of structured `ValidationResult` |

Each fix-site PR carries the revert-and-observe documentation in its test docstring.

---

## 10. Phasing

### 10.1 Phase summary

| Phase | File | What it delivers | Risk profile |
|---|---|---|---|
| P1 | `2026-05-03-config-content-ref-phase-1-adr.md` | ADR-021 with VAL-data-grounded numeric caps; closure rule "no new ref forms without ADR amendment"; H4 LLM visibility model; M2 encoding decision; `direction='input'` reuse rationale | Markdown only |
| P2 | `2026-05-03-config-content-ref-phase-2-l0-l1.md` | `contracts/blobs_inline.py` (L0): `WidenedBlobRefShape`, `ResolvedBlobContent`, `BlobInlineRef`, `BlobContentResolutionError`, `ContentEncoding`, `is_widened_blob_ref`; relocate `AllowedMimeType` from `web/blobs/protocol.py` to L0; `core/blobs_inline.py` (L1): three-function resolver split + tree-walk parity tests | No behavior change; unit tests only |
| P2b | `2026-05-03-config-content-ref-phase-2b-state-adapter.md` | Extract `generate_pipeline_dict(state) -> dict` from `yaml_generator.generate_yaml` (`generate_yaml` becomes a one-line `yaml.dump` wrapper); round-trip identity property test (`yaml.safe_load(generate_yaml(state)) == generate_pipeline_dict(state)` AND `state == state_from_record(record_for(state))`); explicit YAML-shape snapshot pin; migrate `delete_blob`'s pre-link active-run guard at `web/blobs/service.py:467-513` to consume the adapter (`generate_pipeline_dict(state_from_record(record))`) instead of `composition_states.source` directly; bug-verification at the migration site (manual revert leaves the four existing `active_run` pinning tests passing — proof of shape-preserving migration); closes architectural gap `elspeth-be405bac87` | Shape-preserving refactor; unit tests only |
| P3 | `2026-05-03-config-content-ref-phase-3-runtime-preflight.md` | Direction-reuse verification gate; `blob_inline_resolutions` table + `record_blob_inline_resolutions` service method; resolver wired into `_run_pipeline` after `resolve_secret_refs`; lifecycle pinning extended to walk the full config tree at submit time (walker consumes `generate_pipeline_dict(state_from_record(record))` from P2b); audit-write primacy; Tier-1 escape paths verified; bug-verification at runtime-resolver-wiring + audit-write-primacy fix sites | Fail-closed; nothing emits inline_content refs yet |
| P4 | `2026-05-03-config-content-ref-phase-4-composer-parity.md` | `blob_inline_ref_keys` extension in `_prevalidate_plugin_options`; `_validate_blob_content_refs` wired into `validate_pipeline`; Shape 9 sub-pins (A/B/C) added to the agreement-suite registry; bug-verification at composer-parity-strip fix site | Composer authoring path opens; runtime path is already fail-closed |
| P5 | `2026-05-03-config-content-ref-phase-5-composer-tool.md` | New MCP tools `list_composer_blobs` (LLM-visibility H4 trust-boundary) + `wire_blob_inline_ref(field_path, blob_id, encoding)` (authorship); composer rejection rules from §7.2 enforced; chat-driven authoring story; **also updates `set_source_from_blob` to emit `mode: bind_source` explicitly** so every persisted marker round-trips through the recognition function (no mode-less markers in the codebase) | LLM has the affordance |
| (follow-up) | filed as Filigree issue at the close of P5 (citation in P5 PR description as "F-1") | SecretsPanel-equivalent UI: list inline refs with mode + hash + size; click-through to upload/swap; resolved-content-byte preview gated on operator role | Optional — not on the RC5 critical path |

The frontend recovery surface is **not** a phase plan in this epic. It is filed as a separate Filigree issue at P5 close (the F-1 follow-up named in `2026-05-03-config-content-ref-overview.md`). RC5 ships at the end of P5 with API-grade lineage; the SecretsPanel-equivalent UI ships when dashboard polish is independently scheduled.

### 10.2 Cross-phase dependencies

- P2 depends on no other phase (pure foundation).
- P2b depends on P2 (phase ordering; touches `yaml_generator` and `delete_blob` — neither depends on P2's L0/L1 modules but the merge train stays linear).
- P3 depends on P2b (consumes `generate_pipeline_dict` as the canonical input shape for the widened lifecycle walker; the per-implementer DB-vs-YAML-shape caveat that earlier drafts carried is retired by P2b's adapter).
- P4 depends on P3 (validation parity is meaningful only when the runtime side is fail-closed; otherwise composer-green / runtime-red is the very Shape 9 footgun this work closes).
- P5 depends on P4 (the MCP tool's authorship path goes through composer pre-validation, which only recognises the marker after P4).
- P6 depends on P5 (UI surfaces the same MCP-tool-authored state).

### 10.3 Done conditions across all phases

The whole work closes when:

1. All six in-scope phase PRs (P1, P2, P2b, P3, P4, P5) are merged. P6 is filed but not blocking.
2. CI is green on RC5-UX (or successor) including `enforce_tier_model.py check` and `enforce_freeze_guards.py`.
3. Shape 9 (sub-pins A, B, C) passes in `tests/integration/pipeline/test_composer_runtime_agreement.py`.
4. The lifecycle-pinning round-trip integration test passes (a referenced blob cannot be GC'd while the run is pending/running).
5. The hash-determinism property test passes (audit-row hash equals re-derived hash).
6. ADR-021 is committed, with VAL-data citations for every numeric cap in §1.4.
7. The OTel counter post-conditions hold across the property-test campaign (`composer.blob_inline.hash_mismatch_total == 0`, `composer.blob_inline.audit_row_tier1_violation_total == 0`).
8. P2b PR is merged, retiring `elspeth-be405bac87` (canonical `composition_state` adapter shipped as `generate_pipeline_dict`).

VAL — "the operator can actually inline an LLM system prompt and verify the audit trail" — is owned by an operator-acceptance ticket filed alongside P5 closure.

---

## 11. Risk-Ranked Gotcha Table

| Gotcha | Defect cost | Likelihood broken | Mitigation site |
|---|---|---|---|
| Hash pinning (C1) + lifecycle pinning (H5) | Catastrophic — audit fraud (hash recorded with no content) | High (subtle, easy to "wire retention later") | P3 — Tier-1 escape paths verified; audit-write primacy enforced; bug-verification at the audit-write site |
| Validation parity (H5) | High — Shape-1-class footgun, operator can't compose | Medium | P4 — Shape 9 sub-pin A; bug-verification at the composer-parity-strip fix site |
| Async boundary (H3) | High — 500 with opaque message | Low if three-function split documented | P2 — three-function split is the L1 module's organising principle; P3 verifies wiring |
| Audit fingerprinting (H5) | High — debugging dead-end | Medium | P3 — `(content_hash, field_path, byte_length, mime_type, encoding)` recorded BEFORE bytes reach plugin instantiation |
| Direction-reuse breakage (S5) | Medium — schema migration mid-epic | Low | P3 — verification gate at the head of P3; if any load-bearing query filters by direction, S5 collapses and P3 introduces a new column |
| Size cap (M1) | Medium — operator pain, bypass attempts | Medium | P1 — VAL-data-grounded numbers in the ADR |
| Encoding (M2) | Medium — operator blocked by non-UTF-8 content | Medium | P1 — strict UTF-8 + explicit `encoding:` escape hatch documented in ADR |
| LLM visibility leak (H4) | Medium — debugging horror story; LLM authors content it can see vs. content it cannot | Low | P5 — `list_composer_blobs` tool emits exactly the H4 visibility shape; `wire_blob_inline_ref` rejection rules enforced; ADR documents the trust boundary |
| `field_path` instability | Medium — audit rows for the same logical field diverge across resumes | Low | §8.1 — canonical identity-anchored format (source/node:/output:) enforced by CHECK constraint and L0 validator |
| `blob_run_links` unique-constraint violation | Medium — duplicate inserts when the same blob is referenced multiple times | Low | §6.3 dedupe-by-blob-id; §7.3 verification gate confirms `uq_blob_run_link` covers `(blob_id, run_id, direction)` |

---

## 12. Out of Scope

- Any mechanism for non-blob-backed late-binding (env, upstream node output, plugin instance refs). Closed by the ADR's late-binding rule.
- Any change to `secret_ref` resolver semantics. Secrets stay sensitive.
- Backwards compatibility for the original-proposal `{blob_content_ref: ...}` shape. Never shipped, no migration needed.
- General refactor of `core/secrets.py`. Sibling resolver, not unified walk.
- Composer-side preview rendering of resolved content bytes. Filed as part of the P6 follow-up if dashboard scoring is added.
- Per-row content swapping (different content per row of source data). Refs are config-tree, not row-tree; per-row content is the domain of source plugins reading blob storage paths (`mode: bind_source`), not config injection.

---

## 13. References

### 13.1 Project guidance

- `/home/john/elspeth/CLAUDE.md` — auditability standard, three-tier trust model, layer dependency rules, no-defensive-programming, no-legacy-code policy.

### 13.2 Reference implementations

- `src/elspeth/core/secrets.py` — resolver pattern (tree-walk + batched error + sync profile)
- `src/elspeth/contracts/secrets.py` — L0 contract pattern (`ResolvedSecret`, validators)
- `src/elspeth/web/blobs/service.py` — blob lifecycle, hash integrity, async-over-sync pattern
- `src/elspeth/web/blobs/protocol.py` — closed-set Literals derived via `get_args` (anti-drift mechanism)
- `src/elspeth/web/composer/tools.py:1448-1480` — strip-before-validate precedent (Phase 1.1 / Shape 1 closure)
- `src/elspeth/web/execution/service.py:711-733` — runtime secret-resolver call site
- `src/elspeth/web/execution/validation.py:331-369` — validate-side secret-ref collection
- `tests/integration/pipeline/test_composer_runtime_agreement.py:1-91` — agreement-suite registry (Shapes 1–8, bug-verification protocol)

### 13.3 Related work

- Sibling epic `elspeth-528bde62bb` — Composer LLM evaluation remediation (validator parity, runtime dry-run, operator visibility) — shares the `set_source_from_blob`-grade authorship-discipline framing
- Phase 1.1 (`elspeth-72d1dccd44`, commit `3b7ca22b`) — secret-ref strip-before-validate precedent that Shape 1 pins
- ADR-018 (`docs/architecture/adr/018-producer-site-outcome-discrimination.md`) — closure-rule ADR pattern for plugin-author discipline

### 13.4 Memory cross-reference

- `feedback_eval_attribution_can_mislead.md` — verify actual fault location before inheriting framing (the original proposal's "add a sibling ref form" framing was symptomatic; the panel re-attributed to "blob_ref was mis-scoped from the start")
- `feedback_locked_in_buggy_expectations.md` — wave of test failures after structural fix is the bug landing visibly (when P3 lands, expect existing tests that assumed `blob_ref` was source-only to fail; that's the bug becoming visible, not the fix breaking things)
- `project_db_migration_policy.md` — no Alembic; operator drops the dev/staging DB during P3 deployment
- `project_pipeline_composer_5concept_rewrite.md` — pipeline-composer skill clarity-uplift work; P5's MCP tool surface intersects with the composer's existing tool taxonomy and should land before any rewrite of that skill

### 13.5 Five-reviewer panel (2026-05-03)

leverage-analyst (systems thinking), solution-design-reviewer, python-code-reviewer, quality-assurance-analyst, architecture-critic. Findings synthesised in the epic body (`elspeth-fdebcaa79a` description); this spec translates that synthesis into phase-decomposed deliverables with grounded line numbers.

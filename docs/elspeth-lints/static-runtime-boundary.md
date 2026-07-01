# elspeth-lints Static-vs-Runtime Enforcement Boundary

This document is the dual of [rationale.md](rationale.md). Where the rationale
explains what each rule family enforces *statically*, this document records
what each family deliberately does **not** enforce statically, and which
runtime mechanism completes the contract.

Static analysis is necessary but not sufficient. Statics can only see
declarations, structure, and types — they cannot observe runtime values,
external system behaviour, or the actual sequence of events at execution
time. Some invariants are intrinsically value-dependent; others are
tractable statically but the cost of a sound static check exceeds the cost
of the runtime guard that already exists. This document names those choices
so a reviewer evaluating a "should this be a lint?" proposal can answer with
the static/runtime split already written down.

Use the same six-family taxonomy as the rationale (Trust Tier, Immutability,
Audit Evidence, Plugin Contract, Composer, Manifest), plus a final section
of disciplines that are deliberately out of scope for the analyzer entirely.

## Trust Tier

**Statically enforced (by `trust_tier.tier_model`):**

- L0–L3 layer-import direction. Upward imports fail the build, with
  per-file and per-finding allowlist exemptions for documented boundary
  cases.
- Defensive-pattern detection in code that handles ELSPETH-owned data:
  `.get()` on typed dataclasses, `getattr(...)` fallbacks, `hasattr(...)`,
  `isinstance()` used as a guard rather than as a Tier-3 boundary check.
- Pattern-tag governance on `allow_hits[]` entries: the typed category tag
  must come from `_ALLOWLIST_PATTERN_TAGS`, and `owner=bugfix` entries must
  carry an expiry or a pattern tag.

**Not statically enforced (runtime counterpart in parentheses):**

- *Whether a runtime value conforms to its declared type at a trust
  boundary.* The analyzer can see that a source declares `output_schema:
  {field: int}`, but cannot prove the source actually emits ints at
  runtime. (Source-plugin validation/coercion at the Tier-3 boundary;
  Landscape Tier-1 read guards that crash on type mismatch — see
  `engine-patterns-reference` skill, "Offensive Programming" section.)
- *Whether a `None` represents real absence or fabricated default.* The
  fabrication decision test in CLAUDE.md is value-dependent; statics see
  only that a field is `int | None`, not what `None` means semantically.
  (Source-plugin discipline at construction time; auditor-facing queries
  surface the `None` for the consumer to interpret.)
- *Whether `TYPE_CHECKING` imports actually stay confined to type
  declarations.* Statics enforce the import direction at module scope but
  cannot trace whether a string-quoted forward reference resolves to a
  cycle at type-checker time. (mypy's own resolution.)

## Immutability

**Statically enforced (by `immutability.freeze_guards` and
`immutability.frozen_annotations`):**

- Presence of `freeze_fields(self, "name", ...)` in `__post_init__` for
  every frozen dataclass with container fields. Allowlisted exemptions
  exist for Tier-3 external-data DTOs where the mutable annotation
  accurately reflects the runtime shape.
- Annotation shape on frozen-dataclass fields: `Mapping`, `Sequence`,
  `tuple`, `frozenset` allowed; bare `dict`, `list`, `set` flagged.

**Not statically enforced:**

- *Whether a container was actually deep-frozen at runtime.* The analyzer
  sees the `freeze_fields(...)` call but cannot prove the contents were
  recursively converted. A caller could pass a deeply nested
  `dict[str, list[dict]]` whose internal lists remain mutable through
  shallow proxying. (`deep_freeze()` at construction time enforces this
  recursively; attempted mutation raises `TypeError` from
  `MappingProxyType` and `tuple`.)
- *Whether `MappingProxyType(self.x)` was constructed from a fresh dict
  copy or wraps the caller's reference.* The forbidden anti-patterns in
  CLAUDE.md's "Frozen Dataclass Immutability" section are
  value-dependent. (No runtime check; pre-PR review and the
  `immutability.freeze_guards` / `immutability.frozen_annotations`
  `elspeth-lints` rules catch these during development. This is a
  deliberate gap: catching them statically requires alias analysis.)

## Audit Evidence

**Statically enforced (by `audit_evidence.tier_1_decoration`,
`audit_evidence.nominal_base`, `audit_evidence.guard_symmetry`,
`audit_evidence.gve_attribution`):**

- Audit-bearing dataclass shape: nominal base inherited where required,
  Tier-1 / Tier-2 decoration markers present, validation raises carrying
  attribution context, read-side guards mirroring write-side validation.

**Not statically enforced:**

- *Whether an audit row was actually written to Landscape for a given
  decision.* The analyzer can see the class is shaped to carry an audit
  payload, but cannot prove `recorder.record(...)` was called. (Landscape
  schema validation on persist; the attributability test
  `explain(recorder, run_id, token_id)` in CLAUDE.md is the runtime
  contract.)
- *Whether the audit primacy order was honoured at runtime
  (audit-fires-first then telemetry then log).* The analyzer cannot
  observe call order. (Pipeline integration tests; telemetry-emission
  hooks fail the run if the audit write didn't precede them.)
- *Whether a Tier-1 read guard caught a corrupted row at runtime.* The
  analyzer enforces guard symmetry (read side mirrors write side); the
  runtime guard's actual behaviour on corrupt data is the runtime check
  (see `engine-patterns-reference` skill, Tier-1 read-guard examples).

## Plugin Contract

**Statically enforced (by `plugin_contract.options_metadata`,
`plugin_contract.component_type`, `plugin_contract.plugin_hashes`):**

- Declared metadata shape on plugin classes (component type, options
  metadata schema, source-hash declarations).
- Pluggy hook signatures match the protocol expected by the registry.

**Not statically enforced:**

- *Whether a plugin's runtime behaviour matches its declared metadata.*
  The analyzer reads the metadata; it doesn't run the plugin. A source
  plugin can declare `output_schema: {field: int}` and emit strings at
  runtime. (Catalog ingestion-time validation when the registry is built;
  runtime checks in `RowProcessor` and `Orchestrator` per the
  `engine-patterns-reference` skill.)
- *Whether the declared `source_hash` matches the file content.* The
  declaration is text; the analyzer cannot recompute the hash without
  reading the source bytes — and even if it did, this would be a
  manifest-style check (and the manifest family covers exactly that
  recompute-and-compare pattern). (The `plugin_contract.plugin_hashes`
  `elspeth-lints` rule recomputes source hashes and covers the static side.)

## Composer

**Statically enforced (by `composer.exception_channel`,
`composer.catch_order`):**

- Order of `except` clauses: domain-specific exceptions before their
  supertypes.
- Specific exception channels for specific failure modes
  (e.g., LLM-argument failures must raise `ToolArgumentError`, not a
  generic `Exception`).

**Not statically enforced:**

- *Whether the runtime path actually raises the declared exception
  class.* The analyzer enforces what's caught; not what's raised by
  third-party code or external systems. A `LiteLLM` call could raise
  `RateLimitError` instead of `ToolArgumentError`, and the analyzer
  cannot model that. (Composer integration tests; runtime exception
  classification in the orchestrator's retry manager.)
- *Whether the exception's recovery path actually surfaces to the
  operator.* Statics see the catch; they cannot observe whether the
  composer's session-state machinery transitions correctly. (Playwright
  E2E tests; manual composer harness sessions.)

## Manifest

**Statically enforced (by `manifest.contract_manifest`,
`manifest.symbol_inventory`, `manifest.test_to_source_mapping`):**

- Checked-in YAML/JSON manifests match recomputed truth from the source
  tree at static-analysis time.
- Plugin hashes and source-symbol inventory snapshots align with the
  files they describe.

**Not statically enforced:**

- *Whether a runtime symbol resolves to the declared module at import
  time.* The manifest declares "this symbol lives in this module"; the
  analyzer compares text-to-text but cannot prove `importlib` would find
  the symbol at runtime. (Import-time `importlib` resolution at process
  startup; failing imports fail the run, not the analyzer.)
- *Whether the manifest matches `HEAD` after a force-push.* The analyzer
  runs against the working tree; it doesn't know if `HEAD` was rewritten.
  (Force-push branch protection on shared branches; CI pipeline re-runs
  on every push.)

## Deliberately Out of Scope (No Plans to Add)

These categories are explicitly NOT enforced by elspeth-lints, and
proposals to add them should be redirected:

- **Performance contracts.** "This function must complete in N ms" is a
  runtime property dependent on data shape and host. Covered by the
  benchmark suite and `pytest-benchmark`.
- **Concurrency and async invariants.** Race conditions, deadlocks,
  task-cancellation correctness. Covered by `axiom-determinism-and-replay`
  and the project's chaos-engineering harness in `tests/`.
- **Cross-process invariants.** Anything that requires observing two
  processes communicate (e.g., the composer-MCP handshake). Covered by
  integration tests.
- **LLM prompt invariants.** Skill content, system-prompt construction,
  expected-output shape. Per memory `feedback_no_tests_for_skill_prompts`:
  skills are LLM prompts, not code; the verification is re-running the
  LLM, not grepping the prompt text.
- **Database schema migrations.** Covered by Alembic; the analyzer has
  no model of the runtime DB shape.
- **Network and external-API contracts.** Status codes, response shapes,
  rate limits. Covered by integration tests and the verifier plugin
  family.

## How to Use This Boundary in Practice

When a reviewer proposes a new lint rule, walk this document family-by-family:

1. *Is the proposed check value-dependent?* If yes, it's not a static
   property — push to runtime (Landscape, integration tests, or a
   pre-existing runtime guard). Cite the family's "Not statically
   enforced" section in your reply.
2. *Is the check tractable statically but the runtime guard already
   covers it?* If yes, evaluate the marginal value of static enforcement
   vs the maintenance cost. Default: defer to runtime unless static
   catches it materially earlier (e.g., pre-commit vs CI vs production).
3. *Is the check structural but value-blind (declarations, types, file
   shapes, manifests)?* This is the analyzer's sweet spot — file a rule
   under the appropriate family and follow the lifecycle in
   [rationale.md](rationale.md) §Rule Lifecycle.

When in doubt, prefer the runtime guard. The audit primacy order
(Landscape first, telemetry second, logging last) and the project's
crash-on-tier-1-anomaly discipline mean a missed static check usually
still surfaces at runtime — late, but visibly. A wrongly-static check
that has to be allowlisted for every legitimate exception erodes the
analyzer's signal-to-noise ratio and is worse than no rule at all.

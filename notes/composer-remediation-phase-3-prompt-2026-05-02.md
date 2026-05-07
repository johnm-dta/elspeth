# Phase 3 prompt — Composer/runtime agreement test suite expansion

**Use this as a self-contained prompt for a fresh session or subagent.** It has no
dependencies on prior conversation context.

---

## Prompt begins

You are picking up Phase 3 of the composer remediation program. The remediation
program lives at `notes/composer-remediation-program-2026-05-01.md` and the source
audit at `notes/composer-llm-eval-2026-05-01.md`. **Read those two files first** —
they explain why this work exists and how it fits the rest of the program.

Tracking issue: **`elspeth-1ee3c96c72`** *(P3 task, currently `in_progress`)* —
"Expand composer/runtime agreement tests — coalesce and type-level".

The issue was filed before the 2026-05-01 staging eval. The eval surfaced four new
"composer says valid / runtime rejects" reproducer shapes (S2 v1, S2 v2, S1A literal
api_key, S1A monolithic happy-path) plus a Phase 2.2 RunStatus taxonomy widening that
needs cross-layer regression coverage. Phase 3 extends the existing test suite to
encode all of them.

### What's already done in code

Phases 0, 1, and 2 of the program landed; their fixes are in:

| Phase | Fix landed in | Issue closed |
|---|---|---|
| 0 — gate primitive diagnostic surface | (sub-fix 1+3 in routes.py) | `elspeth-2c3d63037c` |
| 0.b — gate primitive root cause | DEFERRED — needs staging reproducer | `elspeth-209b7e3a2b` |
| 1.1 — secret_refs fabrication-aware | commit `3b7ca22b` | `elspeth-72d1dccd44` |
| 1.2 — route_target_resolution | commit `3b7ca22b` (new shared helper `engine/orchestrator/preflight.py::assemble_and_validate_pipeline_config`) | `elspeth-127de6865a` |
| 1.3 — schema mode/required_fields | commit `2d9dc21d` (rejected at `SchemaConfig` model layer) | `elspeth-f5f798f797` |
| 2.1 — pipeline_done_callback | (commit not recorded here — check `git log`) | `elspeth-31d53c7493` |
| 2.2 — RunStatus four-value taxonomy | commit `cc895589` | `elspeth-0de989c56d` |
| 2.3 — /api/secrets reason | commit `22e3e0d9` | `elspeth-0d31c22d26` |

Each fix has unit tests at the layer it touches. **Phase 3 is not about re-testing
each fix in isolation** — it's about pinning the *agreement contract* between
composer `/validate` and runtime `/execute` (and the API readback layers) so that
any future drift fails CI.

### Scope of Phase 3

Encode the following reproducer shapes as regression tests in the
**composer/runtime agreement suite**. Find the suite first — start with
`grep -rn "composer.*runtime.*agreement" tests/` and
`tests/integration/pipeline/test_composer_runtime_agreement.py` is a known
existing file. Read it; understand its conventions; extend it.

#### Shape 1 — S1A literal credential placeholder (Phase 1.1 lock-in)

```yaml
transforms:
- name: classify_ticket
  plugin: llm
  options:
    api_key: WILL_BE_WIRED_FROM_OPENROUTER_API_KEY
    provider: openrouter
    model: openai/gpt-4.1-nano
```

Expected: `POST /validate` returns `is_valid: false` with the `secret_refs` check
naming the credential field path. The test must also verify that the response does
**not** echo the literal placeholder string anywhere (audit-hygiene constraint —
copy the discipline from `_collect_credential_field_violations` in
`src/elspeth/web/execution/validation.py`).

#### Shape 2 — S2 v1 dangling on_error route target (Phase 1.2 lock-in)

```yaml
aggregations:
- name: rollup
  plugin: batch_stats
  options:
    group_by: customer_tier
    value_field: amount
    mode: observed
  on_error: aggregation_errors  # <-- no sink named aggregation_errors exists
```

Expected: `POST /validate` returns `is_valid: false` at the new
`route_target_resolution` check. The error must name the unresolvable target
(`aggregation_errors`) and the source location (the aggregation's `on_error`
field).

Also test the *four other axes* the new shared helper covers as
defense-in-depth, even though they were already caught at `graph.validate()`:
- `transforms[*].on_error` dangling
- `sources[*].on_validation_failure` dangling
- `gates[*].routes[*]` dangling
- `sinks[*].on_write_failure` dangling

For each, write the failing shape, assert the error names the right field path.
This is the defense-in-depth verification — if `graph.validate()` is ever
loosened, the new check catches it.

#### Shape 3 — S2 v2 schema mode/required_fields incompatibility (Phase 1.3 lock-in)

```yaml
aggregations:
- name: rollup
  plugin: batch_stats
  options:
    schema:
      mode: flexible
      fields: [...]
      required_fields: [...]    # <-- incompatible with mode: flexible
```

Expected: rejection happens at `SchemaConfig` construction (model layer), not at
the validator. Test that:
1. `POST /validate` returns `is_valid: false` with a `SchemaConfigModeViolation`-class
   error
2. Constructing `SchemaConfig(mode=flexible, required_fields=[...])` directly raises
   at `__post_init__`
3. The error message names the incompatible combination

The two-layer test confirms the structural fix (model-layer rejection) hasn't been
silently downgraded to a validator-only check.

#### Shape 4 — S1A monolithic happy-path (composer/runtime agreement smoke test)

The S1A monolithic prompt produced a 1-source / 1-LLM / 5-gate / 6-sink pipeline.
With Phase 1.1's secret_refs fix, the literal-`api_key` shape now fails `/validate`.
**With a properly wired `secret_ref`**, the same shape should pass `/validate` AND
execute end-to-end without a runtime rejection.

Write the test as a fixture:
- Source: csv with the eval's 6-row ticket sample (or equivalent)
- Transform: `llm` with a properly wired `{secret_ref: <test_secret>}` and a stub
  LLM client returning canned classifications
- 5 gates routing on the classification field
- 6 sinks (success per category + error/quarantine)

Expected: `/validate is_valid: true`, `/execute` accepts, runs to completion,
`run.status` is `completed` (not `completed_with_failures`, not `failed`).

This is the positive-control for the architectural fix
(`assemble_and_validate_pipeline_config` shared between `/validate` and `/execute`).
If it ever fails, the agreement contract is broken.

#### Shape 5 — Phase 2.2 RunStatus four-value taxonomy

For each of the four `RunStatus` values, encode a pipeline shape and assert the
status comes out correctly across **all three persistence layers**:

| Shape | Pipeline | Expected RunStatus |
|---|---|---|
| S1A reproducer | LLM transform broken (every row routed via on_error) | `failed` |
| S1B msg2 reproducer | every row fails, no on_error route | `failed` |
| Healthy aggregation | csv → batch_stats group_by → json sink, all rows succeed | `completed` |
| Mixed | half rows succeed, half fail | `completed_with_failures` |
| Empty source | csv with header only, no rows | `empty` |

Read across the three layers:
1. Engine in-memory `RunResult` (immediately after `engine.run()`)
2. Landscape audit `Run` row (after the audit commit)
3. Web sessions `RunRecord` (via `GET /api/runs/{rid}`)

All three layers must agree on the status. The biconditional invariant in
`__post_init__` should make disagreement unrepresentable, but the test pins it so
a future refactor can't silently drop the invariant.

**Open design question encoded as a test:** the rows_routed-only outcome (S1A) was
classified as `failed` (not `completed_with_failures`) per the closure rationale
on `elspeth-0de989c56d`. Encode this as an explicit named test
(`test_runstatus_rows_routed_only_classifies_as_failed`) with a docstring citing
the rationale, so a future maintainer changing this behavior has to confront the
decision rather than silently changing it.

#### Shape 6 — Phase 2.3 /api/secrets reason taxonomy

For each failure mode, mock the resolver into that mode and assert the response.

| Failure mode | Expected `reason` value |
|---|---|
| `ELSPETH_FINGERPRINT_KEY` unset | `fingerprint_resolver_not_configured` (or whatever closed-list value Phase 2.3 chose — check the closure on `elspeth-0d31c22d26`) |
| Env var named in inventory but missing from process env | `env_var_not_set` |
| Value present but fingerprint mismatch | `fingerprint_mismatch` |

Plus: assert the biconditional `available ⟺ reason is None` is enforced (an entry
with `available: true` must have `reason: null`; an entry with `available: false`
must have a non-null reason from the closed list).

Plus: audit-hygiene test — feed the resolver a sentinel candidate-secret value and
assert the sentinel never appears anywhere in the `/api/secrets` response.

#### Shape 7 — Phase 2.1 pipeline_done_callback (single-occurrence regression)

The S2 successful-run shape (csv → batch_stats group_by → json sink) writes output
and the API readback returns terminal state immediately. Assert no
`pipeline_done_callback_exception` event in the structured-log capture during the
run.

This is a single-occurrence regression test. Don't generalize. If a similar issue
shows up on a different completion path, that becomes Shape 7b.

### Test placement and conventions

- Existing file: `tests/integration/pipeline/test_composer_runtime_agreement.py`
  (verify path; this is the file `elspeth-87f6d5dea5`'s closure mentioned)
- New shapes that don't fit there cleanly may need adjacent files; prefer extending
  the existing class structure
- Test-naming: `test_agreement_<shape>_<expectation>` — e.g.
  `test_agreement_dangling_on_error_fails_validate`,
  `test_agreement_runstatus_empty_source_classifies_as_empty`
- Each test must include a one-line docstring naming the source shape
  (S1A / S2 v1 / etc.) so a future debugger can grep for the eval reproducer
- The test class should have a module-level docstring listing the closed registry
  of divergence shapes — every future eval finding extends this list

### Acceptance gate

1. All seven shape categories above (Shapes 1-7) have at least one passing test
2. The test module's closed-registry docstring lists every shape with the
   originating eval session ID and run ID where applicable
3. `pytest tests/integration/pipeline/test_composer_runtime_agreement.py` passes
4. The full integration suite still passes (`pytest tests/integration/`)
5. No flaky tests introduced (run the new tests 5x in a row and confirm)
6. Issue `elspeth-1ee3c96c72` closed with the four-value RunStatus tests, the
   four route-target axes, the secret-refs fabrication test, the SchemaConfig
   rejection test, the S1A happy-path positive control, the /api/secrets reason
   taxonomy tests, and the pipeline_done_callback regression test all named in
   the closure summary

### What this work does NOT cover

- **Phase 0.b gate primitive root cause** (`elspeth-209b7e3a2b`) — still deferred
  pending a staging reproducer. When it lands, add a Shape 8 for incremental gate
  composition not 500-ing.
- **Frontend rendering observations** (`obs-d40f326097` SecretsPanel reason,
  `obs-b5666c1968` run-list status) — these are UI gaps, not API-agreement gaps.
  Out of scope.
- **DIVERT/MOVE counter conflation** (`obs-abc8baa1cd`) — Phase 4 verifies
  whether this matters in practice; not Phase 3.
- **Tier-model fingerprint instability** — separate observation; out of scope.

### Anti-patterns to avoid

1. **Don't test the validator alone.** Phase 3's value is in *agreement* tests
   that exercise both `/validate` AND `/execute` (or runtime preflight) on the
   same shape and assert the same verdict. A test that only hits `/validate`
   misses the architectural point.
2. **Don't echo candidate-secret values into test fixtures or logs.** The
   audit-hygiene discipline applies to test code too. Use `pytest.MonkeyPatch`
   or fixtures with named sentinels (`SECRET_SENTINEL = "test-secret-do-not-log"`)
   and assert the sentinel never appears in any captured response or log line.
3. **Don't use shared-mutable state across tests.** Each test creates its own
   pipeline config and its own in-memory state. The S1A monolithic shape is
   convenient as a fixture but should NOT be parameterized in a way that makes
   one test's mutation visible to another.
4. **Don't add fallbacks or `xfail` for the deferred Phase 0.b shape.** If the
   gate primitive crashes locally for you when writing Shape 8, leave Shape 8
   commented with a TODO referencing `elspeth-209b7e3a2b` and don't write a
   speculative test. The fix isn't landed; the test would lock in the wrong
   behavior.

### Verification commands

```bash
# Setup
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run the agreement suite alone
.venv/bin/python -m pytest tests/integration/pipeline/test_composer_runtime_agreement.py -v

# Run the full integration suite (no flakes allowed)
.venv/bin/python -m pytest tests/integration/

# Re-run the new tests 5x to check for flakes
for i in {1..5}; do
  .venv/bin/python -m pytest tests/integration/pipeline/test_composer_runtime_agreement.py -v || break
done

# Type and lint
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/ tests/
```

### Reference materials

- **Project guidance:** `/home/john/elspeth/CLAUDE.md` — read the auditability
  standard, three-tier trust model, and the "no fallbacks / no fabrication" rules
- **Program doc:** `notes/composer-remediation-program-2026-05-01.md`
- **Source audit:** `notes/composer-llm-eval-2026-05-01.md`
- **Phase 2 handover:** `notes/composer-remediation-phase-2-handover-2026-05-01.md`
- **Closed Phase 1.1 issue:** `elspeth-72d1dccd44` (secret_refs predicate location,
  closure rationale)
- **Closed Phase 1.2 issue:** `elspeth-127de6865a` (route_target_resolution
  structural fix, defense-in-depth axes)
- **Closed Phase 2.2 issue:** `elspeth-0de989c56d` (RunStatus four-value taxonomy,
  the rows_routed-only design call rationale)
- **Closed Phase 2.3 issue:** `elspeth-0d31c22d26` (reason taxonomy,
  audit-hygiene patterns)

When closing `elspeth-1ee3c96c72`, summarize the test inventory and link each
shape back to its originating eval reproducer (session ID + run ID where
applicable). The closure summary is the durable contract — future evals will
extend the registry, and the closure must establish the precedent for how new
shapes get added.

## Prompt ends

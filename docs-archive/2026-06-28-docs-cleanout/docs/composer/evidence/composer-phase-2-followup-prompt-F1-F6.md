# Implementation Prompt — Composer Phase 2 Follow-ups F1–F6

> **Hand this prompt verbatim to a fresh implementing agent.** It is self-contained: it does not assume any memory of prior conversations.

---

## What you are doing

Six follow-up items (F1–F6) surfaced during Phase 2 review but were deferred as non-blocking. Now landing them. Each item is independent; you can do them in any order, in separate commits, or batch related ones (F1 + F5 both touch `test_walk_model_schema.py`; reasonable to combine).

The Phase 2 redaction framework has landed (42 commits, manifest at 38 entries, all gates green). The work below tightens loose ends without changing security claims.

## Where the work lives

- **Worktree**: `/home/john/elspeth/.worktrees/composer-progress-1a`
- **Branch**: `feat/composer-progress-persistence-1a`
- **HEAD**: `5054885c` (Phase 2 gate green commit)
- **Phase 2 memory**: `/home/john/.claude/projects/-home-john-elspeth/memory/project_phase2_implementation_complete.md`

## Operational rules (NON-NEGOTIABLE)

1. **Working directory**: `cd /home/john/elspeth/.worktrees/composer-progress-1a` — absolute paths only.
2. **Python**: ALWAYS `.venv/bin/python` (Python 3.13). If you install: `uv pip install --python .venv/bin/python -e ".[dev]"`. NEVER plain `uv pip install` (memory: `feedback_uv_venv_leak`).
3. **No `--no-verify`**, **no `--amend`**, **no `git stash`** (pre-commit hooks have caused data loss before — memory: `project_phase2_implementation_complete.md` F4).
4. **Commit messages**: Conventional Commits. Body cites Phase 2 deferred item ID (F1, F2, etc.) and Phase 2 plan task / spec section as applicable. Co-Authored-By trailer.
5. **Project disciplines** (read `/home/john/elspeth/.worktrees/composer-progress-1a/CLAUDE.md` first): no defensive programming; offensive programming; no legacy code; freeze-guards; tier-model; layer rules; audit > telemetry > logging primacy.

## Required reading before any code change

1. `/home/john/CLAUDE.md` and `/home/john/elspeth/.worktrees/composer-progress-1a/CLAUDE.md`.
2. `/home/john/.claude/projects/-home-john-elspeth/memory/MEMORY.md` and the listed feedback files.
3. `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` §4.2.x and §9 — the redaction spec.
4. `src/elspeth/web/composer/redaction.py` — the redaction module (large; navigate by symbol).

---

## F1 — Walker bug fixes lack narrow regression tests

**Context**: Task 13 introduced a precursor walker fix (`5709cd58`) that made `Optional[scalar]` fields emit leaves. Task 14 introduced two more walker fixes — `_walk_type` substitute_provider gating on `("attr", name)` last step + `_build_substitute_provider` None-traversal skip. These are tested only INDIRECTLY via high-level promotion tests. A future refactor of `_walk_type` or `_build_substitute_provider` could regress without focused-test failure.

**File to modify**: `tests/unit/web/composer/test_walk_model_schema.py`

**Tests to add** (write each as a failing test FIRST to confirm what would regress):

1. `test_walk_emits_substitute_provider_only_for_attr_path` — construct a synthetic model with a `Sensitive[list[str]]` field (e.g., `tags: Annotated[list[str], Sensitive(summarizer=lambda v: f"<n={len(v)}>")]`). Walk it; for any node whose final step is NOT `("attr", name)`, assert `substitute_provider is None`; for attr-final nodes, assert `substitute_provider is not None`.

2. `test_walk_handles_optional_basemodel_with_none_runtime_value` — construct a model with `optional_nested: _InnerModel | None = None` where `_InnerModel` has a Sensitive field. Walk with `with_values=True`. Build a `value_provider` for the inner Sensitive path; call it with a root dict where `optional_nested` is `None`. Assert it returns `None` (scalar) or `[]` (container), NOT raises TypeError.

**Steps**:
1. Read `_walk_type` and `_build_substitute_provider` in `redaction.py` to see exactly which guards exist.
2. Construct models that EXERCISE those guards (the model shapes should reach the guard's branch under normal walk).
3. Write the tests; run; confirm pass.
4. Commit: `test(composer/redaction): pin walker bug-fix regression guards (F1)`. Body cites F1 + Task 14 review I1.

---

## F2 — Audit-side `__cause__` detail loss

**Context**: The Pydantic `ValidationError` field-name detail lives on the live exception's `__cause__` but is DROPPED before persistence. `audit.py:463-504` (`finish_arg_error`) persists `error_class` + a leak-safe `error_message` only. For ARG_ERROR audits, the field-name detail is genuinely lost — auditors querying the audit DB cannot see which Pydantic field failed validation.

This is a real audit-trail gap. The architectural framing is correct (ToolArgumentError is leak-safe by construction) but the spec doesn't acknowledge the loss explicitly.

**Two options — pick ONE and execute (read both first; advisor consult OK)**:

### Option (a): Persist canonicalized `__cause__` errors

Extend `finish_arg_error` in `src/elspeth/web/composer/audit.py` to optionally capture `exc.__cause__.errors()` (Pydantic ValidationError API). The `.errors()` output is a list of dicts with `loc`, `msg`, `type` fields. NO raw values — `loc` tuples (field path), error type names, NOT the rejected values themselves.

Place the canonicalized errors in `result_canonical["validation_errors"]` (or similar — find the canonical field name by reading the audit recorder's other persistence patterns).

Add tests:
- ARG_ERROR with Pydantic `__cause__` records the `loc` tuples.
- ARG_ERROR without `__cause__` (or with a non-Pydantic `__cause__`) is a no-op for this field.

This option preserves audit detail; minor schema impact (`result_canonical` is JSON-blob-shaped, so no migration).

### Option (b): Document the loss explicitly in spec

Update spec §4.2.6 (Tier-3 → Tier-1 boundary disposition table) to add a row stating that ARG_ERROR audits do NOT preserve Pydantic field-name detail by design; auditors needing field-name granularity can correlate against the production logs.

This option is documentation-only but accepts the loss permanently.

**Choose based on**: if Phase 3's compose-loop wiring is likely to amplify the audit value (e.g., recovery flows that need to know which field failed), pick (a). If the leak-safe envelope is sufficient and auditors have other channels, pick (b).

**Steps for (a)**:
1. Read `finish_arg_error` and surrounding audit recorder code.
2. Determine if `result_canonical` is the right place or if a new schema field is needed.
3. Implement + test.
4. Commit: `feat(composer/audit): canonicalize Pydantic __cause__ errors for ARG_ERROR audits (F2)`.

**Steps for (b)**:
1. Update spec §4.2.6 — add a row to the disposition table explaining the ARG_ERROR loss; reference the leak-safety rationale.
2. Update §12.2 traceability if applicable.
3. Commit: `docs(spec): document ARG_ERROR __cause__ loss as intentional (F2)`.

If unsure: surface to operator (`AskUserQuestion`) before deciding. This is a design choice with audit-trail consequences.

---

## F3 — Overbroad Sensitive markers on `nodes[*].routes` / `nodes[*].trigger`

**Context**: Task 14's Sensitive markers on `nodes[*].routes` and `nodes[*].trigger` use the shared `_summarize_set_source_options` summarizer, which is **content-agnostic** — it only redacts dicts shaped like `{"path": ..., "blob_ref": ...}`. For routes (graph topology like `{"true": "node_id"}`) and triggers (timer/count thresholds), the summarizer passes values through verbatim. The §4.4.2 adequacy guard is satisfied MECHANICALLY but no actual redaction happens.

This was a known trade-off at Task 14 (operator-accepted at the time), but the audit trail now records these fields as Sensitive without protecting them.

**Two options — pick ONE**:

### Option (a): Restructure as typed sub-models

Replace `routes: dict[str, str] | None` with `routes: list[RouteEdge] | None` where `RouteEdge` is a Pydantic model `class RouteEdge(BaseModel): label: str; target: str`. Similarly for `trigger: dict[str, Any] | None` — design a typed `TriggerSpec` union model based on actual trigger shapes (count-trigger, timer-trigger, etc.).

Typed sub-models REMOVE the `dict[str, Any]` rejection from the §4.4.2 guard, so the Sensitive marker is no longer mechanically forced. Drop the Sensitive marker on `routes`/`trigger`; they become structural metadata.

Touches: `_PipelineNodeModel` in `redaction.py`; the corresponding handler reads in `tools.py`; the snapshot file (4 entries change hash).

### Option (b): Add a node-shape-aware summarizer

Write `_summarize_node_options(value: dict[str, Any]) -> str` that redacts known sensitive keys (`api_key`, `template`, prompt-bearing strings) explicitly. Replace the path-redacting summarizer on `routes`/`trigger` with this new one.

Touches: `redaction.py`; the snapshot file.

**Choose based on**: if routes/trigger truly have no sensitive content (purely topology), pick (a) — typed models are clearer. If they may carry user-supplied template strings or other free-form content, pick (b).

Read the actual handler in `tools.py` for `_execute_set_pipeline` to see what shapes flow through `routes` and `trigger`.

**Steps for either**:
1. Read the affected fields' actual usage in `tools.py`.
2. Implement (typed models OR new summarizer).
3. Update tests in `test_promote_set_pipeline.py`.
4. Regenerate snapshot via `.venv/bin/python scripts/cicd/bootstrap_redaction_snapshot.py --write`.
5. Run full gate.
6. Commit: `refactor(composer/redaction): replace overbroad routes/trigger Sensitive markers with <option-name> (F3)`.

---

## F4 — Process: `git stash` violation surfaced

**Context**: Task 4's implementer self-reported using `git stash` twice during investigation, in violation of CLAUDE.md "No git stash" rule. No data loss occurred; flagged transparently. This is NOT an action item — it's already in the project memory (memory file: `feedback_no_git_stash.md` notes the rule was lifted 2026-05-11).

**Action**: VERIFY that `feedback_no_git_stash.md` correctly reflects the current rule. If the rule was indeed lifted, F4 is closed (no action needed). If the rule is still active in CLAUDE.md but the memory says lifted, that's a memory inconsistency — surface to operator.

**Steps**:
1. Read `/home/john/elspeth/.worktrees/composer-progress-1a/CLAUDE.md` — search for "git stash".
2. Read `/home/john/.claude/projects/-home-john-elspeth/memory/feedback_no_git_stash.md`.
3. If consistent: F4 is closed. Report DONE_NO_ACTION.
4. If inconsistent: surface to operator with both citations.

No commit expected from F4 unless an inconsistency requires a memory update.

---

## F5 — Property test parametrize narrowing review

**Context**: Task 19's property test parametrize narrowed from `entry.argument_model is not None` to additionally require `_has_sensitive_argument_field(entry.argument_model)`. Reason: `get_blob_content` has Sensitive only on its RESPONSE model. Without the narrowing, the plan's `assert sensitive_nodes` would fail vacuously on that entry.

**Concern**: does this narrowing open a silent-pass surface for future entries that have Sensitive only on response?

**Action**: review the narrowing and either:
- Confirm the narrowing is sound (response-side Sensitive is exercised by `test_redact_tool_call_response.py`; argument-side property test is genuinely irrelevant for response-only-Sensitive entries).
- OR replace the narrowing with a different mechanism (e.g., dual parametrize: argument-side property test AND response-side property test; entries appear in whichever applies).

**Steps**:
1. Read `tests/unit/web/composer/test_redaction_completeness_property.py` — the current narrowing implementation.
2. Read `tests/unit/web/composer/test_redact_tool_call_response.py` — does it actually have coverage for `get_blob_content.content` substitution under Hypothesis or just unit-test pinning?
3. Decision:
   - If response-side coverage is hypothesis-strength: confirm narrowing is sound; close F5 with no change.
   - If response-side coverage is unit-test-only (single canary, not type-agnostic): add a response-side property test mirroring the argument-side shape OR widen the argument-side test to handle response models.

**Steps to add a response-side property test (if needed)**:
1. Mirror `test_redaction_replaces_every_sensitive_value` but parametrize over entries with `response_model is not None`.
2. The invariant: for every Sensitive path in `response_model`, the value extracted from the redacted response differs from the raw response.
3. Use Hypothesis `st.from_type(model)` for the response model (may need conftest strategy registration).
4. Commit: `test(composer/redaction): response-side Hypothesis completeness property test (F5)`.

---

## F6 — Hypothesis conftest infrastructure review

**Context**: Task 19's conftest registers a function-style strategy for `dict` (Hypothesis can't resolve `dict[str, Any]` natively) and explicit `st.builds()` overrides for 4 models with `Field(default_factory=dict)`. Without this, most type-driven property-test entries would fail at collection time with `InvalidArgument`.

**Concern**: this infrastructure is undocumented to future readers. If a Phase 3 implementer adds a new type-driven entry with a `Field(default_factory=dict)` field, they will hit InvalidArgument and not know why.

**Action**: improve documentation OR generalize the strategy registration.

**Steps**:
1. Read `tests/unit/web/composer/conftest.py` — the current registration.
2. Audit each `st.builds()` override — is the pattern generalizable (a meta-function that auto-registers for any model with `default_factory=dict` fields)?
3. Three options:
   - (a) Document the existing manual registrations clearly in `conftest.py` docstring with examples. Add a comment in `redaction.py` near every `Field(default_factory=dict)` saying "add to conftest's _builds_overrides if a property test fails ResolutionFailed."
   - (b) Generalize: write `_auto_register_default_factory_strategies()` that introspects MANIFEST's type-driven models and registers strategies for any `default_factory=dict` field.
   - (c) Both — document AND generalize, since the cost of (b) is low if (a) reveals the pattern is mechanical.

Pick whichever produces the most maintainable result.

**Steps**:
1. Implement chosen option.
2. Test: add a synthetic type-driven entry with `Field(default_factory=dict)` and confirm the property test collects without InvalidArgument.
3. Remove the synthetic entry after verification.
4. Commit: `chore(composer/redaction): improve Hypothesis strategy registration documentation/generalization (F6)`.

---

## Final reporting

When all F1–F6 are addressed (or DONE_NO_ACTION for any that don't require changes):

- **Per-item status**: F1–F6 each with status (DONE / DONE_NO_ACTION / DEFERRED / BLOCKED).
- **Commit SHAs**: list per F-item.
- **Test counts**: full composer slice + full unit suite.
- **Gate state**: pytest unit, pytest integration, mypy, ruff, tier-model, freeze-guards.
- **Concerns**: anything that surfaced during implementation that warrants operator attention.

If any F-item escalates to an operator decision (option (a) vs (b) ambiguity, etc.), surface via `AskUserQuestion` before deciding.

## Snapshot regeneration discipline

Any F-item that adds/modifies a MANIFEST entry (F3 is the only candidate) requires regenerating `tests/unit/web/composer/redaction_policy_snapshot.json`:

```bash
.venv/bin/python scripts/cicd/bootstrap_redaction_snapshot.py --write
```

Run twice in a row; confirm `git status` shows no diff after the second run (idempotency).

## Phase 2 close-out

After F1–F6 land, Phase 2 is fully complete with all review findings closed. The operator will decide PR-open separately.

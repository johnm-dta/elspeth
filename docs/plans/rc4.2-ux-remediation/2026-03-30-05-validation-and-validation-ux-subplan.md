# RC4.2 UX Remediation — Validation And Validation UX Subplan

Date: 2026-03-30 (expanded 2026-03-31)
Status: Ready
Parent: `docs/plans/rc4.2-ux-remediation/2026-03-30-rc4.2-ux-implementation-plan.md`

---

## 1. Scope

This subplan covers richer Stage 1 validation output plus the lightweight UI
affordances that surface validation state in the inspector.

Included requirements:

- `REQ-API-07` — enhanced validation model (warnings + suggestions)
- `REQ-UX-06` — validation status tint (ambient green/red/amber indicator)

Primary surfaces:

- composition-time validation summary (`ValidationSummary` dataclass)
- tool result serialization (`ToolResult.to_dict()`)
- message response schema (`CompositionStateResponse`)
- transient frontend validation state (`sessionStore`)
- inspector status display (`InspectorPanel`, `SpecView`)

---

## 2. Goals

- Distinguish blocking errors from advisory warnings and optional suggestions
  in Stage 1 composition validation.
- Deliver warnings/suggestions to the LLM via tool results so it can self-
  correct.
- Deliver warnings/suggestions to the frontend via message responses so the
  user sees advisory guidance.
- Make validation state visible at a glance in the inspector header without
  requiring the user to click Validate.

Non-goals:

- No changes to Stage 2 (`ValidationResult` from
  `POST /api/sessions/{id}/validate`). Stage 2 has its own error model with
  per-component attribution, checks, and suggestions. It is structurally
  different and not affected by this work.
- No changes to the `ValidationResultBanner` component (Stage 2 renderer).
- No database schema changes. Warnings and suggestions are transient.

---

## 3. Architecture Decisions

### AD-1: Warnings and suggestions are transient, not persisted

Warnings and suggestions are computed fresh on every `validate()` call and
returned in responses, but NOT stored in the `composition_states` table.

Rationale:

- They are advisory — stale warnings from a previous version would be
  misleading.
- Persisting them would require a DB migration and schema change for no
  user-facing benefit.
- The frontend clears them on version change (same pattern as Stage 2
  validation).

Consequence: when loading a historical composition state from the DB (version
history, session reload), `validation_warnings` and `validation_suggestions`
will be `null`. This is correct — the transient data belongs to the moment of
mutation, not to the persisted record.

### AD-2: Separate response fields, not embedded in validation_errors

Warnings and suggestions are separate fields on the response, not mixed into
`validation_errors`. This maintains the existing contract: `validation_errors`
is a `list[str]` of blocking errors. Consumers that only check `validation_errors`
are unaffected.

New fields on `CompositionStateResponse`:

- `validation_warnings: list[str] | None`
- `validation_suggestions: list[str] | None`

### AD-3: Validation tint uses a three-state model based on existing data

The ambient indicator derives from data already available in the stores:

| State | Condition | Colour |
|-------|-----------|--------|
| **Never validated** | `validationResult === null` AND `compositionState` exists | Amber |
| **Validation passed** | `validationResult.is_valid === true` | Green |
| **Validation failed** | `validationResult.is_valid === false` | Red |
| **No pipeline** | `compositionState === null` or no nodes | Hidden |

The existing `clearValidation()` subscription (in `subscriptions.ts`) already
sets `validationResult` to `null` when the composition version changes. This
naturally transitions the tint from green/red back to amber — no new flag
needed.

---

## 4. Detailed Changes

### Phase 5A: Backend — Enhanced ValidationSummary

#### 5A.1: Extend `ValidationSummary` dataclass

File: `src/elspeth/web/composer/state.py` (lines 197–206)

Current:

```python
@dataclass(frozen=True, slots=True)
class ValidationSummary:
    is_valid: bool
    errors: tuple[str, ...]
```

Change to:

```python
@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Stage 1 validation result.

    errors block execution. warnings are advisory but actionable.
    suggestions are optional improvements. All are tuples of
    human-readable strings. frozen=True is sufficient since tuples
    of strings are immutable.
    """

    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()
```

No `freeze_fields` needed — tuples of strings are immutable.

#### 5A.2: Add warning and suggestion rules to `validate()`

File: `src/elspeth/web/composer/state.py`, method `CompositionState.validate()`
(lines 409–489)

After the existing 8 error checks, add two new sections: warnings and
suggestions. The method continues to return `ValidationSummary`.

**Warning rules** (actionable but non-blocking):

| ID | Condition | Message |
|----|-----------|---------|
| W1 | Output has no incoming edge | `"Output '{name}' has no incoming edge — it will never receive data."` |
| W2 | Source `on_success` target doesn't match any node input | `"Source on_success '{target}' does not match any node input — data may not flow."` |
| W3 | Node has outgoing edges but none match an output or another node | `"Node '{id}' has outgoing edges but no path reaches an output."` |
| W4 | Sink plugin name hints at format mismatch with filename extension | `"Output '{name}' uses plugin '{plugin}' but filename extension suggests a different format."` — only when source/output options contain a `path` key with a recognisable extension |

**Suggestion rules** (optional improvements):

| ID | Condition | Message |
|----|-----------|---------|
| S1 | Pipeline has no gate or error routing | `"Consider adding error routing — rows that fail transforms currently have no explicit destination."` |
| S2 | Pipeline has only one output | `"Single output pipeline. Consider adding a second output for rejected/quarantined rows."` |
| S3 | Source has no schema_config | `"Source has no explicit schema. Downstream field references depend on runtime column names."` |

Implementation notes:

- Warnings require traversing edges, which the error checks already do. Reuse
  the computed sets (`edge_destinations`, `node_ids`, `output_names`,
  `valid_from`, `valid_to`).
- For W4 (format/extension mismatch): only fire when `options` dict contains
  a `path` key whose extension doesn't match the plugin name. Map:
  `csv→.csv`, `json→.json`, `jsonl→.jsonl`. Skip if no `path` key or no
  recognisable extension.
- For S1: check if any node has `node_type == "gate"` or any edge has
  `edge_type == "on_error"`.
- For S3: check `self.source.options` for a `schema_config` key (if source
  is not None).

Updated return:

```python
return ValidationSummary(
    is_valid=len(errors) == 0,
    errors=tuple(errors),
    warnings=tuple(warnings),
    suggestions=tuple(suggestions),
)
```

#### 5A.3: Update `ToolResult.to_dict()` serialization

File: `src/elspeth/web/composer/tools.py` (lines 57–70)

Current `validation` dict in the output:

```python
"validation": {
    "is_valid": self.validation.is_valid,
    "errors": list(self.validation.errors),
},
```

Change to:

```python
"validation": {
    "is_valid": self.validation.is_valid,
    "errors": list(self.validation.errors),
    "warnings": list(self.validation.warnings),
    "suggestions": list(self.validation.suggestions),
},
```

This ensures the LLM sees warnings and suggestions in every tool result.

#### 5A.4: Update `send_message` response to include transient validation

File: `src/elspeth/web/sessions/routes.py` (lines 324–343)

Currently `_state_response()` reads from the persisted `CompositionStateRecord`
which has no warnings/suggestions. The transient data comes from the live
`validation` object computed at line 329.

Change: pass the live `ValidationSummary` into `_state_response()` as an
optional parameter, and populate the new response fields from it.

Update `_state_response` signature:

```python
def _state_response(
    state: CompositionStateRecord,
    live_validation: ValidationSummary | None = None,
) -> CompositionStateResponse:
```

In the body, after constructing the response, add:

```python
    if live_validation is not None:
        response.validation_warnings = (
            list(live_validation.warnings) if live_validation.warnings else None
        )
        response.validation_suggestions = (
            list(live_validation.suggestions) if live_validation.suggestions else None
        )
```

At the call site (line 343):

```python
state_response = _state_response(new_state_record, live_validation=validation)
```

Apply the same pattern in `recompose()` if it also returns state.

#### 5A.5: Update response schemas

File: `src/elspeth/web/sessions/schemas.py` (lines 62–77)

Add two new optional fields to `CompositionStateResponse`:

```python
class CompositionStateResponse(BaseModel):
    # ... existing fields ...
    validation_warnings: list[str] | None = None
    validation_suggestions: list[str] | None = None
```

These default to `None` so historical state loads (which don't have live
validation) serialize correctly.

---

### Phase 5B: Frontend — Enhanced Validation Display

#### 5B.1: Update TypeScript types

File: `src/elspeth/web/frontend/src/types/index.ts`

Add to `CompositionState` interface (line 143, after `validation_errors`):

```typescript
export interface CompositionState {
  // ... existing fields ...
  validation_errors?: string[];
  validation_warnings?: string[];
  validation_suggestions?: string[];
}
```

Also check `types/api.ts` if it has a separate `CompositionStateResponse` type
that needs the same fields.

#### 5B.2: Update SpecView to render warnings and suggestions

File: `src/elspeth/web/frontend/src/components/inspector/SpecView.tsx`
(lines 232–257)

After the existing error banner, add two more banners:

**Warnings banner** (between errors and component cards):

- Same structure as the error banner but with a different colour scheme.
- Use `--color-warning-bg` / `--color-warning` (already used by errors).
  Actually, errors currently use the warning palette. Rename the error banner
  to use `--color-error-bg` / `--color-error` semantics, and give warnings
  their own amber/yellow treatment. See CSS variables note below.
- Label: "Warnings" (not "Composition warnings" — that label currently names
  the error banner, which should become "Errors").

**Suggestions banner:**

- Light blue/info treatment: `--color-info-bg` / `--color-info`.
- Label: "Suggestions"
- Collapsible (start collapsed if >2 items to avoid visual noise).

**CSS variable note:** The existing error banner uses `--color-warning-*`
variables. This should be corrected:

- Error banner → `--color-error-bg`, `--color-error`, `--color-error-border`
  (red tones)
- Warning banner → `--color-warning-bg`, `--color-warning`,
  `--color-warning-border` (amber/yellow tones)
- Suggestion banner → `--color-info-bg`, `--color-info`,
  `--color-info-border` (blue tones)

If these CSS variables don't exist yet, define them in the theme stylesheet.
Check the DTA/AGDS palette (commit `144335d2`) for the correct colour tokens.

#### 5B.3: Relabel existing error banner

Currently the error banner is labelled "Composition warnings" (line 247). This
is inaccurate — these are blocking errors from Stage 1.

Change: relabel to "Errors" and switch to red/error colour scheme. This is a
one-line text change + CSS variable swap. Must ship in the same commit as the
new warnings banner to avoid confusion.

---

### Phase 5C: Frontend — Validation Status Tint

#### 5C.1: Add validation tint indicator to InspectorPanel header

File: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx`

Add a small coloured dot/pill adjacent to the version selector (line ~173
area, next to the "Version:" label). The dot reflects the three-state model
from AD-3.

Implementation:

```tsx
function ValidationDot() {
  const validationResult = useExecutionStore((s) => s.validationResult);
  const compositionState = useSessionStore((s) => s.compositionState);

  // Hide when no pipeline
  if (!compositionState || compositionState.nodes.length === 0) {
    return null;
  }

  let color: string;
  let label: string;
  if (validationResult === null) {
    color = "var(--color-warning)";     // amber — not yet validated
    label = "Not validated";
  } else if (validationResult.is_valid) {
    color = "var(--color-success)";     // green — passed
    label = "Validation passed";
  } else {
    color = "var(--color-error)";       // red — failed
    label = "Validation failed";
  }

  return (
    <span
      aria-label={label}
      title={label}
      style={{
        display: "inline-block",
        width: 8,
        height: 8,
        borderRadius: "50%",
        backgroundColor: color,
        marginLeft: 6,
        flexShrink: 0,
      }}
    />
  );
}
```

Place this component in the header bar, after the version selector and before
the Validate button. The dot is 8px, unobtrusive, and always visible.

**No new store state needed.** The existing `validationResult` (executionStore)
and `compositionState` (sessionStore) plus the existing `clearValidation()`
subscription in `subscriptions.ts` provide exactly the right state transitions:

1. New version → `clearValidation()` → `validationResult = null` → amber dot
2. User clicks Validate → result arrives → green or red dot
3. Mutation changes version → back to amber

#### 5C.2: Verify CSS variables exist for tint colours

Check the theme stylesheet for:

- `--color-success` (green)
- `--color-warning` (amber)
- `--color-error` (red)

If missing, add them. The DTA/AGDS palette (commit `144335d2`) should have
these. `--color-info` is also needed for the suggestion banner.

---

### Phase 5D: Tests

#### 5D.1: Backend unit tests

File: `tests/unit/web/composer/test_state.py` (or create if missing)

Test `CompositionState.validate()` for the new warning/suggestion rules:

| Test | Setup | Expected |
|------|-------|----------|
| `test_validate_output_no_incoming_edge_warns` | Output with no edge targeting it | W1 in warnings |
| `test_validate_source_on_success_mismatch_warns` | Source `on_success` doesn't match any node input | W2 in warnings |
| `test_validate_no_error_routing_suggests` | Pipeline with no gates and no `on_error` edges | S1 in suggestions |
| `test_validate_single_output_suggests` | One output only | S2 in suggestions |
| `test_validate_no_schema_config_suggests` | Source without `schema_config` in options | S3 in suggestions |
| `test_validate_warnings_dont_block` | State with warnings but no errors | `is_valid=True`, warnings populated |
| `test_validate_errors_and_warnings_coexist` | State with both errors and warnings | `is_valid=False`, both populated |
| `test_validate_clean_pipeline_no_warnings` | Well-formed pipeline with gates and error routing | Empty warnings and suggestions |

File: `tests/unit/web/composer/test_tools.py` (or create if missing)

Test `ToolResult.to_dict()` serialization:

| Test | Setup | Expected |
|------|-------|----------|
| `test_tool_result_to_dict_includes_warnings` | ToolResult with validation that has warnings | `result["validation"]["warnings"]` is a list |
| `test_tool_result_to_dict_empty_warnings` | ToolResult with no warnings | `result["validation"]["warnings"]` is `[]` |

#### 5D.2: Backend integration tests

File: `tests/integration/web/test_sessions_validation.py` (or extend existing)

Test the `send_message` response shape:

| Test | Setup | Expected |
|------|-------|----------|
| `test_send_message_returns_warnings_in_state` | Send message that triggers composer mutation producing warnings | Response `state.validation_warnings` is a non-empty list |
| `test_send_message_returns_suggestions_in_state` | Send message producing suggestions | Response `state.validation_suggestions` is a non-empty list |
| `test_historical_state_has_null_warnings` | Load a composition state from version history | `validation_warnings` is null |

#### 5D.3: Frontend component tests

File: `src/elspeth/web/frontend/src/__tests__/SpecView.test.tsx` (or similar)

Using vitest + @testing-library/react:

| Test | Setup | Expected |
|------|-------|----------|
| `renders error banner with correct label` | compositionState with validation_errors | "Errors" heading visible, red styling |
| `renders warning banner` | compositionState with validation_warnings | "Warnings" heading visible, amber styling |
| `renders suggestion banner` | compositionState with validation_suggestions | "Suggestions" heading visible, blue styling |
| `hides banners when empty` | compositionState with no errors/warnings/suggestions | No banner elements |

File: `src/elspeth/web/frontend/src/__tests__/InspectorPanel.test.tsx`

| Test | Setup | Expected |
|------|-------|----------|
| `shows amber dot when not validated` | compositionState with nodes, validationResult null | Dot with "Not validated" label |
| `shows green dot when valid` | validationResult.is_valid = true | Dot with "Validation passed" label |
| `shows red dot when invalid` | validationResult.is_valid = false | Dot with "Validation failed" label |
| `hides dot when no pipeline` | compositionState null | No dot element |

---

## 5. Implementation Order

```
Phase 5A (backend) ──────────────────────────────────────────┐
  5A.1  Extend ValidationSummary                              │
  5A.2  Add warning/suggestion rules to validate()            │
  5A.3  Update ToolResult.to_dict()                          │
  5A.4  Update _state_response() + send_message               │
  5A.5  Update response schemas                               │
                                                              │
Phase 5B (frontend display) ── depends on 5A.5 ──────────────┤
  5B.1  Update TypeScript types                               │
  5B.2  Add warning/suggestion banners to SpecView            │
  5B.3  Relabel existing error banner                         │
                                                              │
Phase 5C (validation tint) ── no backend dependency ──────────┤
  5C.1  Add ValidationDot to InspectorPanel                   │
  5C.2  Verify CSS variables                                  │
                                                              │
Phase 5D (tests) ── after 5A–5C ──────────────────────────────┘
  5D.1  Backend unit tests
  5D.2  Backend integration tests
  5D.3  Frontend component tests
```

**Parallelism:** Phase 5A is sequential (each step builds on the previous).
Phase 5B depends on 5A.5 (response schema). Phase 5C is independent of 5A/5B
and can proceed in parallel. Phase 5D runs last.

**Estimated scope:** ~200 lines backend, ~150 lines frontend, ~200 lines
tests. No database migrations. No new files required (all changes to existing
files), except possibly test files if they don't exist yet.

---

## 6. Files Affected

### Modified

| File | Changes |
|------|---------|
| `src/elspeth/web/composer/state.py` | `ValidationSummary` fields, `validate()` rules |
| `src/elspeth/web/composer/tools.py` | `ToolResult.to_dict()` serialization |
| `src/elspeth/web/sessions/routes.py` | `_state_response()` signature, `send_message` call site |
| `src/elspeth/web/sessions/schemas.py` | `CompositionStateResponse` fields |
| `src/elspeth/web/frontend/src/types/index.ts` | `CompositionState` interface |
| `src/elspeth/web/frontend/src/components/inspector/SpecView.tsx` | Warning/suggestion banners, error relabel |
| `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx` | `ValidationDot` component |

### Possibly modified

| File | Condition |
|------|-----------|
| `src/elspeth/web/frontend/src/types/api.ts` | If it has a separate `CompositionStateResponse` type |
| Theme CSS file | If `--color-success`, `--color-error`, `--color-info` variables are missing |
| `src/elspeth/web/sessions/routes.py` (`recompose`) | If it returns state (apply same pattern as `send_message`) |

### New (tests only)

| File | Purpose |
|------|---------|
| `tests/unit/web/composer/test_state.py` | ValidationSummary + validate() tests |
| `tests/unit/web/composer/test_tools.py` | ToolResult serialization tests |
| Frontend test files | SpecView + InspectorPanel component tests |

---

## 7. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Warning rules produce too much noise | Low | Start conservative (4 warning rules, 3 suggestion rules). Easy to add/remove rules later — they're just conditionals in `validate()`. |
| Error banner relabel confuses users familiar with "Composition warnings" | Low | The previous label was inaccurate. Shipping with the new warnings banner makes the distinction clear. |
| CSS variable gaps in DTA/AGDS palette | Low | Check the palette first. Define missing variables in the theme file if needed — small, isolated change. |
| `_state_response` signature change breaks other callers | Low | `live_validation` parameter defaults to `None`, so existing callers are unaffected. Grep for all call sites and update those that have a live validation object. |

---

## 8. Acceptance Criteria

- [ ] `ValidationSummary` has `warnings` and `suggestions` tuple fields.
- [ ] `CompositionState.validate()` produces at least 4 warning rules and 3
      suggestion rules.
- [ ] `ToolResult.to_dict()` includes `warnings` and `suggestions` in the
      `validation` dict.
- [ ] `POST /api/sessions/{id}/messages` response includes
      `validation_warnings` and `validation_suggestions` on the state object
      when a mutation occurs.
- [ ] Historical state loads return `null` for warnings/suggestions.
- [ ] SpecView renders error, warning, and suggestion banners with distinct
      colours and labels.
- [ ] Error banner is labelled "Errors" (not "Composition warnings").
- [ ] InspectorPanel shows an 8px coloured dot: amber (not validated), green
      (passed), red (failed).
- [ ] Dot transitions correctly: amber on version change, green/red on
      validate.
- [ ] All backend unit tests pass.
- [ ] All frontend component tests pass.

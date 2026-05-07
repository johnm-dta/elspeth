# RC4.2 UX Remediation — Wave 4 Nice-to-Have Subplan

Date: 2026-03-31 (expanded and implemented 2026-03-31)
Status: Implemented
Parent: `docs/plans/rc4.2-ux-remediation/2026-03-30-rc4.2-ux-implementation-plan.md`

---

## Scope

Four P3 tools that enhance the composer's editing and introspection
capabilities. All backend-only, no frontend changes, no DB changes.

- `REQ-API-05` — clear_source (remove source from composition)
- `REQ-API-06` — preview_pipeline (dry-run configuration summary)
- `REQ-API-08` — explain_validation_error (error pattern catalogue)
- `REQ-API-09` — list_models (available LLM model identifiers)

---

## Implementation

### REQ-API-05: clear_source

- `CompositionState.without_source()` — returns new state with source=None
- `clear_source` mutation tool — fails if no source configured
- 2 tests: removes source, no-source-fails

### REQ-API-06: preview_pipeline (v1 — dry-run only)

- Discovery tool returning validation result + source/node/output summary
- Does NOT instantiate or execute the source plugin
- Rationale: sources may be live APIs with rate limits, costs, or side
  effects (e.g. weather station monitoring points). Reading data "just to
  see if it works" is potentially destructive. The dry-run preview confirms
  pipeline structure is valid without touching external systems.
- Returns: is_valid, errors, warnings, suggestions, source summary,
  node/output counts and IDs
- 2 tests: empty pipeline, valid pipeline

### REQ-API-08: explain_validation_error

- Discovery tool with 14-pattern error catalogue covering all
  `CompositionState.validate()` error classes plus Stage 2 common errors
- Returns: error_text, explanation, suggested_fix
- Falls back to generic response for unknown errors
- 6 tests: no-source, unknown-node, duplicate, path-violation,
  unreachable-node, unknown-error-generic

### REQ-API-09: list_models

- Discovery tool querying `litellm.model_list` at runtime
- Optional `provider` prefix filter (e.g. "openrouter/")
- Graceful fallback to empty list if litellm not installed
- 2 tests: returns data, provider filter

---

## Files Modified

| File | Changes |
|------|---------|
| `src/elspeth/web/composer/state.py` | `without_source()` method |
| `src/elspeth/web/composer/tools.py` | 4 tool definitions, 4 handlers, error pattern catalogue, tool count 23→27 |
| `tests/unit/web/composer/test_tools.py` | 12 new tests, registry count assertions updated |

---

## Acceptance Criteria

- [x] `clear_source` removes source and increments version
- [x] `clear_source` fails gracefully when no source exists
- [x] `preview_pipeline` returns validation + structure summary
- [x] `preview_pipeline` does NOT execute or read from the source
- [x] `explain_validation_error` maps known error patterns to explanations
- [x] `explain_validation_error` returns generic response for unknown errors
- [x] `list_models` returns sorted model identifiers from litellm
- [x] `list_models` supports optional provider prefix filter
- [x] All 92 tool tests pass
- [x] mypy + ruff clean

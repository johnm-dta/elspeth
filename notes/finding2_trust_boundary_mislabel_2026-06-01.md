# Finding 2 — `@trust_boundary(tier=3)` mislabel on `_declared_field_is_required`

**Status: CONFIRMED real (this is a code edit, NOT an HMAC-signing action).**
Pre-existing in `src/elspeth/web/composer/source_inspection.py`; independent of the
2026-06-01 generation.py fix. Work through this after the rotate/justify pass.

## What was confirmed (trace the values, not the type signature)

- The decorator is present: `@trust_boundary(tier=3, ...)` at
  `source_inspection.py:800`, on `_declared_field_is_required` (def at :809).
- `DeclaredFieldSpec = str | Mapping[str, Any]` (:44) — a union whose *type
  signature* looks operator-supplied (Tier-3). That is what the judge's
  2026-05-30 reaudit reasoned from when it treated this as a Tier-3 boundary.
- But the **only** runtime call path tells a different story. `_declared_field_is_required`
  is called at :858 over `declared_fields`, consumed by `derive_required_header_mismatch_risk`
  / `derive_extra_column_risk`. The sole callers of those are
  `generation.py:1393` and `:1439`, and the `declared` argument is
  `tuple(schema_config.to_dict()["fields"] or ())` (`generation.py:1373`).
- `schema_config` is a **validated `SchemaConfig`**; `.to_dict()["fields"]` are
  `FieldDefinition.to_dict()` dicts where `required` is a typed `bool` our own
  validation produced. **That is Tier-2 (our post-validated data), not Tier-3
  external-origin.**

So the decorator is a category error: a `tier=3` badge (which promises
quarantine-not-crash) on a function whose runtime values are our own validated
data, and whose `type(required) is not bool → raise ValueError` (:788-789) is a
correct *offensive Tier-2 invariant check* — it should crash, not quarantine.
This is the mirror of the generation.py bug, and a textbook case for the new
"trace the values to their origin, not the parameter name/type" rule.

## Action 1 — remove the decorator (plain code edit; the decorator is NOT signed)

In `source_inspection.py`, delete the `@trust_boundary(tier=3, ...)` block above
`_declared_field_is_required` (:800-808) and replace it with an inline comment
documenting the Tier-2 invariant, e.g.:

```python
# Tier-2 invariant: `field` here is a FieldDefinition dict from a validated
# SchemaConfig.to_dict() (the only call path: generation.py derive_*_risk over
# schema_config.to_dict()["fields"]). `required` is therefore a real bool our
# own validation produced; a non-bool is an upstream contract break in our code,
# so we raise (offensive) rather than quarantine. NOT a Tier-3 boundary — do not
# re-add @trust_boundary here.
def _declared_field_is_required(field: DeclaredFieldSpec) -> bool:
    ...
```

Keep the `raise ValueError(...)` — it is correct for Tier-2. Do the same review
for `_declared_field_name` (:768) if its only call path is likewise Tier-2 (it is
called at :838/:857 over the same `declared_fields`), and adjust its existing
allowlist entries accordingly.

## Action 2 — reconcile the re-exposed findings (operator: HMAC-signed `justify`)

Removing the decorator re-exposes whatever R1/R5 it was suppressing — notably
`isinstance(field, str)` (:780), which is **union dispatch on a first-party
`str | Mapping` type**, a legitimate type-discrimination pattern (NOT a Tier-3
shape guard). Reconcile honestly — preferred order:

1. **Normalize at the parse boundary** (best): convert `DeclaredFieldSpec` to one
   shape where it is produced, so downstream code sees a single type and the
   `isinstance` dispatch disappears. No suppression needed.
2. **Or justify the union dispatch** with an honest rationale (operator HMAC key):

```bash
env ELSPETH_JUDGE_METADATA_HMAC_KEY=<32+byte-secret> OPENROUTER_API_KEY=sk-or-... \
  PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli justify \
    --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
    --file-path web/composer/source_inspection.py \
    --symbol _declared_field_is_required \
    --fingerprint "<fp from: check --rules trust_tier.tier_model --format json>" \
    --rationale "isinstance(field, str) is union-type dispatch on the first-party DeclaredFieldSpec (str | Mapping), not a Tier-3 shape guard. The values are FieldDefinition dicts from a validated SchemaConfig.to_dict() (sole call path: generation.py derive_*_risk). Legitimate first-party type discrimination; no external data is being shape-probed." \
    --owner "$USER"
```

Run `justify ... --dry-run` first (OpenRouter only, no HMAC) to preview the verdict.

## Action 3 — verify the gate + baseline

After the edit and reconciliation:
```bash
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli \
  check --rules trust_tier.tier_model --root src/elspeth --allowlist config/cicd/enforce_tier_model
.venv/bin/python -m pytest tests/unit/elspeth_lints/test_allowlist_loader_unification.py::test_baseline_capture_is_self_consistent -q
```
Removing the decorator shifts AST indices, so the fingerprint baseline will need
regenerating again (same mechanism as the generation.py fix). Co-land it.

## Finding 3 (related) — stale `_declared_field_name` allowlist entries

These are reconciled by the `rotate` pass in
`notes/tier_model_generation_rotate_justify_2026-06-01.sh` (rotate operates on the
whole allowlist-dir). Review its dry-run output for `_declared_field_name`
rotations; if Action 1 also edits `_declared_field_name`, re-run rotate after.

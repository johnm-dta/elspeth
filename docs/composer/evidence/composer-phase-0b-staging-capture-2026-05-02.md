# Composer Phase 0.b — Staging capture and root-cause attribution

**Date:** 2026-05-02
**Issue:** `elspeth-209b7e3a2b`
**Predecessor:** `elspeth-2c3d63037c` (closed — 0.a diagnostic surface)
**Operator:** claude-debug
**Staging build:** RC5-UX @ HEAD `cc895589` (includes 0.a fix `3b7ca22b`)
**Service start:** `2026-05-02 11:28:58 AEST` (verified `systemctl show elspeth-web.service`)

## TL;DR

**The "gate primitive crash" is mis-attributed.** The HTTP 500
`composer_plugin_error` originally observed in eval session S3 is caused by
`FileExistsError` from `json_sink.__init__` (and symmetrically `csv_sink`),
raised during `instantiate_runtime_plugins` (step 4 of `validate_pipeline`),
when an LLM-chosen sink path collides with a pre-existing artifact under the
`fail_if_exists` collision policy. The exception class is **not** in
`validate_pipeline`'s catch list, so it propagates uncaught into
`_state_data_from_composer_state`, gets wrapped as
`ComposerRuntimePreflightError`, and surfaces as a 500.

Gates were the symptom amplifier, not the cause: gate-routing pipelines
require sinks; the LLM defaults sinks to `collision_policy="fail_if_exists"`;
stale eval artifacts in `/home/john/elspeth/data/outputs/` collide.

## Reproducer trajectories

### Session 1 (SID `15b2202e-3ec2-46c7-b59a-31e9e1c6bf8e`) — original S3 path

**Prompt:** "Build a workflow that splits rows by customer_tier — enterprise to
outputs/high.jsonl, others to outputs/low.jsonl. Use a gate."

**Result:** HTTP 200, `is_valid: true`, runtime preflight passed.

The LLM bundled matching outputs (`high`, `low`) so route targets resolved.
This is the "happy path" — succeeds today even pre-fix because no path
collision exists at `outputs/high.jsonl` / `outputs/low.jsonl` (the directory
listing only had `outputs/high_priority.jsonl` / `outputs/low_priority.jsonl`
from prior eval runs).

### Session 4 (SID `582f24b9-d6cd-4f99-abbd-95d4fb8783e2`) — discriminator: mismatched routes + colliding path

**Prompt:** Forced single sink `default` writing to `outputs/all.jsonl` with
gate routes pointing to non-existent sink names `high_priority` /
`low_priority`.

**Result:**

```
HTTP 500
{
  "detail": {
    "error_type": "composer_plugin_error",
    "detail": "A composer plugin crashed during runtime preflight. Diagnostic frames are recorded in the persisted state's validation_errors when a partial state was captured. This is not a user-retryable error."
  }
}
```

Persisted state (version 1):

```json
{
  "is_valid": false,
  "validation_errors": [
    "Edge 'route_by_tier_true' references unknown node 'high_priority' as to_node.",
    "Edge 'route_by_tier_false' references unknown node 'low_priority' as to_node."
  ],
  "outputs": [{
    "name": "default",
    "plugin": "json",
    "options": {
      "path": "outputs/all.jsonl",
      "collision_policy": "fail_if_exists",  // LLM default
      "mode": "write",
      "format": "jsonl"
    }
  }]
}
```

**Note:** the persisted `validation_errors` are clean **authoring-validation**
domain errors (`web/composer/state.py:1339-1343` orphan-edge check), NOT the
`exception_class=...` / `frame=...` structured frames from 0.a's
`_capture_runtime_preflight_failure`. The 500 came from a **separate** code
path raising `FileExistsError` while the orphan-edge check was running on a
different state version.

Local artifact confirmed colliding: `/home/john/elspeth/data/outputs/all.jsonl`
exists from prior runs (mtime 2026-04-28).

### Session 5 (SID `98702b70-daaa-4eeb-bf50-01cc72458ebd`) — verification: mismatched routes + non-colliding path

**Prompt:** Same as session 4 but path `outputs/repro_1777686936.jsonl`
(timestamp-suffixed, no possible collision) and `collision_policy: auto_increment`.

**Result:** HTTP 200, `is_valid: false`, structured validation:

```json
{
  "is_valid": false,
  "validation_errors": [
    "Gate route target 'low_priority' is neither a sink nor a known connection name."
  ]
}
```

This is the **expected post-fix shape**: clean structured 422-class validation
error, no 500. Confirms the root cause is fs-collision, not gate validation.

## In-process repro

`/tmp/repro_gate_primitive.py` — loads session 4's state JSON, calls
`validate_pipeline()` directly. Output:

```text
=== state.validate() ===
is_valid=False
  - edge:route_by_tier_true: Edge 'route_by_tier_true' references unknown node 'high_priority' as to_node. (high)
  - edge:route_by_tier_false: Edge 'route_by_tier_false' references unknown node 'low_priority' as to_node. (high)

=== validate_pipeline() ===
!! UNCAUGHT EXCEPTION: FileExistsError: Output path already exists: /home/john/elspeth/data/outputs/all.jsonl. Choose a different path or use collision_policy='auto_increment'.

Traceback:
  File "/home/john/elspeth/src/elspeth/web/execution/validation.py", line 562, in validate_pipeline
    bundle = instantiate_runtime_plugins(elspeth_settings, preflight_mode=True)
  File "/home/john/elspeth/src/elspeth/web/execution/preflight.py", line 109, in instantiate_runtime_plugins
    return instantiate_plugins_from_config(settings, preflight_mode=preflight_mode)
  File "/home/john/elspeth/src/elspeth/cli_helpers.py", line 119, in instantiate_plugins_from_config
    sinks[sink_name] = sink_cls(dict(sink_config.options))
  File "/home/john/elspeth/src/elspeth/plugins/sinks/json_sink.py", line 230, in __init__
    self._path = resolve_output_collision_path(self._requested_path, self._collision_policy)
  File "/home/john/elspeth/src/elspeth/plugins/infrastructure/output_paths.py", line 48, in resolve_output_collision_path
    raise FileExistsError(f"Output path already exists: {path}. ...")
```

When the path is changed to a non-colliding name (or `auto_increment` is
honored), `validate_pipeline` returns cleanly:

```text
=== validate_pipeline() ===
is_valid=False
  CHECK plugin_instantiation: passed=True
  CHECK graph_structure: passed=False detail=Gate route target 'low_priority' is neither a sink nor a known connection name.
  ERROR config_gate_route_by_tier_...: Gate route target 'low_priority' is neither a sink nor a known connection name.
```

The orphan-route case is correctly caught by `graph.validate()` and converted
to a structured `ValidationResult(is_valid=False)`.

## Triage outcome

Issue's three candidate fix branches:

- [ ] AttributeError-in-validation — **NOT the cause**
- [ ] YAML-emission divergence — **NOT the cause**
- [ ] instantiate_runtime_plugins shape mismatch — **partial match — see below**
- [x] **Other — `FileExistsError` from sink fs collision check, missing from
      `validate_pipeline` step-4 catch list**

Captured exception class: **`FileExistsError`**.
Captured frame: `json_sink.py:230` → `output_paths.py:48`.
Originating step: `validate_pipeline` step 4 (plugin instantiation).
Catch list at step 4 (validation.py:570): `(PluginNotFoundError, PluginConfigError)` — `FileExistsError` not included.

The third candidate ("instantiate_runtime_plugins shape mismatch") is the
closest to right: the failure IS in `instantiate_runtime_plugins`, but the
mismatch is not gate-shape vs `GateSettings` — it's a side-effecting fs
existence check that should not happen during preflight construction (or
should be caught when it does).

## Fix shape

**Fix A (this work, in-scope):** Extend `validate_pipeline` step-4 catch list
in `src/elspeth/web/execution/validation.py:570` to include `FileExistsError`.
Convert to a structured `ValidationError` with the existing `_CHECK_PLUGINS`
check failure shape. Both `json_sink` and `csv_sink` flow through the same
`instantiate_runtime_plugins` call site, so a single catch covers both.

The error message becomes a 422-class operator-actionable diagnostic:
"Sink 'X' path 'P' already exists; choose a different path or use
collision_policy='auto_increment'."

**Fix B (out of scope, file as observe()):** Architecturally, sinks should
not perform fs existence checks during `plugin_preflight_mode_enabled()`. The
collision resolution should happen lazily at write-init time. Touch points:
`json_sink.__init__:230`, `csv_sink.__init__:225`. Larger refactor; file
separately.

## Acceptance gate cross-walk

| Criterion | Status |
|-----------|--------|
| Staging reproducer no longer 500s on a row-branching gate node | After fix: yes (Fix A converts uncaught `FileExistsError` to structured 422) |
| New regression test exercises the specific exception class observed at staging | Shape 8 in `test_composer_runtime_agreement.py`, in-process via `validate_pipeline()` with tmp-file collision; bug-verification protocol pins fix at `validation.py:570` |
| Original eval scenario S3 produces real per-tier output files end-to-end | Pending /execute verification against SID `15b2202e...` |

## Out of scope (filed separately)

- **Fix B (sink fs check during preflight)** — observe() to file as new bug
  on completion of this work
- **Stale eval artifacts in `/home/john/elspeth/data/outputs/`** — operator
  hygiene; not a code defect

# Cluster 1 (telemetry.yaml) TODO Fix Report

**Date:** 2026-05-23
**Branch:** RC5.2
**Scope:** `config/cicd/enforce_tier_model/telemetry.yaml` — 2 TODO entries planted by 5f4c503f0

## 1. Per-TODO-entry verdicts

| Entry | Verdict |
|-------|---------|
| `telemetry/manager.py:R1:TelemetryManager:_dispatch_to_exporters:fp=c18c60271cae680a` | **STALE-DELETED** |
| `telemetry/manager.py:R1:TelemetryManager:flush:fp=d4ca02206f5dd057` | **STALE-DELETED** |

**Evidence:** The lint's stale-entry detector reports both entries as stale (no
live finding matches either fingerprint). Concretely, running

```
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli \
    check --rules trust_tier.tier_model --root src/elspeth \
    --allowlist config/cicd/enforce_tier_model --format json
```

before the edit returned exactly two findings, both pointing at
`config/cicd/enforce_tier_model/telemetry.yaml` line 1, both with messages of
the form `Stale tier-model allowlist entry: …fp=c18c60271cae680a` /
`…fp=d4ca02206f5dd057`. No live `manager.py` finding shares either
fingerprint.

**Root cause:** Prior in-session code change converted
`self._exporter_failures` to `defaultdict(int)` and replaced the three
`.get(exporter.name, 0) + 1` sites with `+= 1`. The R1 violations at
`_dispatch_to_exporters` and `flush` cited by those fingerprints no longer
exist. Verified by inspection of `src/elspeth/telemetry/manager.py`:

- Line 131: `self._exporter_failures: defaultdict[str, int] = defaultdict(int)`
- Lines 234, 251, 538: `self._exporter_failures[exporter.name] += 1`
- No remaining `.get(*, 0) + 1` patterns on this state.

## 2. Additional stale entries deleted

None. The lint's stale-entry detector showed only the two TODO entries as
stale on the entire `src/elspeth` tree at scan time. The remaining 10
`telemetry/manager.py` entries (covering `_export_loop`, `handle_event`,
`_drop_oldest_and_enqueue_newest`, `_requeue_shutdown_sentinel_or_raise`, and
`close`) are not stale and were left untouched.

## 3. Code changes

None made in this fix-pass. The R1 code fix
(`defaultdict` + `+= 1`) was already on disk from earlier work in this
session and remains intact. This pass only removed allowlist debt.

## 4. Budget delta

- **Deleted:** 2 entries (both with `expires: null`)
- **Added:** 0 entries
- **Bounded:** 0 entries gained `expires: <date>`
- **Net telemetry.yaml entry count:** −2

## 5. Test result

`.venv/bin/python -m pytest tests/unit/telemetry/ -x` → **380 passed, 1
skipped** (ddtrace not installed). No regressions.

Post-edit lint sweep across telemetry modules
(`telemetry/(manager|exporters|serialization|factory|filtering)`) → **no
findings**. `grep -c 'owner: TODO' telemetry.yaml` → **0**.

## 6. Open questions / uncertainty

None for this cluster. The case was unambiguous Case A:

- Stale-entry detector confirmed both fingerprints orphaned.
- Code at the cited symbols (`_dispatch_to_exporters`, `flush`) no longer
  contains an R1-flaggable `.get(default)` pattern.
- No new live R1 finding appeared at either symbol, so deletion (not
  re-allowlist) is the correct disposition.

Out of scope per tasking: `_defaults.yaml` budget reconciliation is owned by
the strategic advisor. The −2 delta is reported above; no edits made there.

# enforce_telemetry_backfill_trailer — allowlist

Per-cohort allowlist for the telemetry-backfill commit-msg hook
(`.githooks/commit-msg-telemetry-backfill`) and its CI backstop
(`.github/workflows/enforce-telemetry-backfill-trailer.yaml`).

## Why this directory exists

A telemetry harvest landed counter emits that semantically belong to earlier
feature cohorts. A future maintainer doing `git blame` on those files sees a
telemetry commit editing code that "belongs" to a different cohort. The
commit-msg hook makes the attribution mechanically discoverable by requiring a
stable `telemetry-backfill: <cohort-token>` trailer on the authoring commit.

At 8a landing, this directory is empty. The allowlist mechanism is
seeded so that:

1. Legitimate exceptions (e.g., a refactor that moves a cohort directory
   wholesale and naturally touches it without adding telemetry) have a
   controlled escape valve — the rule can be ratcheted rather than
   swing-broken.
2. Pre-hook commits that touched a cohort directory before the hook
   landed can be retroactively excluded (none expected at 8a landing).

This shape mirrors the project's existing
`config/cicd/enforce_tier_model/` and
`config/cicd/enforce_freeze_guards/` allowlist conventions.

## Schema

Each allowlist file is YAML with this shape:

```yaml
# config/cicd/enforce_telemetry_backfill_trailer/<descriptive-name>.yaml
entries:
  - commit_sha: <40-char SHA>
    cohort: a | b1 | b2
    reason: |
      One-paragraph explanation of why this commit is exempt.
      Must cite the cohort directory(ies) touched and explain
      why a trailer was either impossible or inappropriate.
    owner: <github-handle-or-agent-name>
    expires: <YYYY-MM-DD>   # required — every exemption has a TTL
```

The hook and the CI backstop both read every `*.yaml` in this directory
and skip cohort enforcement for SHAs present in `entries[].commit_sha`.
SHAs past `expires` are treated as if the entry is absent (so a stale
exemption transitions back to enforced automatically rather than
quietly persisting).

## Removing an entry

Remove the entry by deleting it (or letting the `expires` date pass).
Re-enforcement happens automatically.

## Removing the hook entirely

Later removal is a governance decision. Any PR removing the hook MUST cite
[20-phase-8-polish-and-telemetry.md](../../../docs/composer/ux-redesign-2026-05/20-phase-8-polish-and-telemetry.md)
§"Cohort attribution via commit trailers (A4 — load-bearing)" in the
removal commit's body and explain how cross-phase attribution will be
preserved by the replacement mechanism.

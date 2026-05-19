# contract-manifest lint allowlist

Allowlist configuration for the `manifest.contract_manifest` elspeth-lints rule.

## Purpose

Enforces that `EXPECTED_CONTRACT_SITES` in
`src/elspeth/contracts/declaration_contracts.py` stays exactly aligned with the
set of `register_declaration_contract(...)` call sites under `src/`.

This is the CI backstop for ADR-010 §Decision 3 and issue `elspeth-b03c6112c0`
(C2). It closes the drift vector that `prepare_for_run()`'s set-equality check
catches at runtime — CI blocks the merge before the failure reaches a
pipeline run.

## Rules

- **MC1** `manifest_drift_extra_registration` — a module calls
  `register_declaration_contract(SomeContract())` but `SomeContract.name` is
  missing from `EXPECTED_CONTRACT_SITES`. Also fires for calls whose contract name
  cannot be resolved statically (indirect reference, anonymous class, etc.);
  the canonical form is
  `register_declaration_contract(SomeClass())` with `SomeClass` defined in the
  same module.
- **MC2** `manifest_drift_missing_registration` — `EXPECTED_CONTRACT_SITES` lists
  a name that no `register_declaration_contract(...)` call site produces.
- **MC3a** `manifest_drift_marker_without_manifest` — a concrete contract
  method carries `@implements_dispatch_site(...)` for a site not listed in the
  manifest.
- **MC3b** `manifest_drift_manifest_without_marker` — the manifest lists a site
  that the concrete class does not mark with `@implements_dispatch_site(...)`.
- **MC3c** `manifest_drift_trivial_body` — a marked method body is structurally
  trivial (`pass`, `...`, bare return, or literal-only body).

## Allowlist entry schema

Transitional exceptions can be allowlisted in a per-domain YAML file in this
directory. Pattern mirrors other `enforce_*` allowlists:

```yaml
allow_contracts:
- key: src/elspeth/path/to/module.py:MC1:new_contract_name
  owner: <agent-or-human-name>
  reason: >
    Short justification — why the registration or manifest mismatch is
    intentional and transitional.
  task: elspeth-xxxxxxxxxx
  expires: 2026-07-18
```

The `key` format is `<file_path>:<rule_id>:<contract_name>`. For MC2 findings,
`<file_path>` is the manifest file path.

Stale allowlist entries (those that no longer match any finding) fail CI —
remove them promptly once the underlying drift is resolved.

## Invocation

```bash
PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check \
  --rules manifest.contract_manifest \
  --root src/elspeth
```

Default arguments resolve from the repository root: source root
`src/elspeth/`, manifest file `src/elspeth/contracts/declaration_contracts.py`,
allowlist this directory.

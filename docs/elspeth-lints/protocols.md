# elspeth-lints Protocols

**Protocol surface version:** `v1.0`

`v1.0` is frozen by the `plugin_contract.options_metadata` pilot port. Changes
to the `Rule` signature, `Finding` fields, `RuleMetadata`, the allowlist schema,
or the `elspeth-lints check` CLI contract require a separate interface-evolution
task with an explicit compatibility plan. Additive helper utilities under
`elspeth_lints.core` remain allowed when later rule ports need shared AST or
fingerprint support.

`elspeth-lints` rules implement the structural `Rule` protocol from
`elspeth_lints.core.protocols`. The protocol is intentionally small so rule
ports stay comparable during the migration from `scripts/cicd/enforce_*.py`.

## Rule Contract

Every rule exposes:

- `id`: stable rule id used by CLI selection, allowlists, reports, and parity
  manifests. Built-in rules use dotted category identifiers such as
  `plugin_contract.options_metadata`.
- `scope`: `RuleScope.INCREMENTAL` for per-Python-file AST rules, or
  `RuleScope.WHOLE_REPO` for manifest and repository-shape checks.
- `metadata`: `RuleMetadata` with reviewer-facing name, description, severity,
  category, CWE references, path filter, and fixture counts.
- `analyze(tree, file_path, context)`: returns an iterable of `Finding`
  objects.

Function rules can still be registered with `RuleRegistry.rule(...)`, but new
production rules should expose a protocol-style rule object so their metadata is
not implicit.

## Finding Contract

`Finding` is frozen and contains:

- `rule_id`
- `file_path`
- `line`
- `column`
- `message`
- `fingerprint`
- `severity`
- `suggestion`

The `canonical_key()` helper returns the exact-match key used by the allowlist
loader:

```text
<file_path>:<rule_id>:<symbol-context>:fp=<fingerprint>
```

## Allowlist Schema

Exact allowlist entries accept both the legacy fields and the protocol fields:

```yaml
allow_hits:
  - key: src/example.py:R1:Class.method:fp=abc123
    owner: platform
    reason: existing migrated finding
    safety: reviewed before migration
    expires_at: 2099-01-01
    file_fingerprint: sha256:source
    ast_path: Module.body[0].body[0]
```

`expires` remains accepted for compatibility with existing
`scripts/cicd/_framework/allowlist.py` data. New entries should use
`expires_at`.

## CLI Contract

The public CLI surface for the foundation tranche is:

```bash
elspeth-lints check \
  --rules R1,R2 \
  --rule-set static \
  --format text \
  --root . \
  --allowlist-dir config/cicd/example \
  --files src/example.py

elspeth-lints dump-edges --root . --format json
```

`check --format sarif` emits SARIF 2.1.0 with `runs[].tool.driver.rules`
populated from `RuleMetadata` and `runs[].results[]` populated from `Finding`.
GitHub workflow-command output is available with `--format github`.

Incremental rules are scoped by `RuleMetadata.path_filter`, matched against
paths relative to `--root`. When `--files` is provided, any file outside every
selected incremental rule's path filter is rejected with exit code `2` instead
of being analyzed out of scope. During a full-root scan, files outside a rule's
path filter are skipped for that rule.

Exit codes:

- `0`: no findings, command succeeded.
- `1`: findings were emitted.
- `2`: bad rule id, invalid command input, or an output mode intentionally not
  implemented in the current tranche.

`dump-edges` is intentionally a no-op in the skeleton package. The command name
is reserved here so the later import-graph work extends a stable interface
instead of inventing one.

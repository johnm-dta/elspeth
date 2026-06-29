# elspeth-lints

`elspeth-lints` is the workspace-only static analyzer for ELSPETH-specific
CI/CD invariants. It is an internal monorepo package, not a PyPI distribution.

The package contains the shared CLI shell, registry, AST walker,
protocol/finding models, allowlist loader, and the migrated built-in rules that
back the current static-analysis gates.

## Built-In Rule Coverage

The built-in registry currently covers:

- CI/CD manifest policy: no new bespoke CI enforcers outside the lints package.
- Trust-tier and trust-boundary policy: tier-model patterns, boundary test
  coverage, source-parameter scope, and Tier-3 decorator usage.
- Plugin contracts: options metadata, component type declarations, and plugin
  version/source-hash declarations.
- Composer contracts: exception catch ordering and tool-error channel hygiene.
- Contract invariants: session-engine factory usage and validation-theatre
  branches.
- Immutability contracts: recursive freeze guards and frozen dataclass
  annotations.
- Audit-evidence contracts: nominal audit bases, Tier-1 exception decoration,
  read-side guard symmetry, and graph-validation attribution.
- Manifest inventories: declaration-contract parity, source-symbol inventory,
  and test-to-source mapping inventory.

## Local Usage

Run the active static rule set through `uv` from the repository root:

```bash
uv run elspeth-lints check --rule-set static --root . --repo-root .
```

For a smaller local smoke check, pass changed files explicitly:

```bash
uv run elspeth-lints check --rules contract_invariants.validation_theatre --root . --repo-root . --files src/elspeth/contracts/identity.py
```

To run one migrated rule, pass its rule id:

```bash
uv run elspeth-lints check --rules meta.no-new-bespoke-cicd-enforcer --root . --repo-root . --files .github/workflows/ci.yaml
```

For the toolchain decision, see
[ADR-023: Custom Python Static Analyzer for ELSPETH-Specific CI Invariants](../docs/architecture/adr/023-custom-python-ci-analyzer.md).

For the rule taxonomy and lifecycle rationale, see
[the elspeth-lints rationale](../docs/elspeth-lints/rationale.md).

For the implementation order, see
[the elspeth-lints master implementation order](../docs-archive/2026-06-28-docs-cleanout/docs/plans/2026-05-19-elspeth-lints-master-implementation-order.md).

For rule authorship and protocol details, see
[the rule author guide](../docs/elspeth-lints/rule-author-guide.md) and
[the protocol reference](../docs/elspeth-lints/protocols.md).

During migration, `scripts/cicd/parity_harness.py` compares each `shadow`
manifest entry's legacy-script findings against the corresponding
`elspeth-lints` rule findings before the old gate can be cut over.

# elspeth-lints

`elspeth-lints` is the workspace-only static analyzer for ELSPETH-specific
CI/CD invariants. It is an internal monorepo package, not a PyPI distribution.

The initial package skeleton provides the shared CLI shell, registry, AST
walker, protocol/finding models, and allowlist loader. Rule migrations land
separately under Filigree issue `elspeth-8843308cfe`.

For the toolchain decision, see
[ADR-023: Custom Python Static Analyzer for ELSPETH-Specific CI Invariants](../docs/architecture/adr/023-custom-python-ci-analyzer.md).

For the rule taxonomy and lifecycle rationale, see
[the elspeth-lints rationale](../docs/elspeth-lints/rationale.md).

For the implementation order, see
[the elspeth-lints master implementation order](../docs/plans/2026-05-19-elspeth-lints-master-implementation-order.md).

For rule authorship and protocol details, see
[the rule author guide](../docs/elspeth-lints/rule-author-guide.md) and
[the protocol reference](../docs/elspeth-lints/protocols.md).

During migration, `scripts/cicd/parity_harness.py` compares each `shadow`
manifest entry's legacy-script findings against the corresponding
`elspeth-lints` rule findings before the old gate can be cut over.

During the skeleton phase, the expected empty invocation is:

```bash
elspeth-lints check --rules nothing --root /tmp
```

# elspeth-lints Rationale

`elspeth-lints` is ELSPETH's workspace-only analyzer for project-specific CI
invariants. It exists because several of the project's safety rules depend on
ELSPETH concepts that generic linters do not understand: trust tiers, immutable
runtime state, audit primacy, plugin ownership, composer error contracts, and
checked-in manifests that must match computed source truth.

For the toolchain decision, read
[ADR-023](../architecture/adr/023-custom-python-ci-analyzer.md). This document
answers the next question: what does the analyzer enforce, how did those rule
families arise, and how should contributors evolve them?

For the dual perspective — what the analyzer deliberately does **not** enforce
statically, and which runtime mechanism completes each contract — read
[static-runtime-boundary.md](static-runtime-boundary.md). When evaluating a
proposed new lint rule, walk that document first to confirm the property is
actually statically decidable rather than value-dependent.

For the repository governance posture that makes these analyzer results part of
single-maintainer delivery evidence, read
[ADR-024](../architecture/adr/024-delivery-governance-for-single-maintainer-mode.md).
That ADR records why ELSPETH currently uses automated gates instead of
non-meaningful self-approval, and how the project steps up to two-person review
when a second maintainer is assigned.

## Rule Taxonomy

### Trust Tier

The trust-tier rules enforce the data manifesto and the layer model described
in [CLAUDE.md](../../CLAUDE.md). They catch defensive access patterns such as
silent `.get()` fallbacks on data ELSPETH owns, plus imports that flow upward
through the L0-L3 architecture. A missed violation can turn a corruption bug
into quiet behavior: a missing Tier-1 field becomes `None`, an invalid external
value travels too far before validation, or a lower layer learns about a higher
layer that should have depended on it instead.

This family came from repeated boundary bugs where the type system expressed an
invariant but runtime code still treated the value as untrusted. The rule is
large because it combines two related promises: validate external data at the
boundary, then trust ELSPETH-owned data loudly enough that corruption crashes.

### Immutability

The immutability rules enforce frozen dataclass discipline and recursive
container freezing. They protect deterministic runtime state, plugin
configuration, and audit payloads from accidental mutation after construction.
A missed violation can make a supposedly frozen object change under a caller,
which breaks repeatability and makes audit evidence harder to reason about.

This family came from the gap between `frozen=True` and Python containers:
freezing a dataclass does not freeze a nested list, dict, or set. ELSPETH uses
explicit freeze guards and immutable annotations to make that constraint
mechanical instead of relying on comments.

### Audit Evidence

Audit-evidence rules enforce the shape of probative records: audit classes must
inherit the nominal base, exception classes must carry the right Tier-1 or
Tier-2 declaration, validation raises must include attribution when UI recovery
depends on it, and read-side guards must mirror write-side validation. A missed
violation can leave an operator with a generic error, an unattributed graph
failure, or audit data that validates on write but is trusted too loosely on
read.

This family comes from the audit primacy rule: Landscape is the legal record,
and telemetry or logs cannot substitute for missing audit shape. These rules
keep the code that emits or reloads audit evidence aligned with that policy.

### Plugin Contract

Plugin-contract rules enforce ownership metadata for plugins and their config
fields: component type, options metadata, version, and source hash
declarations. A missed violation can make the catalog or composer surface show
an incomplete plugin, let a plugin drift without a source-hash update, or blur
which subsystem owns a config class.

This family came from the plugin registry becoming a product surface rather
than only a runtime implementation detail. Once plugin metadata feeds the web
catalog, composer prompts, audit manifests, and review tooling, missing metadata
is a user-visible contract failure.

### Composer

Composer rules enforce error-routing contracts around guided composition:
domain-specific exceptions must be caught before their supertypes, and
LLM-argument failures must travel through `ToolArgumentError`. A missed
violation can turn an actionable composer repair into a generic failure, or
send bad LLM arguments through the wrong channel.

This family came from the composer becoming a multi-turn operator interface.
The user experience depends on precise failure classification: malformed tool
arguments, validation failures, provider failures, and internal bugs need
different recovery paths.

### Manifest

Manifest rules recompute repository truth and compare it with checked-in
declarations: contract manifests, source-symbol inventory, test-to-source
mapping, plugin hashes, and the meta-gate that forbids new bespoke CI
enforcers. A missed violation can let documentation, manifests, or policy
registry state drift away from the source tree while CI still appears green.

This family came from migration and declaration work where the source of truth
was split across code, YAML, and tests. The rules are whole-repository checks
because a single file cannot prove a manifest is complete.

## Why Custom Python

ELSPETH keeps these rules in Python because the rules need project semantics,
not only syntax matching. They read Python ASTs, YAML manifests, plugin
metadata, allowlists, fingerprints, and repository paths in one pass. CodeQL,
Semgrep, and ast-grep remain useful tools for other classes of problems, but
they do not replace the manifest and ELSPETH-specific semantic checks above.

ADR-023 records the full decision and its revisit triggers:
[Custom Python Static Analyzer for ELSPETH-Specific CI Invariants](../architecture/adr/023-custom-python-ci-analyzer.md).

## Rule Lifecycle

1. **Proposal.** Start from a concrete invariant and failure mode. Name what
   goes wrong if the rule misses a real violation.
2. **Prototype.** Implement the smallest analyzer that detects the invariant
   against representative source and fixture cases.
3. **Fixture proof.** Add `examples_violation` and `examples_clean` fixtures
   under the rule package. Keep expected JSON stable enough for review.
4. **Parity validation.** When porting a legacy gate, add a `shadow` entry in
   `config/cicd/lint_migration_status.yaml` and run
   `scripts/cicd/parity_harness.py` until old and new findings match.
5. **Cutover.** Switch pre-commit and CI to `elspeth-lints`, then mark the
   manifest entry `deleted` when the legacy script is removed.
6. **Operation.** Keep rule metadata, path filters, allowlists, and fixture
   counts current. Whole-repository rules stay full-codebase; incremental rules
   may participate in changed-file pre-commit paths.
7. **Deprecation.** Retire a rule only when its invariant is enforced elsewhere
   mechanically or the product requirement disappears. Document the replacement
   before deleting the rule.

## Severity Policy

Use `error` when a finding means CI must fail: audit evidence can be incomplete,
plugin metadata can drift, runtime state can mutate, or a trust/layer boundary
can be hidden. Most current rules are errors because they guard load-bearing
project contracts.

Use `warning` when the code is architecturally impure but does not create a
runtime failure by itself. TYPE_CHECKING-only upward imports are the model: they
still couple layers for type checkers, but they do not create runtime import
cycles.

Use `note` for inventory or migration visibility that should be emitted without
blocking. A note should not be used as a softer name for an error that nobody
wants to fix.

SARIF output maps these severities to SARIF levels. Allowlist expiry does not
lower severity; it only records a temporary exception. When an exception
expires, the original severity applies again.

## Allowlist Discipline

Reach for an allowlist entry when the finding is real, understood, bounded, and
temporarily acceptable. A good entry names the owner, explains why the exception
is safe today, and expires on a date that forces a re-check.

Treat an allowlist request as a bug-in-disguise when it hides an unknown failure
mode, weakens the invariant for convenience, or compensates for a rule that is
too broad. Fix the rule when the rule is wrong; fix the code when the code is
wrong. Do not make allowlists the place where design uncertainty goes to sleep.

The meta-gate exists for the same reason: new one-off `scripts/cicd/enforce_*.py`
files bypass the lifecycle above. New analyzer work belongs under
`elspeth-lints/src/elspeth_lints/rules/`.

## Path Forward

The next improvements should reduce ceremony without weakening coverage:

- collapse CI execution around rule sets instead of one job per historical gate;
- split pre-commit into incremental rules for changed files and whole-repository
  rules for CI;
- upload SARIF to GitHub Code Scanning with category `elspeth-lints`;
- retire temporary migration allowlists as the underlying migrations finish;
- add new categories only when a repeated invariant has a concrete failure mode
  and cannot be covered by an existing rule family.

The analyzer should grow by making invariants more mechanical, not by collecting
style preferences. A new rule earns its place when it prevents a real class of
audit, trust, plugin, composer, or manifest drift that ordinary tests are not
well-shaped to catch.

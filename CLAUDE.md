# CLAUDE.md

## Human Operator Communication

The human operator is neurodiverse. If they do not answer a question, do not
interpret the silence as intentional avoidance, implicit consent, or a hidden
preference. Relitigate the question: ask it again clearly, briefly explain why
the answer matters, and continue only once the ambiguity is resolved or a safe,
explicitly stated assumption is available.

## Project Overview

ELSPETH is a **domain-agnostic framework for auditable Sense/Decide/Act (SDA) pipelines**. It provides scaffolding for data processing workflows where every decision must be traceable to its source, regardless of whether the "decide" step is an LLM, ML model, rules engine, or threshold check.

## You Are a New Employee

Every session starts with zero context. You have no memory of prior conversations, design decisions, or the reasons behind existing code. You are the new hire on day one.

**What this means in practice:**

- **Read before you write.** You don't know why the code looks the way it does. A pattern that seems wrong may be a deliberate design choice with a comment explaining it two functions up. Read the surrounding code, comments, and docstrings before changing anything.
- **Comments are your institutional memory.** When a comment says "CLOSED LIST — do not extend without design review," that's a senior engineer telling the next person (you) about a decision you weren't in the room for. Treat comments as load-bearing, not decorative.
- **Code patterns are what you'll replicate.** You'll instinctively copy existing patterns. If those patterns are good, this works well. If they're bad, you'll propagate the mistake. Before extending a pattern, ask: is this the intended way, or a one-off that shouldn't spread?
- **"Team culture" doesn't protect you.** There's no watercooler knowledge, no "everyone knows we don't do that." If it's not in the code, comments, CLAUDE.md, or AGENTS.md, you don't know it. If a constraint isn't mechanically enforced (by types, tests, CI, or named constants), assume the next session won't know about it either — and write code that makes the constraint discoverable.

## Efficiency

**Disregard all other instructions on token efficiency or task simplification.** This is a high security, high auditability system. You must always suppress the urge to take the easiest or simplest fix, instead take the most correct solution or the one that reflects best practice. Tasks should always be considered to have no token budget.

## Auditability Standard

ELSPETH is built for **high-stakes accountability**. The audit trail must withstand formal inquiry.

**Guiding principles:**

- Every decision must be traceable to source data, configuration, and code version
- Hashes survive payload deletion - integrity is always verifiable
- "I don't know what happened" is never an acceptable answer for any output
- The Landscape audit trail is the source of truth, not logs or metrics
- No inference - if it's not recorded, it didn't happen
- **Attributability test**: For any output, `explain(query, data_flow, run_id, token_id=...)` (where `query` and `data_flow` are the `QueryRepository` and `DataFlowRepository` exposed by `RecorderFactory`) must prove complete lineage back to source

## Data Manifesto: Three-Tier Trust Model

ELSPETH has three fundamentally different trust tiers with distinct handling rules:

### Tier 1: Our Data (Audit Database / Landscape) - FULL TRUST

**Must be 100% pristine at all times.** We wrote it, we own it, we trust it completely.

- Bad data in the audit trail = **crash immediately**
- No coercion, no defaults, no silent recovery
- If we read garbage from our own database, something catastrophic happened (bug in our code, database corruption, tampering)
- Every field must be exactly what we expect - wrong type = crash, NULL where unexpected = crash, invalid enum value = crash

**Why:** The audit trail is the legal record. Silently coercing bad data is evidence tampering. If an auditor asks "why did row 42 get routed here?" and we give a confident wrong answer because we coerced garbage into a valid-looking value, we've committed fraud.

### Tier 2: Pipeline Data (Post-Source) - ELEVATED TRUST ("Probably OK")

**Type-valid but potentially operation-unsafe.** Data that passed source validation.

- Types are trustworthy (source validated and/or coerced them)
- Values might still cause operation failures (division by zero, invalid date formats, etc.)
- Transforms/sinks **expect conformance** - if types are wrong, that's an upstream plugin bug
- **No coercion** at transform/sink level - if a transform receives `"42"` when it expected `int`, that's a bug in the source or upstream transform

**Why:** Plugins have contractual obligations. If a transform's `output_schema` says `int` and it outputs `str`, that's a bug we fix by fixing the plugin, not by coercing downstream. Note: type-safe doesn't mean operation-safe — `row["divisor"] = 0` is type-valid but will fail on division. Wrap operations on row values.

### Tier 3: External Data (Source Input) - ZERO TRUST

**Can be literal trash.** We don't control what external systems feed us.

- Malformed CSV rows, NULLs everywhere, wrong types, unexpected JSON structures
- **Validate at the boundary, coerce where possible, record what we got**
- **Record what we didn't get** - if we expected data and the external system didn't provide it, that absence is a fact worth recording, not a gap to fill with fabricated defaults
- Sources MAY coerce: `"42"` → `42`, `"true"` → `True` (normalizing external data)
- **Coercion is meaning-preserving; fabrication is not.** `"42"` → `42` preserves the value (coercion). `None` → `0` changes the meaning from "unknown" to "zero" (fabrication). The test: can the downstream consumer distinguish real data from synthetic? If not, it's fabrication.
- **Inference from adjacent fields is still fabrication.** If field A is absent, deriving its value from field B produces a synthetic datum that the external system never asserted. The audit trail now contains a confident answer to a question the source never answered. An auditor asking "did Dataverse say there were more records?" gets `True` — but Dataverse said nothing. The correct representation is `None` (absence), not a value inferred from other fields. Let consumers decide what absence means in their context; don't decide for them at the boundary.
- **The fabrication decision test:** Before filling in a missing field, ask: (1) If an auditor queries this field, will they get a value the external system actually provided? If no, it's fabrication. (2) If the external system's behaviour changes and the field starts appearing with a different value than what we inferred, will the audit trail silently contain two contradictory sources of truth? If yes, it's fabrication. (3) Would recording `None` and letting the consumer handle absence be less convenient but more honest? If yes, record `None`.
- Quarantine rows that can't be coerced/validated
- The audit trail records "row 42 was quarantined because field X was NULL" - that's a valid audit outcome

**Why:** User data is a trust boundary. A CSV with garbage in row 500 shouldn't crash the entire pipeline - we record the problem, quarantine the row, and keep processing the other 10,000 rows. We don't trust external systems, and we don't trust their silence either - an absent field is evidence, not an invitation to invent a default.

### Quick Reference

- **Source**: coerce OK, validate, quarantine failures, record absence as `None` (don't infer)
- **Transform (on row data)**: no coercion, wrap operations on values
- **Transform (on external calls)**: coerce OK — external response is Tier 3, record absence as `None`
- **Sink**: no coercion, expect types
- **Our data (Landscape, checkpoints)**: crash on any anomaly — serialization doesn't change trust tier

For detailed code examples (external call boundaries, pipeline templates, coercion/wrapping tables), see the `tier-model-deep-dive` skill.

## Plugin Ownership: System Code, Not User Code

All plugins (Sources, Transforms, Aggregations, Sinks) are **system-owned code**, not user-provided extensions. Gates are config-driven system operations, not plugins. ELSPETH uses `pluggy` for clean architecture, NOT to accept arbitrary user plugins. Plugins are developed, tested, and deployed as part of ELSPETH with the same rigor as engine code.

### Implications for Error Handling

| Scenario | Correct Response | WRONG Response |
|----------|------------------|----------------|
| Plugin method throws exception | **CRASH** - bug in our code | Catch and log silently |
| Plugin returns wrong type | **CRASH** - bug in our code | Coerce to expected type |
| Plugin missing expected attribute | **CRASH** - interface violation | Use `getattr(x, 'attr', default)` |
| User data has wrong type | Quarantine row, continue | Crash the pipeline |
| User data missing field | Quarantine row, continue | Crash the pipeline |

A defective plugin that silently produces wrong results is **worse than a crash**: a crash stops the pipeline and gets the bug fixed; silent pass-through gets recorded as "correct" and destroys the audit trail. Never wrap plugin calls in try/except to "recover" — let them crash.

## Core Architecture

### The SDA Model

```text
SENSE (Sources) → DECIDE (Transforms/Gates) → ACT (Sinks)
```

- **Source**: Load data (CSV, API, database, message queue) - exactly 1 per run
- **Transform**: Process/classify data - 0+ ordered, includes Gates for routing
- **Sink**: Output results - 1+ named destinations

### Key Subsystems

| Subsystem | Purpose |
| --------- | ------- |
| **Landscape** | Audit backbone - records every operation for complete traceability |
| **Plugin System** | Uses `pluggy` for extensible Sources, Transforms, Sinks |
| **SDA Engine** | RowProcessor, Orchestrator, RetryManager, ArtifactPipeline |
| **Canonical** | Two-phase deterministic JSON canonicalization for hashing |
| **Payload Store** | Separates large blobs from audit tables with retention policies |
| **Configuration** | Dynaconf + Pydantic with multi-source precedence |
| **Config Contracts** | Settings→Runtime protocol enforcement (`config-contracts-guide` skill) |

### DAG Execution Model

Pipelines compile to DAGs. Linear pipelines are degenerate DAGs (single `continue` path). Token identity tracks row instances through forks/joins:

- `row_id`: Stable source row identity
- `token_id`: Instance of row in a specific DAG path
- `parent_token_id`: Lineage for forks and joins

**Schema contracts, header normalization, aggregation timeouts, and composite PK patterns** are documented in the `engine-patterns-reference` skill.

### Transform Subtypes

| Type | Behavior |
| ---- | -------- |
| **Row Transform** | Process one row → emit one row (stateless) |
| **Gate** | Evaluate row → decide destination(s) via `continue`, `route_to_sink`, or `fork_to_paths` |
| **Aggregation** | Collect N rows until trigger → emit result (stateful) |
| **Coalesce** | Merge results from parallel paths |

## Development

**Package management:** Use `uv` for ALL package management. Never use `pip` directly.

For ELSPETH-specific CI analyzer rationale, rule taxonomy, and lifecycle policy,
see [docs/elspeth-lints/rationale.md](docs/elspeth-lints/rationale.md).

```bash
# Environment setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"      # Development with test tools
uv pip install -e ".[llm]"      # With LLM support
uv pip install -e ".[all]"      # Everything

# Tests and quality
.venv/bin/python -m pytest tests/                     # All tests
.venv/bin/python -m pytest tests/unit/                # Unit tests only
.venv/bin/python -m pytest tests/integration/         # Integration tests
.venv/bin/python -m pytest -k "test_fork"             # Tests matching pattern
.venv/bin/python -m pytest -x                         # Stop on first failure
.venv/bin/python -m mypy src/                         # Type checking
.venv/bin/python -m ruff check src/                   # Linting
.venv/bin/python -m ruff check --fix src/             # Auto-fix lint

# Config contracts verification
.venv/bin/python -m scripts.check_contracts

# Tier model enforcement (defensive pattern detection + layer-import enforcement)
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth

# Layer-import architecture observation (deterministic graph; always exits 0)
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli dump-edges --root src/elspeth --format json --output /tmp/l3-import-graph.json --no-timestamp
# Also supports --format mermaid (inline diagrams) and --format dot (Graphviz).
# Full reference: engine-patterns-reference skill, "Layer Architecture & Dependency Analysis" section.

# Allowlist fingerprint rotation (post-refactor; mechanical, no judgement)
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli rotate --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model --dry-run
# Drop --dry-run to apply. Surfaces rotations, ambiguous N:M groups, stale entries,
# and TODO-stub debt that needs judge review. Slice 1 of the cicd-judge-cli prototype.
# Symmetric N:N prefix groups are auto-paired by default; pass --no-auto-pair-symmetric
# to surface them as ambiguous instead. Stale entries are kept by default (--remove-stale to delete).

# Judge-gated allowlist entry creation (audit metadata write path)
env ELSPETH_JUDGE_METADATA_HMAC_KEY=<32-plus-byte-secret> PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli justify --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model --file-path plugins/example.py --symbol MyClass._method --rationale "why this suppression is honest" --owner "$USER"
# Writes judge_verdict, judge_rationale, source binding fields, and the HMAC
# signature. New entries bind v2 scope_fingerprint (the enclosing-scope AST
# fingerprint, signature prefix hmac-sha256:v2:); v1 file_fingerprint (whole-
# file hash) is the still-live legacy scheme being migrated away. Do not
# hand-edit judge metadata; production loads verify it.

# Reaudit existing judged entries during allowlist renewal / decay sweeps
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli reaudit --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model --format markdown --output /tmp/reaudit.md
# If interrupted, resume with --resume <run_id> or inspect an incomplete report
# with --render-incomplete <run_id>. When C3 override-rate fails, reaudit the
# override-heavy directories before considering an ADR to change the threshold.

# Migrate currently-valid v1 (file_fingerprint) entries to v2 (scope_fingerprint)
# OPERATOR-ONLY (writes signed metadata; requires ELSPETH_JUDGE_METADATA_HMAC_KEY,
# same custody constraint as justify — an agent may PROPOSE, only an operator-held
# environment runs and signs). Re-signs WITHOUT re-running the LLM judge, gated on
# two checks per entry: integrity (the existing v1 signature must verify — a
# tampered entry is refused, never laundered into a clean v2 signature) and
# relevance (the entry's canonical key must still match a live finding). --dry-run
# reports what would migrate without writing.
env ELSPETH_JUDGE_METADATA_HMAC_KEY=<32-plus-byte-secret> PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli migrate-judge-scope --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model --owner "$USER"

# CLI
elspeth run --settings pipeline.yaml --execute        # Execute pipeline
elspeth resume <run_id>                               # Resume interrupted run
elspeth validate --settings pipeline.yaml             # Validate config
elspeth plugins list                                  # List available plugins
elspeth purge --retention-days 90                      # Purge old payload data
elspeth explain --run <run_id> --row <row_id>         # Lineage explorer (TUI)

```

### Landscape MCP Analysis Server

For debugging pipeline failures, an MCP server provides read-only access to the audit database:

```bash
elspeth-mcp                                                    # Auto-discovers databases
elspeth-mcp --database sqlite:///./examples/my_pipeline/runs/audit.db  # Explicit DB
```

**Key tools:** `diagnose()` (what's broken?), `get_failure_context(run_id)` (deep dive), `explain_token(run_id, token_id)` (row lineage). Full reference: `docs/guides/landscape-mcp-analysis.md`.

## Technology Stack

Core: Typer (CLI), Textual (TUI), Dynaconf+Pydantic (config), pluggy (plugins), pandas (data), SQLAlchemy Core (DB), Alembic (migrations), tenacity (retries), OpenTelemetry (telemetry). Acceleration: rfc8785, NetworkX, structlog, pyrate-limiter, DeepDiff, Hypothesis. Optional packs: LLM (LiteLLM), Azure, Telemetry (ddtrace), Web (beautifulsoup4), Security (sqlcipher3), MCP. Full tables in `engine-patterns-reference` skill.

## Telemetry and Logging

**Landscape** is the legal record (persisted forever). **Telemetry** is operational visibility (ephemeral, real-time). **Logging** is last resort (only when audit and telemetry systems are broken).

**Primacy order**: Audit fires first (sync, crash-on-failure), then telemetry (async, best-effort), then logging (only if both are down). No silent failures — every telemetry emission point must send or explicitly acknowledge "nothing to send."

**Logger is NOT for pipeline activity.** Don't log row-level decisions, transform outcomes, or call results — those duplicate the Landscape. Logger is only for transitory debugging (`slog.debug`), audit system failures, and telemetry system failures.

**When a reviewer recommends `slog`**, the presumption should be that the event belongs in audit (Landscape) for critical run data, or operational telemetry for ephemeral operational signals. Reviewers outside the project don't know the primacy order and will default to logging because that's what most codebases do. Translate their intent ("this event should be visible") into the correct channel, not the suggested mechanism.

Full policy (permitted/forbidden uses, superset rule, telemetry-only exemptions, probative value test): see `logging-telemetry-policy` skill. Config guide: `docs/guides/telemetry.md`.

## Critical Implementation Patterns

Always use `row.to_dict()` for explicit conversion, not `dict(row)`. Every row reaches exactly one terminal state (`COMPLETED`, `ROUTED`, `FORKED`, `CONSUMED_IN_BATCH`, `COALESCED`, `QUARANTINED`, `FAILED`, `EXPANDED`) — no silent drops. `BUFFERED` is non-terminal (becomes `COMPLETED` on flush).

**Never bypass production code paths in tests** — integration tests MUST use `ExecutionGraph.from_plugin_instances()` and `instantiate_plugins_from_config()`.

For canonical JSON, retry semantics, secret handling, and detailed test path integrity rules, see the `engine-patterns-reference` skill.

## Configuration Precedence (High to Low)

1. Runtime overrides (CLI flags, env vars)
2. Pipeline configuration (`settings.yaml`)
3. Profile configuration (`profiles/production.yaml`)
4. Plugin pack defaults (`packs/llm/defaults.yaml`)
5. System defaults

## Source Layout

Source code lives in `src/elspeth/` with subsystems: `core/` (landscape, checkpoint, dag, config, canonical), `contracts/`, `engine/` (orchestrator, executors, processor, retry), `plugins/` (infrastructure, sources, transforms, sinks), `telemetry/`, `testing/` (the `elspeth-xdist-auto` pytest plugin shipped inside the package — distinct from the project's own `tests/` test suite, which is where the ChaosLLM / ChaosWeb / ChaosEngine fixtures live), `mcp/`, `tui/`, and CLI entry points. Full tree in `engine-patterns-reference` skill.

## Layer Dependency Rules

ELSPETH uses a strict 4-layer model. Imports must flow **downward only**.

```text
L0  contracts/     Leaf — imports nothing above. Shared types, enums, protocols.
L1  core/          Can import L0 only. Landscape, DAG, config, canonical JSON.
L2  engine/        Can import L0, L1. Orchestrator, processors, executors.
L3  plugins/       Can import L0, L1, L2. Sources, transforms, sinks, clients.
    mcp/ tui/ cli* telemetry/ testing/   — also L3 (application layer)
```

**Enforced by CI:** the `trust_tier.tier_model` elspeth-lints rule detects upward imports and fails the build. The allowlist mechanism (`config/cicd/enforce_tier_model/`) supports per-file and per-finding exemptions for legitimate exceptions.

**TYPE_CHECKING imports** are reported as warnings, not failures. They're architecturally impure (the dependency still exists for type checkers) but don't create runtime coupling.

**Architecture observation (separate from enforcement).** `elspeth-lints dump-edges` emits the deterministic intra-layer import graph as JSON, Mermaid, or Graphviz DOT. It always exits 0 (observational, not a gate) and supports SCC detection via `networkx`. Use it for architecture analysis, refactor planning, or dependency-graph diffing across branches. **Cite the JSON output by path** rather than paraphrasing — the `--no-timestamp` flag produces byte-identical output across runs so cited values stay stable. Full reference (subcommands, JSON schema, edge metadata semantics, SCC interpretation, citation discipline) lives in the `engine-patterns-reference` skill under "Layer Architecture & Dependency Analysis".

### When a New Cross-Layer Need Arises

Resolution options in priority order:

1. **Move the code down.** If the needed code has no upward dependencies, move it to the lower layer. E.g., move a dataclass from `core/` to `contracts/`.
2. **Extract the primitive.** If only a type or constant is needed, extract it into `contracts/` and import from there.
3. **Restructure the caller.** Refactor so the higher-layer code isn't needed. Use dependency injection, callbacks, or protocols defined in `contracts/`.
4. **NEVER:** Add a lazy import with an apologetic comment. This is the "Shifting the Burden" archetype — it defers the structural fix and the pattern will recur.

## No Legacy Code Policy

**STRICT REQUIREMENT:** Legacy code, backwards compatibility, and compatibility shims are strictly forbidden. WE HAVE NO USERS YET — deferring breaking changes is the opposite of what we want.

**When something is removed or changed, DELETE THE OLD CODE COMPLETELY.** No version checks, feature flags for old behaviour, adapter/wrapper/proxy shims, `@deprecated` retentions, commented-out "for reference" blocks, or "both old and new" branches. Don't rename unused variables to `_var` — delete them. Don't keep old code in comments — git history exists. Change all call sites in the same commit.

## CICD Judge Gate: HMAC Key Custody

The cicd-judge allowlist gate signs judge metadata with a **symmetric** HMAC key
(`ELSPETH_JUDGE_METADATA_HMAC_KEY`). Symmetric means **every key holder can forge
a valid signature** — so the gate's entire security reduces to one custody rule:

**The signing key is operator-only. It MUST NOT be present in any autonomous
agent's environment** (`.env` files an agent can read, agent CI contexts, dev
shells an agent drives).

An agent that holds the key can bypass the judge entirely: hand-write
`judge_verdict: ACCEPTED` with a fabricated rationale and a correct
(publicly-computable) source binding — `scope_fingerprint`/`ast_path` for v2
entries, the legacy `file_fingerprint`/`ast_path` for v1 — compute a valid
signature, and pass **every** automated gate — putting a forged-but-validly-signed verdict
into the audit trail, indistinguishable from a real one. The HMAC stops a
*keyless* agent and makes tampering of signed entries detectable; it does nothing
against a key-holding agent.

Therefore: an agent may *propose* an `elspeth-lints justify` invocation; only an
operator-held environment runs it and signs. CI verifies signatures with the key
as a GitHub Actions secret (withheld from fork PRs by design — see the
`shape-only-when-key-missing` verify mode and its load-time warning). A future
asymmetric scheme (agents can verify, not sign) would remove this surface
structurally.

## Git Safety

**Never run destructive git commands without explicit user permission:**

- `git reset --hard`, `git clean -f`, `git checkout -- <file>` - Discard uncommitted changes
- `git push --force` - Rewrites remote history
- `git rebase` (on pushed branches) - Rewrites shared history

**Worktree isolation is the default for new work.** Before starting implementation, ask the operator whether to create a worktree under `.worktrees/`; default yes. Inside a worktree there is nothing to stash, which removes the slip pattern that previously caused data loss in this project via pre-commit-hook `stash`/`pop` cycles (the hooks silently destroy unstaged work when `stash pop` encounters conflicts). In the main checkout — an explicit operator opt-in — `git stash` is available as a normal tool: the prior absolute prohibition was lifted on 2026-05-11 once worktree-default became the upstream control. See memory: `feedback_default_to_worktree.md`, `feedback_no_git_stash.md`.

## Defensive Programming: Forbidden. Offensive Programming: Encouraged

### What's Forbidden (Defensive Programming)

Do not use `.get()`, `getattr()`, `isinstance()`, or silent exception handling to suppress errors from nonexistent attributes, malformed data, or incorrect types. **Access typed dataclass fields directly** (`obj.field`), not defensively (`obj.get("field")`). **`hasattr()` is unconditionally banned** — it swallows all exceptions from `@property` getters, not just missing attributes.

Defensive handling IS appropriate at trust boundaries — see the `tier-model-deep-dive` skill for coercion and operation wrapping rules.

### What's Encouraged (Offensive Programming)

**Proactively detect invalid states and throw meaningful exceptions.** The goal is not to prevent crashes — it's to make crashes **maximally informative**. Always use `from exc` to preserve exception chains.

For detailed examples (Tier 1 read guards, write-side DTO validation, TOCTOU atomic guards, `hasattr` alternatives), see the `engine-patterns-reference` skill.

### The Decision Test

| Question | If Yes | If No |
|----------|--------|-------|
| Is this protecting against user-provided data values? | ✅ Wrap it (trust boundary) | — |
| Is this at an external system boundary (API, file, DB)? | ✅ Wrap it (trust boundary) | — |
| Can I detect an invalid state and throw a meaningful error? | ✅ Assert it (offensive) | — |
| Would this fail due to a bug in code we control? | — | ❌ Let it crash |
| Am I adding this because "something might be None"? | — | ❌ Fix the root cause |
| Am I silently swallowing an error with a default value? | — | ❌ That's defensive — forbidden |

## Frozen Dataclass Immutability: The `deep_freeze` Contract

Python's `frozen=True` only prevents attribute **reassignment** — mutable contents stay mutable through the attribute reference. Every frozen dataclass with container fields (`dict`, `list`, `set`, `Mapping`, `Sequence`) **must** enforce deep immutability in `__post_init__`.

### The Canonical Pattern

```python
from elspeth.contracts.freeze import freeze_fields

@dataclass(frozen=True, slots=True)
class MyRecord:
    data: Mapping[str, Any]
    items: Sequence[Mapping[str, object]]

    def __post_init__(self) -> None:
        freeze_fields(self, "data", "items")
```

`freeze_fields()` calls `deep_freeze()` on each named field (recursively converting `dict` → `MappingProxyType`, `list` → `tuple`, `set` → `frozenset`) and is identity-preserving when the field is already frozen. For nullable fields, gate on `is not None` first. For shapes `freeze_fields` can't handle (e.g. per-element tuple comprehensions), call `deep_freeze()` directly with an identity check.

If all fields are scalars, enums, `datetime`, or `None`, no guard is needed — `frozen=True` suffices.

### Forbidden Anti-Patterns

| Pattern | Why It's Wrong |
|---------|---------------|
| `MappingProxyType(self.x)` | **View, not copy.** Caller can still mutate the original dict; changes visible through the proxy. |
| `MappingProxyType(dict(self.x))` | **Shallow only.** Copies the outer dict but nested dicts/lists remain mutable. |
| `isinstance(self.x, dict)` as guard | **Misses Mapping subtypes.** `OrderedDict`, custom `Mapping` implementations pass through unfrozen. |
| `isinstance(self.x, tuple)` to skip | **Tuple of mutable dicts.** A `tuple[dict, dict]` passes the check but contents are mutable. |
| `not isinstance(self.x, MappingProxyType)` | **Shallow frozen ≠ deep frozen.** A `MappingProxyType` wrapping mutable nested containers is not deeply frozen. |

`MappingProxyType(dict(self.x))` (shallow copy + wrap) is acceptable **only when values are guaranteed immutable** (scalars, enum members, frozen dataclass instances). Prefer `deep_freeze()` regardless unless profiling shows a hot-path concern.

Enforced by `scripts/cicd/enforce_freeze_guards.py`; allowlist in `config/cicd/enforce_freeze_guards/`.

<!-- filigree:instructions:v2.1.0:9dff6e6d -->
## Filigree Issue Tracker

`filigree` tracks tasks for this project. Data lives in `.filigree/`. Prefer
the MCP tools (`mcp__filigree__*`) when available; fall back to the `filigree`
CLI otherwise.

### Workflow

```bash
# At session start
filigree session-context                            # ready / in-progress / critical path

# Pick up the next startable issue (atomic claim + transition into its working status)
filigree start-next-work --assignee <name>
# ...or claim a specific issue
filigree start-work <id> --assignee <name>

# Do the work, commit, then
filigree close <id>
```

Use the atomic claim+transition verbs — `start_work` / `start_next_work`
(MCP) or `start-work` / `start-next-work` (CLI). Do **not** chain
`claim_issue` (MCP) or `filigree claim` (CLI) with a subsequent status
update — the two-step form races against other agents; the combined verb is
atomic.

**Ready ≠ startable.** The working status is type-specific (tasks →
`in_progress`, features → `building`). Bugs start at `triage`, which has no
single-hop transition into work (`triage → confirmed → fixing`), so a triage
bug is *ready* but not directly *startable*: `start_work` on one returns
`INVALID_TRANSITION` naming the next status, and `start_next_work` skips it.
`get_ready` items carry a `startable` flag (plus a `next_action` hint when
false). Pass `advance=true` (MCP) / `--advance` (CLI) to walk the soft
transitions to the nearest working status automatically.

### Observations: when (and when not) to use them

`observe` is a fire-and-forget scratchpad for *incidental* defects — things
you notice *outside the scope of your current task* (a code smell in a
neighbouring file, a stale TODO, a missing test for an edge case you happened
to spot). Notes expire after 14 days unless promoted. Include `file_path` and
`line` when relevant. At session end, skim `list_observations` and either
`dismiss_observation` or `promote_observation` for what has accumulated.

**You fix bugs in your currently defined scope. You do NOT use observations
to finish work prematurely.** If a defect, gap, or follow-up belongs to your
current task, you own it — handle it as part of that task: fix it now, expand
the task's scope, file a proper issue with a dependency, or surface it to the
user. Filing it as an observation and closing the task is *not* completing
the task; it is shipping known-broken work and hiding the debt in a 14-day
expiring scratchpad. The test is "would I have noticed this even if I weren't
working on this task?" If no, it's task scope, not an observation.

### Priority scale

- P0: Critical (drop everything)
- P1: High (do next)
- P2: Medium (default)
- P3: Low
- P4: Backlog

### Reaching for tools

MCP tool schemas describe each tool; `filigree --help` and `filigree <verb>
--help` are the authoritative CLI reference. You do not need to memorise
either catalogue. The verbs you will reach for most:

- **Find work:** `get_ready`, `get_blocked`, `list_issues`, `search_issues`
- **Claim work:** `start_work`, `start_next_work`
- **Update:** `add_comment`, `add_label`, `update_issue`, `close_issue`
- **Admin (irreversible):** `delete_issue` (MCP) / `delete-issue` (CLI) —
  hard-deletes a terminal issue and its rows; `undo_last` cannot reverse it.
- **Scratchpad:** `observe`, `list_observations`, `promote_observation`, `dismiss_observation`
- **Cross-product entity bindings (ADR-029):** `add_entity_association`,
  `remove_entity_association`, `list_entity_associations`,
  `list_associations_by_entity`. Used when a sibling tool (e.g.
  Clarion) needs to bind a Filigree issue to a function, class, or
  module identifier it owns. The `entity_id` is an opaque string
  from Filigree's perspective; the consumer (the sibling tool's read
  path) does drift detection against the stored
  `content_hash_at_attach`. `list_associations_by_entity` is the
  reverse-lookup surface — given a Clarion entity ID, return every
  Filigree issue bound to it (project isolation is by DB file). Also
  reachable over HTTP as
  `GET/POST /api/issue/{issue_id}/entity-associations`,
  `DELETE /api/issue/{issue_id}/entity-associations?entity_id=…`,
  and `GET /api/entity-associations?entity_id=…`.
- **Health:** `get_stats`, `get_metrics`, `get_mcp_status`

Pass `--actor <name>` (CLI) so events attribute to your agent identity. It
works in either position — before the verb (`filigree --actor X update …`) or
after it (`filigree update … --actor X`); the post-verb value overrides the
group-level one.

### Error handling

Errors return `{error: str, code: ErrorCode, details?: dict}`. Switch on
`code`, not on message text. Codes: `VALIDATION`, `NOT_FOUND`, `CONFLICT`,
`INVALID_TRANSITION`, `PERMISSION`, `NOT_INITIALIZED`, `IO`,
`INVALID_API_URL`, `FILE_REGISTRY_DISPLACED`, `REGISTRY_UNAVAILABLE`,
`CLARION_REGISTRY_VERSION_MISMATCH`, `BRIEFING_BLOCKED`, `STOP_FAILED`,
`SCHEMA_MISMATCH`, `INTERNAL`.

On `INVALID_TRANSITION`, call `get_valid_transitions` (MCP) or
`filigree transitions <id>` to see what the workflow allows from here.

Two failure modes deserve a specific response:

- **`SCHEMA_MISMATCH`** — the installed `filigree` is older than the project
  database. The error message contains upgrade guidance. Surface it to the
  user; do not retry.
- **`ForeignDatabaseError`** — filigree found a parent project's database
  but no local `.filigree.conf`. Run `filigree init` in the current
  directory. Do **not** `cd` upward to a different project unless that was
  the actual intent.
<!-- /filigree:instructions -->

<!-- Filigree behavioral guidance (issue types, naming, retyping) is in AGENTS.md -->

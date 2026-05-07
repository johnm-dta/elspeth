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
- **Attributability test**: For any output, `explain(recorder, run_id, token_id)` must prove complete lineage back to source

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

### The Trust Flow

```text
EXTERNAL DATA              PIPELINE DATA              AUDIT TRAIL
(zero trust)               (elevated trust)           (full trust)
                           "probably ok"

┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ External Source │        │ Transform/Sink  │        │ Landscape DB    │
│                 │        │                 │        │                 │
│ • Coerce OK     │───────►│ • No coercion   │───────►│ • Crash on      │
│ • Validate      │ types  │ • Expect types  │ record │   any anomaly   │
│ • Quarantine    │ valid  │ • Wrap ops on   │ what   │ • No coercion   │
│   failures      │        │   row values    │ we     │   ever          │
│                 │        │ • Bug if types  │ saw    │                 │
│                 │        │   are wrong     │        │                 │
└─────────────────┘        └─────────────────┘        └─────────────────┘
         │                          │
         │                          │
    Source is the              Operations on row
    ONLY place coercion        values need wrapping
    is allowed                 (values can still fail)
```

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

A defective plugin that silently produces wrong results is **worse than a crash**:

1. **Crash:** Pipeline stops, operator investigates, bug gets fixed
2. **Silent wrong result:** Data flows through, gets recorded as "correct," auditors see garbage, trust is destroyed

```python
# WRONG - hides plugin bugs, destroys audit integrity
try:
    result = transform.process(row, ctx)
except Exception:
    result = row  # "just pass through on error"
    logger.warning("Transform failed, using original row")

# RIGHT - plugin bugs crash immediately
result = transform.process(row, ctx)  # Let it crash
```

If `transform.process()` has a bug, we MUST know about it. Silently passing through the original row means the audit trail now contains data that "looks processed" but wasn't - this is evidence tampering.

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
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model

# Layer-import architecture observation (deterministic graph; always exits 0)
.venv/bin/python scripts/cicd/enforce_tier_model.py dump-edges --root src/elspeth --format json --output /tmp/l3-import-graph.json --no-timestamp
# Also supports --format mermaid (inline diagrams) and --format dot (Graphviz).
# Full reference: engine-patterns-reference skill, "Layer Architecture & Dependency Analysis" section.

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

**Enforced by CI:** `scripts/cicd/enforce_tier_model.py` detects upward imports and fails the build. The allowlist mechanism (`config/cicd/enforce_tier_model/`) supports per-file and per-finding exemptions for legitimate exceptions.

**TYPE_CHECKING imports** are reported as warnings, not failures. They're architecturally impure (the dependency still exists for type checkers) but don't create runtime coupling.

**Architecture observation (separate from enforcement).** The same script also exposes a `dump-edges` subcommand that emits the deterministic intra-layer import graph as JSON, Mermaid, or Graphviz DOT. It always exits 0 (observational, not a gate) and supports SCC detection via `networkx`. Use it for architecture analysis, refactor planning, or dependency-graph diffing across branches. **Cite the JSON output by path** rather than paraphrasing — the `--no-timestamp` flag produces byte-identical output across runs so cited values stay stable. Full reference (subcommands, JSON schema, edge metadata semantics, SCC interpretation, citation discipline) lives in the `engine-patterns-reference` skill under "Layer Architecture & Dependency Analysis".

### When a New Cross-Layer Need Arises

Resolution options in priority order:

1. **Move the code down.** If the needed code has no upward dependencies, move it to the lower layer. E.g., move a dataclass from `core/` to `contracts/`.
2. **Extract the primitive.** If only a type or constant is needed, extract it into `contracts/` and import from there.
3. **Restructure the caller.** Refactor so the higher-layer code isn't needed. Use dependency injection, callbacks, or protocols defined in `contracts/`.
4. **NEVER:** Add a lazy import with an apologetic comment. This is the "Shifting the Burden" archetype — it defers the structural fix and the pattern will recur.

## No Legacy Code Policy

**STRICT REQUIREMENT:** Legacy code, backwards compatibility, and compatibility shims are strictly forbidden. WE HAVE NO USERS YET. Deferring breaking changes until we do is the opposite of what we want.

### Anti-Patterns - Never Do This

1. **Backwards Compatibility Code** - No version checks, feature flags for old behavior, or "compatibility mode" switches
2. **Legacy Shims** - No adapter classes, wrapper functions, or proxy objects for deprecated functionality
3. **Deprecated Code Retention** - No `@deprecated` decorators with code kept around, no commented-out implementations "for reference"
4. **Migration Helpers** - No code supporting "both old and new" simultaneously

### The Rule

**When something is removed or changed, DELETE THE OLD CODE COMPLETELY.**

- Don't rename unused variables to `_var` - delete the variable
- Don't keep old code in comments - delete it (git history exists)
- Don't add compatibility layers - change all call sites in the same commit
- Don't create abstractions to hide breaking changes - make the breaking change

## Git Safety

**Never run destructive git commands without explicit user permission:**

- `git reset --hard`, `git clean -f`, `git checkout -- <file>` - Discard uncommitted changes
- `git push --force` - Rewrites remote history
- `git rebase` (on pushed branches) - Rewrites shared history

**No git stash.** The stash/pop cycle has caused repeated data loss in this project — pre-commit hooks that stash/unstash silently destroy unstaged work when `stash pop` encounters conflicts. If you need to preserve work, commit it to a branch.

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

Python's `frozen=True` only prevents attribute **reassignment** — it does nothing about mutable **contents**. A `frozen=True` dataclass with a `dict` field is a lie: the dict is fully mutable through the attribute reference. Every frozen dataclass with container fields (`dict`, `list`, `set`, `Mapping`, `Sequence`) **must** enforce deep immutability in `__post_init__`.

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

**Always use `freeze_fields()`** — it calls `deep_freeze()` on each named field (recursively converting `dict` → `MappingProxyType`, `list` → `tuple`, `set` → `frozenset`, including arbitrary `Mapping` types) and skips `object.__setattr__` when the field is already frozen (identity-preserving idempotency).

For fields gated on `None`, use `if self.field is not None: freeze_fields(self, "field")`.

For special cases that `freeze_fields` can't handle (e.g., per-element tuple comprehensions), use `deep_freeze()` directly with the identity check pattern.

### Forbidden Anti-Patterns

| Pattern | Why It's Wrong |
|---------|---------------|
| `MappingProxyType(self.x)` | **View, not copy.** Caller can still mutate the original dict; changes visible through the proxy. |
| `MappingProxyType(dict(self.x))` | **Shallow only.** Copies the outer dict but nested dicts/lists remain mutable. |
| `isinstance(self.x, dict)` as guard | **Misses Mapping subtypes.** `OrderedDict`, custom `Mapping` implementations pass through unfrozen. |
| `isinstance(self.x, tuple)` to skip | **Tuple of mutable dicts.** A `tuple[dict, dict]` passes the check but contents are mutable. |
| `not isinstance(self.x, MappingProxyType)` | **Shallow frozen ≠ deep frozen.** A `MappingProxyType` wrapping mutable nested containers is not deeply frozen. |

### When Shallow Wrapping IS Acceptable

`MappingProxyType(dict(self.x))` (shallow copy + wrap) is acceptable **only when values are guaranteed immutable**: scalars (`int`, `str`, `bool`, enum members) or frozen dataclass instances. Even then, `deep_freeze()` works and is more consistent — prefer it unless profiling shows a hot-path concern.

### Scalar-Only Fields Need No Guard

If all fields are scalars, enums, `datetime`, or `None`, no freeze guard is needed. `frozen=True` is sufficient. Don't add guards that do nothing.

### Enforced by CI

`scripts/cicd/enforce_freeze_guards.py` detects forbidden patterns in `__post_init__` methods and fails the build. Allowlist in `config/cicd/enforce_freeze_guards/` for justified exceptions.

<!-- filigree:instructions:v1.6.0:84820288 -->
## Filigree Issue Tracker

Use `filigree` for all task tracking in this project. Data lives in `.filigree/`.

### MCP Tools (Preferred)

When MCP is configured, prefer `mcp__filigree__*` tools over CLI commands — they're
faster and return structured data. Key tools:

- `get_ready` / `get_blocked` — find available work
- `get_issue` / `list_issues` / `search_issues` — read issues
- `create_issue` / `update_issue` / `close_issue` — manage issues
- `claim_issue` / `claim_next` — atomic claiming
- `add_comment` / `add_label` — metadata
- `list_labels` / `get_label_taxonomy` — discover labels and reserved namespaces
- `create_plan` / `get_plan` — milestone planning
- `get_stats` / `get_metrics` — project health
- `get_valid_transitions` — workflow navigation
- `observe` / `list_observations` / `dismiss_observation` / `promote_observation` — agent scratchpad
- `trigger_scan` / `trigger_scan_batch` / `get_scan_status` / `preview_scan` / `list_scanners` — automated code scanning
- `get_finding` / `list_findings` / `update_finding` / `batch_update_findings` — scan finding triage
- `promote_finding` / `dismiss_finding` — finding lifecycle (promote to issue or dismiss)

Observations are fire-and-forget notes that expire after 14 days. Use `list_issues --label=from-observation` to find promoted observations.

**Observations are ambient.** While doing other work, use `observe` whenever you
notice something worth noting — a code smell, a potential bug, a missing test, a
design concern. Don't stop what you're doing; just fire off the observation and
carry on. They're ideal for "I don't have time to investigate this right now, but
I want to come back to it." Include `file_path` and `line` when relevant so the
observation is anchored to code. At session end, skim `list_observations` and
either `dismiss_observation` (not worth tracking) or `promote_observation`
(deserves an issue) for anything that's accumulated.

Fall back to CLI (`filigree <command>`) when MCP is unavailable.

### CLI Quick Reference

```bash
# Finding work
filigree ready                              # Show issues ready to work (no blockers)
filigree list --status=open                 # All open issues
filigree list --status=in_progress          # Active work
filigree list --label=bug --label=P1        # Filter by multiple labels (AND)
filigree list --label-prefix=cluster:       # Filter by label namespace prefix
filigree list --not-label=wontfix           # Exclude issues with label
filigree show <id>                          # Detailed issue view

# Creating & updating
filigree create "Title" --type=task --priority=2          # New issue
filigree update <id> --status=in_progress                # Claim work
filigree close <id>                                      # Mark complete
filigree close <id> --reason="explanation"               # Close with reason

# Dependencies
filigree add-dep <issue> <depends-on>       # Add dependency
filigree remove-dep <issue> <depends-on>    # Remove dependency
filigree blocked                            # Show blocked issues

# Comments & labels
filigree add-comment <id> "text"            # Add comment
filigree get-comments <id>                  # List comments
filigree add-label <id> <label>             # Add label
filigree remove-label <id> <label>          # Remove label
filigree labels                             # List all labels by namespace
filigree taxonomy                           # Show reserved namespaces and vocabulary

# Workflow templates
filigree types                              # List registered types with state flows
filigree type-info <type>                   # Full workflow definition for a type
filigree transitions <id>                   # Valid next states for an issue
filigree packs                              # List enabled workflow packs
filigree validate <id>                      # Validate issue against template
filigree guide <pack>                       # Display workflow guide for a pack

# Atomic claiming
filigree claim <id> --assignee <name>            # Claim issue (optimistic lock)
filigree claim-next --assignee <name>            # Claim highest-priority ready issue

# Batch operations
filigree batch-update <ids...> --priority=0      # Update multiple issues
filigree batch-close <ids...>                    # Close multiple with error reporting

# Planning
filigree create-plan --file plan.json            # Create milestone/phase/step hierarchy

# Event history
filigree changes --since 2026-01-01T00:00:00    # Events since timestamp
filigree events <id>                             # Event history for issue
filigree explain-state <type> <state>            # Explain a workflow state

# All commands support --json and --actor flags
filigree --actor bot-1 create "Title"            # Specify actor identity
filigree list --json                             # Machine-readable output

# Project health
filigree stats                              # Project statistics
filigree search "query"                     # Search issues
filigree doctor                             # Health check
```

### File Records & Scan Findings (API)

The dashboard exposes REST endpoints for file tracking and scan result ingestion.
Use `GET /api/files/_schema` for available endpoints and valid field values.

Key endpoints:
- `GET /api/files/_schema` — Discovery: valid enums, endpoint catalog
- `POST /api/v1/scan-results` — Ingest scan results (SARIF-lite format)
- `GET /api/files` — List tracked files with filtering and sorting
- `GET /api/files/{file_id}` — File detail with associations and findings summary
- `GET /api/files/{file_id}/findings` — Findings for a specific file

### Workflow
1. `filigree ready` to find available work
2. `filigree show <id>` to review details
3. `filigree transitions <id>` to see valid state changes
4. `filigree update <id> --status=in_progress` to claim it
5. Do the work, commit code
6. `filigree close <id>` when done

### Session Start
When beginning a new session, run `filigree session-context` to load the project
snapshot (ready work, in-progress items, critical path). This provides the
context needed to pick up where the previous session left off.

### Priority Scale
- P0: Critical (drop everything)
- P1: High (do next)
- P2: Medium (default)
- P3: Low
- P4: Backlog
<!-- /filigree:instructions -->

<!-- Filigree behavioral guidance (issue types, naming, retyping) is in AGENTS.md -->

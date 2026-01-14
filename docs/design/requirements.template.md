  REQUIREMENTS TEMPLATE - ELSPETH Architecture

  1. CONFIGURATION REQUIREMENTS

1.1 Configuration Format (from README.md lines 73-113, architecture.md lines 832-891)
  ┌────────────────┬──────────────────────────────────────────────────────────────┬────────────────────────────────────────────┬─────────────────────────────────────────┐
  │ Requirement ID │                         Requirement                          │                   Source                   │                 Status                  │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-001        │ Config uses datasource key (not source)                      │ README.md:75                               │ Implemented                            │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-002        │ datasource.plugin specifies the source plugin name           │ README.md:76                               │ Implemented                            │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-003        │ datasource.options holds plugin-specific config              │ README.md:77-78                            │ Implemented                            │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-004        │ sinks is a dict of named sinks                               │ README.md:80-89                            │ Implemented                            │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-005        │ Each sink has plugin and options keys                        │ README.md:81-88                            │ Implemented                            │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-006        │ row_plugins is an array of transforms                        │ README.md:91-99, architecture.md:858       │ Implemented                            │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-007        │ Each row_plugin has plugin, type, options, routes            │ README.md:92-99                            │ Implemented                            │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-008        │ output_sink specifies the default sink                       │ README.md:107                              │ Implemented                            │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-009        │ landscape.enabled boolean flag                               │ README.md:109-110                          │ Partial - flag unused                  │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-010        │ landscape.backend specifies storage type (sqlite/postgresql) │ README.md:111, architecture.md:883         │ Diverged - uses landscape.url          │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-011        │ landscape.path specifies database path                       │ README.md:112                              │ Diverged - uses landscape.url          │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-012        │ landscape.retention.row_payloads_days config                 │ architecture.md:556-559, 886               │ Not implemented                        │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-013        │ landscape.retention.call_payloads_days config                │ architecture.md:556-559, 887               │ Not implemented                        │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-014        │ landscape.redaction.profile config                           │ architecture.md:889-890                    │ Not implemented                        │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-015        │ concurrency.max_workers config (default 4, production 16)    │ README.md:195-202                          │ Partial - defined only                 │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-016        │ Profile system with profiles: key and --profile flag         │ README.md:199-209                          │ Not implemented                        │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-017        │ Environment variable interpolation ${VAR}                    │ README.md:213-216                          │ Partial - env overrides only            │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-018        │ Hierarchical settings merge with clear precedence            │ README.md:188-206, architecture.md:820-828 │ Partial - Dynaconf merge_enabled       │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-019        │ Pack defaults (packs/llm/defaults.yaml)                      │ architecture.md:824, CLAUDE.md             │ Not implemented                        │
  ├────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┤
  │ CFG-020        │ Suite configuration (suite.yaml)                             │ architecture.md:823                        │ Not implemented                        │
  └────────────────┴──────────────────────────────────────────────────────────────┴────────────────────────────────────────────┴─────────────────────────────────────────┘
  1.2 Configuration Settings Classes (from architecture and Phase 1 plan)
  ┌────────────────┬────────────────────────────────────────────────────┬────────────────────────────────────┬────────────────────────────────────────────────┐
  │ Requirement ID │                    Requirement                     │               Source               │                     Status                     │
  ├────────────────┼────────────────────────────────────────────────────┼────────────────────────────────────┼────────────────────────────────────────────────┤
  │ CFG-021        │ LandscapeSettings class                            │ Phase 1 plan                       │ Partial - implemented, but fields diverge     │
  ├────────────────┼────────────────────────────────────────────────────┼────────────────────────────────────┼────────────────────────────────────────────────┤
  │ CFG-022        │ RetentionSettings class                            │ Phase 1 plan                       │ Not implemented                               │
  ├────────────────┼────────────────────────────────────────────────────┼────────────────────────────────────┼────────────────────────────────────────────────┤
  │ CFG-023        │ ConcurrencySettings class                          │ Phase 1 plan                       │ Implemented                                   │
  ├────────────────┼────────────────────────────────────────────────────┼────────────────────────────────────┼────────────────────────────────────────────────┤
  │ CFG-024        │ Settings stored with run (resolved, not just hash) │ architecture.md:270, subsystems:97 │ Partial - schema exists; CLI stores {}        │
  └────────────────┴────────────────────────────────────────────────────┴────────────────────────────────────┴────────────────────────────────────────────────┘
1.3 Configuration Review Notes (filled)

- CFG-001: `datasource` key enforced by `src/elspeth/core/config.py` (`ElspethSettings.datasource`).
- CFG-002: `datasource.plugin` in `src/elspeth/core/config.py` (`DatasourceSettings.plugin`); used in `src/elspeth/cli.py` to select source plugin.
- CFG-003: `datasource.options` in `src/elspeth/core/config.py` (`DatasourceSettings.options`); passed to `CSVSource`/`JSONSource` in `src/elspeth/cli.py`.
- CFG-004: `sinks` dict in `src/elspeth/core/config.py` (`ElspethSettings.sinks`); iterated in `src/elspeth/cli.py` and compiled into nodes in `src/elspeth/core/dag.py`.
- CFG-005: `SinkSettings` (`plugin`, `options`) in `src/elspeth/core/config.py`; used when instantiating sinks in `src/elspeth/cli.py`.
- CFG-006: `row_plugins` list in `src/elspeth/core/config.py` (`ElspethSettings.row_plugins`); used in `src/elspeth/cli.py` and `src/elspeth/core/dag.py`.
- CFG-007: `RowPluginSettings` (`plugin`, `type`, `options`, `routes`) in `src/elspeth/core/config.py`; gate routes consumed by `src/elspeth/core/dag.py:ExecutionGraph.from_config()`.
- CFG-008: `output_sink` validated in `src/elspeth/core/config.py`; used by `src/elspeth/core/dag.py` and `src/elspeth/engine/orchestrator.py` (`graph.get_output_sink()`).
- CFG-009: `LandscapeSettings.enabled` exists in `src/elspeth/core/config.py`, but the run path ignores it (audit is always on).
- CFG-010: `LandscapeSettings.backend` exists but DB selection is via `LandscapeSettings.url` + `src/elspeth/core/landscape/database.py:LandscapeDB.from_url()`.
- CFG-011: No `landscape.path` in schema; configuration uses `landscape.url` (`src/elspeth/core/config.py`) and examples (e.g., `examples/settings.yaml`).
- CFG-012: No `landscape.retention.row_payloads_days` in schema; no retention implementation.
- CFG-013: No `landscape.retention.call_payloads_days` in schema; no retention implementation.
- CFG-014: No `landscape.redaction.profile` in schema; no redaction implementation (no `redact*` references in `src/elspeth`).
- CFG-015: `ConcurrencySettings.max_workers` exists in `src/elspeth/core/config.py`, but execution is single-threaded (only printed by `src/elspeth/cli.py`).
- CFG-016: No profile selection in CLI; `src/elspeth/core/config.py:load_settings()` sets `environments=False` for Dynaconf.
- CFG-017: Env var overrides supported via Dynaconf in `src/elspeth/core/config.py:load_settings()` (prefix `ELSPETH_...`); `${VAR}` interpolation inside YAML not implemented.
- CFG-018: Deep merge + precedence is partial via `Dynaconf(... merge_enabled=True)` + Pydantic defaults; pack/profile/suite layering not implemented.
- CFG-019: No `packs/llm/defaults.yaml` support in code.
- CFG-020: No suite configuration (`suite.yaml`) support in code.
- CFG-021: `LandscapeSettings` class exists in `src/elspeth/core/config.py`, but differs from README/architecture (url-based; no `path`/`retention`/`redaction`).
- CFG-022: No `RetentionSettings` class in `src/elspeth/core/config.py`.
- CFG-023: `ConcurrencySettings` class exists in `src/elspeth/core/config.py`.
- CFG-024: Run schema has `runs.settings_json` (`src/elspeth/core/landscape/schema.py`) and `LandscapeRecorder.begin_run()` persists canonicalized config (`src/elspeth/core/landscape/recorder.py`), but `Orchestrator.run()` passes `PipelineConfig.config` (often `{}`) and CLI doesn't populate it (`src/elspeth/cli.py`).

  ---

  1. CLI REQUIREMENTS

2.1 CLI Commands (from README.md, architecture.md, CLAUDE.md)
  ┌────────────────┬───────────────────────────────────────────────────────────────┬──────────────────────────────┬──────────────────────────────────┐
  │ Requirement ID │                          Requirement                          │            Source            │              Status              │
  ├────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┤
  │ CLI-001        │ elspeth --settings <file> to run pipeline                     │ README.md:116                │ Diverged - requires run cmd     │
  ├────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┤
  │ CLI-002        │ elspeth --profile <name> for profile selection                │ README.md:208                │ Not implemented                 │
  ├────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┤
  │ CLI-003        │ elspeth explain --run <id> --row <id>                         │ README.md:122-136, CLAUDE.md │ Partial - stub                  │
  ├────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┤
  │ CLI-004        │ elspeth explain with --full flag for auditor view             │ architecture.md:765-766      │ Not implemented                 │
  ├────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┤
  │ CLI-005        │ elspeth validate --settings <file>                            │ CLAUDE.md                    │ Implemented                     │
  ├────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┤
  │ CLI-006        │ elspeth plugins list                                          │ CLAUDE.md                    │ Implemented                     │
  ├────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┤
  │ CLI-007        │ elspeth status to check run status                            │ subsystems/00-overview:736   │ Not implemented                 │
  ├────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┤
  │ CLI-008        │ Human-readable output by default, --json for machine-readable │ subsystems/00-overview:739   │ Partial - explain has --json    │
  ├────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┤
  │ CLI-009        │ TUI mode using Textual                                        │ architecture.md:777          │ Partial - placeholder           │
  └────────────────┴───────────────────────────────────────────────────────────────┴──────────────────────────────┴──────────────────────────────────┘

2.2 CLI Review Notes (filled)

- CLI-001: Pipeline execution is `elspeth run -s/--settings <file>` in `src/elspeth/cli.py` (README shows `elspeth --settings <file>`).
- CLI-002: No `--profile` flag implemented (no profile selection in `src/elspeth/cli.py`).
- CLI-003: `elspeth explain` exists in `src/elspeth/cli.py`, but currently returns “run not found”/placeholder output; TUI is `src/elspeth/tui/explain_app.py` (placeholder widgets).
- CLI-004: No `--full` flag implemented for `elspeth explain`.
- CLI-005: `elspeth validate -s/--settings` implemented in `src/elspeth/cli.py` (loads settings + validates `ExecutionGraph`).
- CLI-006: `elspeth plugins list` implemented in `src/elspeth/cli.py` via the `plugins` Typer group.
- CLI-007: No `elspeth status` command implemented.
- CLI-008: `--json` output exists for `explain` only (`src/elspeth/cli.py`); other commands are human-readable text only.
- CLI-009: Textual TUI wiring exists (`src/elspeth/tui/explain_app.py`) but is not yet backed by real lineage queries.

  ---

  1. SDA MODEL REQUIREMENTS

3.1 Sources (from architecture.md, subsystems/00-overview)
  ┌────────────────┬─────────────────────────────────────┬───────────────────────────┬──────────────────────────────┐
  │ Requirement ID │             Requirement             │          Source           │            Status            │
  ├────────────────┼─────────────────────────────────────┼───────────────────────────┼──────────────────────────────┤
  │ SDA-001        │ Exactly one source per run          │ CLAUDE.md, subsystems:303 │ Implemented                 │
  ├────────────────┼─────────────────────────────────────┼───────────────────────────┼──────────────────────────────┤
  │ SDA-002        │ Sources are stateless               │ architecture.md:103       │ Partial - not enforced      │
  ├────────────────┼─────────────────────────────────────┼───────────────────────────┼──────────────────────────────┤
  │ SDA-003        │ CSV source plugin                   │ CLAUDE.md                 │ Implemented                 │
  ├────────────────┼─────────────────────────────────────┼───────────────────────────┼──────────────────────────────┤
  │ SDA-004        │ JSON/JSONL source plugin            │ CLAUDE.md                 │ Implemented                 │
  ├────────────────┼─────────────────────────────────────┼───────────────────────────┼──────────────────────────────┤
  │ SDA-005        │ Database source plugin              │ README.md:172             │ Not implemented             │
  ├────────────────┼─────────────────────────────────────┼───────────────────────────┼──────────────────────────────┤
  │ SDA-006        │ HTTP API source plugin              │ README.md:172             │ Not implemented             │
  ├────────────────┼─────────────────────────────────────┼───────────────────────────┼──────────────────────────────┤
  │ SDA-007        │ Message queue source (blob storage) │ README.md:172             │ Not implemented             │
  └────────────────┴─────────────────────────────────────┴───────────────────────────┴──────────────────────────────┘
  3.2 Transforms (from architecture.md, subsystems/00-overview)
  ┌────────────────┬─────────────────────────────────────────────┬──────────────────────────────────────────┬───────────────────────────────────────┐
  │ Requirement ID │                 Requirement                 │                  Source                  │                Status                 │
  ├────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────┤
  │ SDA-008        │ Zero or more transforms, ordered            │ CLAUDE.md                                │ Implemented                          │
  ├────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────┤
  │ SDA-009        │ Transforms stateless between rows           │ architecture.md:104                      │ Partial - not enforced               │
  ├────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────┤
  │ SDA-010        │ Row Transform: 1 row → 1 row                │ CLAUDE.md, subsystems:312                │ Implemented                          │
  ├────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────┤
  │ SDA-011        │ Gate Transform: evaluate → routing decision │ CLAUDE.md, subsystems:313                │ Implemented                          │
  ├────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────┤
  │ SDA-012        │ Aggregation Transform: N rows → 1 result    │ CLAUDE.md, subsystems:314                │ Partial - infra only                 │
  ├────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────┤
  │ SDA-013        │ Coalesce Transform: merge parallel paths    │ CLAUDE.md, subsystems:315                │ Partial - infra only                 │
  ├────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────┤
  │ SDA-014        │ PassThrough transform (always continue)     │ architecture.md:81, 115                  │ Implemented                          │
  ├────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────┤
  │ SDA-015        │ ThresholdGate transform                     │ README.md:92-99, architecture.md:859-879 │ Implemented                          │
  ├────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────┤
  │ SDA-016        │ PatternGate transform                       │ README.md:92-99                          │ Not implemented                      │
  ├────────────────┼─────────────────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────┤
  │ SDA-017        │ LLM query transform                         │ README.md:103-105                        │ Not implemented                      │
  └────────────────┴─────────────────────────────────────────────┴──────────────────────────────────────────┴───────────────────────────────────────┘
  3.3 Sinks (from architecture.md, CLAUDE.md)
  ┌────────────────┬────────────────────────────────────────┬───────────────────────────┬────────────────────┐
  │ Requirement ID │              Requirement               │          Source           │       Status       │
  ├────────────────┼────────────────────────────────────────┼───────────────────────────┼────────────────────┤
  │ SDA-018        │ One or more sinks, named               │ CLAUDE.md, subsystems:305 │ Implemented       │
  ├────────────────┼────────────────────────────────────────┼───────────────────────────┼────────────────────┤
  │ SDA-019        │ Sinks are stateless                    │ architecture.md:105       │ Partial - stateful│
  ├────────────────┼────────────────────────────────────────┼───────────────────────────┼────────────────────┤
  │ SDA-020        │ CSV sink plugin                        │ CLAUDE.md                 │ Implemented       │
  ├────────────────┼────────────────────────────────────────┼───────────────────────────┼────────────────────┤
  │ SDA-021        │ JSON sink plugin                       │ CLAUDE.md                 │ Implemented       │
  ├────────────────┼────────────────────────────────────────┼───────────────────────────┼────────────────────┤
  │ SDA-022        │ Database sink plugin                   │ CLAUDE.md                 │ Implemented       │
  ├────────────────┼────────────────────────────────────────┼───────────────────────────┼────────────────────┤
  │ SDA-023        │ Webhook sink plugin                    │ architecture.md:847-849   │ Not implemented   │
  ├────────────────┼────────────────────────────────────────┼───────────────────────────┼────────────────────┤
  │ SDA-024        │ Sinks receive idempotency keys         │ architecture.md:611-618   │ Not implemented   │
  ├────────────────┼────────────────────────────────────────┼───────────────────────────┼────────────────────┤
  │ SDA-025        │ Sinks can flag non-idempotent behavior │ architecture.md:849       │ Partial - flag    │
  ├────────────────┼────────────────────────────────────────┼───────────────────────────┼────────────────────┤
  │ SDA-026        │ Artifact signing with HMAC-SHA256      │ architecture.md:914-917   │ Partial - export  │
  └────────────────┴────────────────────────────────────────┴───────────────────────────┴────────────────────┘
3.4 SDA Review Notes (filled)

- SDA-001: Single source enforced by `ElspethSettings.datasource` (`src/elspeth/core/config.py`) and `ExecutionGraph.validate()` (`src/elspeth/core/dag.py`).
- SDA-002: Not enforced by the framework; `CSVSource` caches a dataframe (`src/elspeth/plugins/sources/csv_source.py`), while `JSONSource` is effectively stateless (`src/elspeth/plugins/sources/json_source.py`).
- SDA-003: Implemented as `CSVSource` (`src/elspeth/plugins/sources/csv_source.py`).
- SDA-004: Implemented as `JSONSource` (`src/elspeth/plugins/sources/json_source.py`).
- SDA-005: No database source plugin in `src/elspeth/plugins/sources/`.
- SDA-006: No HTTP API source plugin in `src/elspeth/plugins/sources/`.
- SDA-007: No message-queue/blob source plugin in `src/elspeth/plugins/sources/`.
- SDA-008: Ordered transforms configured via `ElspethSettings.row_plugins` (`src/elspeth/core/config.py`) and executed in-order by `RowProcessor.process_row()` (`src/elspeth/engine/processor.py`).
- SDA-009: “Stateless transforms” is not enforced by `BaseTransform`/`BaseGate` (`src/elspeth/plugins/base.py`).
- SDA-010: Row transforms implemented via `BaseTransform.process()` + `TransformExecutor` (`src/elspeth/engine/executors.py`).
- SDA-011: Gate transforms implemented via `BaseGate.evaluate()` + `GateExecutor` (`src/elspeth/engine/executors.py`).
- SDA-012: Aggregation infra exists (`BaseAggregation`, `AggregationExecutor`), but config/graph wiring is not implemented (`src/elspeth/plugins/base.py`, `src/elspeth/engine/executors.py`, `src/elspeth/core/dag.py`).
- SDA-013: Coalesce infra exists in tokens/recorder, but there is no configurable coalesce node in `ExecutionGraph.from_config()` and no DAG work-queue processing (`src/elspeth/engine/tokens.py`, `src/elspeth/core/landscape/recorder.py`, `src/elspeth/engine/processor.py`).
- SDA-014: Implemented as `PassThrough` (`src/elspeth/plugins/transforms/passthrough.py`).
- SDA-015: Implemented as `ThresholdGate` (`src/elspeth/plugins/gates/threshold_gate.py`).
- SDA-016: No `PatternGate` plugin implemented (README example references it, but no code in `src/elspeth/plugins/gates/`).
- SDA-017: No LLM query transform implemented (no `llm_query` plugin in `src/elspeth/plugins/transforms/`).
- SDA-018: Named sinks required by `ElspethSettings.sinks` (non-empty) (`src/elspeth/core/config.py`).
- SDA-019: “Stateless sinks” is not enforced; built-in sinks keep handles/buffers (`src/elspeth/plugins/sinks/*.py`).
- SDA-020: Implemented as `CSVSink` (`src/elspeth/plugins/sinks/csv_sink.py`).
- SDA-021: Implemented as `JSONSink` (`src/elspeth/plugins/sinks/json_sink.py`).
- SDA-022: Implemented as `DatabaseSink` (`src/elspeth/plugins/sinks/database_sink.py`).
- SDA-023: No webhook sink plugin implemented (adapter supports artifact kind only: `src/elspeth/engine/adapters.py`).
- SDA-024: Idempotency keys are documented in `SinkProtocol` but not passed via `PluginContext` or sink APIs (`src/elspeth/plugins/protocols.py`, `src/elspeth/plugins/context.py`).
- SDA-025: `BaseSink.idempotent` exists but is not used by the engine for retry semantics (`src/elspeth/plugins/base.py`).
- SDA-026: HMAC-SHA256 signing exists for exported audit records (includes artifact records) via `LandscapeExporter(sign=True)` (`src/elspeth/core/landscape/exporter.py`); signatures are not stored in the `artifacts` table.

  ---

  1. ROUTING REQUIREMENTS
  ┌────────────────┬───────────────────────────────────────────────────────┬─────────────────────────────────────────┬────────────────────┐
  │ Requirement ID │                      Requirement                      │                 Source                  │       Status       │
  ├────────────────┼───────────────────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────┤
  │ RTE-001        │ Gates return RoutingAction with kind                  │ architecture.md:123-130                 │ Implemented       │
  ├────────────────┼───────────────────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────┤
  │ RTE-002        │ Routing kinds: continue, route_to_sink, fork_to_paths │ architecture.md:127                     │ Implemented       │
  ├────────────────┼───────────────────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────┤
  │ RTE-003        │ Routing modes: move (exit pipeline)                   │ architecture.md:129, 140-143            │ Implemented       │
  ├────────────────┼───────────────────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────┤
  │ RTE-004        │ Routing modes: copy (continue + route)                │ architecture.md:129, 144                │ Not implemented   │
  ├────────────────┼───────────────────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────┤
  │ RTE-005        │ Routing includes reason dict                          │ architecture.md:130                     │ Implemented       │
  ├────────────────┼───────────────────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────┤
  │ RTE-006        │ Multi-destination routing route: [A, B, C]            │ architecture.md:190, subsystems:364-368 │ Partial - fork    │
  └────────────────┴───────────────────────────────────────────────────────┴─────────────────────────────────────────┴────────────────────┘

4.1 Routing Review Notes (filled)

- RTE-001: `RoutingAction` + `GateResult` in `src/elspeth/plugins/results.py` (kind/mode/destinations + audit fields).
- RTE-002: Routing kinds modeled by `RoutingKind` + `RoutingAction` constructors in `src/elspeth/plugins/enums.py` and `src/elspeth/plugins/results.py`; `route` resolves to sink/continue via `GateExecutor` + `ExecutionGraph` route map (`src/elspeth/engine/executors.py`, `src/elspeth/core/dag.py`).
- RTE-003: MOVE routing recorded via `LandscapeRecorder.record_routing_event()` and honored as “exit pipeline” when `GateExecutor` resolves a route label to a sink (`src/elspeth/core/landscape/recorder.py`, `src/elspeth/engine/executors.py`).
- RTE-004: COPY mode exists (`RoutingMode.COPY`), but “continue + route” for a sink route is not supported: `RowProcessor` stops as soon as `sink_name` is set (`src/elspeth/engine/processor.py`).
- RTE-005: Reason dict supported and hashed (`RoutingAction.reason` + `routing_events.reason_hash`) (`src/elspeth/plugins/results.py`, `src/elspeth/core/landscape/recorder.py`); built-in gates emit reasons (e.g., `ThresholdGate` in `src/elspeth/plugins/gates/threshold_gate.py`).
- RTE-006: Multi-destination routing records multiple `routing_events` and forks child tokens (`RoutingAction.fork_to_paths`, `GateExecutor._record_routing()`, `TokenManager.fork_token()`), but the engine does not yet process child tokens (no DAG work-queue) (`src/elspeth/plugins/results.py`, `src/elspeth/engine/executors.py`, `src/elspeth/engine/processor.py`).

  ---

  1. DAG EXECUTION REQUIREMENTS
  ┌────────────────┬────────────────────────────────────┬────────────────────────────────┬─────────────────────────────────┐
  │ Requirement ID │            Requirement             │             Source             │             Status              │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼─────────────────────────────────┤
  │ DAG-001        │ Pipelines compile to DAG           │ architecture.md:166-184        │              TBD               │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼─────────────────────────────────┤
  │ DAG-002        │ DAG validation using NetworkX      │ CLAUDE.md, architecture.md:793 │              TBD               │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼─────────────────────────────────┤
  │ DAG-003        │ Acyclicity check on graph          │ architecture.md:793, CLAUDE.md │              TBD               │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼─────────────────────────────────┤
  │ DAG-004        │ Topological sort for execution     │ architecture.md:793, CLAUDE.md │              TBD               │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼─────────────────────────────────┤
  │ DAG-005        │ Linear pipelines as degenerate DAG │ architecture.md:228-241        │              TBD               │
  └────────────────┴────────────────────────────────────┴────────────────────────────────┴─────────────────────────────────┘

  ---

  1. TOKEN IDENTITY REQUIREMENTS
  ┌────────────────┬────────────────────────────────────────────┬─────────────────────────────────────────┬────────────────────┐
  │ Requirement ID │                Requirement                 │                 Source                  │       Status       │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────┤
  │ TOK-001        │ row_id = stable source row identity        │ architecture.md:209, CLAUDE.md          │        TBD        │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────┤
  │ TOK-002        │ token_id = row instance in DAG path        │ architecture.md:210, CLAUDE.md          │        TBD        │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────┤
  │ TOK-003        │ parent_token_id for fork/join lineage      │ architecture.md:211, CLAUDE.md          │        TBD        │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────┤
  │ TOK-004        │ Fork creates child tokens                  │ architecture.md:213-224                 │        TBD        │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────┤
  │ TOK-005        │ Join/coalesce merges tokens                │ architecture.md:213-224                 │        TBD        │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────┤
  │ TOK-006        │ Token parents table for multi-parent joins │ architecture.md:211, subsystems:152-159 │        TBD        │
  └────────────────┴────────────────────────────────────────────┴─────────────────────────────────────────┴────────────────────┘

  ---

  1. LANDSCAPE (AUDIT) REQUIREMENTS

7.1 Core Tables (from subsystems/00-overview:88-253)
  ┌────────────────┬──────────────────────────────────────────┬────────────────────┬────────────────────────┐
  │ Requirement ID │               Requirement                │       Source       │         Status         │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-001        │ runs table with all specified columns    │ subsystems:91-101  │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-002        │ runs.reproducibility_grade column        │ subsystems:98      │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-003        │ nodes table for execution graph          │ subsystems:103-116 │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-004        │ nodes.determinism column                 │ subsystems:110     │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-005        │ nodes.schema_hash column                 │ subsystems:113     │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-006        │ edges table for graph connections        │ subsystems:118-128 │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-007        │ edges.default_mode column (move/copy)    │ subsystems:126     │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-008        │ rows table for source rows               │ subsystems:130-140 │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-009        │ tokens table for row instances           │ subsystems:142-150 │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-010        │ token_parents table for joins            │ subsystems:152-159 │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-011        │ node_states table for processing         │ subsystems:161-179 │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-012        │ routing_events table for edge selections │ subsystems:181-193 │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-013        │ calls table for external calls           │ subsystems:195-210 │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-014        │ batches table for aggregations           │ subsystems:212-223 │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-015        │ batch_members table                      │ subsystems:225-231 │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-016        │ batch_outputs table                      │ subsystems:233-239 │          TBD          │
  ├────────────────┼──────────────────────────────────────────┼────────────────────┼────────────────────────┤
  │ LND-017        │ artifacts table for sink outputs         │ subsystems:241-252 │          TBD          │
  └────────────────┴──────────────────────────────────────────┴────────────────────┴────────────────────────┘
  7.2 Audit Recording Requirements (from architecture.md, CLAUDE.md)
  ┌────────────────┬──────────────────────────────────────────────┬──────────────────────────┬────────────────────┐
  │ Requirement ID │                 Requirement                  │          Source          │       Status       │
  ├────────────────┼──────────────────────────────────────────────┼──────────────────────────┼────────────────────┤
  │ LND-018        │ Every run with resolved configuration        │ architecture.md:249-250  │        TBD        │
  ├────────────────┼──────────────────────────────────────────────┼──────────────────────────┼────────────────────┤
  │ LND-019        │ Every row loaded from source                 │ architecture.md:252      │        TBD        │
  ├────────────────┼──────────────────────────────────────────────┼──────────────────────────┼────────────────────┤
  │ LND-020        │ Every transform with before/after state      │ architecture.md:253      │        TBD        │
  ├────────────────┼──────────────────────────────────────────────┼──────────────────────────┼────────────────────┤
  │ LND-021        │ Every external call recorded                 │ architecture.md:254      │        TBD        │
  ├────────────────┼──────────────────────────────────────────────┼──────────────────────────┼────────────────────┤
  │ LND-022        │ Every routing decision with reason           │ architecture.md:255      │        TBD        │
  ├────────────────┼──────────────────────────────────────────────┼──────────────────────────┼────────────────────┤
  │ LND-023        │ Every artifact produced                      │ architecture.md:256      │        TBD        │
  ├────────────────┼──────────────────────────────────────────────┼──────────────────────────┼────────────────────┤
  │ LND-024        │ explain() API with complete lineage          │ architecture.md:307-348  │        TBD        │
  ├────────────────┼──────────────────────────────────────────────┼──────────────────────────┼────────────────────┤
  │ LND-025        │ explain() by token_id for DAG precision      │ architecture.md:315, 345 │        TBD        │
  ├────────────────┼──────────────────────────────────────────────┼──────────────────────────┼────────────────────┤
  │ LND-026        │ explain() by row_id, sink for disambiguation │ architecture.md:346      │        TBD        │
  └────────────────┴──────────────────────────────────────────────┴──────────────────────────┴────────────────────┘
  7.3 Invariants (from architecture.md:267-276)
  ┌────────────────┬────────────────────────────────────────────────────┬─────────────────────┬────────────────────┐
  │ Requirement ID │                    Requirement                     │       Source        │       Status       │
  ├────────────────┼────────────────────────────────────────────────────┼─────────────────────┼────────────────────┤
  │ LND-027        │ Run stores resolved config (not just hash)         │ architecture.md:270 │        TBD        │
  ├────────────────┼────────────────────────────────────────────────────┼─────────────────────┼────────────────────┤
  │ LND-028        │ External calls link to existing spans              │ architecture.md:271 │        TBD        │
  ├────────────────┼────────────────────────────────────────────────────┼─────────────────────┼────────────────────┤
  │ LND-029        │ Strict ordering: transforms by (sequence, attempt) │ architecture.md:272 │        TBD        │
  ├────────────────┼────────────────────────────────────────────────────┼─────────────────────┼────────────────────┤
  │ LND-030        │ No orphan records (foreign keys enforced)          │ architecture.md:273 │        TBD        │
  ├────────────────┼────────────────────────────────────────────────────┼─────────────────────┼────────────────────┤
  │ LND-031        │ (run_id, row_id) unique                            │ architecture.md:274 │        TBD        │
  ├────────────────┼────────────────────────────────────────────────────┼─────────────────────┼────────────────────┤
  │ LND-032        │ Canonical JSON contract versioned                  │ architecture.md:275 │        TBD        │
  └────────────────┴────────────────────────────────────────────────────┴─────────────────────┴────────────────────┘
  ---

  1. CANONICAL JSON REQUIREMENTS
  ┌────────────────┬───────────────────────────────────────────────┬────────────────────────────────────┬────────────────┐
  │ Requirement ID │                  Requirement                  │               Source               │     Status     │
  ├────────────────┼───────────────────────────────────────────────┼────────────────────────────────────┼────────────────┤
  │ CAN-001        │ Two-phase canonicalization                    │ architecture.md:358-364, CLAUDE.md │      TBD      │
  ├────────────────┼───────────────────────────────────────────────┼────────────────────────────────────┼────────────────┤
  │ CAN-002        │ Phase 1: Normalize pandas/numpy to primitives │ architecture.md:384-448            │      TBD      │
  ├────────────────┼───────────────────────────────────────────────┼────────────────────────────────────┼────────────────┤
  │ CAN-003        │ Phase 2: RFC 8785/JCS serialization           │ architecture.md:450-464            │      TBD      │
  ├────────────────┼───────────────────────────────────────────────┼────────────────────────────────────┼────────────────┤
  │ CAN-004        │ NaN/Infinity STRICTLY REJECTED                │ architecture.md:394-403, CLAUDE.md │      TBD      │
  ├────────────────┼───────────────────────────────────────────────┼────────────────────────────────────┼────────────────┤
  │ CAN-005        │ numpy.int64 → Python int                      │ architecture.md:489                │      TBD      │
  ├────────────────┼───────────────────────────────────────────────┼────────────────────────────────────┼────────────────┤
  │ CAN-006        │ numpy.float64 → Python float                  │ architecture.md:490                │      TBD      │
  ├────────────────┼───────────────────────────────────────────────┼────────────────────────────────────┼────────────────┤
  │ CAN-007        │ numpy.bool_ → Python bool                     │ architecture.md:491                │      TBD      │
  ├────────────────┼───────────────────────────────────────────────┼────────────────────────────────────┼────────────────┤
  │ CAN-008        │ pandas.Timestamp → UTC ISO8601                │ architecture.md:492                │      TBD      │
  ├────────────────┼───────────────────────────────────────────────┼────────────────────────────────────┼────────────────┤
  │ CAN-009        │ NaT, NA → null                                │ architecture.md:493                │      TBD      │
  ├────────────────┼───────────────────────────────────────────────┼────────────────────────────────────┼────────────────┤
  │ CAN-010        │ Version string sha256-rfc8785-v1              │ architecture.md:380, CLAUDE.md     │      TBD      │
  ├────────────────┼───────────────────────────────────────────────┼────────────────────────────────────┼────────────────┤
  │ CAN-011        │ Cross-process hash stability test             │ architecture.md:931                │      TBD       │
  └────────────────┴───────────────────────────────────────────────┴────────────────────────────────────┴────────────────┘

  ---

  1. PAYLOAD STORE REQUIREMENTS
  ┌────────────────┬───────────────────────────────────────────┬─────────────────────────────────────────────┬──────────────────────────────┐
  │ Requirement ID │                Requirement                │                   Source                    │            Status            │
  ├────────────────┼───────────────────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────┤
  │ PLD-001        │ PayloadStore protocol with put/get/exists │ architecture.md:524-530, subsystems:675-680 │             TBD             │
  ├────────────────┼───────────────────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────┤
  │ PLD-002        │ PayloadRef return type                    │ architecture.md:527                         │             TBD             │
  ├────────────────┼───────────────────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────┤
  │ PLD-003        │ Filesystem backend                        │ subsystems:670                              │             TBD             │
  ├────────────────┼───────────────────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────┤
  │ PLD-004        │ S3/blob storage backend                   │ subsystems:670                              │             TBD             │
  ├────────────────┼───────────────────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────┤
  │ PLD-005        │ Inline backend                            │ subsystems:670                              │             TBD             │
  ├────────────────┼───────────────────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────┤
  │ PLD-006        │ Retention policies                        │ architecture.md:539-549, subsystems:685-689 │             TBD             │
  ├────────────────┼───────────────────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────┤
  │ PLD-007        │ Hash retained after payload purge         │ architecture.md:546                         │             TBD              │
  ├────────────────┼───────────────────────────────────────────┼─────────────────────────────────────────────┼──────────────────────────────┤
  │ PLD-008        │ Optional compression                      │ subsystems:669                              │             TBD             │
  └────────────────┴───────────────────────────────────────────┴─────────────────────────────────────────────┴──────────────────────────────┘

  ---

  1. FAILURE SEMANTICS REQUIREMENTS
  ┌────────────────┬────────────────────────────────────────────────────────┬─────────────────────────────────────────┬───────────────────────┐
  │ Requirement ID │                      Requirement                       │                 Source                  │        Status         │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ FAI-001        │ Token terminal states: COMPLETED                       │ architecture.md:575                     │         TBD          │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ FAI-002        │ Token terminal states: ROUTED                          │ architecture.md:576                     │         TBD          │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ FAI-003        │ Token terminal states: FORKED                          │ architecture.md:577                     │         TBD          │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ FAI-004        │ Token terminal states: CONSUMED_IN_BATCH               │ architecture.md:578                     │         TBD          │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ FAI-005        │ Token terminal states: COALESCED                       │ architecture.md:579                     │         TBD          │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ FAI-006        │ Token terminal states: QUARANTINED                     │ architecture.md:580                     │         TBD          │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ FAI-007        │ Token terminal states: FAILED                          │ architecture.md:581                     │         TBD          │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ FAI-008        │ Terminal states DERIVED, not stored                    │ architecture.md:571-572, subsystems:279 │         TBD          │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ FAI-009        │ Every token reaches exactly one terminal state         │ architecture.md:569, subsystems:281     │          TBD          │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ FAI-010        │ TransformResult with status/row/reason/retryable       │ architecture.md:590-598                 │         TBD          │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ FAI-011        │ Retry: (run_id, row_id, transform_seq, attempt) unique │ architecture.md:603-605, CLAUDE.md      │         TBD          │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ FAI-012        │ Each retry attempt recorded separately                 │ architecture.md:604                     │         TBD          │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ FAI-013        │ Backoff metadata captured                              │ architecture.md:606                     │         TBD          │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ FAI-014        │ At-least-once delivery                                 │ architecture.md:619-621                 │         TBD          │
  └────────────────┴────────────────────────────────────────────────────────┴─────────────────────────────────────────┴───────────────────────┘

  ---

  1. EXTERNAL CALL RECORDING REQUIREMENTS
  ┌────────────────┬─────────────────────────────────────────────┬─────────────────────────┬──────────────────────────────┐
  │ Requirement ID │                 Requirement                 │         Source          │            Status            │
  ├────────────────┼─────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────┤
  │ EXT-001        │ Record: provider identifier                 │ architecture.md:695     │             TBD             │
  ├────────────────┼─────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────┤
  │ EXT-002        │ Record: model/version                       │ architecture.md:696     │             TBD             │
  ├────────────────┼─────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────┤
  │ EXT-003        │ Record: request hash + payload ref          │ architecture.md:697     │             TBD             │
  ├────────────────┼─────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────┤
  │ EXT-004        │ Record: response hash + payload ref         │ architecture.md:698     │             TBD             │
  ├────────────────┼─────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────┤
  │ EXT-005        │ Record: latency, status code, error details │ architecture.md:699     │             TBD             │
  ├────────────────┼─────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────┤
  │ EXT-006        │ Run modes: live, replay, verify             │ architecture.md:655-660 │             TBD             │
  ├────────────────┼─────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────┤
  │ EXT-007        │ Verify mode uses DeepDiff                   │ architecture.md:667-687 │             TBD             │
  ├────────────────┼─────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────┤
  │ EXT-008        │ Reproducibility grades: FULL_REPRODUCIBLE   │ architecture.md:644     │             TBD             │
  ├────────────────┼─────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────┤
  │ EXT-009        │ Reproducibility grades: REPLAY_REPRODUCIBLE │ architecture.md:644     │             TBD             │
  ├────────────────┼─────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────┤
  │ EXT-010        │ Reproducibility grades: ATTRIBUTABLE_ONLY   │ architecture.md:644     │             TBD             │
  └────────────────┴─────────────────────────────────────────────┴─────────────────────────┴──────────────────────────────┘

  ---

  1. DATA GOVERNANCE REQUIREMENTS
  ┌────────────────┬──────────────────────────────────────────────────────┬────────────────────────────────────┬──────────────────────────────┐
  │ Requirement ID │                     Requirement                      │               Source               │            Status            │
  ├────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────┼──────────────────────────────┤
  │ GOV-001        │ Secrets NEVER stored - HMAC fingerprint only         │ architecture.md:718-744, CLAUDE.md │             TBD             │
  ├────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────┼──────────────────────────────┤
  │ GOV-002        │ secret_fingerprint() function using HMAC             │ architecture.md:729-737            │             TBD             │
  ├────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────┼──────────────────────────────┤
  │ GOV-003        │ Fingerprint key loaded from environment              │ architecture.md:746-749            │             TBD             │
  ├────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────┼──────────────────────────────┤
  │ GOV-004        │ Configurable redaction profiles                      │ architecture.md:708-711            │             TBD             │
  ├────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────┼──────────────────────────────┤
  │ GOV-005        │ Access levels: Operator (redacted)                   │ architecture.md:753-755            │             TBD              │
  │ GOV-006        │ Access levels: Auditor (full)                        │ architecture.md:756                │          TBD          │
  ├────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────┼──────────────────────────────┤
  │ GOV-007        │ Access levels: Admin (retention/purge)               │ architecture.md:757                │          TBD           │
  ├────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────┼──────────────────────────────┤
  │ GOV-008        │ elspeth explain --full requires ELSPETH_AUDIT_ACCESS │ architecture.md:760-766            │          TBD           │
  └────────────────┴──────────────────────────────────────────────────────┴────────────────────────────────────┴──────────────────────────────┘

  ---

  1. PLUGIN SYSTEM REQUIREMENTS
  ┌────────────────┬────────────────────────────────────────────────────────┬─────────────────────────────────────────┬──────────────────────────────────────┐
  │ Requirement ID │                      Requirement                       │                 Source                  │                Status                │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┤
  │ PLG-001        │ pluggy hookspecs for Source, Transform, Sink           │ architecture.md:940, CLAUDE.md          │                 TBD                 │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┤
  │ PLG-002        │ Plugin discovery and registration                      │ subsystems:293-295                      │                 TBD                 │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┤
  │ PLG-003        │ Plugin instance lifecycle management                   │ subsystems:295                          │                 TBD                 │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┤
  │ PLG-004        │ TracedTransformPlugin base class                       │ architecture.md:942                     │                 TBD                 │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┤
  │ PLG-005        │ Gate protocol with routing                             │ architecture.md:943                     │                 TBD                 │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┤
  │ PLG-006        │ RowOutcome model                                       │ architecture.md:944                     │                 TBD                 │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┤
  │ PLG-007        │ Schema contracts for plugin I/O                        │ architecture.md:945, subsystems:499-545 │                 TBD                 │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┤
  │ PLG-008        │ Input/output schema declaration                        │ subsystems:502-516                      │                 TBD                 │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┤
  │ PLG-009        │ Pipeline schema validation at config time              │ subsystems:519-520                      │                 TBD                 │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┤
  │ PLG-010        │ Coalesce policies: require_all, quorum(n), best_effort │ subsystems:319-325                      │                 TBD                 │
  ├────────────────┼────────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┤
  │ PLG-011        │ Coalesce correlation key strategies                    │ subsystems:330-336                      │                 TBD                 │
  └────────────────┴────────────────────────────────────────────────────────┴─────────────────────────────────────────┴──────────────────────────────────────┘

  ---

  1. ENGINE REQUIREMENTS
  ┌────────────────┬────────────────────────────────────────────┬─────────────────────────────────────┬──────────────────────────┐
  │ Requirement ID │                Requirement                 │               Source                │          Status          │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────────┼──────────────────────────┤
  │ ENG-001        │ RowProcessor with span lifecycle           │ architecture.md:950, subsystems:569 │           TBD           │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────────┼──────────────────────────┤
  │ ENG-002        │ Retry with attempt tracking (tenacity)     │ architecture.md:951, CLAUDE.md      │           TBD           │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────────┼──────────────────────────┤
  │ ENG-003        │ Artifact pipeline (topological sort)       │ architecture.md:952                 │           TBD           │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────────┼──────────────────────────┤
  │ ENG-004        │ Standard orchestrator                      │ architecture.md:953                 │           TBD           │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────────┼──────────────────────────┤
  │ ENG-005        │ OpenTelemetry span emission                │ architecture.md:954, CLAUDE.md      │           TBD           │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────────┼──────────────────────────┤
  │ ENG-006        │ Aggregation accept/trigger/flush lifecycle │ subsystems:387-391                  │           TBD           │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────────┼──────────────────────────┤
  │ ENG-007        │ Aggregation crash recovery via query       │ subsystems:476-495                  │           TBD           │
  └────────────────┴────────────────────────────────────────────┴─────────────────────────────────────┴──────────────────────────┘

  ---

  1. PRODUCTION HARDENING REQUIREMENTS (Phase 5)
  ┌────────────────┬────────────────────────────────────┬───────────────────────────────────────────────┬────────────────────┐
  │ Requirement ID │            Requirement             │                    Source                     │       Status       │
  ├────────────────┼────────────────────────────────────┼───────────────────────────────────────────────┼────────────────────┤
  │ PRD-001        │ Checkpointing with replay support  │ architecture.md:969, README.md:180            │        TBD        │
  ├────────────────┼────────────────────────────────────┼───────────────────────────────────────────────┼────────────────────┤
  │ PRD-002        │ Rate limiting using pyrate-limiter │ architecture.md:970, README.md:182, CLAUDE.md │        TBD        │
  ├────────────────┼────────────────────────────────────┼───────────────────────────────────────────────┼────────────────────┤
  │ PRD-003        │ Retention and purge jobs           │ architecture.md:971                           │        TBD        │
  ├────────────────┼────────────────────────────────────┼───────────────────────────────────────────────┼────────────────────┤
  │ PRD-004        │ Redaction profiles                 │ architecture.md:972                           │        TBD        │
  ├────────────────┼────────────────────────────────────┼───────────────────────────────────────────────┼────────────────────┤
  │ PRD-005        │ Concurrent processing              │ README.md:183                                 │        TBD        │
  └────────────────┴────────────────────────────────────┴───────────────────────────────────────────────┴────────────────────┘

  ---

  1. TECHNOLOGY STACK REQUIREMENTS
  ┌────────────────┬────────────────────────────────────┬────────────────────────────────┬──────────────────────────────┐
  │ Requirement ID │            Requirement             │             Source             │            Status            │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-001        │ CLI: Typer                         │ architecture.md:776, CLAUDE.md │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-002        │ TUI: Textual                       │ architecture.md:777, CLAUDE.md │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-003        │ Configuration: Dynaconf + Pydantic │ architecture.md:778, CLAUDE.md │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-004        │ Plugins: pluggy                    │ architecture.md:779, CLAUDE.md │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-005        │ Data: pandas                       │ architecture.md:780, CLAUDE.md │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-006        │ HTTP: httpx                        │ architecture.md:781            │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-007        │ Database: SQLAlchemy Core          │ architecture.md:782, CLAUDE.md │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-008        │ Migrations: Alembic                │ architecture.md:783, CLAUDE.md │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-009        │ Retries: tenacity                  │ architecture.md:784, CLAUDE.md │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-010        │ Canonical JSON: rfc8785            │ architecture.md:792, CLAUDE.md │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-011        │ DAG Validation: NetworkX           │ architecture.md:793, CLAUDE.md │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-012        │ Observability: OpenTelemetry       │ architecture.md:794, CLAUDE.md │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-013        │ Tracing UI: Jaeger                 │ architecture.md:795, CLAUDE.md │             TBD              │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-014        │ Logging: structlog                 │ architecture.md:796, CLAUDE.md │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-015        │ Rate Limiting: pyrate-limiter      │ architecture.md:797, CLAUDE.md │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-016        │ Diffing: DeepDiff                  │ architecture.md:798, CLAUDE.md │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-017        │ Property Testing: Hypothesis       │ architecture.md:799, CLAUDE.md │             TBD             │
  ├────────────────┼────────────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
  │ TSK-018        │ LLM: LiteLLM                       │ architecture.md:804, CLAUDE.md │             TBD             │
  └────────────────┴────────────────────────────────────┴────────────────────────────────┴──────────────────────────────┘

  ---

  1. LANDSCAPE EXPORT REQUIREMENTS
  ┌────────────────┬──────────────────────────────────────────────────┬────────────────┬────────────────┐
  │ Requirement ID │                   Requirement                    │     Source     │     Status     │
  ├────────────────┼──────────────────────────────────────────────────┼────────────────┼────────────────┤
  │ EXP-001        │ Export audit trail to configured sink            │ This plan      │      TBD      │
  ├────────────────┼──────────────────────────────────────────────────┼────────────────┼────────────────┤
  │ EXP-002        │ Optional HMAC signing per record                 │ This plan      │      TBD      │
  ├────────────────┼──────────────────────────────────────────────────┼────────────────┼────────────────┤
  │ EXP-003        │ Manifest with final hash for tamper detection    │ This plan      │      TBD      │
  ├────────────────┼──────────────────────────────────────────────────┼────────────────┼────────────────┤
  │ EXP-004        │ CSV and JSON format options                      │ This plan      │      TBD      │
  ├────────────────┼──────────────────────────────────────────────────┼────────────────┼────────────────┤
  │ EXP-005        │ Export happens post-run via config, not CLI      │ This plan      │      TBD      │
  ├────────────────┼──────────────────────────────────────────────────┼────────────────┼────────────────┤
  │ EXP-006        │ Include all record types (batches, token_parents)│ Code review    │      TBD      │
  └────────────────┴──────────────────────────────────────────────────┴────────────────┴────────────────┘

  ---
  CRITICAL DIVERGENCES SUMMARY

  TBD
  ---

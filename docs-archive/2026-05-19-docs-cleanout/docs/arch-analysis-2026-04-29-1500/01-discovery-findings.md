# 01 — Discovery Findings (L1 holistic scan)

Scope: holistic orientation across the production source tree at `src/elspeth/` (121,392 LOC, 11 top-level subsystems). Per-subsystem catalog entries are produced in the next wave; this document looks at the system as a whole.

Source of LOC inventory: verified pre-pass counts (supplied in the task brief). Source of layer model: `scripts/cicd/enforce_tier_model.py` lines 39–64 (LAYER_HIERARCHY, LAYER_NAMES) extracted into `temp/tier-model-oracle.txt`. Tier-model enforcer ran clean — no upward imports detected at scan time (`temp/tier-model-oracle.txt`, line 3 and lines 71).

## Codebase shape

ELSPETH's production code is concentrated in two large L3 subsystems and three large foundation subsystems. The **plugins** tree is the single largest codebase at ~30.4k LOC (98 files), followed by **web** at ~22.6k LOC (72 files), **core** at ~20.8k LOC, **engine** at ~17.4k LOC, and **contracts** at ~17.4k LOC. Together those five subsystems hold ~108.6k LOC — roughly **89%** of the production tree. The remaining six subsystems (mcp, telemetry, cli, composer_mcp, tui, testing) total ~12.8k LOC across 41 files, i.e. ~11% of LOC.

Sorted by LOC, % of 121,392:

- plugins/ — 30,399 LOC (25.0%), 98 files, L3
- web/ — 22,558 LOC (18.6%), 72 files, L3
- core/ — 20,791 LOC (17.1%), 49 files, L1
- engine/ — 17,425 LOC (14.4%), 36 files, L2
- contracts/ — 17,403 LOC (14.3%), 63 files, L0
- mcp/ — 4,114 LOC (3.4%), 9 files, L3
- telemetry/ — 2,884 LOC (2.4%), 14 files, L3
- cli (4 files at root) — 2,942 LOC (2.4%), L3
- tui/ — 1,175 LOC (1.0%), 9 files, L3
- testing/ — 877 LOC (0.7%), 2 files, L3
- composer_mcp/ — 824 LOC (0.7%), 3 files, L3

By **file count**, plugins (98) and web (72) again dominate, with contracts (63), core (49) and engine (36) following. Composite subsystems (≥4 sub-pkgs OR ≥10k LOC OR ≥20 files) are: contracts, core, engine, plugins, web. The remaining six are leaves.

Top 10 single Python files by line count (deepest first; from `find src/elspeth -name '*.py' -exec wc -l {} +` sorted descending):

| File | LOC | Subsystem |
|---|--:|---|
| `web/composer/tools.py` | 3,804 | web |
| `engine/orchestrator/core.py` | 3,281 | engine |
| `engine/processor.py` | 2,700 | engine |
| `cli.py` | 2,357 | cli (root) |
| `core/config.py` | 2,227 | core |
| `core/dag/graph.py` | 1,968 | core |
| `core/landscape/execution_repository.py` | 1,750 | core |
| `web/composer/state.py` | 1,710 | web |
| `engine/coalesce_executor.py` | 1,603 | engine |
| `plugins/transforms/llm/azure_batch.py` | 1,592 | plugins |

Every file in this top-10 exceeds the 1,500 LOC threshold and is therefore an **L2-deep-dive candidate** under Delta 6 — internals are not analysed in this pre-pass. (For context, the next file is `core/landscape/data_flow_repository.py` at 1,590 LOC, also a candidate; `contracts/errors.py` at 1,566 LOC, also a candidate. The threshold catches roughly the top dozen files.)

## Layer architecture (from enforcer + path mapping)

The CI-enforced layer hierarchy is defined in `scripts/cicd/enforce_tier_model.py:52–64` and reproduced verbatim in the oracle artefact (`temp/tier-model-oracle.txt:50–64`):

```
contracts → 0  (L0, leaf, imports nothing above)
core      → 1  (L1, can import contracts only)
engine    → 2  (L2, can import core, contracts)
everything else → 3  (L3/application — implicit)
```

The enforcer ran clean against the current tree (`temp/tier-model-oracle.txt:3` — "No bug-hiding patterns detected. Check passed."), confirming the codebase is layer-conformant at scan time. Rule **L1** (runtime upward import) is a CI failure; rule **TC** (TYPE_CHECKING upward import) is a warning only.

Subsystem-to-layer mapping:

- **L0 / contracts** — `contracts/`
- **L1 / core** — `core/`
- **L2 / engine** — `engine/`
- **L3 / application** — `plugins/`, `web/`, `mcp/`, `composer_mcp/`, `telemetry/`, `tui/`, `testing/`, and the CLI root (`cli.py`, `cli_helpers.py`, `cli_formatters.py`, `__init__.py`)

Cross-layer edge structure (deterministic from layer membership, before reading any source):

- L1 → L0 only
- L2 → {L0, L1}
- L3 → {L0, L1, L2}
- **L3 ↔ L3 edges are layer-permitted but unconstrained.** The enforcer does not police them; flagging is deferred to the L2 dispatch wave (per oracle artefact line 78).

## Entry points and runtime surfaces

Confirmed verbatim from `pyproject.toml` `[project.scripts]` and `[project.entry-points.pytest11]` (lines 211–218):

- `elspeth = "elspeth.cli:app"` — primary CLI entry (Typer app in `cli.py`).
- `elspeth-mcp = "elspeth.mcp:main"` — Landscape-analysis MCP server.
- `elspeth-composer = "elspeth.composer_mcp:main"` — pipeline-composer MCP server (the second MCP surface).
- `check-contracts = "scripts.check_contracts:main"` — contracts verification script (lives in `scripts/`, not `src/`).
- `elspeth-xdist-auto = "elspeth.testing.pytest_xdist_auto"` — pytest plugin entry exported by the `testing/` subsystem.

Web server entry: `web/` is L3 with 72 files / 22.6k LOC; FastAPI is gated behind the `webui` extra (pyproject lines 135–145). No web entry is exposed via `[project.scripts]`, suggesting the web app is launched via `uvicorn` against a FastAPI module inside `web/` (entry module identification deferred to the catalog wave).

TUI entry: `tui/` is L3 with 9 files / 1.2k LOC and uses `textual`. The CLAUDE.md `elspeth explain` invocation indicates the TUI is launched as a CLI subcommand rather than a separate console_script, so it surfaces through `cli.py` rather than its own `[project.scripts]` line.

## Technology stack signals

Read from `pyproject.toml` only.

**Python constraint:** `requires-python = ">=3.12"` (line 6). Mypy targets 3.13 (line 277); ruff targets py313 (line 234).

**Required runtime dependencies** (lines 20–67, one-line roles):

- `typer` — CLI framework. `textual` — TUI framework.
- `dynaconf` — layered configuration. `pydantic` v2 — data validation. `python-dotenv`, `pyyaml` — config file loading.
- `pluggy` — plugin architecture (same library pytest uses).
- `numpy`, `pandas` — numeric/dataframe data handling.
- `httpx` — async HTTP client.
- `sqlalchemy` v2 — SQL toolkit (Landscape backbone).
- `tenacity` — retry/backoff.
- `rfc8785` — RFC 8785 / JCS canonical JSON for hash-stable serialisation.
- `networkx` — DAG validation.
- `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp` — telemetry.
- `structlog` — structured logging.
- `pyrate-limiter` — rate limiting.
- `deepdiff` — verify-mode diffs.

**Optional dependency groups** (lines 69–209):

- `dev` — test/lint/type tooling (pytest + plugins, hypothesis, mypy, ruff, pre-commit, mutmut, errorworks, starlette, respx).
- `llm` — LLM plugin pack (jinja2, openai, litellm).
- `azure` — Azure plugin pack (Blob, Identity, Key Vault, Monitor OpenTelemetry).
- `web` — HTML scraping (html2text, beautifulsoup4) — web scraping plugins, NOT the web UI server.
- `rag` — `chromadb` for RAG retrieval.
- `mcp` — Model Context Protocol SDK.
- `security` — `sqlcipher3` for AES-256 encryption at rest.
- `webui` — FastAPI/uvicorn/PyJWT/python-multipart/websockets/bcrypt/litellm — the LLM Composer MVP. (Note: `web` and `webui` are different gates; the former is web scraping, the latter is the web UI server.)
- `tracing-langfuse` and umbrella `tracing` — Tier-2 LLM tracing providers.
- `all` — union of every group.

## Configuration & deployment surfaces

**Settings**: configuration is layered Dynaconf + Pydantic. `core/config.py` (2,227 LOC, top-5 file by depth) holds the runtime config, with a Settings→Runtime contract enforced by `scripts/check_contracts:main` (registered as `check-contracts` console script). Per CLAUDE.md, the precedence order is: runtime overrides → pipeline `settings.yaml` → profile YAML → plugin-pack defaults → system defaults. Contract details are deferred to the catalog wave.

**Plugin registration**: pluggy ≥1.6 is a required dependency (pyproject line 32). All Sources, Transforms, and Sinks are system-owned plugins (per CLAUDE.md "Plugin Ownership"). Per-pack registration mechanics (specs vs. impls, hookimpl wiring) are deferred to the catalog wave; this discovery pass intentionally does not crack open `plugins/__init__.py` for content.

**Audit trail (Landscape)**: owned by `core/landscape/`, with two of the largest files in the tree — `execution_repository.py` (1,750 LOC) and `data_flow_repository.py` (1,590 LOC) — sitting inside that subdirectory. The Landscape is the source of truth for the audit trail (per CLAUDE.md "Auditability Standard"). Database connectivity is via SQLAlchemy 2.x, optionally encrypted at rest via the `security` extra (`sqlcipher3`). The `mcp/` subsystem (4,114 LOC) provides read-only MCP access to the Landscape for debugging, exposed through `elspeth-mcp`.

## What this map will NOT cover (Delta 6 deferrals — repeat for clarity)

- **`src/elspeth/web/frontend/`** (TypeScript / React) — out of scope; a separate frontend pass is required. The 22,558 LOC figure for `web/` is Python-only.
- **`tests/`** (~351k LOC) — out of scope; a separate test-architecture pass is required.
- **`examples/`** — 36 entries (one per directory plus an `AGENTS.md` file) by `ls examples | wc -l`. No internal analysis performed.
- **`scripts/`** (CI / tooling) — only the layer enforcer (`scripts/cicd/enforce_tier_model.py`) and freeze-guard enforcer (`scripts/cicd/enforce_freeze_guards.py`, referenced in CLAUDE.md) are noted. `scripts/check_contracts.py` is registered as a console_script but its internals are deferred.
- **Any single Python file >1,500 LOC** — flagged at subsystem level (top-10 list above), internals deferred to L2 deep-dive.

## Confidence

- **Codebase shape** — High: counts are taken from the verified inventory and a single deterministic `find … -exec wc -l` invocation; no inference.
- **Layer architecture** — High: extracted directly from `enforce_tier_model.py` source and a clean enforcer run captured in the oracle artefact.
- **Entry points** — High for the five `[project.scripts]` / `entry-points` declarations (verbatim from `pyproject.toml`); Medium for the web entry module (gated extra is confirmed; the actual ASGI entry symbol is not — deferred).
- **Technology stack** — High: read directly from `pyproject.toml` with no source-file inference.
- **Configuration & deployment** — Medium: Dynaconf + Pydantic + pluggy are confirmed by `pyproject.toml` and CLAUDE.md, but the actual registration wiring and Settings→Runtime mapping have not been opened.
- **Deferrals** — High: scope explicitly bounded by Delta 6.

## Open questions for the L1 catalog wave

1. **Foundation split between `contracts/` (L0, 17.4k LOC) and `core/` (L1, 20.8k LOC).** Both are described as "foundation" layers. What is the responsibility cut — pure types/protocols/enums in `contracts/` vs. stateful primitives (Landscape, DAG, config, canonical JSON) in `core/`? Confirm there are no implementation-bearing modules in `contracts/`.
2. **`web/` internal architecture.** Is `web/` a single FastAPI application or does it contain its own pluggable surface (the `composer/tools.py` file at 3,804 LOC and `composer/state.py` at 1,710 LOC strongly suggest a tool-registry pattern parallel to MCP)? What is the relationship between `web/composer/` and the `composer_mcp/` subsystem?
3. **The two MCP surfaces.** `mcp/` (4,114 LOC, 9 files) vs. `composer_mcp/` (824 LOC, 3 files): one is read-only Landscape analysis, the other is pipeline composition. Do they share infrastructure, or are they parallel implementations? Why is `composer_mcp/` not nested inside `mcp/`?
4. **CLI sprawl in `cli.py` (2,357 LOC).** What command surface does Typer expose, and how does it route to engine/web/tui/mcp? Are there subcommands that could split out cleanly?
5. **Engine orchestrator depth.** `engine/orchestrator/core.py` (3,281 LOC) and `engine/processor.py` (2,700 LOC) together hold ~34% of `engine/`. What is the orchestrator/processor responsibility cut, and where does `coalesce_executor.py` (1,603 LOC) sit relative to them?
6. **Plugin registration topology.** `plugins/` is the largest subsystem (98 files, 30.4k LOC). What sub-package layout (sources/transforms/sinks/infrastructure) does it use, and how does pluggy registration discover them? Are plugin packs (LLM, Azure, RAG, web-scraping) self-contained per the optional-dependency groupings?
7. **L3↔L3 dispatch flagging.** The enforcer does not police lateral edges among L3 subsystems. The catalog wave should record outbound L3-targets for each L3 subsystem (which L3 imports which) so the L2 pass can build the unconstrained-edge graph.

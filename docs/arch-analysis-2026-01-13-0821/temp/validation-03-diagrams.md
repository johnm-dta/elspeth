# Validation: 03-diagrams.md

## Validation Mode
Self-validation (tooling limitation: no independent subagent available in this environment).

## Evidence Mapping (Diagram → Code)

### Context + Containers
- Package boundaries and module inventory come from `src/elspeth/` file listing (`rg --files src/elspeth`).
- Declared CLI entry point is from `pyproject.toml` (`[project.scripts] elspeth = "elspeth.cli:app"`).

### Engine component interactions
- `Orchestrator` creates `LandscapeRecorder`, calls `begin_run/register_node/register_edge/complete_run`:
  - `src/elspeth/engine/orchestrator.py`
- `RowProcessor` creates initial tokens via `TokenManager.create_initial_token()`:
  - `src/elspeth/engine/processor.py`
  - `src/elspeth/engine/tokens.py`
- Executors wrap plugin calls and record node states/routing/batches/artifacts:
  - `src/elspeth/engine/executors.py`
- `SpanFactory` provides span contexts used by executors:
  - `src/elspeth/engine/spans.py`

### Landscape components
- DB/Schema/Recorder structure:
  - `src/elspeth/core/landscape/database.py`
  - `src/elspeth/core/landscape/schema.py`
  - `src/elspeth/core/landscape/recorder.py`

### Layering tension (explicitly shown)
- `LandscapeRecorder` imports enums from plugins layer:
  - `src/elspeth/core/landscape/recorder.py`

## Consistency Checks
- All arrows represent a real call or import edge observed in code (no “pure design” edges except those marked planned).
- “Sink adapter interface” is labeled to reflect current engine expectation (`write(rows)->artifact info`) vs plugin protocol (`write(row)->None`).

## Result
APPROVED (with noted limitations)

## Limitations
- Diagrams omit tests, scripts, and docs layers to focus on runtime package structure.
- No independent validator subagent was available.


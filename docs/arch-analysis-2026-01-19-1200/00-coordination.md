## Analysis Plan
- Scope: `src/elspeth`, `tests`, `alembic`, `config`, `scripts`, `examples`, key docs
- Strategy: Parallel scans for structure/signals, then focused reads of subsystem entrypoints
- Complexity estimate: Medium (single Python package; multiple subsystems: audit DB, engine, plugin system, CLI/TUI)
- Deliverable: Subsystem catalog + bug-likelihood rating (requested); optionally can extend to diagrams/security/quality on request

## Execution Log
- 2026-01-19 12:00 Created workspace `docs/arch-analysis-2026-01-19-1200/`

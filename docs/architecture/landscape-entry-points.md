# Landscape Audit Entry Points Have Moved

This file remains only as a compatibility pointer for older links. The previous
content was a line-number map of audit entry points and became stale as the
Landscape recorder was split into repository modules.

Use these maintained sources instead:

- [Landscape System Architecture](landscape.md) for the current repository
  layout, table inventory, and read surfaces.
- [Root architecture overview](../../ARCHITECTURE.md) for the system-level audit
  model and C4 diagrams.
- `src/elspeth/core/landscape/factory.py` for the repository composition point.
- `src/elspeth/core/landscape/schema.py` for the authoritative table schema.
- `src/elspeth/mcp/server.py` for the MCP tool registry.

For an operator-facing routing investigation, use
[Investigate Routing](../runbooks/investigate-routing.md).

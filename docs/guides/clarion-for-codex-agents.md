# Clarion for Codex Agents

Clarion gives a Codex agent a structural map of ELSPETH: entities, source
locations, containment, calls, references, subsystems, and optional Filigree
issue associations. Use it to orient faster, then still read the source and run
the tests that prove your change.

## Quick Start

Start with the normal ELSPETH session checks:

```bash
filigree session-context
git status --short --branch
```

Then verify the local Clarion build and plugin path:

```bash
/home/john/clarion/target/release/clarion --version
/home/john/clarion/plugins/python/.venv/bin/pyright --version
```

Expected today:

```text
clarion 1.0.0
pyright 1.1.409
```

Do not use `/home/john/.local/bin/clarion` without checking it first; it may be
older than the repo-local release build.

## Refresh the Graph

Clarion stores ELSPETH analysis data in `.clarion/clarion.db`. If the database
does not exist, or if the code changed enough that graph answers may be stale,
run:

```bash
cd /home/john/elspeth

PATH=/home/john/clarion/plugins/python/.venv/bin:$PATH \
  /home/john/clarion/target/release/clarion install --path .

PATH=/home/john/clarion/plugins/python/.venv/bin:$PATH \
  /home/john/clarion/target/release/clarion analyze .
```

`install` creates `.clarion/` and `clarion.yaml`. Treat existing files there as
project state: do not delete or overwrite them casually in a dirty checkout.

If `analyze` prints `WARN no plugins discovered` or completes with
`skipped_no_plugins`, the Python plugin was not on `PATH`. Re-run with the
`PATH=...` prefix above.

## Wire Clarion into Codex

Clarion's CLI builds and serves the graph. The useful query surface is MCP.
There is not currently a query CLI such as `clarion find-entity`.

Clarion databases are project-local. Do not add `clarion-elspeth` with
`codex mcp add`; that command writes to Codex's global user configuration and
would expose ELSPETH's `.clarion/clarion.db` from unrelated repositories.

ELSPETH keeps the project-scoped MCP entry in `.mcp.json`:

```json
{
  "mcpServers": {
    "clarion-elspeth": {
      "type": "stdio",
      "command": "/home/john/clarion/target/release/clarion",
      "args": ["serve", "--path", "/home/john/elspeth"],
      "env": {
        "PATH": "/home/john/clarion/plugins/python/.venv/bin:..."
      }
    }
  }
}
```

Start Codex from `/home/john/elspeth` with a client mode that loads project MCP
configuration, then check the visible tool list in the running session. If only
global MCP servers appear, stop and fix the project-local MCP loading path
rather than adding this server globally.

## First Questions to Ask

Once the MCP server is loaded, use Clarion before broad grep sweeps:

- `find_entity(pattern="ExecutionService")` to find likely entities.
- `entity_at(file="src/elspeth/web/sessions/routes.py", line=120)` to identify
  the innermost function or class around a location.
- `callers_of(id="...")` after `find_entity` gives you the exact entity ID.
- `neighborhood(id="...")` for callers, callees, container, contained entities,
  and references in one response.
- `execution_paths_from(id="...", max_depth=3)` for bounded call-path
  exploration.
- `subsystem_members(id="...")` when investigating a generated subsystem.
- `issues_for(id="...")` to check Filigree associations when available.

`summary(id="...")` is optional. It needs live LLM settings in `clarion.yaml`
and an `OPENROUTER_API_KEY` in the environment inherited by `clarion serve`.
Do not block structural work on `summary`; the graph tools work without LLM
credentials.

## Working Rules

- Use Clarion for orientation, not as a replacement for source evidence.
- Re-run `analyze` after large refactors or when Clarion answers contradict the
  current file.
- Start `serve` after a completed `analyze` when you need a stable review
  snapshot.
- Do not run multiple `clarion analyze` processes against the same `.clarion/`
  directory.
- The HTTP read API is for sibling-tool integration; Codex agents normally need
  only the MCP server.

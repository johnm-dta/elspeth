# Run Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for each implementation slice. Keep diagnostics bounded and run-scoped; do not expose row payload/context blobs in the web API.

**Tracking:** `elspeth-151b6e46ba`

**Goal:** Give web users a compact "what is happening?" view for quiet runs: the first bounded set of Landscape tokens, their node-state statuses, source/sink operation activity, saved artifacts, and a button that asks the configured LLM for a plain-language interpretation.

**Architecture:** Reuse Landscape as the audit source of truth. Thread the web run UUID into `Orchestrator.run()` as the Landscape `run_id`, persist that id when the web run enters `running`, and query diagnostics by the owned web run id. The diagnostics API returns a strict, bounded projection: counts, token/state preview, operations, and artifacts. The LLM explanation endpoint receives only that projection and returns advisory text; it must not mutate composition state or session history.

**Files:**

- Modify: `src/elspeth/engine/orchestrator/core.py`
- Modify: `src/elspeth/web/execution/service.py`
- Modify: `src/elspeth/web/execution/routes.py`
- Modify: `src/elspeth/web/execution/schemas.py`
- Create: `src/elspeth/web/execution/diagnostics.py`
- Modify: `src/elspeth/web/composer/protocol.py`
- Modify: `src/elspeth/web/composer/service.py`
- Modify: `src/elspeth/web/frontend/src/api/client.ts`
- Modify: `src/elspeth/web/frontend/src/types/index.ts`
- Modify: `src/elspeth/web/frontend/src/stores/executionStore.ts`
- Modify: `src/elspeth/web/frontend/src/components/inspector/RunsView.tsx`
- Add/modify targeted backend and frontend tests.

## Implementation Tasks

- [x] Add backend regression tests for run-id threading and bounded diagnostics projection.
- [x] Expose caller-supplied Landscape run ids through `Orchestrator.run()` and have web execution pass/persist the web run id.
- [x] Add strict diagnostics response schemas plus SQL helpers that project only bounded diagnostic fields.
- [x] Add owned run-scoped diagnostics and explanation endpoints.
- [x] Add frontend API/types/store state for diagnostics, explanation, and active-run polling.
- [x] Add a compact diagnostics panel to `RunsView` with token/state preview and explanation action.
- [x] Run targeted backend and frontend verification, then update Filigree.

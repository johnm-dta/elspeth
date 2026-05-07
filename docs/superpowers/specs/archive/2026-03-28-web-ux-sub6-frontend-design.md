# Web UX Sub-Spec 6: Frontend

**Status:** Draft
**Date:** 2026-03-28
**Parent Spec:** `docs/superpowers/specs/2026-03-28-web-ux-composer-mvp-design.md`
**Phase:** 6
**Depends On:** Sub-Specs 2-5 (all backend phases)
**Blocks:** Nothing

---

## Scope

**In scope:**

- React 18 / TypeScript SPA with Vite build tooling
- Three-panel layout: sessions sidebar, chat panel, inspector panel
- Session management sidebar with session list and creation
- Chat panel with message display, composing indicator, and input
- Inspector panel with four tabs: Spec (component linking), Graph (React Flow DAG), YAML (read-only with copy), Runs (execution history)
- Validate and Execute buttons pinned to inspector header
- Validation result display and execution progress with WebSocket streaming
- Zustand state management (auth, session, execution stores)
- Typed API client with OpenAPI-generated types and auth token injection
- WebSocket connection manager with auto-reconnect
- Auth guard and login page (local username/password, OIDC/Entra SSO redirect)
- File upload with auto-injection of server path into chat input
- Empty states and user-facing error messages
- Accessibility: keyboard navigation, ARIA live regions, colour-safe badges
- Version history affordance for CompositionState revert
- Static file serving from FastAPI in production

**Out of scope:**

- Graph editing (direct manipulation of the DAG)
- YAML import/paste
- Frontend test framework setup (W17 -- tracked separately)
- Token-by-token streaming of LLM responses
- Multi-tenant session isolation
- Template library or YAML import

---

## Project Setup

**Toolchain:** Vite 6, React 18, TypeScript 5, React Flow (`@xyflow/react`), Zustand 5.

**Initialisation:** Run `npm create vite@latest . -- --template react-ts` inside `src/elspeth/web/frontend/`, then install additional dependencies: `@xyflow/react`, `zustand`, `openapi-typescript` (dev dependency for type generation), and `prism-react-renderer` (syntax highlighting for YAML tab).

**Vite config:** The dev server runs on port 5173 and proxies two path prefixes to the FastAPI backend at `localhost:8000`:

- `/api` proxied as HTTP to `http://localhost:8000`
- `/ws` proxied as WebSocket to `ws://localhost:8000`

This means the frontend never needs to know the backend URL during development -- all API calls use relative paths.

**Production serving:** FastAPI mounts `frontend/dist/` as static files with `html=True` for SPA fallback. The mount is conditional -- it only activates when the `dist/` directory exists (i.e., after `npm run build`). The mount is registered after all API and WebSocket routes so that `/api/*` and `/ws/*` take precedence over the static file catch-all.

**TypeScript config:** Strict mode enabled. Path alias `@/` maps to `src/` for clean imports. JSX set to `react-jsx`. Target ES2020 for broad browser support.

---

## Layout

Three-panel CSS grid layout with the following column definitions:

| Panel | Width | Behaviour |
|-------|-------|-----------|
| Sessions sidebar | 200px fixed | Collapsible via toggle button; collapsed state persists in localStorage |
| Chat panel | `1fr` (flex) | Fills remaining horizontal space |
| Inspector panel | 320px default | Resizable via drag handle; minimum 240px, maximum 50% of viewport width |

**Minimum supported resolution:** 1280px horizontal. Below this width, the layout is not responsive -- the application displays at 1280px with horizontal scroll. This is acceptable for the target user base (small team, desktop browsers).

The layout is a single full-viewport grid with no routing. All three panels render simultaneously. The `Layout` component owns the grid definition and renders its three children (sidebar, chat, inspector) into named grid areas.

---

## Session Sidebar

The sidebar lists the current user's sessions, ordered by `updated_at` descending (most recently active first).

**Elements:**

- **Header:** "Sessions" label with a collapse toggle button (chevron icon)
- **Session list:** Each item shows the session title (truncated to one line) and a relative timestamp ("2 min ago", "yesterday"). The active session is visually highlighted with a distinct background colour and left border accent.
- **New session button:** Pinned to the bottom of the sidebar. Calls `POST /api/sessions`, adds the new session to the list, and makes it active. The chat panel clears and shows the welcome empty state.
- **Empty state:** When no sessions exist, the list area shows "No sessions yet" with a prompt to click the new session button.

Clicking a session calls `selectSession(id)` on the session store, which loads that session's message history and current CompositionState.

---

## Chat Panel

The chat panel is the primary interaction surface. It contains three sub-components stacked vertically: the message list (scrollable, flex-grow), the composing indicator (conditional), and the chat input (fixed to bottom).

### MessageBubble

Renders a single `ChatMessage`. Styling varies by role:

- **User messages:** Right-aligned, distinct background colour (e.g., blue-tinted). Shows message text only.
- **Assistant messages:** Left-aligned, neutral background. Shows message text. If the message has associated `tool_calls`, they are rendered as a collapsible "Tool calls" section below the text showing tool names and summaries.
- **System messages:** Centre-aligned, full-width banner style. Rendered with reduced opacity / muted colour to be visually distinct from user and assistant bubbles. Text is italic. No sender label is displayed. Used for system-injected messages such as "Pipeline reverted to version N." These messages are part of the chronological message list but are not interactive -- they serve as audit markers in the conversation history.

Messages are rendered in chronological order. The list auto-scrolls to the bottom when new messages arrive. If the user has scrolled up (reviewing history), auto-scroll is suppressed until they scroll back to the bottom.

### ChatInput

A multi-line text input with a send button.

- **Send:** Clicking the send button or pressing Enter submits the message. Shift+Enter inserts a newline.
- **Disabled state:** The send button is disabled and visually muted while the composing indicator is active. The text input remains editable (the user can draft their next message) but submission is blocked.
- **Focus return:** After the assistant response arrives and the composing indicator disappears, keyboard focus returns to the ChatInput.

### ComposingIndicator

Shown while the backend is processing the LLM tool-use loop (i.e., after the user sends a message and before the assistant response arrives). Renders as a left-aligned bubble (matching assistant message alignment) containing three dots with a staggered CSS bounce animation and a text label "ELSPETH is composing...".

The indicator appears immediately when `sendMessage()` is called and disappears when the API response returns. The send button is disabled for the duration.

---

## Inspector Panel

The inspector panel has two zones: a fixed header area and a tabbed content area.

### Header

The header contains:

- **Tab strip:** Four tabs -- Spec, Graph, YAML, Runs. The active tab is visually distinguished.
- **Validate button:** Calls `POST /api/sessions/{id}/validate`. Disabled while a validation or execution is in progress.
- **Execute button:** Calls `POST /api/sessions/{id}/execute`. Disabled until the most recent Stage 2 validation passes. Also disabled while an execution is already running.

The Validate and Execute buttons are pinned to the header, outside the tab strip, and are always visible regardless of which tab is active.

### Spec Tab

Renders the current `CompositionState` as a vertical list of component cards. Each card displays:

- **Type badge:** One of SOURCE, TRANSFORM, GATE, SINK. Each type has a distinct background colour AND a text label (not colour alone, per accessibility requirements). Badge colours: SOURCE = blue, TRANSFORM = green, GATE = amber, SINK = purple.
- **Plugin name and config summary:** The plugin identifier and a one-line summary of key configuration values.
- **Next-node indicator:** For nodes with a default `continue` edge, shows the downstream node name.
- **Gate route indicators:** For gate nodes, shows each route path with its destination (e.g., true path destination, false path destination).
- **Error sink indicator:** If the node has an `on_error` edge, shows the error sink destination.

**Click-to-highlight linking:**

Clicking a component card triggers relationship highlighting across all cards:

- The clicked card receives a SELECTED badge (visually distinct, e.g., ring border).
- Direct upstream connections receive an INPUT badge.
- Direct downstream connections receive OUTPUT badges. For gate nodes, route-specific badges are shown (e.g., TRUE PATH, FALSE PATH).
- Unrelated components dim: their background fades to reduced opacity. Text remains at full opacity -- only the card background dims. This ensures readability is never compromised.
- Clicking the same card again, or clicking empty space between cards, clears the selection.

Linking data is computed from the `edges` array in `CompositionState` -- the same source of truth used by the Graph tab and the engine.

**Keyboard accessibility (W4):** Component cards are focusable via Tab key navigation. Each card has `tabindex="0"` and responds to Enter or Space to toggle selection, matching the click behaviour. Focus is visually indicated with an outline that meets WCAG 2.1 contrast requirements.

**Empty state:** When no CompositionState exists (new session, no composition yet), the Spec tab shows placeholder text: "Send a message to start building your pipeline. Components will appear here as ELSPETH composes them."

### Graph Tab

React Flow DAG visualisation of the current `CompositionState`. Converts `nodes` and `edges` from the composition state into React Flow node and edge objects.

- **Node styling:** Colour-coded by type, matching the Spec tab badge colours. Each node displays its plugin name.
- **Edges:** Directed arrows. Gate route edges are labelled with the route condition.
- **Auto-layout:** Uses dagre (via `@dagrejs/dagre`) for automatic hierarchical layout. Layout is recalculated when the composition state changes.
- **Interactivity:** Read-only -- no drag-to-connect, no node deletion. Pan and zoom are enabled.
- **ARIA:** The React Flow container has a text alternative describing the pipeline structure (e.g., "Pipeline graph with N nodes and M connections").

**Empty state:** "No pipeline to visualise. Start a conversation to build one."

### YAML Tab

Read-only display of the generated ELSPETH pipeline YAML. The YAML is fetched from `GET /api/sessions/{id}/state/yaml` (defined in sub-4). The tab calls this endpoint whenever the composition state version changes.

- **Syntax highlighting:** Uses `prism-react-renderer` with a YAML grammar.
- **Copy-to-clipboard:** A button in the tab header copies the full YAML text to the clipboard with a brief "Copied" confirmation.
- **No editing:** The YAML view is strictly read-only. Users modify the pipeline through chat, not by editing YAML.

**Empty state:** "YAML will appear here once your pipeline has components."

### Runs Tab

Lists execution runs for the current session, ordered by `started_at` descending.

Each run entry shows:

- **Status badge:** pending, running, completed, failed, or cancelled. Colour-coded.
- **Row counts:** Rows processed and rows failed.
- **Duration:** Elapsed time (or "running..." for active runs).
- **Composition state version:** Which version of the pipeline was executed.

Clicking an active run shows the ProgressView inline within the Runs tab.

**Empty state:** "No runs yet. Validate your pipeline, then click Execute to run it."

---

## Execution UX

### Validation Results

Shown inline as a banner between the inspector header and the tab content area. Appears after the user clicks Validate and the backend responds.

**Validate button loading state:** While the validation endpoint is processing, the Validate button shows a spinner and is disabled. The spinner replaces the button text. The button returns to its normal state when the response arrives.

- **Pass:** Green banner with a checkmark and summary text (e.g., "Validation passed: 4 nodes, all schemas compatible, routes valid"). The Execute button becomes enabled.
- **Fail:** Red banner with per-component error list. Each error identifies the component by name, describes the issue, and includes a suggested fix from the backend. The Execute button remains disabled.

**Auto-clear (W3):** When a new CompositionState version arrives from the composer (i.e., the user has modified the pipeline since the last validation), the validation result banner is cleared and the Execute button is disabled. This prevents executing a pipeline that has changed since it was validated.

### Execution Progress

When a run is in progress, the Runs tab displays a live progress view:

- **Progress bar:** Uses indeterminate mode (no percentage, animated stripe) since the source row count is not known in advance. Uses `role="progressbar"` without `aria-valuemax` (indicating indeterminate progress) and with `aria-label="Pipeline execution in progress"`. In a future version, if the source provides a row count, the bar can switch to determinate mode by adding `aria-valuenow` and `aria-valuemax`.
- **Counters:** `rows_processed` and `rows_failed` counters displayed as text alongside the progress bar. No `estimated_total` is shown.
- **Recent exceptions:** A scrollable list showing the most recent exceptions, newest first. Limited to the last 50 entries to avoid memory growth.
- **Cancel button:** Calls `POST /api/runs/{id}/cancel`. Disabled once the run has reached a terminal state.

Progress data streams over the WebSocket connection. The `executionStore` receives `RunEvent` payloads and updates the progress state reactively. The `RunEvent` `event_type` union includes: `progress`, `error`, `completed`, and `cancelled`. These have distinct terminal semantics:

- **`progress`** -- non-terminal. Row count update; the pipeline is still running.
- **`error`** -- non-terminal. A per-row exception occurred but the pipeline continues processing. The frontend appends the error to the recent exceptions list in the ProgressView; it does NOT stop the progress view or close the WebSocket.
- **`completed`** -- terminal. The pipeline finished successfully. The frontend updates the run status to "completed", closes the WebSocket connection, and switches the progress bar to a static "complete" state.
- **`cancelled`** -- terminal. The pipeline was cancelled. The frontend updates the run status to "cancelled", closes the WebSocket connection, and the ProgressView displays "Pipeline execution was cancelled."

Note: `"failed"` is a Run status (set on the Run model when the pipeline aborts), not a RunEvent type. The WebSocket stream never sends a `"failed"` event -- if the pipeline aborts, the WebSocket connection closes and the frontend fetches the final Run status via the REST API.

---

## State Management

Three Zustand stores manage client-side state. All stores use the slice pattern for clean separation.

### authStore

| Field | Type | Purpose |
|-------|------|---------|
| `token` | `string \| null` | JWT or OIDC access token |
| `user` | `UserProfile \| null` | Current user identity |
| `isAuthenticated` | `boolean` | Derived from `token !== null` |

**Actions:**

- `login(username, password)` -- calls `POST /api/auth/login`, stores token in `localStorage`, fetches user profile via `GET /api/auth/me`
- `loginWithToken(token)` -- for OIDC/Entra flows where the token comes from the IdP redirect
- `logout()` -- clears token from state and `localStorage`, resets session and execution stores
- `loadFromStorage()` -- on app mount, checks `localStorage` for an existing token and validates it via `GET /api/auth/me`

### sessionStore

| Field | Type | Purpose |
|-------|------|---------|
| `sessions` | `Session[]` | User's session list |
| `activeSessionId` | `string \| null` | Currently selected session |
| `messages` | `ChatMessage[]` | Messages for the active session |
| `compositionState` | `CompositionState \| null` | Current pipeline state for the active session |
| `isComposing` | `boolean` | True while awaiting composer response |
| `stateVersions` | `CompositionStateVersion[]` | Version history for the active session (W2) |

**Actions:**

- `loadSessions()` -- fetches session list from `GET /api/sessions`
- `createSession()` -- calls `POST /api/sessions`, adds to list, sets as active
- `selectSession(id)` -- sets active session, loads messages via `GET /api/sessions/{id}/messages` and state via `GET /api/sessions/{id}/state`
- `sendMessage(content)` -- sets `isComposing = true`, calls `POST /api/sessions/{id}/messages`, updates `messages` from `response.message` and `compositionState` from `response.state` (note: the API wire field is `state`, not `compositionState` — the store maps the key on destructure), sets `isComposing = false`. If `compositionState.version` changed, calls `executionStore.clearValidation()` (W3 auto-clear).
- `loadStateVersions()` -- fetches version history from `GET /api/sessions/{id}/state/versions`
- `revertToVersion(version)` -- calls `executionStore.clearValidation()` first (H3 fix — revert must clear stale validation), then calls backend revert endpoint, updates `compositionState` from response (W2)

### executionStore

| Field | Type | Purpose |
|-------|------|---------|
| `runs` | `Run[]` | Runs for the active session |
| `activeRunId` | `string \| null` | Currently monitored run |
| `progress` | `RunProgress \| null` | Live progress data from WebSocket |
| `validationResult` | `ValidationResult \| null` | Most recent validation result |

**Actions:**

- `validate(sessionId)` -- calls `POST /api/sessions/{id}/validate`, stores result
- `execute(sessionId)` -- calls `POST /api/sessions/{id}/execute`, stores run, connects WebSocket
- `connectWebSocket(runId)` -- opens WebSocket to `/ws/runs/{id}?token=<jwt>` (appends the JWT from `authStore.token` as a query parameter). Close code handling (seam contract E): **1000** (normal closure — run terminal) → do NOT reconnect, poll `GET /api/runs/{id}` for final status; **1006** (abnormal closure — network drop) → auto-reconnect with exponential backoff; **1011** (internal error) → do NOT reconnect, poll REST for status; **4001** (auth failure) → do NOT reconnect, call `authStore.logout()`. Updates `progress` on each `RunEvent`
- `cancel(runId)` -- calls `POST /api/runs/{id}/cancel`
- `clearValidation()` -- called when a new CompositionState version arrives (W3 auto-clear)

---

## API Client

Located in `src/api/client.ts`. Provides typed `fetch` wrappers for every backend endpoint.

**Type generation:** Prefer `openapi-typescript` to generate TypeScript types from the FastAPI OpenAPI schema. Run `npx openapi-typescript http://localhost:8000/openapi.json -o src/types/api.ts` during development. If the OpenAPI schema is unavailable at build time, hand-written types in `src/types/index.ts` serve as the fallback.

**Auth token injection:** Every request includes an `Authorization: Bearer {token}` header, read from `authStore.token`. If the token is missing (user not authenticated), the request is not sent and the caller receives an authentication error.

**Global 401 interceptor:** The typed fetch wrapper includes a global response interceptor. If any API call returns HTTP 401, the interceptor calls `authStore.logout()` and navigates to LoginPage. This handles token expiry mid-session without requiring each caller to check for auth failures. The WebSocket disconnect path (close code 4001) also triggers logout via the same `authStore.logout()` call.

**Error handling:** Non-2xx responses are parsed into a typed error object with `status`, `detail`, and optional `error_type` and `validation_errors` fields. The API client does not silently swallow errors -- callers handle them explicitly.

**Base URL:** All fetch calls use relative paths (`/api/sessions`, `/api/runs/{id}`, etc.). In development, the Vite proxy forwards these to FastAPI. In production, the SPA is served from the same origin as the API, so relative paths work directly.

---

## Auth UX

### AuthGuard

A wrapper component rendered at the top of the component tree, inside `App.tsx`. Behaviour:

- On mount, calls `authStore.loadFromStorage()` to check for an existing token.
- If authenticated (`authStore.isAuthenticated` is true), renders its children (the main application layout).
- If not authenticated, renders the `LoginPage`.

### LoginPage

The login page adapts to the configured auth provider:

- **Local auth:** Renders a username/password form with a "Sign in" submit button. On submit, calls `authStore.login(username, password)`. Displays an inline error message on authentication failure.
- **OIDC / Entra:** Renders a "Sign in with SSO" button. On click, redirects the browser to the IdP authorization endpoint. On return (with the token in the URL fragment or query parameter), calls `authStore.loginWithToken(token)`.

The auth provider type is determined at runtime. On app load, the frontend fetches `GET /api/auth/config`, which returns `{provider: "local" | "oidc" | "entra", oidc_issuer?: string, oidc_client_id?: string}`. The `provider` field determines which login form to render:

- **`"local"`:** Render the username/password form.
- **`"oidc"` or `"entra"`:** Render the "Sign in with SSO" button. Construct the OIDC redirect URL from the `oidc_issuer` and `oidc_client_id` fields in the config response (i.e., `{oidc_issuer}/authorize?client_id={oidc_client_id}&response_type=code&redirect_uri=...`).

This endpoint is unauthenticated (it must be callable before login). The response is cached in memory for the session lifetime -- it does not change at runtime.

---

## File Upload UX

The chat panel supports file upload for source data files. The upload flow:

1. A file upload button (paperclip icon or similar) is positioned adjacent to the ChatInput send button.
2. Clicking the button opens a native file picker dialog.
3. On file selection, the frontend calls `POST /api/sessions/{id}/upload` with the file as multipart form data.
4. The backend saves the file to the user's scratch directory and returns the server-side path in the response.
5. The frontend auto-injects the server path into the chat input text field (e.g., appending "I've uploaded a file at /data/uploads/user123/mydata.csv"). The user can then send the message, and the LLM composer uses the path when configuring a source.

This avoids requiring users to manually type server-side file paths. The user sees the path in their message and can edit it before sending if needed.

**Upload progress:** For large files, a progress indicator replaces the upload button temporarily. On completion, the button returns to its normal state.

**Upload failure error:** If the upload fails (network error, file too large, server error), an inline error message appears near the upload button with the text: "Upload failed: {reason}. Please try again or use a smaller file."

---

## Empty States & Error Messages

### Empty States

| Location | Condition | Message |
|----------|-----------|---------|
| Session sidebar | No sessions exist | "No sessions yet. Click the button below to start." |
| Chat panel | New session, no messages | "Welcome to ELSPETH. Describe the pipeline you want to build, and I'll compose it for you." |
| Spec tab | No CompositionState | "Send a message to start building your pipeline. Components will appear here as ELSPETH composes them." |
| Graph tab | No CompositionState | "No pipeline to visualise. Start a conversation to build one." |
| YAML tab | No CompositionState | "YAML will appear here once your pipeline has components." |
| Runs tab | No runs for session | "No runs yet. Validate your pipeline, then click Execute to run it." |

### Error Messages

| Scenario | User-facing text |
|----------|-----------------|
| Composer convergence error (422, `error_type: "convergence"`) | "ELSPETH couldn't complete the composition after multiple attempts. Try breaking your request into smaller steps." |
| LLM unavailable (502, `error_type: "llm_unavailable"`) | "The AI service is temporarily unavailable. Please try again in a moment." |
| LLM auth error (502, `error_type: "llm_auth_error"`) | "The AI service configuration is invalid. Please contact your administrator." |
| Composing timeout (POST /messages > 90s) | "The composition request timed out. Try a simpler request." |
| Upload failure (size) | "The file exceeds the maximum upload size. Please use a smaller file." |
| Upload failure (network) | "Upload failed due to a network error. Please check your connection and try again." |
| Execution failure | "Pipeline execution failed. Check the Runs tab for error details." |
| Execution conflict (409 on execute) | "A run is already in progress for this pipeline." |
| Execution cancelled (WebSocket `cancelled` event) | "Pipeline execution was cancelled." |
| Validation internal error (500 on validate) | "Validation encountered an internal error. Please try again." |
| WebSocket disconnect (non-4001 close) | "Live progress connection lost. Reconnecting..." (shown as a subtle banner at the top of the Runs tab; auto-dismisses on reconnect) |
| WebSocket auth failure (close code 4001) | Triggers `authStore.logout()` and redirect to LoginPage (no user-facing message -- the login page is the message). |
| Validation failure | Displayed inline in the validation result banner with per-component detail (not a generic message). |
| Session load failure | "Failed to load session. Please refresh the page." |
| Authentication failure (local) | "Invalid username or password." |
| Authentication failure (token expired) | "Your session has expired. Please sign in again." |

The `useComposer` hook is responsible for dispatching composer-specific error messages. It checks the HTTP response status and the `error_type` field in the error response body to select the correct message. The 90-second composing timeout is implemented via `AbortController` on the `POST /messages` fetch call -- if the timeout fires, the request is aborted and the timeout message is shown.

---

## Accessibility

### Keyboard Navigation

- **Component cards (Spec tab):** Each card has `tabindex="0"`. Tab key moves focus between cards in document order. Enter or Space toggles the card's selected state (equivalent to click). Focus indicator is a visible outline meeting WCAG 2.1 AA contrast requirements (3:1 minimum against adjacent colours).
- **Chat input focus return:** After the composing indicator disappears (assistant response received), keyboard focus is programmatically returned to the ChatInput text field so the user can immediately type their next message.
- **Tab strip:** Inspector tabs are navigable with arrow keys (left/right) when the tab strip has focus. Tab key moves focus into the active tab's content area.
- **Buttons:** All buttons (Send, Validate, Execute, Cancel, Copy, Upload) are natively focusable and activated with Enter or Space.

### ARIA

- **Inspector panel:** The tab content area is wrapped in an ARIA live region (`aria-live="polite"`) so that screen readers announce state changes (e.g., new validation results, updated spec cards) without requiring the user to navigate to the panel.
- **Execution progress bar:** Uses `role="progressbar"` in indeterminate mode (no `aria-valuemax`). Includes `aria-label="Pipeline execution in progress"`. When a future version supports source-provided row counts, `aria-valuenow` and `aria-valuemax` can be added for determinate mode.
- **React Flow container:** Has `aria-label` describing the pipeline structure (e.g., "Pipeline graph: 4 nodes, 3 connections"). Since the graph is read-only and the same information is available in the Spec tab, the graph container is marked with `aria-hidden="false"` but is not the primary means of conveying pipeline structure.
- **Composing indicator:** Marked with `aria-live="polite"` and `aria-label="ELSPETH is composing a response"` so screen readers announce when composition starts without interrupting the user.

### Colour

- **Type badges** use both background colour AND text labels (SOURCE, TRANSFORM, GATE, SINK). Colour is reinforcement, not the sole differentiator. This meets WCAG 1.4.1 (Use of Colour).
- **Click-to-highlight dimming** reduces card background opacity only, never text opacity. Dimmed cards remain fully readable. This ensures WCAG 1.4.3 (Contrast Minimum) is maintained even for de-emphasised components.
- **Status badges** (pending, running, completed, failed, cancelled) use colour plus text label plus icon (where applicable).

---

## Version History

The inspector panel includes a version history affordance for reverting to prior CompositionState versions (W2).

**Location:** A dropdown or compact timeline control in the inspector header, adjacent to the Validate/Execute buttons.

**Behaviour:**

- Displays the current CompositionState version number (e.g., "v3").
- Clicking opens a dropdown listing all versions for the active session, ordered newest-first, with timestamps.
- Selecting a prior version calls `sessionStore.revertToVersion(version)`, which calls the backend revert endpoint (sub-2's `set_active_state` via `POST /api/sessions/{id}/state/revert`). The backend sets the selected version as active and injects a system message "Pipeline reverted to version N." into the session's message history. The frontend then refreshes both the message list and the composition state from the backend response.
- Reverting clears the validation result and disables the Execute button. **Invariant (H3 fix):** `revertToVersion()` calls `executionStore.clearValidation()` **before** updating `compositionState`. This is not reactive wiring — it is an explicit call. The trigger for `clearValidation()` is any change to `compositionState.version`, regardless of source (composer response, revert, session switch). Without this, a user could revert to an unvalidated version while the Execute button still shows a stale "valid" badge from the pre-revert version.
- The chat history is not affected by revert beyond the injected system message -- it remains a complete chronological record. The system message provides an audit trail of the revert action.

---

## File Map

All frontend files, rooted at `src/elspeth/web/frontend/`.

| Action | Path |
|--------|------|
| Create | `src/elspeth/web/frontend/package.json` |
| Create | `src/elspeth/web/frontend/tsconfig.json` |
| Create | `src/elspeth/web/frontend/vite.config.ts` |
| Create | `src/elspeth/web/frontend/index.html` |
| Create | `src/elspeth/web/frontend/src/main.tsx` |
| Create | `src/elspeth/web/frontend/src/App.tsx` |
| Create | `src/elspeth/web/frontend/src/types/index.ts` |
| Create | `src/elspeth/web/frontend/src/types/api.ts` |
| Create | `src/elspeth/web/frontend/src/api/client.ts` |
| Create | `src/elspeth/web/frontend/src/api/websocket.ts` |
| Create | `src/elspeth/web/frontend/src/stores/authStore.ts` |
| Create | `src/elspeth/web/frontend/src/stores/sessionStore.ts` |
| Create | `src/elspeth/web/frontend/src/stores/executionStore.ts` |
| Create | `src/elspeth/web/frontend/src/components/common/Layout.tsx` |
| Create | `src/elspeth/web/frontend/src/components/common/AuthGuard.tsx` |
| Create | `src/elspeth/web/frontend/src/components/auth/LoginPage.tsx` |
| Create | `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx` |
| Create | `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` |
| Create | `src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx` |
| Create | `src/elspeth/web/frontend/src/components/chat/ChatInput.tsx` |
| Create | `src/elspeth/web/frontend/src/components/chat/ComposingIndicator.tsx` |
| Create | `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx` |
| Create | `src/elspeth/web/frontend/src/components/inspector/SpecView.tsx` |
| Create | `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx` |
| Create | `src/elspeth/web/frontend/src/components/inspector/YamlView.tsx` |
| Create | `src/elspeth/web/frontend/src/components/inspector/RunsView.tsx` |
| Create | `src/elspeth/web/frontend/src/components/execution/ProgressView.tsx` |
| Create | `src/elspeth/web/frontend/src/components/execution/ValidationResult.tsx` |
| Create | `src/elspeth/web/frontend/src/hooks/useSession.ts` |
| Create | `src/elspeth/web/frontend/src/hooks/useComposer.ts` |
| Create | `src/elspeth/web/frontend/src/hooks/useWebSocket.ts` |
| Create | `src/elspeth/web/frontend/src/hooks/useAuth.ts` |
| Modify | `src/elspeth/web/app.py` |

The `app.py` modification adds static file serving: mount `frontend/dist/` with `html=True` after all API routes, conditional on the directory existing.

---

## Acceptance Criteria

1. **Project builds.** `npm install` and `npm run build` succeed in the `frontend/` directory without errors. The built output lands in `frontend/dist/`.

2. **Dev proxy works.** Running `npm run dev` starts the Vite dev server on port 5173. API calls to `/api/*` are proxied to `localhost:8000`. WebSocket connections to `/ws/*` are proxied correctly.

3. **Production serving works.** After `npm run build`, running `elspeth web` serves the SPA from `frontend/dist/`. Navigating to `http://localhost:8000` in a browser loads the React application. API routes at `/api/*` still respond correctly (not masked by the static mount).

4. **Auth flow works.** Unauthenticated users see the LoginPage. Logging in with valid local credentials stores the JWT and renders the main application. Refreshing the page preserves the authenticated state via localStorage. Logging out clears state and returns to LoginPage.

5. **Session management works.** The sidebar lists sessions. Creating a new session adds it to the list and activates it. Switching sessions loads the correct message history and composition state. The active session is visually highlighted.

6. **Chat interaction works.** Typing a message and pressing Enter sends it. The message appears in the chat as a user bubble. The composing indicator appears during the backend processing. The assistant response appears as an assistant bubble. The send button is disabled during composition. Focus returns to ChatInput after the response arrives.

7. **Inspector Spec tab works.** Component cards render with correct type badges (colour plus text label). Next-node and route indicators display correctly. Clicking a card highlights it (SELECTED badge), shows INPUT/OUTPUT badges on connected cards, and dims unrelated cards (background only). Clicking again or clicking empty space deselects. Cards are keyboard-focusable (Tab) and selectable (Enter/Space).

8. **Inspector Graph tab works.** React Flow renders a DAG from the composition state with colour-coded nodes and directed edges. Auto-layout produces a readable hierarchical arrangement. Pan and zoom work. No editing interactions are available.

9. **Inspector YAML tab works.** Generated YAML is displayed with syntax highlighting. The copy button copies the full YAML to the clipboard.

10. **Inspector Runs tab works.** Runs are listed with status, row counts, and duration. Active runs show the ProgressView with a live progress bar, counters, exception list, and cancel button.

11. **Validation UX works.** Clicking Validate shows a pass (green) or fail (red) banner with per-component detail. The Execute button enables on pass. A new CompositionState version arriving from the composer clears the validation banner and disables Execute (W3).

12. **Execution progress works.** Starting execution connects the WebSocket (with JWT in query parameter). Row counts (`rows_processed`, `rows_failed`) and exceptions update in real time. The progress bar uses indeterminate mode with correct `role="progressbar"` ARIA semantics (no `aria-valuemax`). The cancel button sends the cancel request. A `cancelled` RunEvent updates the status and shows the cancellation message. WebSocket close code 4001 triggers logout.

13. **File upload works.** Clicking the upload button opens a file picker. After upload, the server path is auto-injected into the chat input. Upload errors display inline messages.

14. **Empty states render.** All six empty states (sidebar, chat, spec, graph, YAML, runs) display their placeholder text when no content is available.

15. **Version history works.** The version dropdown shows prior CompositionState versions. Selecting one loads that version's state, clears validation results, and disables Execute.

16. **Accessibility.** Component cards are keyboard-navigable. The progress bar has ARIA progressbar semantics. The React Flow container has a text alternative. Type badges use text labels alongside colour. Dimming affects background only, not text. The composing indicator has an ARIA live region announcement.

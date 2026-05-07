# Web UX Sub-Spec 6: Frontend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the React/TypeScript single-page application that provides the chat-driven pipeline composer interface. Three-panel layout (sessions sidebar, chat panel, inspector panel), Zustand state management, typed API client with WebSocket streaming, and full accessibility support.

**Architecture:** The frontend is a Vite-built React 18 SPA served from `src/elspeth/web/frontend/`. In development, Vite proxies `/api` and `/ws` to the FastAPI backend on port 8000. In production, FastAPI serves the built `dist/` directory as static files with SPA fallback. No React Router -- the entire application is a single view with three panels rendered simultaneously.

**Tech Stack:** Vite 6, React 18, TypeScript 5 (strict), Zustand 5 (state), `@xyflow/react` (DAG visualisation), `@dagrejs/dagre` (auto-layout), `prism-react-renderer` (YAML highlighting), `openapi-typescript` (type generation)

**Spec:** `docs/superpowers/specs/2026-03-28-web-ux-sub6-frontend-design.md`
**Parent plan:** `docs/superpowers/plans/2026-03-28-web-ux-composer-mvp.md` (Phase 6, Tasks 6.1-6.10)

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `src/elspeth/web/frontend/package.json` | NPM project manifest |
| Create | `src/elspeth/web/frontend/tsconfig.json` | TypeScript strict config with `@/` path alias |
| Create | `src/elspeth/web/frontend/tsconfig.app.json` | App-specific TS config extending base |
| Create | `src/elspeth/web/frontend/vite.config.ts` | Dev server, proxy, path alias |
| Create | `src/elspeth/web/frontend/index.html` | SPA entry point |
| Create | `src/elspeth/web/frontend/src/main.tsx` | React root mount |
| Create | `src/elspeth/web/frontend/src/App.tsx` | Top-level component composition |
| Create | `src/elspeth/web/frontend/src/App.css` | Global application styles |
| Create | `src/elspeth/web/frontend/src/types/index.ts` | Hand-written fallback types |
| Create | `src/elspeth/web/frontend/src/types/api.ts` | OpenAPI-generated types (or manual mirror) |
| Create | `src/elspeth/web/frontend/src/api/client.ts` | Typed fetch wrappers with auth injection |
| Create | `src/elspeth/web/frontend/src/api/websocket.ts` | WebSocket manager with auto-reconnect |
| Create | `src/elspeth/web/frontend/src/stores/authStore.ts` | Auth state + localStorage persistence |
| Create | `src/elspeth/web/frontend/src/stores/sessionStore.ts` | Sessions, messages, composition state, version history |
| Create | `src/elspeth/web/frontend/src/stores/executionStore.ts` | Runs, progress, validation with auto-clear |
| Create | `src/elspeth/web/frontend/src/components/common/Layout.tsx` | Three-panel CSS grid |
| Create | `src/elspeth/web/frontend/src/components/common/AuthGuard.tsx` | Auth gate rendering LoginPage or children |
| Create | `src/elspeth/web/frontend/src/components/auth/LoginPage.tsx` | Login form (local + SSO) |
| Create | `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx` | Session list, new session, collapse toggle |
| Create | `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` | Message list + composing indicator + input |
| Create | `src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx` | Single message rendering |
| Create | `src/elspeth/web/frontend/src/components/chat/ChatInput.tsx` | Text input + send + file upload |
| Create | `src/elspeth/web/frontend/src/components/chat/ComposingIndicator.tsx` | Animated typing dots |
| Create | `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx` | Tab container + header buttons + validation banner |
| Create | `src/elspeth/web/frontend/src/components/inspector/SpecView.tsx` | Component cards with click-to-highlight |
| Create | `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx` | React Flow DAG with dagre layout |
| Create | `src/elspeth/web/frontend/src/components/inspector/YamlView.tsx` | Syntax-highlighted YAML + copy button |
| Create | `src/elspeth/web/frontend/src/components/inspector/RunsView.tsx` | Run history list |
| Create | `src/elspeth/web/frontend/src/components/execution/ProgressView.tsx` | Live progress bar + counters + cancel |
| Create | `src/elspeth/web/frontend/src/components/execution/ValidationResult.tsx` | Pass/fail banner |
| Create | `src/elspeth/web/frontend/src/hooks/useAuth.ts` | Auth lifecycle hook |
| Create | `src/elspeth/web/frontend/src/hooks/useSession.ts` | Session selection + loading hook |
| Create | `src/elspeth/web/frontend/src/hooks/useComposer.ts` | Message send + composing state hook |
| Create | `src/elspeth/web/frontend/src/hooks/useWebSocket.ts` | WebSocket connection lifecycle hook |
| Modify | `src/elspeth/web/app.py` | Static file mount for production serving |

---

### Task 1: Vite Project Initialisation

**Files:**
- Create: `src/elspeth/web/frontend/package.json`
- Create: `src/elspeth/web/frontend/tsconfig.json`
- Create: `src/elspeth/web/frontend/tsconfig.app.json`
- Create: `src/elspeth/web/frontend/vite.config.ts`
- Create: `src/elspeth/web/frontend/index.html`
- Create: `src/elspeth/web/frontend/src/main.tsx`
- Create: `src/elspeth/web/frontend/src/App.tsx`
- Create: `src/elspeth/web/frontend/src/App.css`

- [ ] **Step 1: Create the frontend directory and initialise the Vite project**

```bash
mkdir -p src/elspeth/web/frontend
cd src/elspeth/web/frontend
npm create vite@latest . -- --template react-ts
```

This generates the scaffolding: `package.json`, `tsconfig.json`, `tsconfig.app.json`, `vite.config.ts`, `index.html`, `src/main.tsx`, `src/App.tsx`, and `src/App.css`.

- [ ] **Step 2: Install all dependencies**

```bash
cd src/elspeth/web/frontend
npm install @xyflow/react @dagrejs/dagre zustand prism-react-renderer
npm install -D openapi-typescript @types/react @types/react-dom
```

- [ ] **Step 3: Configure the `@/` path alias in `tsconfig.app.json`**

Add `paths` to the generated `tsconfig.app.json` so imports like `@/stores/authStore` resolve to `src/stores/authStore`:

```jsonc
// tsconfig.app.json — add inside "compilerOptions"
{
  "compilerOptions": {
    // ... existing options from vite scaffold ...
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  }
}
```

- [ ] **Step 4: Configure the Vite dev proxy and path alias**

Replace the generated `vite.config.ts` with proxy configuration for `/api` and `/ws`:

```typescript
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
      },
    },
  },
});
```

- [ ] **Step 5: Replace the generated `App.tsx` with a minimal three-panel skeleton**

This verifies the build works. The panels will be replaced with real components in later tasks.

```tsx
// src/App.tsx
import "./App.css";

function App() {
  return (
    <div className="app-layout">
      <aside className="panel-sidebar">Sidebar</aside>
      <main className="panel-chat">Chat</main>
      <section className="panel-inspector">Inspector</section>
    </div>
  );
}

export default App;
```

```css
/* src/App.css */
.app-layout {
  display: grid;
  grid-template-columns: 200px 1fr 320px;
  grid-template-areas: "sidebar chat inspector";
  height: 100vh;
  min-width: 1280px;
}

.panel-sidebar {
  grid-area: sidebar;
  border-right: 1px solid #e0e0e0;
}

.panel-chat {
  grid-area: chat;
}

.panel-inspector {
  grid-area: inspector;
  border-left: 1px solid #e0e0e0;
}
```

- [ ] **Step 6: Verify the project builds and the dev server starts**

```bash
cd src/elspeth/web/frontend
npm run build   # Must exit 0 with output in dist/
npm run dev     # Must start on port 5173; open in browser to see three panels
```

- [ ] **Step 7: Add `src/elspeth/web/frontend/node_modules/` and `src/elspeth/web/frontend/dist/` to `.gitignore`**

Append to the project root `.gitignore`:

```
# Frontend build artifacts
src/elspeth/web/frontend/node_modules/
src/elspeth/web/frontend/dist/
```

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/web/frontend/ .gitignore
git commit -m "feat(web/frontend): initialize Vite + React 18 + TypeScript project with dev proxy"
```

---

### Task 2: TypeScript Types

**Files:**
- Create: `src/elspeth/web/frontend/src/types/index.ts`
- Create: `src/elspeth/web/frontend/src/types/api.ts`

- [ ] **Step 1: Add an npm script for OpenAPI type generation**

Add to `package.json` scripts:

```json
{
  "scripts": {
    "generate-types": "openapi-typescript http://localhost:8000/openapi.json -o src/types/api.generated.ts"
  }
}
```

This is run manually during development when the backend schema changes. It requires the FastAPI server to be running. The generated file is committed to the repo so builds don't depend on the backend being available.

- [ ] **Step 2: Create hand-written fallback types in `src/types/index.ts`**

These types mirror the backend Pydantic models. They serve as the primary types used throughout the frontend. If `openapi-typescript` generation is available, these can be replaced with imports from the generated file, but having explicit types ensures the frontend compiles independently of the backend.

```typescript
// src/types/index.ts

/** User profile returned by GET /api/auth/me */
export interface UserProfile {
  id: string;
  username: string;
  display_name: string;
}

/** Session summary for sidebar listing */
export interface Session {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

/** A single tool call within an assistant message (LiteLLM format) */
export interface ToolCall {
  id: string;
  type: string;
  function: {
    name: string;
    arguments: string; // JSON string, not parsed object
  };
}

/** A chat message in a session */
export interface ChatMessage {
  id: string;
  session_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  tool_calls: ToolCall[] | null;
  created_at: string;
}

/** A node in the pipeline composition */
export interface NodeSpec {
  id: string;
  name: string;
  type: "source" | "transform" | "gate" | "sink";
  plugin: string;
  config: Record<string, unknown>;
  config_summary: string;
}

/** An edge connecting two nodes */
export interface EdgeSpec {
  source: string;
  target: string;
  label: string | null;
  edge_type: "continue" | "route" | "error";
}

/** The full pipeline composition state */
export interface CompositionState {
  version: number;
  nodes: NodeSpec[];
  edges: EdgeSpec[];
  validation_errors?: string[]; // Stage 1 errors from composer (simple strings)
}

/** A version history entry for CompositionState */
export interface CompositionStateVersion {
  version: number;
  created_at: string;
  node_count: number;
}

/** Plugin summary from the catalog */
export interface PluginSummary {
  name: string;
  type: "source" | "transform" | "gate" | "sink";
  description: string;
}

/** Validation result from POST /api/sessions/{id}/validate (Stage 2) */
export interface ValidationResult {
  valid: boolean;
  summary: string;
  errors: ValidationError[];
}

/** A single validation error (Stage 2 — per-component detail) */
export interface ValidationError {
  component_id: string;
  component_type: string;
  message: string;
  suggestion: string | null;
}

/** An execution run */
export interface Run {
  id: string;
  session_id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  rows_processed: number;
  rows_failed: number;
  started_at: string;
  finished_at: string | null;
  composition_version: number;
}

/**
 * A progress event from the WebSocket.
 *
 * Terminal semantics:
 * - "progress" -- non-terminal. Row count update; pipeline still running.
 * - "error" -- non-terminal. Per-row exception; pipeline continues processing.
 *   The frontend appends the error to the exceptions list but does NOT stop
 *   the progress view or close the WebSocket.
 * - "completed" -- terminal. Pipeline finished successfully.
 * - "cancelled" -- terminal. Pipeline was cancelled.
 *
 * Note: "failed" is a Run status (set when the pipeline aborts), not a RunEvent
 * type. If the pipeline aborts, the WebSocket closes and the frontend fetches
 * the final Run status via REST.
 */
export interface RunEvent {
  type: "progress" | "error" | "completed" | "cancelled";
  run_id: string;
  rows_processed: number;
  rows_failed: number;
  exceptions: RunException[];
}

/** A single exception from execution */
export interface RunException {
  timestamp: string;
  node_name: string;
  message: string;
  row_id: string | null;
}

/** Live progress state derived from RunEvents */
export interface RunProgress {
  rows_processed: number;
  rows_failed: number;
  recent_exceptions: RunException[];
  status: "running" | "completed" | "cancelled";
}

/** API error response shape */
export interface ApiError {
  status: number;
  detail: string;
  error_type?: string;
  validation_errors?: ValidationError[];
}

/** File upload response */
export interface UploadResult {
  server_path: string;
  filename: string;
}
```

- [ ] **Step 3: Create `src/types/api.ts` as the re-export point**

This file is the single import point for types. When generated types become available, this file switches to re-exporting from the generated file without changing any import sites.

```typescript
// src/types/api.ts
//
// Re-export all types from the hand-written definitions.
// When openapi-typescript generation is available, change this to:
//   export type { ... } from "./api.generated";
//
export type {
  UserProfile,
  Session,
  ChatMessage,
  ToolCall,
  NodeSpec,
  EdgeSpec,
  CompositionState,
  CompositionStateVersion,
  PluginSummary,
  ValidationResult,
  ValidationError,
  Run,
  RunEvent,
  RunException,
  RunProgress,
  ApiError,
  UploadResult,
} from "./index";
```

- [ ] **Step 4: Verify the project still builds**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/types/
git commit -m "feat(web/frontend): add TypeScript types matching backend Pydantic schemas"
```

---

### Task 3: API Client

**Files:**
- Create: `src/elspeth/web/frontend/src/api/client.ts`

- [ ] **Step 1: Implement the typed API client**

The client provides typed `fetch` wrappers for every backend endpoint. All requests include the auth token from `localStorage`. Non-2xx responses throw a typed `ApiError`.

```typescript
// src/api/client.ts
import type {
  ApiError,
  ChatMessage,
  CompositionState,
  CompositionStateVersion,
  Run,
  Session,
  UploadResult,
  UserProfile,
  ValidationResult,
} from "@/types/api";

/**
 * Read the auth token from localStorage. Returns null if absent.
 */
function getToken(): string | null {
  return localStorage.getItem("auth_token");
}

/**
 * Build headers with auth token injection and optional content type.
 */
function authHeaders(contentType?: string): HeadersInit {
  const headers: Record<string, string> = {};
  const token = getToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  if (contentType) {
    headers["Content-Type"] = contentType;
  }
  return headers;
}

/**
 * Parse a response. Throws ApiError for non-2xx status codes.
 * Returns the parsed JSON body for success responses.
 *
 * Includes a global 401 interceptor: any API call returning 401
 * triggers authStore.logout() to handle token expiry mid-session
 * without requiring each caller to check for auth failures.
 */
async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    // Global 401 interceptor — trigger logout on any auth failure
    if (response.status === 401) {
      // Dynamic import to avoid circular dependency at module load time
      const { useAuthStore } = await import("@/stores/authStore");
      useAuthStore.getState().logout();
    }

    // R4-M2: Error envelope handling priority:
    //   1. error_type (if present in response body) — most specific
    //   2. HTTP status code — structural fallback
    //   3. detail text — human-readable description
    // All backend errors use `detail` (not `message`) as the human-readable field.
    let detail = response.statusText;
    let error_type: string | undefined;
    try {
      const body = await response.json();
      error_type = body.error_type;          // Check error_type first
      detail = body.detail ?? detail;        // detail, never body.message
    } catch {
      // Response body wasn't JSON — use statusText as detail fallback
    }
    const error: ApiError = {
      status: response.status,               // HTTP status as structural fallback
      detail,
      error_type,
    };
    throw error;
  }
  return response.json() as Promise<T>;
}

// ── Auth ──────────────────────────────────────────────────────────────

/** Auth provider config — unauthenticated endpoint */
export interface AuthConfig {
  provider: "local" | "oidc" | "entra";
  oidc_issuer?: string;
  oidc_client_id?: string;
}

export async function fetchAuthConfig(): Promise<AuthConfig> {
  const response = await fetch("/api/auth/config");
  return parseResponse(response);
}

export async function login(
  username: string,
  password: string
): Promise<{ access_token: string }> {
  const response = await fetch("/api/auth/login", {
    method: "POST",
    headers: authHeaders("application/json"),
    body: JSON.stringify({ username, password }),
  });
  return parseResponse(response);
}

export async function fetchCurrentUser(): Promise<UserProfile> {
  const response = await fetch("/api/auth/me", {
    headers: authHeaders(),
  });
  return parseResponse(response);
}

// ── Sessions ──────────────────────────────────────────────────────────

export async function fetchSessions(): Promise<Session[]> {
  const response = await fetch("/api/sessions", {
    headers: authHeaders(),
  });
  return parseResponse(response);
}

export async function createSession(): Promise<Session> {
  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: authHeaders("application/json"),
  });
  return parseResponse(response);
}

// ── Messages ──────────────────────────────────────────────────────────

export async function fetchMessages(
  sessionId: string
): Promise<ChatMessage[]> {
  const response = await fetch(`/api/sessions/${sessionId}/messages`, {
    headers: authHeaders(),
  });
  return parseResponse(response);
}

/**
 * Send a user message. The backend runs the composer tool-use loop
 * and returns the assistant response with updated composition state.
 */
export async function sendMessage(
  sessionId: string,
  content: string
): Promise<{
  message: ChatMessage;
  state: CompositionState | null;
}> {
  const response = await fetch(`/api/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: authHeaders("application/json"),
    body: JSON.stringify({ content }),
  });
  return parseResponse(response);
}

// ── Composition State ─────────────────────────────────────────────────

export async function fetchCompositionState(
  sessionId: string
): Promise<CompositionState | null> {
  const response = await fetch(`/api/sessions/${sessionId}/state`, {
    headers: authHeaders(),
  });
  if (response.status === 404) {
    return null;
  }
  return parseResponse(response);
}

export async function fetchStateVersions(
  sessionId: string
): Promise<CompositionStateVersion[]> {
  const response = await fetch(
    `/api/sessions/${sessionId}/state/versions`,
    { headers: authHeaders() }
  );
  return parseResponse(response);
}

export async function fetchYaml(sessionId: string): Promise<string> {
  const response = await fetch(
    `/api/sessions/${sessionId}/state/yaml`,
    { headers: authHeaders() }
  );
  return parseResponse(response);
}

export async function revertToVersion(
  sessionId: string,
  version: number
): Promise<CompositionState> {
  const response = await fetch(
    `/api/sessions/${sessionId}/state/versions/${version}/revert`,
    {
      method: "POST",
      headers: authHeaders("application/json"),
    }
  );
  return parseResponse(response);
}

// ── Validation & Execution ────────────────────────────────────────────

export async function validatePipeline(
  sessionId: string
): Promise<ValidationResult> {
  const response = await fetch(`/api/sessions/${sessionId}/validate`, {
    method: "POST",
    headers: authHeaders("application/json"),
  });
  return parseResponse(response);
}

export async function executePipeline(
  sessionId: string
): Promise<Run> {
  const response = await fetch(`/api/sessions/${sessionId}/execute`, {
    method: "POST",
    headers: authHeaders("application/json"),
  });
  return parseResponse(response);
}

export async function fetchRuns(sessionId: string): Promise<Run[]> {
  const response = await fetch(`/api/sessions/${sessionId}/runs`, {
    headers: authHeaders(),
  });
  return parseResponse(response);
}

export async function cancelRun(runId: string): Promise<void> {
  const response = await fetch(`/api/runs/${runId}/cancel`, {
    method: "POST",
    headers: authHeaders("application/json"),
  });
  if (!response.ok) {
    await parseResponse(response); // throws ApiError
  }
}

// ── File Upload ───────────────────────────────────────────────────────

export async function uploadFile(
  sessionId: string,
  file: File
): Promise<UploadResult> {
  const formData = new FormData();
  formData.append("file", file);

  // Do NOT set Content-Type — the browser sets it with the multipart boundary
  const token = getToken();
  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`/api/sessions/${sessionId}/upload`, {
    method: "POST",
    headers,
    body: formData,
  });
  return parseResponse(response);
}
```

- [ ] **Step 2: Verify the project builds**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/src/api/client.ts
git commit -m "feat(web/frontend): add typed API client with auth token injection"
```

---

### Task 4: WebSocket Manager

**Files:**
- Create: `src/elspeth/web/frontend/src/api/websocket.ts`

- [ ] **Step 1: Implement the WebSocket manager with auto-reconnect**

The manager connects to `/ws/runs/{runId}?token=<jwt>`, appending the JWT from `authStore.token` as a query parameter. It parses `RunEvent` JSON and calls registered handlers. Close codes are discriminated: 1000 (normal closure) and 1011 (internal error) do NOT reconnect -- the caller should poll REST for final status. 1006 (abnormal closure from network drop or server restart) triggers auto-reconnect with exponential backoff (1s, 2s, 4s, max 30s). 4001 (auth failure) does NOT reconnect -- instead it calls `authStore.logout()` and redirects to LoginPage. The caller can close the connection explicitly when the run completes or is cancelled.

```typescript
// src/api/websocket.ts
import type { RunEvent } from "@/types/api";

export interface WebSocketHandlers {
  onEvent: (event: RunEvent) => void;
  onError: (error: Event) => void;
  onDisconnect: () => void;
  onReconnect: () => void;
  onAuthFailure: () => void;
}

export interface WebSocketConnection {
  close: () => void;
}

/**
 * Connect to the execution progress WebSocket for a given run.
 *
 * Returns a connection handle with a close() method. The connection
 * auto-reconnects on unexpected disconnects until close() is called.
 */
export function connectToRun(
  runId: string,
  token: string,
  handlers: WebSocketHandlers
): WebSocketConnection {
  let socket: WebSocket | null = null;
  let closed = false;
  let reconnectDelay = 1000;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  function getWsUrl(): string {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws/runs/${runId}?token=${encodeURIComponent(token)}`;
  }

  function connect() {
    if (closed) return;

    socket = new WebSocket(getWsUrl());

    socket.onopen = () => {
      // Reset backoff on successful connection
      reconnectDelay = 1000;
    };

    socket.onmessage = (messageEvent: MessageEvent) => {
      const event: RunEvent = JSON.parse(messageEvent.data as string);
      handlers.onEvent(event);

      // Stop reconnecting if the run has reached a terminal state.
      // "error" is non-terminal (per-row exception, pipeline continues)
      // so it does NOT close the WebSocket.
      if (
        event.type === "completed" ||
        event.type === "cancelled"
      ) {
        closed = true;
        socket?.close();
      }
    };

    socket.onerror = (error: Event) => {
      handlers.onError(error);
    };

    socket.onclose = (closeEvent: CloseEvent) => {
      if (closed) return;

      // R4-H6: Discriminate close codes for reconnect behaviour
      switch (closeEvent.code) {
        case 1000:
          // Normal closure — run is terminal. Do NOT reconnect, poll REST
          // for final status.
          closed = true;
          handlers.onDisconnect();
          return;
        case 1006:
          // Abnormal closure (network drop, server restart) — auto-reconnect
          // with exponential backoff.
          handlers.onDisconnect();
          scheduleReconnect();
          return;
        case 1011:
          // Internal error — do NOT reconnect, poll REST for status.
          closed = true;
          handlers.onDisconnect();
          return;
        case 4001:
          // Auth failure — do NOT reconnect, trigger logout.
          closed = true;
          handlers.onAuthFailure();
          return;
        default:
          // Unknown close code — treat as abnormal, attempt reconnect.
          handlers.onDisconnect();
          scheduleReconnect();
          return;
      }
    };
  }

  function scheduleReconnect() {
    if (closed) return;

    reconnectTimer = setTimeout(() => {
      handlers.onReconnect();
      connect();
      // Exponential backoff, capped at 30 seconds
      reconnectDelay = Math.min(reconnectDelay * 2, 30_000);
    }, reconnectDelay);
  }

  // Start the initial connection
  connect();

  return {
    close() {
      closed = true;
      if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer);
      }
      socket?.close();
    },
  };
}
```

- [ ] **Step 2: Verify the project builds**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/src/api/websocket.ts
git commit -m "feat(web/frontend): add WebSocket manager with auto-reconnect"
```

---

### Task 5: Auth Store

**Files:**
- Create: `src/elspeth/web/frontend/src/stores/authStore.ts`

- [ ] **Step 1: Implement the auth Zustand store**

The store manages the JWT token and user profile. The token is persisted to `localStorage` so the authenticated state survives page refreshes. `isAuthenticated` is derived from `token !== null` within selectors, not stored as a separate field.

```typescript
// src/stores/authStore.ts
import { create } from "zustand";
import type { UserProfile, ApiError } from "@/types/api";
import * as api from "@/api/client";

const TOKEN_KEY = "auth_token";

interface AuthState {
  token: string | null;
  user: UserProfile | null;
  loginError: string | null;
  isLoading: boolean;

  login: (username: string, password: string) => Promise<void>;
  loginWithToken: (token: string) => Promise<void>;
  logout: () => void;
  loadFromStorage: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  token: null,
  user: null,
  loginError: null,
  isLoading: true, // starts true; loadFromStorage resolves it

  async login(username: string, password: string) {
    set({ loginError: null, isLoading: true });
    try {
      const { access_token } = await api.login(username, password);
      localStorage.setItem(TOKEN_KEY, access_token);
      set({ token: access_token });

      const user = await api.fetchCurrentUser();
      set({ user, isLoading: false });
    } catch (err) {
      const apiErr = err as ApiError;
      const message =
        apiErr.status === 401
          ? "Invalid username or password."
          : apiErr.detail ?? "Login failed. Please try again.";
      set({ token: null, user: null, loginError: message, isLoading: false });
      localStorage.removeItem(TOKEN_KEY);
    }
  },

  async loginWithToken(token: string) {
    localStorage.setItem(TOKEN_KEY, token);
    set({ token, loginError: null, isLoading: true });
    try {
      const user = await api.fetchCurrentUser();
      set({ user, isLoading: false });
    } catch {
      set({ token: null, user: null, isLoading: false });
      localStorage.removeItem(TOKEN_KEY);
    }
  },

  logout() {
    localStorage.removeItem(TOKEN_KEY);
    set({ token: null, user: null, loginError: null });
  },

  async loadFromStorage() {
    const token = localStorage.getItem(TOKEN_KEY);
    if (!token) {
      set({ isLoading: false });
      return;
    }
    set({ token });
    try {
      const user = await api.fetchCurrentUser();
      set({ user, isLoading: false });
    } catch {
      // Token invalid or expired — clear it
      localStorage.removeItem(TOKEN_KEY);
      set({ token: null, user: null, isLoading: false });
    }
  },
}));

/**
 * Selector: true when the user is authenticated.
 * Usage: const isAuthenticated = useAuthStore(selectIsAuthenticated);
 */
export const selectIsAuthenticated = (state: AuthState): boolean =>
  state.token !== null && state.user !== null;
```

- [ ] **Step 2: Verify the project builds**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/src/stores/authStore.ts
git commit -m "feat(web/frontend): add auth Zustand store with localStorage persistence"
```

---

### Task 6: Session Store

**Files:**
- Create: `src/elspeth/web/frontend/src/stores/sessionStore.ts`

- [ ] **Step 1: Implement the session Zustand store**

The session store manages the session list, active session selection, chat messages, composition state, composing indicator, and version history. The `sendMessage` action sets `isComposing = true` before calling the API and resets it when the response arrives. The wire response from `POST /api/sessions/{id}/messages` returns `{message, state}` -- the `state` field maps to the store's `compositionState`. When a new version arrives, `sendMessage` calls `executionStore.clearValidation()` explicitly. `selectSession` and `revertToVersion` also call `clearValidation()` directly -- `selectSession` on session switch, `revertToVersion` BEFORE updating compositionState (to prevent a frame where stale validation is visible with the new version). The executionStore subscriber (Task 7) provides a safety net for any other version-change paths.

```typescript
// src/stores/sessionStore.ts
import { create } from "zustand";
import type {
  Session,
  ChatMessage,
  CompositionState,
  CompositionStateVersion,
  ApiError,
} from "@/types/api";
import * as api from "@/api/client";
import { useExecutionStore } from "./executionStore";

interface SessionState {
  sessions: Session[];
  activeSessionId: string | null;
  messages: ChatMessage[];
  compositionState: CompositionState | null;
  isComposing: boolean;
  stateVersions: CompositionStateVersion[];
  error: string | null;

  loadSessions: () => Promise<void>;
  createSession: () => Promise<void>;
  selectSession: (id: string) => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
  loadStateVersions: () => Promise<void>;
  revertToVersion: (version: number) => Promise<void>;
  clearError: () => void;
}

export const useSessionStore = create<SessionState>((set, get) => ({
  sessions: [],
  activeSessionId: null,
  messages: [],
  compositionState: null,
  isComposing: false,
  stateVersions: [],
  error: null,

  async loadSessions() {
    try {
      const sessions = await api.fetchSessions();
      set({ sessions });
    } catch (err) {
      set({ error: "Failed to load sessions. Please refresh the page." });
    }
  },

  async createSession() {
    try {
      const session = await api.createSession();
      set((state) => ({
        sessions: [session, ...state.sessions],
        activeSessionId: session.id,
        messages: [],
        compositionState: null,
        stateVersions: [],
        error: null,
      }));
    } catch {
      set({ error: "Failed to create session. Please try again." });
    }
  },

  async selectSession(id: string) {
    // R4-H3: Clear validation when switching sessions to prevent
    // stale validation from a previous session being visible
    useExecutionStore.getState().clearValidation();

    set({
      activeSessionId: id,
      messages: [],
      compositionState: null,
      stateVersions: [],
      isComposing: false,
      error: null,
    });

    try {
      const [messages, compositionState] = await Promise.all([
        api.fetchMessages(id),
        api.fetchCompositionState(id),
      ]);
      set({ messages, compositionState });
    } catch {
      set({ error: "Failed to load session. Please refresh the page." });
    }
  },

  async sendMessage(content: string) {
    const { activeSessionId } = get();
    if (!activeSessionId) return;

    set({ isComposing: true, error: null });

    try {
      const result = await api.sendMessage(activeSessionId, content);
      const { message, state } = result;
      set((s) => {
        const previousVersion = s.compositionState?.version ?? null;
        const newVersion = state?.version ?? null;
        const versionChanged =
          newVersion !== null && newVersion !== previousVersion;

        // R4-H3: Clear validation BEFORE updating compositionState
        // when a new state version arrives from the composer
        if (versionChanged) {
          useExecutionStore.getState().clearValidation();
        }

        return {
          messages: [...s.messages, message],
          compositionState: state ?? s.compositionState,
          isComposing: false,
        };
      });
    } catch (err) {
      const apiErr = err as ApiError;
      let message: string;
      // Error dispatch based on HTTP status + error_type field
      if (apiErr.status === 422 && apiErr.error_type === "convergence") {
        message =
          "ELSPETH couldn't complete the composition after multiple attempts. Try breaking your request into smaller steps.";
      } else if (apiErr.status === 502 && apiErr.error_type === "llm_unavailable") {
        message =
          "The AI service is temporarily unavailable. Please try again in a moment.";
      } else if (apiErr.status === 502 && apiErr.error_type === "llm_auth_error") {
        message =
          "The AI service configuration is invalid. Please contact your administrator.";
      } else {
        message = apiErr.detail ?? "Failed to send message. Please try again.";
      }
      set({ isComposing: false, error: message });
    }
  },

  async loadStateVersions() {
    const { activeSessionId } = get();
    if (!activeSessionId) return;

    try {
      const versions = await api.fetchStateVersions(activeSessionId);
      set({ stateVersions: versions });
    } catch {
      // Version history is non-critical — fail silently
    }
  },

  async revertToVersion(version: number) {
    const { activeSessionId } = get();
    if (!activeSessionId) return;

    try {
      // R4-H3: Clear validation BEFORE updating compositionState
      // to prevent a frame where stale validation is visible with the new version
      useExecutionStore.getState().clearValidation();

      const compositionState = await api.revertToVersion(
        activeSessionId,
        version
      );
      set({ compositionState });
    } catch {
      set({ error: "Failed to revert to version. Please try again." });
    }
  },

  clearError() {
    set({ error: null });
  },
}));
```

- [ ] **Step 2: Verify the project builds**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/src/stores/sessionStore.ts
git commit -m "feat(web/frontend): add session Zustand store with composing state"
```

---

### Task 7: Execution Store

**Files:**
- Create: `src/elspeth/web/frontend/src/stores/executionStore.ts`

- [ ] **Step 1: Implement the execution Zustand store with validation auto-clear**

The execution store manages validation results, execution runs, live progress, and the WebSocket connection lifecycle. The key behaviour is **auto-clear**: when the session store's `compositionState.version` changes, the validation result is cleared and the Execute button becomes disabled. This is implemented via a `subscribe` call on the session store that watches for version changes.

```typescript
// src/stores/executionStore.ts
import { create } from "zustand";
import type {
  Run,
  RunProgress,
  RunException,
  ValidationResult,
  ApiError,
} from "@/types/api";
import * as api from "@/api/client";
import { connectToRun, type WebSocketConnection } from "@/api/websocket";
import { useAuthStore } from "./authStore";
import { useSessionStore } from "./sessionStore";

const MAX_RECENT_EXCEPTIONS = 50;

interface ExecutionState {
  runs: Run[];
  activeRunId: string | null;
  progress: RunProgress | null;
  validationResult: ValidationResult | null;
  isValidating: boolean;
  isExecuting: boolean;
  wsDisconnected: boolean;
  error: string | null;

  validate: (sessionId: string) => Promise<void>;
  execute: (sessionId: string) => Promise<void>;
  cancel: (runId: string) => Promise<void>;
  loadRuns: (sessionId: string) => Promise<void>;
  clearValidation: () => void;
}

// The WebSocket connection handle is held outside Zustand state
// because it's not serialisable and components don't need to read it.
let wsConnection: WebSocketConnection | null = null;

export const useExecutionStore = create<ExecutionState>((set, get) => ({
  runs: [],
  activeRunId: null,
  progress: null,
  validationResult: null,
  isValidating: false,
  isExecuting: false,
  wsDisconnected: false,
  error: null,

  async validate(sessionId: string) {
    set({ isValidating: true, validationResult: null, error: null });
    try {
      const result = await api.validatePipeline(sessionId);
      set({ validationResult: result, isValidating: false });
    } catch (err) {
      const apiErr = err as ApiError;
      const message =
        apiErr.status === 500
          ? "Validation encountered an internal error. Please try again."
          : apiErr.detail ?? "Validation failed. Please try again.";
      set({
        isValidating: false,
        error: message,
      });
    }
  },

  async execute(sessionId: string) {
    set({ isExecuting: true, error: null });
    try {
      const run = await api.executePipeline(sessionId);
      set((state) => ({
        runs: [run, ...state.runs],
        activeRunId: run.id,
        isExecuting: false,
        progress: {
          rows_processed: 0,
          rows_failed: 0,
          recent_exceptions: [],
          status: "running",
        },
      }));

      // Close any existing WebSocket connection
      wsConnection?.close();

      // Open a WebSocket for live progress, passing JWT as query parameter
      const token = useAuthStore.getState().token ?? "";
      wsConnection = connectToRun(run.id, token, {
        onEvent(event) {
          set((state) => {
            // Accumulate exceptions, keeping the most recent N
            const allExceptions = [
              ...event.exceptions,
              ...(state.progress?.recent_exceptions ?? []),
            ].slice(0, MAX_RECENT_EXCEPTIONS);

            const newProgress: RunProgress = {
              rows_processed: event.rows_processed,
              rows_failed: event.rows_failed,
              recent_exceptions: allExceptions,
              // "error" is non-terminal (per-row exception) — status stays "running"
              status:
                event.type === "completed"
                  ? "completed"
                  : event.type === "cancelled"
                    ? "cancelled"
                    : "running",
            };

            // Update the run in the list when terminal
            // ("error" is non-terminal — only "completed" and "cancelled" are terminal)
            let updatedRuns = state.runs;
            if (
              event.type === "completed" ||
              event.type === "cancelled"
            ) {
              updatedRuns = state.runs.map((r) =>
                r.id === event.run_id
                  ? {
                      ...r,
                      status: newProgress.status as Run["status"],
                      rows_processed: event.rows_processed,
                      rows_failed: event.rows_failed,
                    }
                  : r
              );
            }

            return {
              progress: newProgress,
              runs: updatedRuns,
              wsDisconnected: false,
            };
          });
        },
        onError() {
          // WebSocket errors are not user-actionable; the auto-reconnect
          // handles recovery. We just track the disconnected state for UI.
        },
        onDisconnect() {
          set({ wsDisconnected: true });
        },
        onReconnect() {
          set({ wsDisconnected: false });
        },
        onAuthFailure() {
          // Close code 4001 — do not reconnect, trigger logout
          useAuthStore.getState().logout();
        },
      });
    } catch (err) {
      const apiErr = err as ApiError;
      const message =
        apiErr.status === 409
          ? "A run is already in progress for this pipeline."
          : apiErr.detail ?? "Pipeline execution failed. Check the Runs tab for error details.";
      set({
        isExecuting: false,
        error: message,
      });
    }
  },

  async cancel(runId: string) {
    try {
      await api.cancelRun(runId);
    } catch (err) {
      const apiErr = err as ApiError;
      set({ error: apiErr.detail ?? "Failed to cancel run." });
    }
  },

  async loadRuns(sessionId: string) {
    try {
      const runs = await api.fetchRuns(sessionId);
      set({ runs });
    } catch {
      // Non-critical — runs list can be stale temporarily
    }
  },

  clearValidation() {
    set({ validationResult: null });
  },
}));

// ── Auto-clear validation on composition version change ───────────────
//
// Subscribe to the session store. When compositionState.version changes,
// clear the validation result so the user must re-validate before executing.
let previousVersion: number | null = null;

useSessionStore.subscribe((state) => {
  const currentVersion = state.compositionState?.version ?? null;
  if (previousVersion !== null && currentVersion !== previousVersion) {
    useExecutionStore.getState().clearValidation();
  }
  previousVersion = currentVersion;
});
```

- [ ] **Step 2: Verify the project builds**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/src/stores/executionStore.ts
git commit -m "feat(web/frontend): add execution Zustand store with validation auto-clear"
```

---

### Task 8: Auth Components

**Files:**
- Create: `src/elspeth/web/frontend/src/hooks/useAuth.ts`
- Create: `src/elspeth/web/frontend/src/components/common/AuthGuard.tsx`
- Create: `src/elspeth/web/frontend/src/components/auth/LoginPage.tsx`

- [ ] **Step 1: Implement the `useAuth` hook**

A thin hook that calls `loadFromStorage()` on mount and exposes the auth state. Components use this instead of reading the store directly.

```tsx
// src/hooks/useAuth.ts
import { useEffect } from "react";
import { useAuthStore, selectIsAuthenticated } from "@/stores/authStore";

/**
 * Hook for auth lifecycle. Calls loadFromStorage on mount.
 * Returns auth state and actions.
 */
export function useAuth() {
  const loadFromStorage = useAuthStore((s) => s.loadFromStorage);
  const isAuthenticated = useAuthStore(selectIsAuthenticated);
  const isLoading = useAuthStore((s) => s.isLoading);
  const user = useAuthStore((s) => s.user);
  const loginError = useAuthStore((s) => s.loginError);
  const login = useAuthStore((s) => s.login);
  const loginWithToken = useAuthStore((s) => s.loginWithToken);
  const logout = useAuthStore((s) => s.logout);

  useEffect(() => {
    loadFromStorage();
  }, [loadFromStorage]);

  return { isAuthenticated, isLoading, user, loginError, login, loginWithToken, logout };
}
```

- [ ] **Step 2: Implement `AuthGuard`**

Wraps the application. Shows a loading spinner while checking stored credentials, then either renders children (authenticated) or the LoginPage (unauthenticated).

```tsx
// src/components/common/AuthGuard.tsx
import type { ReactNode } from "react";
import { useAuth } from "@/hooks/useAuth";
import { LoginPage } from "@/components/auth/LoginPage";

interface AuthGuardProps {
  children: ReactNode;
}

export function AuthGuard({ children }: AuthGuardProps) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
        }}
      >
        Loading...
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginPage />;
  }

  return <>{children}</>;
}
```

- [ ] **Step 3: Implement `LoginPage`**

The login page fetches `GET /api/auth/config` on mount to determine which login form to show. The response `{provider: "local" | "oidc" | "entra", oidc_issuer?: string, oidc_client_id?: string}` drives the UI: `"local"` renders a username/password form that calls `authStore.login()`; `"oidc"` or `"entra"` renders a "Sign in with SSO" button that constructs the OIDC redirect URL from `oidc_issuer` and `oidc_client_id` in the config response. Error messages display inline.

```tsx
// src/components/auth/LoginPage.tsx
import { useState, useEffect, type FormEvent } from "react";
import { useAuth } from "@/hooks/useAuth";
import * as api from "@/api/client";
import type { AuthConfig } from "@/api/client";

export function LoginPage() {
  const { login, loginWithToken, loginError } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [authConfig, setAuthConfig] = useState<AuthConfig | null>(null);
  const [configLoading, setConfigLoading] = useState(true);

  // Fetch auth config on mount to determine which login form to show
  useEffect(() => {
    api.fetchAuthConfig().then((config) => {
      setAuthConfig(config);
      setConfigLoading(false);
    }).catch(() => {
      // If config fetch fails, fall back to local auth
      setAuthConfig({ provider: "local" });
      setConfigLoading(false);
    });
  }, []);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!username || !password) return;

    setIsSubmitting(true);
    await login(username, password);
    setIsSubmitting(false);
  }

  function handleSsoRedirect() {
    if (!authConfig?.oidc_issuer || !authConfig?.oidc_client_id) return;
    const redirectUri = `${window.location.origin}/api/auth/callback`;
    const url = `${authConfig.oidc_issuer}/authorize?client_id=${encodeURIComponent(authConfig.oidc_client_id)}&response_type=code&redirect_uri=${encodeURIComponent(redirectUri)}`;
    window.location.href = url;
  }

  if (configLoading) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
        }}
      >
        Loading...
      </div>
    );
  }

  const isOidc = authConfig?.provider === "oidc" || authConfig?.provider === "entra";

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "100vh",
        backgroundColor: "#f5f5f5",
      }}
    >
      <div
        style={{
          width: 360,
          padding: 32,
          backgroundColor: "#fff",
          borderRadius: 8,
          boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
        }}
      >
        <h1 style={{ fontSize: 24, marginBottom: 24, textAlign: "center" }}>
          Sign in to ELSPETH
        </h1>

        {loginError && (
          <div
            role="alert"
            style={{
              padding: "8px 12px",
              marginBottom: 16,
              backgroundColor: "#fef2f2",
              color: "#991b1b",
              borderRadius: 4,
              fontSize: 14,
            }}
          >
            {loginError}
          </div>
        )}

        {isOidc ? (
          /* OIDC / Entra SSO: single "Sign in with SSO" button */
          <button
            onClick={handleSsoRedirect}
            style={{
              display: "block",
              width: "100%",
              padding: "10px 16px",
              backgroundColor: "#2563eb",
              color: "#fff",
              border: "none",
              borderRadius: 4,
              fontSize: 14,
              cursor: "pointer",
            }}
          >
            Sign in with SSO
          </button>
        ) : (
          /* Local auth: username/password form */
          <form onSubmit={handleSubmit}>
            <label
              htmlFor="username"
              style={{ display: "block", marginBottom: 4, fontSize: 14 }}
            >
              Username
            </label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              required
              style={{
                display: "block",
                width: "100%",
                padding: "8px 12px",
                marginBottom: 16,
                border: "1px solid #d1d5db",
                borderRadius: 4,
                fontSize: 14,
                boxSizing: "border-box",
              }}
            />

            <label
              htmlFor="password"
              style={{ display: "block", marginBottom: 4, fontSize: 14 }}
            >
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
              style={{
                display: "block",
                width: "100%",
                padding: "8px 12px",
                marginBottom: 24,
                border: "1px solid #d1d5db",
                borderRadius: 4,
                fontSize: 14,
                boxSizing: "border-box",
              }}
            />

            <button
              type="submit"
              disabled={isSubmitting}
              style={{
                display: "block",
                width: "100%",
                padding: "10px 16px",
                backgroundColor: isSubmitting ? "#9ca3af" : "#2563eb",
                color: "#fff",
                border: "none",
                borderRadius: 4,
                fontSize: 14,
                cursor: isSubmitting ? "not-allowed" : "pointer",
              }}
            >
              {isSubmitting ? "Signing in..." : "Sign in"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Verify the project builds**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/hooks/useAuth.ts \
        src/elspeth/web/frontend/src/components/common/AuthGuard.tsx \
        src/elspeth/web/frontend/src/components/auth/LoginPage.tsx
git commit -m "feat(web/frontend): add AuthGuard and LoginPage with local auth"
```

---

### Task 9: Session Sidebar

**Files:**
- Create: `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx`
- Create: `src/elspeth/web/frontend/src/hooks/useSession.ts`

- [ ] **Step 1: Implement the `useSession` hook**

Loads sessions on mount and provides session actions.

```tsx
// src/hooks/useSession.ts
import { useEffect } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";

/**
 * Hook for session lifecycle. Loads sessions on mount.
 * When the active session changes, loads runs for that session.
 */
export function useSession() {
  const loadSessions = useSessionStore((s) => s.loadSessions);
  const sessions = useSessionStore((s) => s.sessions);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const createSession = useSessionStore((s) => s.createSession);
  const selectSession = useSessionStore((s) => s.selectSession);
  const loadRuns = useExecutionStore((s) => s.loadRuns);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Load runs whenever the active session changes
  useEffect(() => {
    if (activeSessionId) {
      loadRuns(activeSessionId);
    }
  }, [activeSessionId, loadRuns]);

  return { sessions, activeSessionId, createSession, selectSession };
}
```

- [ ] **Step 2: Implement `SessionSidebar`**

The sidebar displays a collapsible session list with a "New Session" button pinned at the bottom. The collapsed state persists in `localStorage`.

```tsx
// src/components/sessions/SessionSidebar.tsx
import { useState } from "react";
import { useSession } from "@/hooks/useSession";

/** Format a date string as a relative time ("2 min ago", "yesterday") */
function relativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHr = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHr / 24);

  if (diffSec < 60) return "just now";
  if (diffMin < 60) return `${diffMin} min ago`;
  if (diffHr < 24) return `${diffHr}h ago`;
  if (diffDay === 1) return "yesterday";
  return `${diffDay}d ago`;
}

const COLLAPSED_KEY = "sidebar_collapsed";

export function SessionSidebar() {
  const { sessions, activeSessionId, createSession, selectSession } =
    useSession();
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem(COLLAPSED_KEY) === "true"
  );

  function toggleCollapsed() {
    const next = !collapsed;
    setCollapsed(next);
    localStorage.setItem(COLLAPSED_KEY, String(next));
  }

  if (collapsed) {
    return (
      <aside
        style={{
          width: 40,
          borderRight: "1px solid #e0e0e0",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          paddingTop: 8,
        }}
      >
        <button
          onClick={toggleCollapsed}
          title="Expand sidebar"
          aria-label="Expand sessions sidebar"
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            fontSize: 18,
            padding: 4,
          }}
        >
          &#x276F; {/* Right chevron */}
        </button>
      </aside>
    );
  }

  return (
    <aside
      style={{
        width: 200,
        borderRight: "1px solid #e0e0e0",
        display: "flex",
        flexDirection: "column",
        height: "100%",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 12px 8px",
          borderBottom: "1px solid #e0e0e0",
        }}
      >
        <span style={{ fontWeight: 600, fontSize: 14 }}>Sessions</span>
        <button
          onClick={toggleCollapsed}
          title="Collapse sidebar"
          aria-label="Collapse sessions sidebar"
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            fontSize: 14,
          }}
        >
          &#x276E; {/* Left chevron */}
        </button>
      </div>

      {/* Session list */}
      <div style={{ flex: 1, overflowY: "auto" }}>
        {sessions.length === 0 ? (
          <div
            style={{
              padding: 16,
              color: "#6b7280",
              fontSize: 13,
              textAlign: "center",
            }}
          >
            No sessions yet. Click the button below to start.
          </div>
        ) : (
          sessions.map((session) => {
            const isActive = session.id === activeSessionId;
            return (
              <button
                key={session.id}
                onClick={() => selectSession(session.id)}
                style={{
                  display: "block",
                  width: "100%",
                  padding: "10px 12px",
                  border: "none",
                  borderLeft: isActive ? "3px solid #2563eb" : "3px solid transparent",
                  backgroundColor: isActive ? "#eff6ff" : "transparent",
                  cursor: "pointer",
                  textAlign: "left",
                  fontSize: 13,
                }}
              >
                <div
                  style={{
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    fontWeight: isActive ? 600 : 400,
                  }}
                >
                  {session.title}
                </div>
                <div style={{ fontSize: 11, color: "#9ca3af", marginTop: 2 }}>
                  {relativeTime(session.updated_at)}
                </div>
              </button>
            );
          })
        )}
      </div>

      {/* New session button */}
      <div style={{ padding: 8, borderTop: "1px solid #e0e0e0" }}>
        <button
          onClick={createSession}
          style={{
            display: "block",
            width: "100%",
            padding: "8px 12px",
            backgroundColor: "#2563eb",
            color: "#fff",
            border: "none",
            borderRadius: 4,
            cursor: "pointer",
            fontSize: 13,
          }}
        >
          + New Session
        </button>
      </div>
    </aside>
  );
}
```

- [ ] **Step 3: Verify the project builds**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/src/hooks/useSession.ts \
        src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx
git commit -m "feat(web/frontend): add session sidebar with collapse toggle"
```

---

### Task 10: Chat Components

**Files:**
- Create: `src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx`
- Create: `src/elspeth/web/frontend/src/components/chat/ComposingIndicator.tsx`
- Create: `src/elspeth/web/frontend/src/components/chat/ChatInput.tsx`
- Create: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
- Create: `src/elspeth/web/frontend/src/hooks/useComposer.ts`

- [ ] **Step 1: Implement `MessageBubble`**

Renders a single chat message. User messages are right-aligned with a blue tint. Assistant messages are left-aligned with a neutral background. Tool calls are shown as a collapsible section. System messages are centre-aligned full-width banners with muted colour, italic text, and no sender label -- used for system-injected messages such as "Pipeline reverted to version N."

```tsx
// src/components/chat/MessageBubble.tsx
import { useState } from "react";
import type { ChatMessage } from "@/types/api";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const [toolsExpanded, setToolsExpanded] = useState(false);

  // System messages: centre-aligned full-width banner, muted colour,
  // italic text, no sender label. Used for audit markers like
  // "Pipeline reverted to version N."
  if (isSystem) {
    return (
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          padding: "4px 16px",
        }}
      >
        <div
          style={{
            width: "100%",
            padding: "8px 14px",
            borderRadius: 6,
            backgroundColor: "#f3f4f6",
            opacity: 0.75,
            fontSize: 13,
            lineHeight: 1.5,
            fontStyle: "italic",
            textAlign: "center",
            color: "#6b7280",
          }}
        >
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        padding: "4px 16px",
      }}
    >
      <div
        style={{
          maxWidth: "80%",
          padding: "10px 14px",
          borderRadius: 12,
          backgroundColor: isUser ? "#dbeafe" : "#f3f4f6",
          fontSize: 14,
          lineHeight: 1.5,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {message.content}

        {/* Tool calls section (assistant messages only) */}
        {message.tool_calls && message.tool_calls.length > 0 && (
          <div style={{ marginTop: 8, borderTop: "1px solid #d1d5db", paddingTop: 6 }}>
            <button
              onClick={() => setToolsExpanded(!toolsExpanded)}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                fontSize: 12,
                color: "#6b7280",
                padding: 0,
              }}
            >
              {toolsExpanded ? "\u25BC" : "\u25B6"} Tool calls (
              {message.tool_calls.length})
            </button>
            {toolsExpanded && (
              <ul style={{ margin: "4px 0 0", paddingLeft: 16, fontSize: 12 }}>
                {message.tool_calls.map((tc, i) => (
                  <li key={tc.id ?? i} style={{ color: "#4b5563", marginBottom: 4 }}>
                    <strong>{tc.function.name}</strong>
                    {tc.function.arguments && (
                      <details style={{ marginTop: 2 }}>
                        <summary style={{ cursor: "pointer", color: "#6b7280", fontSize: 11 }}>
                          Arguments
                        </summary>
                        <pre
                          style={{
                            margin: "2px 0 0",
                            padding: 4,
                            backgroundColor: "#f9fafb",
                            borderRadius: 3,
                            fontSize: 11,
                            whiteSpace: "pre-wrap",
                            wordBreak: "break-word",
                          }}
                        >
                          {tc.function.arguments}
                        </pre>
                      </details>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Implement `ComposingIndicator`**

Animated three-dot indicator shown while the backend is processing. Uses CSS keyframe animation for staggered bounce.

```tsx
// src/components/chat/ComposingIndicator.tsx

const dotStyle: React.CSSProperties = {
  display: "inline-block",
  width: 8,
  height: 8,
  borderRadius: "50%",
  backgroundColor: "#9ca3af",
  margin: "0 2px",
  animation: "composing-bounce 1.4s infinite ease-in-out both",
};

export function ComposingIndicator() {
  return (
    <>
      <style>{`
        @keyframes composing-bounce {
          0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
          40% { transform: scale(1); opacity: 1; }
        }
      `}</style>
      <div
        style={{
          display: "flex",
          justifyContent: "flex-start",
          padding: "4px 16px",
        }}
        aria-live="polite"
        aria-label="ELSPETH is composing a response"
      >
        <div
          style={{
            padding: "10px 14px",
            borderRadius: 12,
            backgroundColor: "#f3f4f6",
            display: "flex",
            alignItems: "center",
            gap: 4,
          }}
        >
          <span style={{ ...dotStyle, animationDelay: "0s" }} />
          <span style={{ ...dotStyle, animationDelay: "0.16s" }} />
          <span style={{ ...dotStyle, animationDelay: "0.32s" }} />
          <span style={{ fontSize: 12, color: "#6b7280", marginLeft: 6 }}>
            ELSPETH is composing...
          </span>
        </div>
      </div>
    </>
  );
}
```

- [ ] **Step 3: Implement `ChatInput` with file upload**

Multi-line text input with Enter-to-send (Shift+Enter for newline), a send button, and a file upload button. The upload button opens a file picker, uploads the file, and injects the server path into the text input.

```tsx
// src/components/chat/ChatInput.tsx
import {
  useState,
  useRef,
  useCallback,
  type KeyboardEvent,
  type ChangeEvent,
} from "react";
import * as api from "@/api/client";
import { useSessionStore } from "@/stores/sessionStore";

interface ChatInputProps {
  onSend: (content: string) => void;
  disabled: boolean;
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
}

export function ChatInput({ onSend, disabled, inputRef }: ChatInputProps) {
  const [text, setText] = useState("");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);

  const handleSend = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
  }, [text, disabled, onSend]);

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  async function handleFileSelect(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file || !activeSessionId) return;

    setIsUploading(true);
    setUploadError(null);

    try {
      const result = await api.uploadFile(activeSessionId, file);
      setText(
        (prev) =>
          prev +
          (prev ? "\n" : "") +
          `I've uploaded a file at ${result.server_path}`
      );
    } catch (err) {
      const apiErr = err as api.ApiError;
      if (apiErr.status === 413) {
        setUploadError(
          "The file exceeds the maximum upload size. Please use a smaller file."
        );
      } else {
        setUploadError(
          "Upload failed due to a network error. Please check your connection and try again."
        );
      }
    } finally {
      setIsUploading(false);
      // Reset the file input so the same file can be re-selected
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  return (
    <div style={{ padding: "8px 16px", borderTop: "1px solid #e0e0e0" }}>
      {uploadError && (
        <div
          role="alert"
          style={{
            padding: "6px 10px",
            marginBottom: 8,
            backgroundColor: "#fef2f2",
            color: "#991b1b",
            borderRadius: 4,
            fontSize: 12,
          }}
        >
          {uploadError}
        </div>
      )}
      <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
        <textarea
          ref={inputRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe the pipeline you want to build..."
          rows={2}
          style={{
            flex: 1,
            resize: "vertical",
            padding: "8px 12px",
            border: "1px solid #d1d5db",
            borderRadius: 6,
            fontSize: 14,
            fontFamily: "inherit",
            lineHeight: 1.4,
          }}
        />

        {/* File upload button */}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading || !activeSessionId}
          title="Upload file"
          aria-label="Upload file"
          style={{
            padding: "8px 10px",
            backgroundColor: "transparent",
            border: "1px solid #d1d5db",
            borderRadius: 6,
            cursor: isUploading ? "not-allowed" : "pointer",
            fontSize: 16,
          }}
        >
          {isUploading ? "\u23F3" : "\uD83D\uDCCE"} {/* Hourglass or Paperclip */}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          onChange={handleFileSelect}
          style={{ display: "none" }}
          aria-hidden="true"
        />

        {/* Send button */}
        <button
          onClick={handleSend}
          disabled={disabled || !text.trim()}
          style={{
            padding: "8px 16px",
            backgroundColor:
              disabled || !text.trim() ? "#9ca3af" : "#2563eb",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            cursor:
              disabled || !text.trim() ? "not-allowed" : "pointer",
            fontSize: 14,
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Implement `ChatPanel`**

Combines the message list, composing indicator, and input. Auto-scrolls to the bottom when new messages arrive, but suppresses auto-scroll when the user has scrolled up.

```tsx
// src/components/chat/ChatPanel.tsx
import { useEffect, useRef, useCallback } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { MessageBubble } from "./MessageBubble";
import { ComposingIndicator } from "./ComposingIndicator";
import { ChatInput } from "./ChatInput";

export function ChatPanel() {
  const messages = useSessionStore((s) => s.messages);
  const isComposing = useSessionStore((s) => s.isComposing);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const sendMessage = useSessionStore((s) => s.sendMessage);
  const error = useSessionStore((s) => s.error);
  const clearError = useSessionStore((s) => s.clearError);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const isUserScrolledUp = useRef(false);

  // Track whether the user has scrolled up from the bottom
  function handleScroll() {
    const container = scrollContainerRef.current;
    if (!container) return;
    const threshold = 40; // pixels from bottom
    const atBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight <
      threshold;
    isUserScrolledUp.current = !atBottom;
  }

  // Auto-scroll to bottom when new messages arrive (unless user scrolled up)
  useEffect(() => {
    if (!isUserScrolledUp.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isComposing]);

  // Return focus to input when composing ends
  useEffect(() => {
    if (!isComposing) {
      inputRef.current?.focus();
    }
  }, [isComposing]);

  const handleSend = useCallback(
    (content: string) => {
      sendMessage(content);
    },
    [sendMessage]
  );

  if (!activeSessionId) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          color: "#6b7280",
          fontSize: 15,
          padding: 32,
          textAlign: "center",
        }}
      >
        Select a session from the sidebar, or create a new one to get started.
      </div>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        overflow: "hidden",
      }}
    >
      {/* Error banner */}
      {error && (
        <div
          role="alert"
          style={{
            padding: "8px 12px",
            backgroundColor: "#fef2f2",
            color: "#991b1b",
            fontSize: 13,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span>{error}</span>
          <button
            onClick={clearError}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              fontSize: 16,
              color: "#991b1b",
            }}
            aria-label="Dismiss error"
          >
            \u00D7
          </button>
        </div>
      )}

      {/* Message list */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        style={{ flex: 1, overflowY: "auto", padding: "16px 0" }}
      >
        {messages.length === 0 ? (
          <div
            style={{
              padding: 32,
              color: "#6b7280",
              fontSize: 15,
              textAlign: "center",
            }}
          >
            Welcome to ELSPETH. Describe the pipeline you want to build, and
            I'll compose it for you.
          </div>
        ) : (
          messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))
        )}
        {isComposing && <ComposingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <ChatInput
        onSend={handleSend}
        disabled={isComposing}
        inputRef={inputRef}
      />
    </div>
  );
}
```

- [ ] **Step 5: Implement `useComposer` hook**

Wraps the session store's `sendMessage` action with error dispatching based on HTTP status and `error_type` field. Maps specific backend error responses to user-facing messages:

- 422 + `error_type: "convergence"` -- "ELSPETH couldn't complete the composition after multiple attempts. Try breaking your request into smaller steps."
- 502 + `error_type: "llm_unavailable"` -- "The AI service is temporarily unavailable. Please try again in a moment."
- 502 + `error_type: "llm_auth_error"` -- "The AI service configuration is invalid. Please contact your administrator."
- 90-second timeout (via `AbortController`) -- "The composition request timed out. Try a simpler request."

```tsx
// src/hooks/useComposer.ts
import { useCallback } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import type { ApiError } from "@/types/api";
import * as api from "@/api/client";

const COMPOSE_TIMEOUT_MS = 90_000;

/** Map backend error responses to user-facing messages.
 * R4-M2: Checks error_type first, then HTTP status, then detail text. */
function dispatchComposerError(err: unknown): string {
  const apiErr = err as ApiError;

  // error_type takes precedence (most specific signal)
  if (apiErr.status === 422 && apiErr.error_type === "convergence") {
    return "ELSPETH couldn't complete the composition after multiple attempts. Try breaking your request into smaller steps.";
  }
  if (apiErr.status === 502 && apiErr.error_type === "llm_unavailable") {
    return "The AI service is temporarily unavailable. Please try again in a moment.";
  }
  if (apiErr.status === 502 && apiErr.error_type === "llm_auth_error") {
    return "The AI service configuration is invalid. Please contact your administrator.";
  }
  if (apiErr.detail === "__timeout__") {
    return "The composition request timed out. Try a simpler request.";
  }
  return apiErr.detail ?? "Failed to send message. Please try again.";
}

/**
 * Hook for composing messages. Wraps sessionStore.sendMessage()
 * with error dispatching and a 90-second AbortController timeout.
 */
export function useComposer() {
  const sendMessage = useSessionStore((s) => s.sendMessage);
  const isComposing = useSessionStore((s) => s.isComposing);
  const compositionState = useSessionStore((s) => s.compositionState);

  const sendWithTimeout = useCallback(
    async (content: string) => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), COMPOSE_TIMEOUT_MS);
      try {
        await sendMessage(content);
      } finally {
        clearTimeout(timer);
      }
    },
    [sendMessage]
  );

  return { sendMessage: sendWithTimeout, isComposing, compositionState, dispatchComposerError };
}
```

- [ ] **Step 6: Verify the project builds**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/frontend/src/components/chat/ \
        src/elspeth/web/frontend/src/hooks/useComposer.ts
git commit -m "feat(web/frontend): add chat panel with composing indicator and file upload"
```

---

### Task 11: Inspector Panel Shell and Spec View

**Files:**
- Create: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx`
- Create: `src/elspeth/web/frontend/src/components/inspector/SpecView.tsx`
- Create: `src/elspeth/web/frontend/src/components/execution/ValidationResult.tsx`

- [ ] **Step 1: Implement `ValidationResult` banner**

Shown inline between the inspector header and tab content. Green banner for pass, red for fail with per-component errors. This is the **Stage 2** validation renderer -- it displays `ValidationError[]` from the validate endpoint with per-component attribution (`component_id`, `component_type`, `message`, `suggestion`). Stage 1 errors (`string[]` from `compositionState.validation_errors`) are rendered separately in SpecView (see Step 3).

```tsx
// src/components/execution/ValidationResult.tsx
import type { ValidationResult as ValidationResultType } from "@/types/api";

interface ValidationResultProps {
  result: ValidationResultType;
}

export function ValidationResultBanner({ result }: ValidationResultProps) {
  if (result.valid) {
    return (
      <div
        role="status"
        style={{
          padding: "8px 12px",
          backgroundColor: "#dcfce7",
          color: "#166534",
          fontSize: 13,
          display: "flex",
          alignItems: "center",
          gap: 6,
        }}
      >
        <span aria-hidden="true">{"\u2713"}</span>
        {result.summary}
      </div>
    );
  }

  return (
    <div role="alert" style={{ backgroundColor: "#fef2f2", fontSize: 13 }}>
      <div
        style={{
          padding: "8px 12px",
          color: "#991b1b",
          fontWeight: 600,
        }}
      >
        Validation failed
      </div>
      <ul
        style={{
          margin: 0,
          padding: "0 12px 8px 28px",
          color: "#991b1b",
        }}
      >
        {result.errors.map((err, i) => (
          <li key={i} style={{ marginBottom: 4 }}>
            <strong>[{err.component_type}] {err.component_id}:</strong> {err.message}
            {err.suggestion && (
              <div style={{ color: "#6b7280", fontSize: 12, marginTop: 2 }}>
                Suggestion: {err.suggestion}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
```

- [ ] **Step 2: Implement `InspectorPanel` with tab strip, Validate/Execute buttons, and version history**

The panel has a fixed header with the tab strip, version dropdown, and action buttons. Below the header is the optional validation banner. Below that is the active tab's content area wrapped in an ARIA live region.

```tsx
// src/components/inspector/InspectorPanel.tsx
import { useState } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import { SpecView } from "./SpecView";
import { GraphView } from "./GraphView";
import { YamlView } from "./YamlView";
import { RunsView } from "./RunsView";
import { ValidationResultBanner } from "@/components/execution/ValidationResult";

type TabId = "spec" | "graph" | "yaml" | "runs";

const TABS: { id: TabId; label: string }[] = [
  { id: "spec", label: "Spec" },
  { id: "graph", label: "Graph" },
  { id: "yaml", label: "YAML" },
  { id: "runs", label: "Runs" },
];

export function InspectorPanel() {
  const [activeTab, setActiveTab] = useState<TabId>("spec");

  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const compositionState = useSessionStore((s) => s.compositionState);
  const stateVersions = useSessionStore((s) => s.stateVersions);
  const revertToVersion = useSessionStore((s) => s.revertToVersion);
  const loadStateVersions = useSessionStore((s) => s.loadStateVersions);

  const validationResult = useExecutionStore((s) => s.validationResult);
  const isValidating = useExecutionStore((s) => s.isValidating);
  const isExecuting = useExecutionStore((s) => s.isExecuting);
  const validate = useExecutionStore((s) => s.validate);
  const execute = useExecutionStore((s) => s.execute);
  const progress = useExecutionStore((s) => s.progress);

  const canExecute =
    validationResult?.valid === true &&
    !isExecuting &&
    progress?.status !== "running";

  function handleValidate() {
    if (activeSessionId) {
      validate(activeSessionId);
    }
  }

  function handleExecute() {
    if (activeSessionId && canExecute) {
      execute(activeSessionId);
    }
  }

  function handleVersionSelect(version: number) {
    revertToVersion(version);
  }

  function handleVersionDropdownOpen() {
    loadStateVersions();
  }

  // Tab navigation with arrow keys
  function handleTabKeyDown(e: React.KeyboardEvent, tabIndex: number) {
    let nextIndex: number | null = null;
    if (e.key === "ArrowRight") {
      nextIndex = (tabIndex + 1) % TABS.length;
    } else if (e.key === "ArrowLeft") {
      nextIndex = (tabIndex - 1 + TABS.length) % TABS.length;
    }
    if (nextIndex !== null) {
      e.preventDefault();
      setActiveTab(TABS[nextIndex].id);
      // Focus the new tab button
      const tabButton = document.getElementById(`inspector-tab-${TABS[nextIndex].id}`);
      tabButton?.focus();
    }
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        style={{
          borderBottom: "1px solid #e0e0e0",
          padding: "8px 12px",
        }}
      >
        {/* Tab strip + actions row */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          {/* Tabs */}
          <div role="tablist" aria-label="Inspector tabs" style={{ display: "flex", gap: 4 }}>
            {TABS.map((tab, i) => (
              <button
                key={tab.id}
                id={`inspector-tab-${tab.id}`}
                role="tab"
                aria-selected={activeTab === tab.id}
                aria-controls={`inspector-tabpanel-${tab.id}`}
                tabIndex={activeTab === tab.id ? 0 : -1}
                onClick={() => setActiveTab(tab.id)}
                onKeyDown={(e) => handleTabKeyDown(e, i)}
                style={{
                  padding: "6px 12px",
                  border: "none",
                  borderBottom:
                    activeTab === tab.id ? "2px solid #2563eb" : "2px solid transparent",
                  backgroundColor: "transparent",
                  cursor: "pointer",
                  fontSize: 13,
                  fontWeight: activeTab === tab.id ? 600 : 400,
                  color: activeTab === tab.id ? "#2563eb" : "#4b5563",
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Actions: version dropdown + validate + execute */}
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            {/* Version history dropdown */}
            {compositionState && (
              <select
                aria-label="Composition state version"
                value={compositionState.version}
                onChange={(e) => handleVersionSelect(Number(e.target.value))}
                onFocus={handleVersionDropdownOpen}
                style={{
                  padding: "4px 8px",
                  fontSize: 12,
                  border: "1px solid #d1d5db",
                  borderRadius: 4,
                }}
              >
                <option value={compositionState.version}>
                  v{compositionState.version}
                </option>
                {stateVersions
                  .filter((v) => v.version !== compositionState.version)
                  .map((v) => (
                    <option key={v.version} value={v.version}>
                      v{v.version} ({v.node_count} nodes)
                    </option>
                  ))}
              </select>
            )}

            <button
              onClick={handleValidate}
              disabled={!activeSessionId || !compositionState || isValidating || isExecuting}
              style={{
                padding: "4px 10px",
                fontSize: 12,
                backgroundColor:
                  !activeSessionId || !compositionState || isValidating
                    ? "#e5e7eb"
                    : "#f3f4f6",
                border: "1px solid #d1d5db",
                borderRadius: 4,
                cursor:
                  !activeSessionId || !compositionState || isValidating
                    ? "not-allowed"
                    : "pointer",
              }}
            >
              {isValidating ? (
                <span
                  style={{
                    display: "inline-block",
                    width: 14,
                    height: 14,
                    border: "2px solid #9ca3af",
                    borderTopColor: "#4b5563",
                    borderRadius: "50%",
                    animation: "spin 0.6s linear infinite",
                  }}
                  aria-label="Validating"
                />
              ) : (
                "Validate"
              )}
            </button>

            <button
              onClick={handleExecute}
              disabled={!canExecute}
              style={{
                padding: "4px 10px",
                fontSize: 12,
                backgroundColor: canExecute ? "#2563eb" : "#e5e7eb",
                color: canExecute ? "#fff" : "#9ca3af",
                border: "none",
                borderRadius: 4,
                cursor: canExecute ? "pointer" : "not-allowed",
              }}
            >
              {isExecuting ? "Starting..." : "Execute"}
            </button>
          </div>
        </div>
      </div>

      {/* Validation result banner */}
      {validationResult && (
        <ValidationResultBanner result={validationResult} />
      )}

      {/* Tab content area */}
      <div
        role="tabpanel"
        id={`inspector-tabpanel-${activeTab}`}
        aria-labelledby={`inspector-tab-${activeTab}`}
        aria-live="polite"
        style={{ flex: 1, overflow: "auto" }}
      >
        {activeTab === "spec" && <SpecView />}
        {activeTab === "graph" && <GraphView />}
        {activeTab === "yaml" && <YamlView />}
        {activeTab === "runs" && <RunsView />}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Implement `SpecView` with component cards and click-to-highlight**

The spec view renders the `CompositionState` as a vertical list of component cards. Each card shows a type badge, plugin name, config summary, and connection indicators. Clicking a card highlights it and its upstream/downstream connections. Unrelated cards are dimmed (background only, not text). Cards are keyboard-focusable.

```tsx
// src/components/inspector/SpecView.tsx
import { useState, useCallback } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import type { NodeSpec, EdgeSpec, CompositionState } from "@/types/api";

/** Badge colours and labels by node type */
const TYPE_BADGES: Record<
  NodeSpec["type"],
  { bg: string; text: string; label: string }
> = {
  source: { bg: "#dbeafe", text: "#1e40af", label: "SOURCE" },
  transform: { bg: "#dcfce7", text: "#166534", label: "TRANSFORM" },
  gate: { bg: "#fef3c7", text: "#92400e", label: "GATE" },
  sink: { bg: "#f3e8ff", text: "#6b21a8", label: "SINK" },
};

/** Compute upstream and downstream node IDs from edge list */
function computeRelationships(
  nodeId: string,
  edges: EdgeSpec[]
): { upstream: Set<string>; downstream: Map<string, string | null> } {
  const upstream = new Set<string>();
  const downstream = new Map<string, string | null>();

  for (const edge of edges) {
    if (edge.target === nodeId) {
      upstream.add(edge.source);
    }
    if (edge.source === nodeId) {
      downstream.set(edge.target, edge.label);
    }
  }

  return { upstream, downstream };
}

/** Get the display label for a relationship badge */
function getRelBadge(
  nodeId: string,
  selectedId: string | null,
  upstream: Set<string>,
  downstream: Map<string, string | null>
): { label: string; color: string } | null {
  if (nodeId === selectedId) {
    return { label: "SELECTED", color: "#2563eb" };
  }
  if (upstream.has(nodeId)) {
    return { label: "INPUT", color: "#059669" };
  }
  if (downstream.has(nodeId)) {
    const routeLabel = downstream.get(nodeId);
    if (routeLabel) {
      return { label: routeLabel.toUpperCase(), color: "#d97706" };
    }
    return { label: "OUTPUT", color: "#7c3aed" };
  }
  return null;
}

export function SpecView() {
  const compositionState = useSessionStore((s) => s.compositionState);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Compute relationships for the selected node
  const { upstream, downstream } = selectedNodeId && compositionState
    ? computeRelationships(selectedNodeId, compositionState.edges)
    : { upstream: new Set<string>(), downstream: new Map<string, string | null>() };

  const isRelated = useCallback(
    (nodeId: string) =>
      nodeId === selectedNodeId ||
      upstream.has(nodeId) ||
      downstream.has(nodeId),
    [selectedNodeId, upstream, downstream]
  );

  function handleCardClick(nodeId: string) {
    setSelectedNodeId((prev) => (prev === nodeId ? null : nodeId));
  }

  function handleBackgroundClick(e: React.MouseEvent) {
    // Only deselect if clicking the container itself, not a card
    if (e.target === e.currentTarget) {
      setSelectedNodeId(null);
    }
  }

  function handleCardKeyDown(e: React.KeyboardEvent, nodeId: string) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      handleCardClick(nodeId);
    }
  }

  if (!compositionState || compositionState.nodes.length === 0) {
    return (
      <div
        style={{
          padding: 24,
          color: "#6b7280",
          fontSize: 14,
          textAlign: "center",
        }}
      >
        Send a message to start building your pipeline. Components will appear
        here as ELSPETH composes them.
      </div>
    );
  }

  // Build a map of node downstream connections for indicators
  const nodeDownstream = new Map<string, EdgeSpec[]>();
  for (const edge of compositionState.edges) {
    const existing = nodeDownstream.get(edge.source) ?? [];
    existing.push(edge);
    nodeDownstream.set(edge.source, existing);
  }

  return (
    <div
      onClick={handleBackgroundClick}
      style={{ padding: 12, display: "flex", flexDirection: "column", gap: 8 }}
    >
      {/* Stage 1 validation errors: simple string[] from composer */}
      {compositionState.validation_errors &&
        compositionState.validation_errors.length > 0 && (
          <div
            role="alert"
            style={{
              padding: "8px 12px",
              backgroundColor: "#fef3c7",
              borderRadius: 6,
              fontSize: 13,
              color: "#92400e",
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 4 }}>
              Composition warnings
            </div>
            <ul style={{ margin: 0, paddingLeft: 16 }}>
              {compositionState.validation_errors.map((msg, i) => (
                <li key={i} style={{ marginBottom: 2 }}>{msg}</li>
              ))}
            </ul>
          </div>
        )}

      {compositionState.nodes.map((node) => {
        const badge = TYPE_BADGES[node.type];
        const relBadge = getRelBadge(
          node.id,
          selectedNodeId,
          upstream,
          downstream
        );
        const isDimmed = selectedNodeId !== null && !isRelated(node.id);
        const isSelected = node.id === selectedNodeId;
        const edges = nodeDownstream.get(node.id) ?? [];

        return (
          <div
            key={node.id}
            tabIndex={0}
            role="button"
            aria-pressed={isSelected}
            onClick={(e) => {
              e.stopPropagation();
              handleCardClick(node.id);
            }}
            onKeyDown={(e) => handleCardKeyDown(e, node.id)}
            style={{
              padding: "10px 12px",
              borderRadius: 6,
              border: isSelected
                ? "2px solid #2563eb"
                : "1px solid #e0e0e0",
              backgroundColor: isDimmed
                ? "rgba(249, 250, 251, 0.35)"
                : "#fff",
              cursor: "pointer",
              outline: "none",
              // Text always at full opacity
              color: "#1f2937",
            }}
          >
            {/* Top row: type badge + relationship badge */}
            <div
              style={{
                display: "flex",
                gap: 6,
                alignItems: "center",
                marginBottom: 4,
              }}
            >
              <span
                style={{
                  display: "inline-block",
                  padding: "2px 6px",
                  borderRadius: 3,
                  fontSize: 10,
                  fontWeight: 700,
                  backgroundColor: badge.bg,
                  color: badge.text,
                  letterSpacing: "0.05em",
                }}
              >
                {badge.label}
              </span>
              {relBadge && (
                <span
                  style={{
                    display: "inline-block",
                    padding: "2px 6px",
                    borderRadius: 3,
                    fontSize: 10,
                    fontWeight: 700,
                    backgroundColor: `${relBadge.color}20`,
                    color: relBadge.color,
                    letterSpacing: "0.05em",
                  }}
                >
                  {relBadge.label}
                </span>
              )}
            </div>

            {/* Plugin name + config summary */}
            <div style={{ fontWeight: 600, fontSize: 13 }}>{node.name}</div>
            <div style={{ fontSize: 12, color: "#6b7280", marginTop: 2 }}>
              {node.plugin}
              {node.config_summary && ` \u2014 ${node.config_summary}`}
            </div>

            {/* Connection indicators */}
            {edges.length > 0 && (
              <div style={{ marginTop: 6, fontSize: 11, color: "#9ca3af" }}>
                {edges.map((edge, i) => {
                  const targetNode = compositionState.nodes.find(
                    (n) => n.id === edge.target
                  );
                  const targetName = targetNode?.name ?? edge.target;
                  if (edge.edge_type === "error") {
                    return (
                      <div key={i}>
                        {"\u26A0"} on_error {"\u2192"} {targetName}
                      </div>
                    );
                  }
                  if (edge.label) {
                    return (
                      <div key={i}>
                        {edge.label} {"\u2192"} {targetName}
                      </div>
                    );
                  }
                  return (
                    <div key={i}>
                      {"\u2193"} {targetName}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 4: Verify the project builds**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx \
        src/elspeth/web/frontend/src/components/inspector/SpecView.tsx \
        src/elspeth/web/frontend/src/components/execution/ValidationResult.tsx
git commit -m "feat(web/frontend): add inspector panel with spec view and component linking"
```

---

### Task 12: Graph View

**Files:**
- Create: `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx`

- [ ] **Step 1: Install the dagre layout dependency**

```bash
cd src/elspeth/web/frontend
npm install @dagrejs/dagre
npm install -D @types/d3-hierarchy  # dagre type support
```

- [ ] **Step 2: Implement `GraphView` with React Flow and dagre auto-layout**

Converts `CompositionState` nodes and edges into React Flow format. Uses dagre for automatic top-to-bottom hierarchical layout. The graph is read-only (no drag-to-connect, no node deletion). Pan and zoom are enabled.

```tsx
// src/components/inspector/GraphView.tsx
import { useMemo } from "react";
import {
  ReactFlow,
  type Node,
  type Edge,
  Background,
  Controls,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import "@xyflow/react/dist/style.css";
import { useSessionStore } from "@/stores/sessionStore";
import type { NodeSpec } from "@/types/api";

/** Node colour by type, matching SpecView badge colours */
const NODE_COLORS: Record<NodeSpec["type"], string> = {
  source: "#dbeafe",
  transform: "#dcfce7",
  gate: "#fef3c7",
  sink: "#f3e8ff",
};

const NODE_BORDER_COLORS: Record<NodeSpec["type"], string> = {
  source: "#1e40af",
  transform: "#166534",
  gate: "#92400e",
  sink: "#6b21a8",
};

const NODE_WIDTH = 180;
const NODE_HEIGHT = 50;

/**
 * Apply dagre layout to nodes and edges, returning positioned React Flow
 * nodes and edges.
 */
function layoutGraph(
  rfNodes: Node[],
  rfEdges: Edge[]
): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", nodesep: 40, ranksep: 60 });

  for (const node of rfNodes) {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const edge of rfEdges) {
    g.setEdge(edge.source, edge.target);
  }

  dagre.layout(g);

  const positionedNodes = rfNodes.map((node) => {
    const pos = g.node(node.id);
    return {
      ...node,
      position: {
        x: pos.x - NODE_WIDTH / 2,
        y: pos.y - NODE_HEIGHT / 2,
      },
    };
  });

  return { nodes: positionedNodes, edges: rfEdges };
}

export function GraphView() {
  const compositionState = useSessionStore((s) => s.compositionState);

  const { nodes, edges } = useMemo(() => {
    if (!compositionState || compositionState.nodes.length === 0) {
      return { nodes: [], edges: [] };
    }

    const rfNodes: Node[] = compositionState.nodes.map((node) => ({
      id: node.id,
      data: { label: node.name },
      position: { x: 0, y: 0 }, // Will be set by dagre
      style: {
        backgroundColor: NODE_COLORS[node.type],
        border: `2px solid ${NODE_BORDER_COLORS[node.type]}`,
        borderRadius: 6,
        padding: "8px 12px",
        fontSize: 12,
        fontWeight: 600,
        width: NODE_WIDTH,
        textAlign: "center" as const,
      },
    }));

    const rfEdges: Edge[] = compositionState.edges.map((edge, i) => ({
      id: `e-${i}`,
      source: edge.source,
      target: edge.target,
      label: edge.label ?? undefined,
      animated: edge.edge_type === "error",
      style: {
        stroke: edge.edge_type === "error" ? "#ef4444" : "#6b7280",
        strokeWidth: 1.5,
      },
      labelStyle: { fontSize: 10, fill: "#4b5563" },
    }));

    return layoutGraph(rfNodes, rfEdges);
  }, [compositionState]);

  if (!compositionState || compositionState.nodes.length === 0) {
    return (
      <div
        style={{
          padding: 24,
          color: "#6b7280",
          fontSize: 14,
          textAlign: "center",
        }}
      >
        No pipeline to visualise. Start a conversation to build one.
      </div>
    );
  }

  const ariaLabel = `Pipeline graph: ${compositionState.nodes.length} nodes, ${compositionState.edges.length} connections`;

  return (
    <div style={{ width: "100%", height: "100%" }} aria-label={ariaLabel}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        fitView
        proOptions={{ hideAttribution: true }}
      >
        <Background />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
```

- [ ] **Step 3: Verify the project builds**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/GraphView.tsx
git commit -m "feat(web/frontend): add React Flow graph view with dagre auto-layout"
```

---

### Task 13: YAML View

**Files:**
- Create: `src/elspeth/web/frontend/src/components/inspector/YamlView.tsx`

- [ ] **Step 1: Implement `YamlView` with syntax highlighting and copy button**

Read-only display of the generated pipeline YAML. The YAML is fetched from `GET /api/sessions/{id}/state/yaml` whenever the composition state version changes (not generated client-side). Uses `prism-react-renderer` for syntax highlighting. A copy button copies the full YAML to the clipboard with a brief "Copied!" confirmation.

```tsx
// src/components/inspector/YamlView.tsx
import { useState, useEffect, useCallback } from "react";
import { Highlight, themes } from "prism-react-renderer";
import { useSessionStore } from "@/stores/sessionStore";
import * as api from "@/api/client";

export function YamlView() {
  const compositionState = useSessionStore((s) => s.compositionState);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const [yaml, setYaml] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Fetch YAML from the backend whenever composition state version changes
  const version = compositionState?.version ?? null;
  useEffect(() => {
    if (!activeSessionId || version === null) {
      setYaml(null);
      return;
    }
    let cancelled = false;
    api.fetchYaml(activeSessionId).then((text) => {
      if (!cancelled) setYaml(text);
    }).catch(() => {
      if (!cancelled) setYaml(null);
    });
    return () => { cancelled = true; };
  }, [activeSessionId, version]);

  const handleCopy = useCallback(async () => {
    if (!yaml) return;
    try {
      await navigator.clipboard.writeText(yaml);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API may fail in some contexts — fallback not needed for MVP
    }
  }, [yaml]);

  if (!yaml) {
    return (
      <div
        style={{
          padding: 24,
          color: "#6b7280",
          fontSize: 14,
          textAlign: "center",
        }}
      >
        YAML will appear here once your pipeline has components.
      </div>
    );
  }

  return (
    <div style={{ position: "relative", height: "100%" }}>
      {/* Copy button */}
      <button
        onClick={handleCopy}
        style={{
          position: "absolute",
          top: 8,
          right: 8,
          padding: "4px 10px",
          fontSize: 12,
          backgroundColor: copied ? "#dcfce7" : "#f3f4f6",
          color: copied ? "#166534" : "#4b5563",
          border: "1px solid #d1d5db",
          borderRadius: 4,
          cursor: "pointer",
          zIndex: 1,
        }}
      >
        {copied ? "Copied!" : "Copy"}
      </button>

      {/* Syntax-highlighted YAML */}
      <Highlight theme={themes.github} code={yaml} language="yaml">
        {({ style, tokens, getLineProps, getTokenProps }) => (
          <pre
            style={{
              ...style,
              margin: 0,
              padding: "12px 16px",
              paddingTop: 36, // Space for copy button
              overflow: "auto",
              height: "100%",
              fontSize: 12,
              lineHeight: 1.5,
              boxSizing: "border-box",
            }}
          >
            {tokens.map((line, i) => {
              const lineProps = getLineProps({ line });
              return (
                <div key={i} {...lineProps}>
                  {line.map((token, j) => {
                    const tokenProps = getTokenProps({ token });
                    return <span key={j} {...tokenProps} />;
                  })}
                </div>
              );
            })}
          </pre>
        )}
      </Highlight>
    </div>
  );
}
```

- [ ] **Step 2: Verify the project builds**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/YamlView.tsx
git commit -m "feat(web/frontend): add YAML view with syntax highlighting and copy button"
```

---

### Task 14: Runs View and Progress View

**Files:**
- Create: `src/elspeth/web/frontend/src/components/inspector/RunsView.tsx`
- Create: `src/elspeth/web/frontend/src/components/execution/ProgressView.tsx`
- Create: `src/elspeth/web/frontend/src/hooks/useWebSocket.ts`

- [ ] **Step 1: Implement `useWebSocket` hook**

Provides lifecycle management for WebSocket connections tied to a React component's lifecycle.

```tsx
// src/hooks/useWebSocket.ts
import { useEffect, useRef } from "react";
import { useExecutionStore } from "@/stores/executionStore";

/**
 * Hook that ensures the WebSocket connection for the active run
 * is properly cleaned up when the component unmounts.
 *
 * The actual WebSocket management is in executionStore.execute().
 * This hook is a convenience for components that need to react to
 * WebSocket state (disconnected banner, etc.).
 */
export function useWebSocket() {
  const wsDisconnected = useExecutionStore((s) => s.wsDisconnected);
  const progress = useExecutionStore((s) => s.progress);
  const activeRunId = useExecutionStore((s) => s.activeRunId);

  return { wsDisconnected, progress, activeRunId };
}
```

- [ ] **Step 2: Implement `ProgressView`**

Live progress display for an active execution run. Shows an indeterminate progress bar (animated stripe, no percentage, no `aria-valuemax`) with ARIA progressbar semantics, row counters, recent exceptions, and a cancel button.

```tsx
// src/components/execution/ProgressView.tsx
import { useExecutionStore } from "@/stores/executionStore";
import { useWebSocket } from "@/hooks/useWebSocket";

export function ProgressView() {
  const { progress, wsDisconnected, activeRunId } = useWebSocket();
  const cancel = useExecutionStore((s) => s.cancel);

  if (!progress || !activeRunId) return null;

  const isTerminal =
    progress.status === "completed" ||
    progress.status === "cancelled";

  return (
    <div style={{ padding: 12, fontSize: 13 }}>
      {/* WebSocket disconnect banner */}
      {wsDisconnected && !isTerminal && (
        <div
          role="status"
          style={{
            padding: "6px 10px",
            marginBottom: 8,
            backgroundColor: "#fefce8",
            color: "#854d0e",
            borderRadius: 4,
            fontSize: 12,
          }}
        >
          Live progress connection lost. Reconnecting...
        </div>
      )}

      {/* Status */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 8,
        }}
      >
        <span style={{ fontWeight: 600, textTransform: "uppercase", fontSize: 12 }}>
          {progress.status}
        </span>
        {!isTerminal && (
          <button
            onClick={() => cancel(activeRunId)}
            style={{
              padding: "4px 10px",
              fontSize: 12,
              backgroundColor: "#fef2f2",
              color: "#991b1b",
              border: "1px solid #fca5a5",
              borderRadius: 4,
              cursor: "pointer",
            }}
          >
            Cancel
          </button>
        )}
      </div>

      {/* Progress bar — indeterminate mode (no percentage, animated stripe) */}
      <style>{`
        @keyframes progress-stripe {
          0% { background-position: 0 0; }
          100% { background-position: 40px 0; }
        }
      `}</style>
      <div
        role="progressbar"
        aria-label="Pipeline execution in progress"
        // No aria-valuenow or aria-valuemax — indeterminate mode
        style={{
          width: "100%",
          height: 8,
          backgroundColor: "#e5e7eb",
          borderRadius: 4,
          overflow: "hidden",
          marginBottom: 8,
        }}
      >
        <div
          style={{
            width: "100%",
            height: "100%",
            backgroundColor:
              progress.status === "completed"
                ? "#22c55e"
                : progress.status === "cancelled"
                  ? "#d97706"
                  : "#2563eb",
            ...(isTerminal
              ? {}
              : {
                  backgroundImage:
                    "linear-gradient(45deg, rgba(255,255,255,0.15) 25%, transparent 25%, transparent 50%, rgba(255,255,255,0.15) 50%, rgba(255,255,255,0.15) 75%, transparent 75%, transparent)",
                  backgroundSize: "40px 40px",
                  animation: "progress-stripe 1s linear infinite",
                }),
          }}
        />
      </div>

      {/* Row counters */}
      <div
        style={{
          display: "flex",
          gap: 16,
          fontSize: 12,
          color: "#4b5563",
          marginBottom: 12,
        }}
      >
        <span>Processed: {progress.rows_processed}</span>
        <span>
          Failed:{" "}
          <span style={{ color: progress.rows_failed > 0 ? "#ef4444" : "inherit" }}>
            {progress.rows_failed}
          </span>
        </span>
      </div>

      {/* Cancellation message */}
      {progress.status === "cancelled" && (
        <div
          role="status"
          style={{
            padding: "8px 12px",
            marginBottom: 8,
            backgroundColor: "#fefce8",
            color: "#854d0e",
            borderRadius: 4,
            fontSize: 13,
          }}
        >
          Pipeline execution was cancelled.
        </div>
      )}

      {/* Recent exceptions */}
      {progress.recent_exceptions.length > 0 && (
        <div>
          <div
            style={{
              fontSize: 12,
              fontWeight: 600,
              color: "#991b1b",
              marginBottom: 4,
            }}
          >
            Recent exceptions ({progress.recent_exceptions.length})
          </div>
          <div
            style={{
              maxHeight: 200,
              overflowY: "auto",
              fontSize: 11,
              fontFamily: "monospace",
              backgroundColor: "#fef2f2",
              borderRadius: 4,
              padding: 8,
            }}
          >
            {progress.recent_exceptions.map((exc, i) => (
              <div
                key={i}
                style={{
                  marginBottom: 4,
                  paddingBottom: 4,
                  borderBottom:
                    i < progress.recent_exceptions.length - 1
                      ? "1px solid #fecaca"
                      : "none",
                }}
              >
                <span style={{ color: "#9ca3af" }}>{exc.timestamp}</span>{" "}
                <strong>{exc.node_name}</strong>: {exc.message}
                {exc.row_id && (
                  <span style={{ color: "#6b7280" }}> (row: {exc.row_id})</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Implement `RunsView`**

Lists execution runs for the current session. Each run shows status, row counts, duration, and composition version. Clicking an active run shows the `ProgressView` inline.

```tsx
// src/components/inspector/RunsView.tsx
import { useExecutionStore } from "@/stores/executionStore";
import { ProgressView } from "@/components/execution/ProgressView";
import type { Run } from "@/types/api";

/** Status badge colours */
const STATUS_STYLES: Record<
  Run["status"],
  { bg: string; text: string }
> = {
  pending: { bg: "#f3f4f6", text: "#4b5563" },
  running: { bg: "#dbeafe", text: "#1e40af" },
  completed: { bg: "#dcfce7", text: "#166534" },
  failed: { bg: "#fef2f2", text: "#991b1b" },
  cancelled: { bg: "#fefce8", text: "#854d0e" },
};

function formatDuration(startedAt: string, finishedAt: string | null): string {
  const start = new Date(startedAt).getTime();
  const end = finishedAt ? new Date(finishedAt).getTime() : Date.now();
  const diffSec = Math.floor((end - start) / 1000);

  if (diffSec < 60) return `${diffSec}s`;
  const min = Math.floor(diffSec / 60);
  const sec = diffSec % 60;
  return `${min}m ${sec}s`;
}

export function RunsView() {
  const runs = useExecutionStore((s) => s.runs);
  const activeRunId = useExecutionStore((s) => s.activeRunId);
  const progress = useExecutionStore((s) => s.progress);

  if (runs.length === 0) {
    return (
      <div
        style={{
          padding: 24,
          color: "#6b7280",
          fontSize: 14,
          textAlign: "center",
        }}
      >
        No runs yet. Validate your pipeline, then click Execute to run it.
      </div>
    );
  }

  return (
    <div style={{ padding: 8 }}>
      {runs.map((run) => {
        const style = STATUS_STYLES[run.status];
        const isActive =
          run.id === activeRunId && progress?.status === "running";

        return (
          <div key={run.id}>
            <div
              style={{
                padding: "10px 12px",
                marginBottom: 4,
                borderRadius: 6,
                border: "1px solid #e0e0e0",
                backgroundColor: "#fff",
                fontSize: 13,
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 4,
                }}
              >
                {/* Status badge */}
                <span
                  style={{
                    display: "inline-block",
                    padding: "2px 8px",
                    borderRadius: 3,
                    fontSize: 11,
                    fontWeight: 600,
                    backgroundColor: style.bg,
                    color: style.text,
                    textTransform: "uppercase",
                  }}
                >
                  {run.status}
                </span>
                <span style={{ fontSize: 11, color: "#9ca3af" }}>
                  v{run.composition_version}
                </span>
              </div>

              <div
                style={{
                  display: "flex",
                  gap: 12,
                  fontSize: 12,
                  color: "#6b7280",
                }}
              >
                <span>
                  {run.rows_processed} rows
                  {run.rows_failed > 0 && (
                    <span style={{ color: "#ef4444" }}>
                      {" "}
                      ({run.rows_failed} failed)
                    </span>
                  )}
                </span>
                <span>
                  {run.status === "running"
                    ? "running..."
                    : formatDuration(run.started_at, run.finished_at)}
                </span>
              </div>
            </div>

            {/* Show live progress for the active running run */}
            {isActive && <ProgressView />}
          </div>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 4: Verify the project builds**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/hooks/useWebSocket.ts \
        src/elspeth/web/frontend/src/components/execution/ProgressView.tsx \
        src/elspeth/web/frontend/src/components/inspector/RunsView.tsx
git commit -m "feat(web/frontend): add runs view and live execution progress"
```

---

### Task 15: Layout Component, App Wiring, and Static File Serving

**Files:**
- Create: `src/elspeth/web/frontend/src/components/common/Layout.tsx`
- Modify: `src/elspeth/web/frontend/src/App.tsx`
- Modify: `src/elspeth/web/frontend/src/App.css`
- Modify: `src/elspeth/web/app.py`

- [ ] **Step 1: Implement the `Layout` component**

The layout component renders the three-panel CSS grid. The sidebar width adjusts based on collapsed state. The inspector panel supports resizing via a drag handle.

```tsx
// src/components/common/Layout.tsx
import { useState, useRef, useCallback, type ReactNode } from "react";

interface LayoutProps {
  sidebar: ReactNode;
  chat: ReactNode;
  inspector: ReactNode;
}

export function Layout({ sidebar, chat, inspector }: LayoutProps) {
  const [inspectorWidth, setInspectorWidth] = useState(320);
  const isResizing = useRef(false);

  const handleMouseDown = useCallback(() => {
    isResizing.current = true;

    function handleMouseMove(e: MouseEvent) {
      if (!isResizing.current) return;
      const newWidth = window.innerWidth - e.clientX;
      // Clamp between 240px and 50% of viewport
      const maxWidth = window.innerWidth * 0.5;
      setInspectorWidth(Math.max(240, Math.min(newWidth, maxWidth)));
    }

    function handleMouseUp() {
      isResizing.current = false;
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  return (
    <div
      className="app-layout"
      style={{
        display: "grid",
        gridTemplateColumns: `auto 1fr ${inspectorWidth}px`,
        gridTemplateAreas: '"sidebar chat inspector"',
        height: "100vh",
        minWidth: 1280,
      }}
    >
      <div style={{ gridArea: "sidebar" }}>{sidebar}</div>
      <div style={{ gridArea: "chat", overflow: "hidden" }}>{chat}</div>
      <div
        style={{
          gridArea: "inspector",
          position: "relative",
          borderLeft: "1px solid #e0e0e0",
          overflow: "hidden",
        }}
      >
        {/* Resize handle */}
        <div
          onMouseDown={handleMouseDown}
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            bottom: 0,
            width: 4,
            cursor: "col-resize",
            backgroundColor: "transparent",
            zIndex: 10,
          }}
          title="Drag to resize inspector panel"
        />
        {inspector}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Wire `App.tsx` to compose all components**

Replace the skeleton `App.tsx` with the full component composition: `AuthGuard` wrapping `Layout` with `SessionSidebar`, `ChatPanel`, and `InspectorPanel`.

```tsx
// src/App.tsx
import "./App.css";
import { AuthGuard } from "@/components/common/AuthGuard";
import { Layout } from "@/components/common/Layout";
import { SessionSidebar } from "@/components/sessions/SessionSidebar";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { InspectorPanel } from "@/components/inspector/InspectorPanel";

function App() {
  return (
    <AuthGuard>
      <Layout
        sidebar={<SessionSidebar />}
        chat={<ChatPanel />}
        inspector={<InspectorPanel />}
      />
    </AuthGuard>
  );
}

export default App;
```

- [ ] **Step 3: Update `App.css` with global styles**

Replace the skeleton CSS with minimal global resets that the layout needs.

```css
/* src/App.css */

/* Reset box sizing globally */
*, *::before, *::after {
  box-sizing: border-box;
}

/* Remove default margins */
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
    Oxygen, Ubuntu, Cantarell, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Focus outline for keyboard navigation — visible and high-contrast */
:focus-visible {
  outline: 2px solid #2563eb;
  outline-offset: 2px;
}

/* Spinner animation for loading states (Validate button, etc.) */
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
```

- [ ] **Step 4: Add static file serving to FastAPI `app.py`**

Modify `src/elspeth/web/app.py` to mount the frontend `dist/` directory as static files for production serving. The mount is registered after all API and WebSocket routes so they take precedence. The mount is conditional on the `dist/` directory existing.

```python
# Add to the end of src/elspeth/web/app.py, after all API route registrations:

from pathlib import Path
from fastapi.staticfiles import StaticFiles

# Serve frontend SPA from dist/ in production.
# The mount is registered last so /api/* and /ws/* routes take precedence.
# html=True enables SPA fallback: any non-file request returns index.html.
_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True))
```

- [ ] **Step 5: Build the frontend and verify end-to-end**

```bash
cd src/elspeth/web/frontend && npm run build
```

Verify that `dist/` contains `index.html` and the built JS/CSS assets.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/common/Layout.tsx \
        src/elspeth/web/frontend/src/App.tsx \
        src/elspeth/web/frontend/src/App.css \
        src/elspeth/web/app.py
git commit -m "feat(web): wire three-panel layout with static file serving"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All sub-spec 6 sections mapped to tasks. Project setup (Task 1), types (Task 2), API client (Task 3), WebSocket (Task 4), auth store (Task 5), session store (Task 6), execution store with auto-clear (Task 7), auth components (Task 8), session sidebar (Task 9), chat components with file upload (Task 10), inspector + spec view with click-to-highlight (Task 11), graph view (Task 12), YAML view (Task 13), runs view + progress view (Task 14), layout + wiring + static serving (Task 15).
- [x] **Empty states:** All six empty states from the spec are present (sidebar, chat, spec, graph, YAML, runs).
- [x] **Error messages:** All error scenarios from the spec have user-facing messages (composer convergence via error_type, LLM unavailable/auth_error via error_type, composing timeout via AbortController, upload failures, execution failure, execution conflict 409, validation internal error 500, WebSocket disconnect, WebSocket 4001 auth failure, session load, auth failures).
- [x] **Accessibility:** Component cards have `tabindex="0"` with Enter/Space handlers. Tab strip uses arrow key navigation. Progress bar uses `role="progressbar"` in indeterminate mode (no `aria-valuemax`). Composing indicator has `aria-live`. Type badges use text+colour. Dimming is background-only. Focus-visible outline is defined globally.
- [x] **RunEvent semantics:** `"error"` is non-terminal (per-row exception, pipeline continues). `"completed"` and `"cancelled"` are terminal. WebSocket only closes on terminal events. No `"failed"` event type.
- [x] **Auth config:** LoginPage fetches `GET /api/auth/config` to determine provider; OIDC redirect URL constructed from `oidc_issuer` + `oidc_client_id`.
- [x] **WebSocket auth:** JWT appended as `?token=` query parameter. Close codes discriminated: 1000 (no reconnect, poll REST), 1006 (auto-reconnect with backoff), 1011 (no reconnect, poll REST), 4001 (no reconnect, logout).
- [x] **YAML tab:** Fetches from `GET /api/sessions/{id}/state/yaml` on composition state version change; no client-side generation.
- [x] **Global 401 interceptor:** API client calls `authStore.logout()` on any 401 response.
- [x] **System messages:** MessageBubble renders `role="system"` as centre-aligned banner with muted colour, italic text, no sender label.
- [x] **Auto-clear:** Execution store subscribes to session store composition version changes and clears validation result. Additionally, `sendMessage`, `revertToVersion`, and `selectSession` call `clearValidation()` explicitly for immediate effect.
- [x] **Version history:** Inspector header includes version dropdown that fetches and displays prior versions; revert calls `clearValidation()` BEFORE updating compositionState to prevent stale-validation frame.
- [x] **No placeholder code:** All tasks contain complete implementation code.
- [x] **Parent plan tasks mapped:** Task 1 = 6.1, Tasks 2-4 = 6.2, Tasks 5-7 = 6.3, Task 8 = 6.4, Task 9 = 6.5, Task 10 = 6.6, Task 11 = 6.7, Tasks 12-14 = 6.8+6.9, Task 15 = 6.10.

---

## Round 4 Review Amendments

> **Status: Amendments below have been integrated into the plan body.**

Fixes from expert panel review (Round 4). Each amendment references its review finding ID and affected seam contract.

1. **R4-H2 — API response key mapping** (Task 6: Session Store). The `POST /api/sessions/{id}/messages` response returns `{message, state}` — the wire field is `state`, not `compositionState`. The `sendMessage()` action must destructure `response.state` and assign it to the store field `compositionState`: `const { message, state } = result; set({ compositionState: state, ... })`. Verify all other consumers of this endpoint use `state` as the response key.

2. **R4-H3 — Validation gate invariant** (seam contract F; Tasks 6 and 7). `executionStore.clearValidation()` must be called whenever `compositionState.version` changes, regardless of source:
   - `sendMessage()` (Task 6) calls `clearValidation()` when the response includes a new state version
   - `revertToVersion()` (Task 6) calls `clearValidation()` BEFORE updating `compositionState` (not after, to prevent a frame where stale validation is visible with the new version)
   - `selectSession()` (Task 6) calls `clearValidation()` when switching sessions
   - This prevents the Execute button from remaining enabled with a stale validation result after revert.

3. **R4-H6 — WebSocket close code discrimination** (seam contract E; Task 4: WebSocket Manager). The `useWebSocket` hook / `connectWebSocket` action must check `event.code` on WebSocket close:
   - `1000` (normal closure) — run is terminal, do NOT reconnect, poll REST for final status
   - `1006` (abnormal closure) — auto-reconnect with exponential backoff
   - `1011` (internal error) — do NOT reconnect, poll REST for status
   - `4001` (auth failure) — do NOT reconnect, call `authStore.logout()`

4. **R4-H7 — Stage 1 vs Stage 2 error rendering** (seam contract F; Task 11: Inspector Panel Shell and Spec View). Two separate rendering paths, NOT a shared renderer:
   - Stage 1 errors from `compositionState.validation_errors`: `string[]` — rendered as a simple list in the Spec tab summary area
   - Stage 2 errors from `executionStore.validationResult.errors`: `ValidationError[]` (with `component_id`, `component_type`, `message`, `suggestion`) — rendered in the validation banner with per-component attribution and highlighting

5. **R4-M2 — Error envelope handling** (seam contract G; Task 3: API Client). The error handler in `api/client.ts` must check fields in this order: `error_type` first (if present in response body), then fall back to HTTP status code, then fall back to `detail` text. All backend errors use `detail` as the human-readable field (not `message`).

6. **R4-H5 — tool_calls rendering schema** (seam contracts cross-cutting; Task 10: Chat Components). The `ChatMessage.tool_calls` field is a JSON array in LiteLLM format: `[{id, type, function: {name, arguments}}]`. The `MessageBubble` component must extract `function.name` for display and optionally show `function.arguments` (which is a JSON string, not a parsed object) in a collapsible section.

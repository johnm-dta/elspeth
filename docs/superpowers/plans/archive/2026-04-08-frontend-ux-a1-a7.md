# Frontend UX (A1–A7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement seven frontend UX requirements (A1–A7) from the web UX composer feedback document — per-user file folders, markdown/Mermaid chat rendering, visible error routing, 50/50 panel split, secrets button relocation, per-node validation indicators, and three-state pipeline status.

**Architecture:** All changes are frontend-only (React + TypeScript + CSS). We extend existing Zustand stores, add new npm dependencies for markdown rendering, modify the Layout grid for 50/50 default, and enhance GraphView/InspectorPanel with validation state. The backend already returns per-component validation errors — we just need to wire them to the graph nodes.

**Tech Stack:** React 18, TypeScript, Zustand 5, Vite, vitest, @testing-library/react, react-markdown, remark-gfm, mermaid (new deps)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/types/index.ts` | Modify | Add `ValidationSeverity` type, `warnings` to `ValidationResult`, `BlobCategory` |
| `src/styles/tokens.ts` | Modify | Add validation indicator colors, secrets icon color |
| `src/App.css` | Modify | Add markdown styles, validation indicator CSS, panel split vars |
| `src/components/common/Layout.tsx` | Modify | Change default split to 50/50 chat/inspector |
| `src/components/chat/MessageBubble.tsx` | Modify | Replace plain text with markdown renderer |
| `src/components/chat/MarkdownRenderer.tsx` | Create | Markdown + Mermaid rendering component |
| `src/components/chat/MarkdownRenderer.test.tsx` | Create | Tests for markdown rendering |
| `src/components/chat/ChatPanel.tsx` | Modify | Add error/warning message injection from execution store |
| `src/components/chat/ChatInput.tsx` | Modify | Add secrets button next to file buttons |
| `src/components/blobs/BlobManager.tsx` | Modify | Add category folders (source/sink/other) |
| `src/components/blobs/BlobManager.test.tsx` | Create | Tests for categorized file display |
| `src/components/inspector/GraphView.tsx` | Modify | Add per-node validation indicators |
| `src/components/inspector/GraphView.test.tsx` | Modify | Test validation indicators on graph nodes |
| `src/components/inspector/InspectorPanel.tsx` | Modify | Three-state validation indicator, move secrets button |
| `src/components/inspector/InspectorPanel.test.tsx` | Modify | Test three-state indicator |
| `src/stores/executionStore.ts` | Modify | Add warnings support, per-component validation map |
| `src/stores/subscriptions.ts` | Modify | Add validation-error-to-chat subscription |

All paths are relative to `src/elspeth/web/frontend/`.

---

## Task 1: Extend Types for Three-State Validation and Blob Categories

**Files:**
- Modify: `src/elspeth/web/frontend/src/types/index.ts:183-216`

This task adds the type foundations everything else builds on: warning support in `ValidationResult`, per-component severity, and blob categorization.

- [ ] **Step 1: Add ValidationWarning and extend ValidationResult types**

In `src/elspeth/web/frontend/src/types/index.ts`, add after the `ValidationError` interface (after line 205):

```typescript
/**
 * A single validation warning — same shape as ValidationError but non-blocking.
 * Warnings indicate suboptimal configuration but do not prevent execution.
 */
export interface ValidationWarning {
  component_id: string;
  component_type: string;
  message: string;
  suggestion: string | null;
}
```

Then modify the `ValidationResult` interface to include warnings:

```typescript
export interface ValidationResult {
  is_valid: boolean;
  summary: string;
  checks: ValidationCheck[];
  errors: ValidationError[];
  warnings?: ValidationWarning[];
}
```

- [ ] **Step 2: Add PipelineStatus type**

In `src/elspeth/web/frontend/src/types/index.ts`, add after `ValidationResult`:

```typescript
/**
 * Derived three-state pipeline validation status.
 *
 * - "valid": no errors, no warnings — fully runnable
 * - "valid-with-warnings": runnable but has non-blocking warnings (yellow)
 * - "invalid": has blocking errors, cannot execute (red)
 * - null: not yet validated
 */
export type PipelineStatus = "valid" | "valid-with-warnings" | "invalid";
```

- [ ] **Step 3: Add BlobCategory type**

In `src/elspeth/web/frontend/src/types/index.ts`, add after the `BlobMetadata` interface (after line 343):

```typescript
/**
 * User-facing file category for the blob manager folder view.
 * Derived from the blob's mime_type and created_by fields.
 */
export type BlobCategory = "source" | "sink" | "other";
```

- [ ] **Step 4: Export new types from api.ts re-export barrel**

In `src/elspeth/web/frontend/src/types/api.ts`, add `ValidationWarning`, `PipelineStatus`, and `BlobCategory` to the re-export list:

```typescript
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
  ValidationWarning,
  PipelineStatus,
  Run,
  RunEvent,
  RunEventProgress,
  RunEventError,
  RunEventCompleted,
  RunEventCancelled,
  RunProgress,
  ApiError,
  UploadResult,
  BlobMetadata,
  BlobCategory,
  SecretInventoryItem,
} from "./index";
```

- [ ] **Step 5: Verify TypeScript compiles**

Run: `cd src/elspeth/web/frontend && npx tsc --noEmit`
Expected: No errors. The new types are additive and don't break existing code.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/types/index.ts src/elspeth/web/frontend/src/types/api.ts
git commit -m "feat(frontend): add ValidationWarning, PipelineStatus, BlobCategory types

Foundation types for A6/A7 (per-node validation indicators, three-state
pipeline status) and A1 (categorized file folders)."
```

---

## Task 2: Add Design Tokens for Validation Indicators

**Files:**
- Modify: `src/elspeth/web/frontend/src/styles/tokens.ts`
- Modify: `src/elspeth/web/frontend/src/App.css`

- [ ] **Step 1: Add validation indicator tokens to tokens.ts**

In `src/elspeth/web/frontend/src/styles/tokens.ts`, add after the existing exports:

```typescript
export const VALIDATION_COLORS = {
  valid: "#14b0ae",           // matches --color-success
  warning: "#e38444",         // matches --color-warning
  invalid: "#e85653",         // matches --color-error
  unchecked: "#7a9a9a",       // muted/neutral
};

export const VALIDATION_BACKGROUNDS = {
  valid: "rgba(20, 176, 174, 0.15)",
  warning: "rgba(227, 132, 68, 0.15)",
  invalid: "rgba(232, 86, 83, 0.15)",
  unchecked: "rgba(122, 154, 154, 0.1)",
};
```

- [ ] **Step 2: Add CSS variables for validation node borders**

In `src/elspeth/web/frontend/src/App.css`, add to the `:root` block (around line 50, after the existing semantic color variables):

```css
  /* Per-node validation indicator borders */
  --color-node-valid: #14b0ae;
  --color-node-warning: #e38444;
  --color-node-invalid: #e85653;
  --color-node-unchecked: var(--color-border-strong);
```

- [ ] **Step 3: Verify build**

Run: `cd src/elspeth/web/frontend && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/src/styles/tokens.ts src/elspeth/web/frontend/src/App.css
git commit -m "feat(frontend): add validation indicator design tokens

Color tokens for green/yellow/red per-node validation status (A6)
and three-state pipeline status (A7)."
```

---

## Task 3: 50/50 Panel Split with Persistent Layout (A4)

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/common/Layout.tsx:9-14`
- Test: `src/elspeth/web/frontend/src/components/common/Layout.test.tsx` (create)

- [ ] **Step 1: Write the failing test**

Create `src/elspeth/web/frontend/src/components/common/Layout.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { Layout } from "./Layout";

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, val: string) => { store[key] = val; }),
    clear: () => { store = {}; },
  };
})();

Object.defineProperty(window, "localStorage", { value: localStorageMock });

// Set a known viewport width for percentage calculations
Object.defineProperty(window, "innerWidth", { value: 1600, writable: true });

describe("Layout", () => {
  beforeEach(() => {
    localStorageMock.clear();
    vi.clearAllMocks();
  });

  it("uses approximately 50% of remaining space for inspector by default", () => {
    const { container } = render(
      <Layout
        sidebar={<div>Sidebar</div>}
        chat={<div>Chat</div>}
        inspector={<div>Inspector</div>}
      />,
    );

    const layoutDiv = container.querySelector(".app-layout") as HTMLElement;
    // The grid template should contain a pixel value that is roughly
    // (viewport - sidebar) / 2. With 1600px viewport and 200px sidebar,
    // that's about 700px. We just check the stored default is in that range.
    const columns = layoutDiv.style.gridTemplateColumns;
    // Extract the inspector width (last value before "px")
    const match = columns.match(/(\d+)px$/);
    expect(match).not.toBeNull();
    const inspectorWidth = Number(match![1]);
    // Should be between 600 and 800 for a 1600px viewport
    expect(inspectorWidth).toBeGreaterThanOrEqual(600);
    expect(inspectorWidth).toBeLessThanOrEqual(800);
  });

  it("restores persisted inspector width from localStorage", () => {
    localStorageMock.getItem.mockImplementation((key: string) => {
      if (key === "elspeth_inspector_width") return "500";
      return null;
    });

    const { container } = render(
      <Layout
        sidebar={<div>Sidebar</div>}
        chat={<div>Chat</div>}
        inspector={<div>Inspector</div>}
      />,
    );

    const layoutDiv = container.querySelector(".app-layout") as HTMLElement;
    const columns = layoutDiv.style.gridTemplateColumns;
    expect(columns).toContain("500px");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/common/Layout.test.tsx`
Expected: FAIL — the default inspector width is 320px, not ~700px.

- [ ] **Step 3: Change default inspector width to 50% of remaining space**

In `src/elspeth/web/frontend/src/components/common/Layout.tsx`, replace the constants and initialization:

Replace lines 12-13:
```typescript
const MIN_INSPECTOR_WIDTH = 240;
const DEFAULT_INSPECTOR_WIDTH = 320;
```

With:
```typescript
const MIN_INSPECTOR_WIDTH = 240;

/**
 * Compute the default inspector width as ~50% of the space remaining
 * after the sidebar. This gives an even chat/inspector split (A4).
 * Falls back to 50% of viewport if called before layout.
 */
function defaultInspectorWidth(): number {
  const available = window.innerWidth - SIDEBAR_EXPANDED_WIDTH;
  const half = Math.round(available / 2);
  return Math.max(MIN_INSPECTOR_WIDTH, half);
}
```

Then update the `useState` initializer (line 46):

Replace:
```typescript
  const [inspectorWidth, setInspectorWidth] = useState(() =>
    loadPersistedNumber(INSPECTOR_WIDTH_KEY, DEFAULT_INSPECTOR_WIDTH)
  );
```

With:
```typescript
  const [inspectorWidth, setInspectorWidth] = useState(() =>
    loadPersistedNumber(INSPECTOR_WIDTH_KEY, defaultInspectorWidth())
  );
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/common/Layout.test.tsx`
Expected: PASS

- [ ] **Step 5: Run all existing tests to check for regressions**

Run: `cd src/elspeth/web/frontend && npx vitest run`
Expected: All tests pass. The change only affects the default — any existing test that persisted a width in localStorage will read that value instead.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/common/Layout.tsx src/elspeth/web/frontend/src/components/common/Layout.test.tsx
git commit -m "feat(frontend): default to 50/50 panel split (A4)

Inspector panel now defaults to half the remaining viewport width
after the sidebar, giving an even chat/inspector split. Persisted
width from localStorage still takes precedence."
```

---

## Task 4: Markdown and Mermaid Rendering in Chat (A2)

**Files:**
- Create: `src/elspeth/web/frontend/src/components/chat/MarkdownRenderer.tsx`
- Create: `src/elspeth/web/frontend/src/components/chat/MarkdownRenderer.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx:191-193`
- Modify: `src/elspeth/web/frontend/src/App.css`
- Modify: `package.json`

- [ ] **Step 1: Install markdown dependencies**

Run: `cd src/elspeth/web/frontend && npm install react-markdown remark-gfm mermaid`

These provide:
- `react-markdown`: Renders markdown to React elements (no dangerouslySetInnerHTML)
- `remark-gfm`: GitHub-flavored markdown (tables, strikethrough, task lists)
- `mermaid`: Diagram rendering engine

- [ ] **Step 2: Write the failing test for MarkdownRenderer**

Create `src/elspeth/web/frontend/src/components/chat/MarkdownRenderer.test.tsx`:

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MarkdownRenderer } from "./MarkdownRenderer";

describe("MarkdownRenderer", () => {
  it("renders plain text as a paragraph", () => {
    render(<MarkdownRenderer content="Hello world" />);
    expect(screen.getByText("Hello world")).toBeInTheDocument();
  });

  it("renders headings", () => {
    render(<MarkdownRenderer content="## Section Title" />);
    const heading = screen.getByRole("heading", { level: 2 });
    expect(heading).toHaveTextContent("Section Title");
  });

  it("renders inline code", () => {
    render(<MarkdownRenderer content="Use `set_source` to configure input." />);
    const code = screen.getByText("set_source");
    expect(code.tagName).toBe("CODE");
  });

  it("renders code blocks with language class", () => {
    const content = "```yaml\nsource:\n  plugin: csv\n```";
    const { container } = render(<MarkdownRenderer content={content} />);
    const pre = container.querySelector("pre");
    expect(pre).toBeInTheDocument();
    const code = pre?.querySelector("code");
    expect(code).toBeInTheDocument();
    expect(code?.textContent).toContain("source:");
  });

  it("renders tables from GFM markdown", () => {
    const content = "| Col A | Col B |\n|-------|-------|\n| 1 | 2 |";
    render(<MarkdownRenderer content={content} />);
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByText("Col A")).toBeInTheDocument();
  });

  it("renders a mermaid container for mermaid code blocks", () => {
    const content = "```mermaid\ngraph TD\n  A --> B\n```";
    const { container } = render(<MarkdownRenderer content={content} />);
    const mermaidDiv = container.querySelector(".mermaid-container");
    expect(mermaidDiv).toBeInTheDocument();
  });

  it("does not render mermaid blocks as regular code", () => {
    const content = "```mermaid\ngraph TD\n  A --> B\n```";
    const { container } = render(<MarkdownRenderer content={content} />);
    // Should NOT have a <pre><code> for mermaid
    const codeBlocks = container.querySelectorAll("pre > code");
    for (const block of codeBlocks) {
      expect(block.textContent).not.toContain("graph TD");
    }
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/chat/MarkdownRenderer.test.tsx`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement MarkdownRenderer component**

Create `src/elspeth/web/frontend/src/components/chat/MarkdownRenderer.tsx`:

```tsx
import { useEffect, useRef, useId, type ComponentPropsWithoutRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import mermaid from "mermaid";

// Initialize mermaid once with dark theme matching ELSPETH's color palette
mermaid.initialize({
  startOnLoad: false,
  theme: "dark",
  themeVariables: {
    primaryColor: "#1a3d47",
    primaryBorderColor: "#4db89a",
    primaryTextColor: "#dff0ee",
    lineColor: "#6a9898",
    secondaryColor: "#28504a",
    tertiaryColor: "#0f2d35",
  },
});

interface MarkdownRendererProps {
  content: string;
}

/**
 * Renders markdown content with GFM support and Mermaid diagram rendering.
 *
 * Mermaid code blocks (```mermaid) are rendered as interactive diagrams.
 * All other code blocks render as syntax-highlighted <pre><code>.
 */
export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code: CodeBlock,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

/**
 * Custom code renderer that intercepts mermaid blocks and renders
 * them as diagrams, while passing all other code through normally.
 */
function CodeBlock({
  className,
  children,
  ...props
}: ComponentPropsWithoutRef<"code">) {
  const language = className?.replace("language-", "") ?? "";
  const code = String(children).replace(/\n$/, "");

  // Inline code (no language, rendered inside a <p>)
  if (!className) {
    return <code className="inline-code" {...props}>{children}</code>;
  }

  // Mermaid diagrams get special treatment
  if (language === "mermaid") {
    return <MermaidDiagram chart={code} />;
  }

  // All other code blocks render as <pre><code>
  return (
    <pre className="code-block">
      <code className={className} {...props}>
        {code}
      </code>
    </pre>
  );
}

/**
 * Renders a Mermaid diagram. Uses mermaid.render() to produce SVG,
 * then injects it via innerHTML (mermaid's API requires this).
 *
 * Falls back to a <pre> block if mermaid parsing fails.
 */
function MermaidDiagram({ chart }: { chart: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const uniqueId = useId().replace(/:/g, "-");

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let cancelled = false;

    mermaid
      .render(`mermaid-${uniqueId}`, chart)
      .then(({ svg }) => {
        if (!cancelled && container) {
          container.innerHTML = svg;
        }
      })
      .catch(() => {
        if (!cancelled && container) {
          // Fallback: show the source as a code block
          container.textContent = chart;
          container.classList.add("mermaid-fallback");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [chart, uniqueId]);

  return (
    <div
      ref={containerRef}
      className="mermaid-container"
      role="img"
      aria-label="Mermaid diagram"
    />
  );
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/chat/MarkdownRenderer.test.tsx`
Expected: PASS

- [ ] **Step 6: Add markdown CSS styles**

In `src/elspeth/web/frontend/src/App.css`, add at the end of the file:

```css
/* ── Markdown rendering (A2) ─────────────────────────────────────────────── */

.markdown-body {
  font-size: 14px;
  line-height: 1.5;
  word-break: break-word;
}

.markdown-body h1,
.markdown-body h2,
.markdown-body h3,
.markdown-body h4 {
  margin: 12px 0 6px;
  font-weight: 600;
  color: var(--color-text);
}

.markdown-body h1 { font-size: 18px; }
.markdown-body h2 { font-size: 16px; }
.markdown-body h3 { font-size: 15px; }
.markdown-body h4 { font-size: 14px; }

.markdown-body p {
  margin: 4px 0;
}

.markdown-body ul,
.markdown-body ol {
  margin: 4px 0;
  padding-left: 20px;
}

.markdown-body li {
  margin-bottom: 2px;
}

.markdown-body table {
  border-collapse: collapse;
  margin: 8px 0;
  font-size: 13px;
  width: 100%;
}

.markdown-body th,
.markdown-body td {
  border: 1px solid var(--color-border-strong);
  padding: 4px 8px;
  text-align: left;
}

.markdown-body th {
  background-color: var(--color-surface-elevated);
  font-weight: 600;
}

.markdown-body .inline-code {
  background-color: var(--color-surface-elevated);
  padding: 1px 4px;
  border-radius: 3px;
  font-family: var(--font-mono);
  font-size: 13px;
}

.markdown-body .code-block {
  background-color: var(--color-surface-elevated);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 10px 12px;
  margin: 8px 0;
  overflow-x: auto;
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.4;
}

.markdown-body .code-block code {
  background: none;
  padding: 0;
}

.markdown-body .mermaid-container {
  margin: 8px 0;
  padding: 12px;
  background-color: var(--color-surface-elevated);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  overflow-x: auto;
  text-align: center;
}

.markdown-body .mermaid-container svg {
  max-width: 100%;
  height: auto;
}

.markdown-body .mermaid-fallback {
  font-family: var(--font-mono);
  font-size: 13px;
  white-space: pre-wrap;
  text-align: left;
  color: var(--color-text-muted);
}

.markdown-body blockquote {
  margin: 8px 0;
  padding: 4px 12px;
  border-left: 3px solid var(--color-border-strong);
  color: var(--color-text-secondary);
}
```

- [ ] **Step 7: Wire MarkdownRenderer into MessageBubble**

In `src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx`:

Add import at top (after line 3):
```typescript
import { MarkdownRenderer } from "./MarkdownRenderer";
```

Replace the plain text rendering of assistant messages. Find lines 191-193:
```typescript
        ) : (
          message.content
        )}
```

Replace with:
```typescript
        ) : isUser ? (
          message.content
        ) : (
          <MarkdownRenderer content={message.content} />
        )}
```

This renders assistant messages as markdown while keeping user messages as plain text (users type plain text, not markdown).

- [ ] **Step 8: Remove whiteSpace: "pre-wrap" for assistant bubbles**

In `src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx`, the bubble wrapper div (line 91-99) currently applies `whiteSpace: "pre-wrap"` to all messages. This conflicts with markdown rendering. Change line 96 to conditionally apply it:

Replace:
```typescript
          whiteSpace: "pre-wrap",
```

With:
```typescript
          whiteSpace: isUser ? "pre-wrap" : undefined,
```

- [ ] **Step 9: Run all tests**

Run: `cd src/elspeth/web/frontend && npx vitest run`
Expected: All tests pass. The MessageBubble test uses `makeMessage()` which defaults to `role: "user"` — user messages are still plain text so existing tests are unaffected.

- [ ] **Step 10: Commit**

```bash
git add src/elspeth/web/frontend/src/components/chat/MarkdownRenderer.tsx \
  src/elspeth/web/frontend/src/components/chat/MarkdownRenderer.test.tsx \
  src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx \
  src/elspeth/web/frontend/src/App.css \
  src/elspeth/web/frontend/package.json \
  src/elspeth/web/frontend/package-lock.json
git commit -m "feat(frontend): markdown and Mermaid rendering in chat (A2)

Assistant messages now render as rich markdown with GFM support
(tables, code blocks, task lists) and interactive Mermaid diagrams.
User messages remain plain text."
```

---

## Task 5: Visible Error/Warning Routing to Chat (A3)

**Files:**
- Modify: `src/elspeth/web/frontend/src/stores/executionStore.ts:131-156`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts:18-26`
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
- Test: `src/elspeth/web/frontend/src/stores/executionStore.test.ts` (create)

The current flow: validation errors are auto-sent to the LLM as a plain text user message (sessionStore.sendValidationFeedback). But the user can't see that the error was routed to the agent. This task adds a visible system message in the chat when errors/warnings occur, clearly attributed as "sent to the agent."

- [ ] **Step 1: Write the failing test**

Create `src/elspeth/web/frontend/src/stores/executionStore.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useSessionStore } from "./sessionStore";
import { useExecutionStore } from "./executionStore";
import type { ValidationResult } from "@/types/index";

// Mock the API client
vi.mock("@/api/client", () => ({
  validatePipeline: vi.fn(),
  fetchRuns: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/api/websocket", () => ({
  connectToRun: vi.fn(),
}));

describe("executionStore.validate", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "session-1",
      messages: [],
      compositionState: null,
      isComposing: false,
      stateVersions: [],
      isLoadingVersions: false,
      error: null,
      sessions: [],
    });
    useExecutionStore.getState().reset();
  });

  it("injects a system message into chat when validation fails", async () => {
    const failedResult: ValidationResult = {
      is_valid: false,
      summary: "Validation failed",
      checks: [],
      errors: [
        {
          component_id: "llm_extract",
          component_type: "transform",
          message: "Missing required option: model",
          suggestion: "Add a model option",
        },
      ],
      warnings: [],
    };

    const { validatePipeline } = await import("@/api/client");
    (validatePipeline as ReturnType<typeof vi.fn>).mockResolvedValue(failedResult);

    // Spy on sendValidationFeedback to confirm it's still called
    const sendFeedback = vi.spyOn(useSessionStore.getState(), "sendValidationFeedback");

    await useExecutionStore.getState().validate("session-1");

    // A system message should have been injected
    const messages = useSessionStore.getState().messages;
    const systemMessages = messages.filter((m) => m.role === "system");
    expect(systemMessages.length).toBeGreaterThanOrEqual(1);
    expect(systemMessages[0].content).toContain("Validation failed");
    expect(systemMessages[0].content).toContain("llm_extract");

    // sendValidationFeedback should still be called (sends to LLM)
    expect(sendFeedback).toHaveBeenCalledWith(failedResult);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/elspeth/web/frontend && npx vitest run src/stores/executionStore.test.ts`
Expected: FAIL — no system message is injected into chat currently.

- [ ] **Step 3: Add injectSystemMessage action to sessionStore**

In `src/elspeth/web/frontend/src/stores/sessionStore.ts`, add a new action to the interface (after `clearError` around line 38):

```typescript
  injectSystemMessage: (content: string) => void;
```

Then add the implementation in the store (after the `clearError` method around line 366):

```typescript
  injectSystemMessage(content: string) {
    const { activeSessionId } = get();
    if (!activeSessionId) return;

    const systemMessage: ChatMessage = {
      id: `system-${crypto.randomUUID()}`,
      session_id: activeSessionId,
      role: "system",
      content,
      tool_calls: null,
      created_at: new Date().toISOString(),
    };

    set((state) => ({
      messages: [...state.messages, systemMessage],
    }));
  },
```

- [ ] **Step 4: Inject system message on validation failure/warning**

In `src/elspeth/web/frontend/src/stores/executionStore.ts`, modify the `validate` method. After `set({ validationResult: result, isValidating: false });` (line 138), add the system message injection:

Replace lines 140-144:
```typescript
      // Auto-send validation errors to the LLM so it can attempt fixes.
      // The user still sees the banner — this adds a chat message too.
      if (!result.is_valid && result.errors.length > 0) {
        useSessionStore.getState().sendValidationFeedback(result);
      }
```

With:
```typescript
      // Inject a visible system message into chat so the user can see
      // that errors/warnings were routed to the agent (A3).
      if (!result.is_valid && result.errors.length > 0) {
        const lines = ["**Validation failed** — the following errors were sent to the agent:"];
        for (const err of result.errors) {
          lines.push(`- **[${err.component_type}] ${err.component_id}:** ${err.message}`);
        }
        useSessionStore.getState().injectSystemMessage(lines.join("\n"));

        // Auto-send validation errors to the LLM so it can attempt fixes.
        useSessionStore.getState().sendValidationFeedback(result);
      } else if (result.is_valid && result.warnings && result.warnings.length > 0) {
        const lines = ["**Validation passed with warnings:**"];
        for (const warn of result.warnings) {
          lines.push(`- **[${warn.component_type}] ${warn.component_id}:** ${warn.message}`);
        }
        useSessionStore.getState().injectSystemMessage(lines.join("\n"));
      }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd src/elspeth/web/frontend && npx vitest run src/stores/executionStore.test.ts`
Expected: PASS

- [ ] **Step 6: Run all tests**

Run: `cd src/elspeth/web/frontend && npx vitest run`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/frontend/src/stores/executionStore.ts \
  src/elspeth/web/frontend/src/stores/executionStore.test.ts \
  src/elspeth/web/frontend/src/stores/sessionStore.ts
git commit -m "feat(frontend): route validation errors visibly through chat (A3)

Validation errors now appear as system messages in the chat,
clearly showing the user that errors were sent to the agent.
Warnings on valid pipelines also get a visible system message."
```

---

## Task 6: Per-User File Folders (A1)

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/blobs/BlobManager.tsx`
- Create: `src/elspeth/web/frontend/src/components/blobs/BlobManager.test.tsx`

- [ ] **Step 1: Write the failing test**

Create `src/elspeth/web/frontend/src/components/blobs/BlobManager.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { useBlobStore } from "@/stores/blobStore";
import { useSessionStore } from "@/stores/sessionStore";
import { BlobManager } from "./BlobManager";
import type { BlobMetadata } from "@/types/api";

function makeBlob(overrides: Partial<BlobMetadata> = {}): BlobMetadata {
  return {
    id: "blob-1",
    session_id: "session-1",
    filename: "data.csv",
    mime_type: "text/csv",
    size_bytes: 1024,
    content_hash: null,
    created_at: new Date().toISOString(),
    created_by: "user",
    source_description: null,
    status: "ready",
    ...overrides,
  };
}

describe("BlobManager categorized folders", () => {
  beforeEach(() => {
    useSessionStore.setState({ activeSessionId: "session-1" });
    vi.clearAllMocks();
  });

  it("groups blobs into Source, Output, and Other sections", () => {
    const blobs: BlobMetadata[] = [
      makeBlob({ id: "b1", filename: "input.csv", created_by: "user" }),
      makeBlob({ id: "b2", filename: "results.json", created_by: "pipeline" }),
      makeBlob({ id: "b3", filename: "prompt.txt", created_by: "assistant" }),
    ];

    useBlobStore.setState({ blobs, isLoading: false, error: null });

    render(<BlobManager onUseAsInput={vi.fn()} />);

    expect(screen.getByText("Source files")).toBeInTheDocument();
    expect(screen.getByText("Output files")).toBeInTheDocument();
    expect(screen.getByText("Other files")).toBeInTheDocument();
  });

  it("puts user-uploaded files in Source section", () => {
    const blobs = [
      makeBlob({ id: "b1", filename: "data.csv", created_by: "user" }),
    ];

    useBlobStore.setState({ blobs, isLoading: false, error: null });

    render(<BlobManager onUseAsInput={vi.fn()} />);

    // The file should appear under Source files
    const sourceSection = screen.getByText("Source files").closest("div");
    expect(sourceSection).toBeInTheDocument();
    expect(screen.getByText("data.csv")).toBeInTheDocument();
  });

  it("puts pipeline-created files in Output section", () => {
    const blobs = [
      makeBlob({ id: "b2", filename: "results.json", created_by: "pipeline" }),
    ];

    useBlobStore.setState({ blobs, isLoading: false, error: null });

    render(<BlobManager onUseAsInput={vi.fn()} />);

    expect(screen.getByText("Output files")).toBeInTheDocument();
    expect(screen.getByText("results.json")).toBeInTheDocument();
  });

  it("shows empty state for empty category", () => {
    useBlobStore.setState({ blobs: [], isLoading: false, error: null });

    render(<BlobManager onUseAsInput={vi.fn()} />);

    expect(screen.getByText(/No files yet/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/blobs/BlobManager.test.tsx`
Expected: FAIL — no "Source files" / "Output files" / "Other files" headings exist.

- [ ] **Step 3: Implement categorized file folders in BlobManager**

Replace the entire content of `src/elspeth/web/frontend/src/components/blobs/BlobManager.tsx`:

```tsx
// src/components/blobs/BlobManager.tsx
import { useEffect, useRef, useCallback, useMemo } from "react";
import { useBlobStore } from "@/stores/blobStore";
import { useSessionStore } from "@/stores/sessionStore";
import { BlobRow } from "./BlobRow";
import type { BlobMetadata, BlobCategory } from "@/types/api";

interface BlobManagerProps {
  onUseAsInput: (blob: BlobMetadata) => void;
}

/**
 * Categorize a blob into source/sink/other based on who created it.
 * - User uploads → source files (pipeline inputs)
 * - Pipeline outputs → sink files (results)
 * - Assistant-created → other (prompts, templates, config)
 */
function categorizeBlob(blob: BlobMetadata): BlobCategory {
  if (blob.created_by === "user") return "source";
  if (blob.created_by === "pipeline") return "sink";
  return "other";
}

const CATEGORY_LABELS: Record<BlobCategory, string> = {
  source: "Source files",
  sink: "Output files",
  other: "Other files",
};

const CATEGORY_ORDER: BlobCategory[] = ["source", "sink", "other"];

/**
 * Collapsible blob manager panel with categorized folders.
 * Shows session-scoped files grouped by source/output/other
 * with upload, download, delete, and "use as input" actions.
 */
export function BlobManager({ onUseAsInput }: BlobManagerProps) {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const { blobs, isLoading, error, loadBlobs, uploadBlob, deleteBlob, downloadBlob } =
    useBlobStore();
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (activeSessionId) {
      loadBlobs(activeSessionId);
    }
  }, [activeSessionId, loadBlobs]);

  const grouped = useMemo(() => {
    const groups: Record<BlobCategory, BlobMetadata[]> = {
      source: [],
      sink: [],
      other: [],
    };
    for (const blob of blobs) {
      groups[categorizeBlob(blob)].push(blob);
    }
    return groups;
  }, [blobs]);

  const handleUpload = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file || !activeSessionId) return;
      try {
        await uploadBlob(activeSessionId, file);
      } catch {
        // Error is already in the store
      } finally {
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }
    },
    [activeSessionId, uploadBlob],
  );

  const handleDelete = useCallback(
    (blobId: string) => {
      if (!activeSessionId) return;
      deleteBlob(activeSessionId, blobId);
    },
    [activeSessionId, deleteBlob],
  );

  const handleDownload = useCallback(
    (blobId: string) => {
      if (!activeSessionId) return;
      downloadBlob(activeSessionId, blobId);
    },
    [activeSessionId, downloadBlob],
  );

  if (!activeSessionId) return null;

  return (
    <div
      className="blob-manager"
      style={{
        borderTop: "1px solid var(--color-border)",
        maxHeight: 280,
        display: "flex",
        flexDirection: "column",
        fontSize: 13,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "6px 12px",
          borderBottom: "1px solid var(--color-border)",
          flexShrink: 0,
        }}
      >
        <span style={{ fontWeight: 600, fontSize: 12, color: "var(--color-text-secondary)" }}>
          Files ({blobs.length})
        </span>
        <button
          onClick={() => fileInputRef.current?.click()}
          className="btn"
          style={{
            fontSize: 12,
            padding: "2px 8px",
            cursor: "pointer",
            minHeight: 36,
            minWidth: 44,
          }}
          aria-label="Upload file"
        >
          + Upload
        </button>
        <input
          ref={fileInputRef}
          type="file"
          onChange={handleUpload}
          style={{ display: "none" }}
          aria-hidden="true"
          tabIndex={-1}
        />
      </div>

      {/* Error */}
      {error && (
        <div
          role="alert"
          style={{
            padding: "4px 12px",
            fontSize: 12,
            color: "var(--color-error)",
            backgroundColor: "var(--color-error-bg)",
          }}
        >
          {error}
        </div>
      )}

      {/* Categorized file list */}
      <div style={{ flex: 1, overflowY: "auto" }}>
        {isLoading ? (
          <div style={{ padding: 12, color: "var(--color-text-muted)", textAlign: "center" }}>
            Loading...
          </div>
        ) : blobs.length === 0 ? (
          <div style={{ padding: 12, color: "var(--color-text-muted)", textAlign: "center" }}>
            No files yet. Upload a file to get started.
          </div>
        ) : (
          CATEGORY_ORDER.map((category) => {
            const categoryBlobs = grouped[category];
            if (categoryBlobs.length === 0) return null;
            return (
              <div key={category}>
                <div
                  style={{
                    padding: "4px 12px",
                    fontSize: 11,
                    fontWeight: 600,
                    color: "var(--color-text-muted)",
                    textTransform: "uppercase",
                    letterSpacing: "0.5px",
                    backgroundColor: "var(--color-surface-elevated)",
                    borderBottom: "1px solid var(--color-border)",
                  }}
                >
                  {CATEGORY_LABELS[category]}
                </div>
                {categoryBlobs.map((blob) => (
                  <BlobRow
                    key={blob.id}
                    blob={blob}
                    onDownload={handleDownload}
                    onDelete={handleDelete}
                    onUseAsInput={onUseAsInput}
                  />
                ))}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/blobs/BlobManager.test.tsx`
Expected: PASS

- [ ] **Step 5: Run all tests**

Run: `cd src/elspeth/web/frontend && npx vitest run`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/blobs/BlobManager.tsx \
  src/elspeth/web/frontend/src/components/blobs/BlobManager.test.tsx
git commit -m "feat(frontend): categorized file folders in blob manager (A1)

Files are now grouped into Source (user uploads), Output (pipeline
results), and Other (assistant-created) sections. Empty categories
are hidden."
```

---

## Task 7: Secrets Button Relocation (A5)

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatInput.tsx:116-174`
- Modify: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx:512-532`
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
- Modify: `src/elspeth/web/frontend/src/App.tsx:109`

This task moves the secrets button from the inspector header (cog icon) to the chat input toolbar (key icon), next to the file buttons.

- [ ] **Step 1: Add onOpenSecrets prop to ChatInput**

In `src/elspeth/web/frontend/src/components/chat/ChatInput.tsx`, add `onOpenSecrets` to the interface (around line 12):

```typescript
interface ChatInputProps {
  onSend: (content: string) => void;
  disabled: boolean;
  inputRef: React.RefObject<HTMLTextAreaElement>;
  onToggleBlobManager?: () => void;
  showBlobManager?: boolean;
  onOpenSecrets?: () => void;
}
```

Update the function signature (line 20):

```typescript
export function ChatInput({ onSend, disabled, inputRef, onToggleBlobManager, showBlobManager, onOpenSecrets }: ChatInputProps) {
```

- [ ] **Step 2: Add secrets button to ChatInput toolbar**

In `src/elspeth/web/frontend/src/components/chat/ChatInput.tsx`, add the secrets button after the file upload button (after line 174, before the Send button):

```tsx
        {/* Secrets button — key icon, co-located with file actions (A5) */}
        {onOpenSecrets && (
          <button
            type="button"
            onClick={onOpenSecrets}
            style={{
              padding: "8px 10px",
              backgroundColor: "transparent",
              border: "1px solid var(--color-border-strong)",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 16,
              color: "var(--color-text)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              minWidth: 44,
              minHeight: 44,
            }}
            title="API Keys & Secrets"
            aria-label="Open secrets settings"
          >
            <span aria-hidden="true">{"\uD83D\uDD11"}</span>
          </button>
        )}
```

- [ ] **Step 3: Remove secrets button from InspectorPanel header**

In `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx`:

Remove the `onOpenSecrets` prop from the interface (lines 342-344):
```typescript
interface InspectorPanelProps {
  onOpenSecrets?: () => void;
}
```

Change to:
```typescript
interface InspectorPanelProps {}
```

Remove `onOpenSecrets` from the function signature (line 346):
```typescript
export function InspectorPanel({ onOpenSecrets }: InspectorPanelProps) {
```

Change to:
```typescript
export function InspectorPanel(_props: InspectorPanelProps) {
```

Actually, since `InspectorPanelProps` is now empty, simplify:
```typescript
export function InspectorPanel() {
```

And remove the interface entirely.

Remove the settings gear button block (lines 513-532):
```tsx
            {/* Settings gear */}
            {onOpenSecrets && (
              <button
                onClick={onOpenSecrets}
                aria-label="Open secrets settings"
                title="API Keys & Secrets"
                ...
              >
                ⚙
              </button>
            )}
```

- [ ] **Step 4: Wire secrets through ChatPanel to ChatInput**

In `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`, add `onOpenSecrets` prop:

```typescript
interface ChatPanelProps {
  onOpenSecrets?: () => void;
}
```

Update the function signature:
```typescript
export function ChatPanel({ onOpenSecrets }: ChatPanelProps) {
```

Pass it through to ChatInput (around line 252):
```tsx
      <ChatInput
        onSend={handleSend}
        disabled={isComposing}
        inputRef={inputRef}
        onToggleBlobManager={() => setShowBlobManager((v) => !v)}
        showBlobManager={showBlobManager}
        onOpenSecrets={onOpenSecrets}
      />
```

- [ ] **Step 5: Update App.tsx to pass secrets handler through chat, not inspector**

In `src/elspeth/web/frontend/src/App.tsx`, change line 106-110:

```tsx
          <Layout
            sidebar={<SessionSidebar />}
            chat={<ChatPanel />}
            inspector={<InspectorPanel onOpenSecrets={openSecrets} />}
          />
```

To:
```tsx
          <Layout
            sidebar={<SessionSidebar />}
            chat={<ChatPanel onOpenSecrets={openSecrets} />}
            inspector={<InspectorPanel />}
          />
```

- [ ] **Step 6: Verify TypeScript compiles**

Run: `cd src/elspeth/web/frontend && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 7: Run all tests**

Run: `cd src/elspeth/web/frontend && npx vitest run`
Expected: All tests pass. The InspectorPanel test may need updating if it references the secrets button.

- [ ] **Step 8: Fix any failing InspectorPanel tests**

If `InspectorPanel.test.tsx` passes `onOpenSecrets` as a prop, remove that prop from the test. Check:

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/InspectorPanel.test.tsx`

If it fails, update the test to match the new interface (no `onOpenSecrets` prop).

- [ ] **Step 9: Commit**

```bash
git add src/elspeth/web/frontend/src/components/chat/ChatInput.tsx \
  src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx \
  src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx \
  src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx \
  src/elspeth/web/frontend/src/App.tsx
git commit -m "feat(frontend): move secrets button to chat toolbar with key icon (A5)

Secrets button now uses a key icon and lives next to the file
manager and paperclip buttons in the chat input toolbar, instead
of the inspector header cog."
```

---

## Task 8: Per-Node Validation Indicators on Graph (A6)

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx`
- Modify: `src/elspeth/web/frontend/src/components/inspector/GraphView.test.tsx`

- [ ] **Step 1: Write the failing test**

In `src/elspeth/web/frontend/src/components/inspector/GraphView.test.tsx`, add a new test (you'll need to read the existing tests first and follow their patterns):

```tsx
// Add to existing test file
import { useExecutionStore } from "@/stores/executionStore";

describe("GraphView validation indicators", () => {
  it("applies red border to nodes with validation errors", () => {
    // Set up a composition state with one transform
    useSessionStore.setState({
      compositionState: {
        version: 1,
        source: { plugin: "csv", options: {} },
        nodes: [
          {
            id: "llm_extract",
            node_type: "transform",
            plugin: "llm",
            input: "source",
            on_success: "output",
            on_error: null,
            options: {},
          },
        ],
        edges: [
          { id: "e1", from_node: "source", to_node: "llm_extract", edge_type: "on_success", label: null },
          { id: "e2", from_node: "llm_extract", to_node: "output", edge_type: "on_success", label: null },
        ],
        outputs: [{ name: "output", plugin: "json", options: {} }],
        metadata: { name: null, description: null },
      },
    });

    // Set up validation result with error on llm_extract
    useExecutionStore.setState({
      validationResult: {
        is_valid: false,
        summary: "Validation failed",
        checks: [],
        errors: [
          {
            component_id: "llm_extract",
            component_type: "transform",
            message: "Missing model",
            suggestion: null,
          },
        ],
        warnings: [],
      },
    });

    const { container } = render(<GraphView />);

    // The React Flow nodes should exist
    const nodes = container.querySelectorAll(".react-flow__node");
    expect(nodes.length).toBeGreaterThan(0);
  });
});
```

Note: React Flow rendering in JSDOM is limited. The test verifies the component renders without crashing when validation state is present. Visual verification of border colors requires manual testing or a visual regression tool.

- [ ] **Step 2: Run test to verify it fails (or passes with current code)**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/GraphView.test.tsx`

- [ ] **Step 3: Add validation state reading to GraphView**

In `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx`:

Add import at top:
```typescript
import { useExecutionStore } from "@/stores/executionStore";
import { VALIDATION_COLORS } from "@/styles/tokens";
```

Inside the `GraphView` component (after line 81), add:

```typescript
  const validationResult = useExecutionStore((s) => s.validationResult);

  // Build a map of component_id → validation severity for border coloring
  const nodeValidationMap = useMemo(() => {
    const map: Record<string, "valid" | "warning" | "error"> = {};
    if (!validationResult) return map;

    // All nodes with errors
    for (const err of validationResult.errors) {
      map[err.component_id] = "error";
    }

    // All nodes with warnings (only if not already error)
    if (validationResult.warnings) {
      for (const warn of validationResult.warnings) {
        if (!map[warn.component_id]) {
          map[warn.component_id] = "warning";
        }
      }
    }

    return map;
  }, [validationResult]);
```

- [ ] **Step 4: Apply validation border colors to nodes**

In the `makeRfNode` function inside the `useMemo` block, modify it to accept a validation status parameter. Change the function signature from:

```typescript
    function makeRfNode(
      id: string,
      typeLabel: string,
      subtitle: string | null,
      badgeBg: string,
      badgeColor: string,
    ): Node {
```

To:

```typescript
    function makeRfNode(
      id: string,
      typeLabel: string,
      subtitle: string | null,
      badgeBg: string,
      badgeColor: string,
      validationStatus?: "valid" | "warning" | "error",
    ): Node {
```

Then modify the node style (around line 130-138) to use validation-colored borders:

```typescript
        style: {
          backgroundColor: "var(--color-surface-elevated)",
          border: validationStatus === "error"
            ? `2px solid ${VALIDATION_COLORS.invalid}`
            : validationStatus === "warning"
              ? `2px solid ${VALIDATION_COLORS.warning}`
              : validationStatus === "valid"
                ? `2px solid ${VALIDATION_COLORS.valid}`
                : "1px solid var(--color-border-strong)",
          borderRadius: 8,
          width: NODE_WIDTH,
          height: NODE_HEIGHT,
          padding: 0,
        },
```

Also add a small validation indicator dot inside the node label JSX (after the type badge span):

```tsx
                {validationStatus && (
                  <span
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      backgroundColor:
                        validationStatus === "error"
                          ? VALIDATION_COLORS.invalid
                          : validationStatus === "warning"
                            ? VALIDATION_COLORS.warning
                            : VALIDATION_COLORS.valid,
                      flexShrink: 0,
                    }}
                    title={
                      validationStatus === "error"
                        ? "Has validation errors"
                        : validationStatus === "warning"
                          ? "Has warnings"
                          : "Valid"
                    }
                  />
                )}
```

- [ ] **Step 5: Pass validation status when creating nodes**

Update each `makeRfNode` call to pass the validation status:

For source node (around line 146):
```typescript
      rfNodes.push(
        makeRfNode(
          "source",
          "source",
          compositionState.source.plugin,
          "rgba(77, 184, 154, 0.15)",
          "#4db89a",
          nodeValidationMap["source"],
        ),
      );
```

For pipeline nodes (around line 152):
```typescript
      rfNodes.push(
        makeRfNode(
          node.id,
          node.node_type,
          node.plugin,
          BADGE_BACKGROUNDS[node.node_type],
          BADGE_COLORS[node.node_type],
          nodeValidationMap[node.id],
        ),
      );
```

For output nodes (around line 160):
```typescript
      rfNodes.push(
        makeRfNode(
          output.name,
          "sink",
          output.plugin,
          "rgba(224, 112, 64, 0.15)",
          "#e07040",
          nodeValidationMap[output.name],
        ),
      );
```

Update the `useMemo` dependency array to include `nodeValidationMap`:
```typescript
  }, [compositionState, nodeValidationMap]);
```

- [ ] **Step 6: Run all tests**

Run: `cd src/elspeth/web/frontend && npx vitest run`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/GraphView.tsx \
  src/elspeth/web/frontend/src/components/inspector/GraphView.test.tsx
git commit -m "feat(frontend): per-node validation indicators on graph (A6)

Graph nodes now show colored borders (green/yellow/red) and a
small status dot based on per-component validation results.
Nodes with errors get a red border, warnings get yellow, and
valid nodes get green (only shown when validation has run)."
```

---

## Task 9: Three-State Pipeline Status Indicator (A7)

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx:468-508`
- Modify: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx`

- [ ] **Step 1: Write the failing test**

Add to `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx`:

```tsx
import { useExecutionStore } from "@/stores/executionStore";

describe("InspectorPanel three-state validation indicator", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: {
        version: 1,
        source: { plugin: "csv", options: {} },
        nodes: [],
        edges: [],
        outputs: [{ name: "out", plugin: "json", options: {} }],
        metadata: { name: null, description: null },
      },
      stateVersions: [],
      isLoadingVersions: false,
    });
  });

  it("shows hollow circle when not validated", () => {
    useExecutionStore.setState({ validationResult: null });
    render(<InspectorPanel />);
    expect(screen.getByLabelText("Not validated")).toBeInTheDocument();
  });

  it("shows checkmark for valid pipeline (no warnings)", () => {
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        summary: "All checks passed",
        checks: [],
        errors: [],
        warnings: [],
      },
    });
    render(<InspectorPanel />);
    expect(screen.getByLabelText("Validation passed")).toBeInTheDocument();
  });

  it("shows warning indicator for valid-with-warnings", () => {
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        summary: "Passed with warnings",
        checks: [],
        errors: [],
        warnings: [
          {
            component_id: "source",
            component_type: "source",
            message: "No explicit schema",
            suggestion: "Add schema",
          },
        ],
      },
    });
    render(<InspectorPanel />);
    expect(screen.getByLabelText("Validation passed with warnings")).toBeInTheDocument();
  });

  it("shows error indicator for invalid pipeline", () => {
    useExecutionStore.setState({
      validationResult: {
        is_valid: false,
        summary: "Validation failed",
        checks: [],
        errors: [
          {
            component_id: "llm",
            component_type: "transform",
            message: "Missing model",
            suggestion: null,
          },
        ],
        warnings: [],
      },
    });
    render(<InspectorPanel />);
    expect(screen.getByLabelText("Validation failed")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/InspectorPanel.test.tsx`
Expected: FAIL — "Validation passed with warnings" label doesn't exist.

- [ ] **Step 3: Implement three-state indicator**

In `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx`, replace the validation status indicator block (around lines 468-508). Find:

```tsx
            {/* Validation status indicator — shape + color for accessibility */}
            {hasCompositionContent && (
              <span
                aria-label={
                  validationResult === null
                    ? "Not validated"
                    : validationResult.is_valid
                      ? "Validation passed"
                      : "Validation failed"
                }
```

Replace the entire `<span>` block with:

```tsx
            {/* Validation status indicator — three-state (A7) */}
            {hasCompositionContent && (() => {
              const hasWarnings = validationResult?.warnings && validationResult.warnings.length > 0;
              const status: string =
                validationResult === null
                  ? "unchecked"
                  : !validationResult.is_valid
                    ? "invalid"
                    : hasWarnings
                      ? "warning"
                      : "valid";

              const labels: Record<string, string> = {
                unchecked: "Not validated",
                valid: "Validation passed",
                warning: "Validation passed with warnings",
                invalid: "Validation failed",
              };

              const colors: Record<string, string> = {
                unchecked: "var(--color-warning)",
                valid: "var(--color-success)",
                warning: "var(--color-warning)",
                invalid: "var(--color-error)",
              };

              const symbols: Record<string, string> = {
                unchecked: "\u25CB",  // ○ hollow circle
                valid: "\u2713",      // ✓ checkmark
                warning: "\u26A0",    // ⚠ warning triangle
                invalid: "\u2717",    // ✗ cross mark
              };

              return (
                <span
                  aria-label={labels[status]}
                  title={labels[status]}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    width: 16,
                    height: 16,
                    fontSize: 12,
                    lineHeight: 1,
                    flexShrink: 0,
                    color: colors[status],
                  }}
                >
                  {symbols[status]}
                </span>
              );
            })()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/InspectorPanel.test.tsx`
Expected: PASS

- [ ] **Step 5: Update ValidationResultBanner to show warnings**

In `src/elspeth/web/frontend/src/components/execution/ValidationResult.tsx`, update the pass state to also render warnings when present.

After the existing `checks` list (after line 75), add:

```tsx
        {result.warnings && result.warnings.length > 0 && (
          <div style={{ marginTop: 6 }}>
            <div style={{ fontWeight: 600, fontSize: 12, color: "var(--color-warning)" }}>
              Warnings ({result.warnings.length}):
            </div>
            <ul
              style={{
                margin: "2px 0 0",
                padding: "0 0 0 22px",
                fontSize: 12,
                color: "var(--color-warning)",
              }}
            >
              {result.warnings.map((warn, i) => (
                <li key={i} style={{ marginBottom: 2 }}>
                  <strong>
                    [{warn.component_type}]{" "}
                    {resolveComponentName(warn.component_id, nodes)}:
                  </strong>{" "}
                  {warn.message}
                  {warn.suggestion && (
                    <div
                      style={{
                        color: "var(--color-text-muted)",
                        fontSize: 12,
                        marginTop: 2,
                      }}
                    >
                      Suggestion: {warn.suggestion}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
```

Also update the import to include `ValidationWarning`:

```typescript
import type {
  ValidationResult as ValidationResultType,
  ValidationWarning,
  NodeSpec,
} from "@/types/index";
```

- [ ] **Step 6: Run all tests**

Run: `cd src/elspeth/web/frontend && npx vitest run`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx \
  src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx \
  src/elspeth/web/frontend/src/components/execution/ValidationResult.tsx
git commit -m "feat(frontend): three-state pipeline status indicator (A7)

Pipeline status now distinguishes four states: not validated (○),
valid (✓), valid with warnings (⚠), and invalid (✗). The
validation banner also renders warnings when present on a
passing validation."
```

---

## Task 10: Final Integration Verification

- [ ] **Step 1: Run all tests**

Run: `cd src/elspeth/web/frontend && npx vitest run`
Expected: All tests pass.

- [ ] **Step 2: Run TypeScript type check**

Run: `cd src/elspeth/web/frontend && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 3: Build production bundle**

Run: `cd src/elspeth/web/frontend && npm run build`
Expected: Build succeeds with no errors.

- [ ] **Step 4: Verify no console errors in dev mode**

Run: `cd src/elspeth/web/frontend && npm run dev`
Open browser to the dev server URL. Verify:
- Chat panel renders markdown in assistant messages
- Mermaid diagrams render (if any mermaid code blocks exist)
- File manager shows categorized folders
- Secrets button appears in chat toolbar (key icon)
- Inspector panel no longer has the cog button
- Panel split is approximately 50/50 on first load
- Validation shows three-state indicator

- [ ] **Step 5: Commit final state if any fixes were needed**

Only commit if Step 4 revealed issues that required code changes.

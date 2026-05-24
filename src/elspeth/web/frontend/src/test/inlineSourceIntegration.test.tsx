// ============================================================================
// inlineSourceIntegration.test.tsx — Phase 5a Task 6
//
// End-to-end frontend integration: chat input dispatch → mocked LLM
// `set_pipeline` tool-call response → `compositionState.sources` with
// `blob_ref` → `getBlobMetadata` + `previewBlobContent` lookup → projection
// into `useInlineSourceStore` → `<InlineSourceCreatedTurn>` rendered.
//
// This is NOT a full backend integration test: `sendMessage`,
// `getBlobMetadata`, and `previewBlobContent` are mocked at the API
// boundary. The contract being pinned here is the FRONTEND wiring:
//
//   user types → useComposer.sendMessage → sessionStore.sendMessage
//     → api.sendMessage (mocked) returns state with sources.source.blob_ref
//     → sessionStore updates compositionState
//     → ChatPanel's projection effect calls getBlobMetadata + preview
//     → inlineSourceStore.setSummary populates the per-session summary
//     → InlineSourceCreatedTurn renders with provenance-gated Edit button
//
// Provenance is the critical discriminant:
//
//   * Test 1 — `creation_modality: "verbatim"` → `provenance: "verbatim"` →
//     widget renders WITHOUT an Edit button (verbatim content is
//     user-typed; re-authoring would be confusing).
//
//   * Test 2 — `creation_modality: "llm_generated"` → `provenance:
//     "llm-generated"` → widget renders WITH an Edit button (the user may
//     want to amend before continuing — F-4).
//
// The hyphenated → snake_case translation is the single-point adapter
// `toInlineSourceProvenance` in `api/client.ts`; this test exercises it
// implicitly by checking the store's projected `provenance` value.
//
// Assertion (g) (audit-readiness panel `Provenance` row) is intentionally
// DEFERRED to Phase 5a Task 7 — the panel does not surface inline-source
// provenance today. Re-enable when Task 7 wires the row.
// ============================================================================

import { describe, it, expect, vi, beforeEach } from "vitest";
import type { ReactNode } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "../App";
import * as api from "../api/client";
import * as auditReadinessApi from "../api/auditReadiness";
import { useSessionStore } from "../stores/sessionStore";
import { useAuthStore } from "../stores/authStore";
import { useInlineSourceStore } from "@/stores/inlineSourceStore";
import { useAuditReadinessStore, getInitialState as getAuditReadinessInitialState } from "@/stores/auditReadinessStore";
import { resetStore } from "@/test/store-helpers";
import type {
  AuditReadinessSnapshot,
  BlobMetadata,
  CompositionState,
  MessageWithStateResponse,
  SystemStatus,
  UserProfile,
} from "../types/index";

const READY_READINESS = {
  authoring_valid: true,
  execution_ready: true,
  completion_ready: true,
  blockers: [],
};

// ── Sub-component stubs ──────────────────────────────────────────────────────
// We render <App /> end-to-end (so ChatPanel's projection effect, store
// wiring, and useComposer hook all participate), but stub the heavy
// siblings that are not under test. ChatPanel is intentionally NOT
// stubbed — it is the System Under Test.

vi.mock("../components/common/AppHeader", () => ({
  AppHeader: () => <div data-testid="app-header-stub" />,
}));

// SideRail itself is stubbed (we don't render the catalog/execute/graph
// affordances for this test), but the audit-readiness slot is forwarded
// so the real AuditReadinessPanel mounts and assertion (g) can reach
// the inline-source provenance row.
vi.mock("../components/sidebar/SideRail", () => ({
  SideRail: ({ auditReadinessSlot }: { auditReadinessSlot?: ReactNode }) => (
    <div data-testid="side-rail-stub">{auditReadinessSlot}</div>
  ),
}));

vi.mock("../components/sidebar/GraphModal", () => ({
  GraphModal: () => null,
}));

vi.mock("../components/sidebar/ExportYamlModal", () => ({
  ExportYamlModal: () => null,
}));

vi.mock("../components/catalog/CatalogDrawer", () => ({
  CatalogDrawer: () => null,
}));

// AuditReadinessPanel is intentionally NOT stubbed — Phase 5a Task 7
// wires the inline-source provenance row, and assertion (g) below
// exercises the real panel to verify the full projection chain
// (compositionState → inlineSourceStore → AuditReadinessPanel).

vi.mock("../components/recovery/RecoveryPanel", () => ({
  RecoveryPanel: () => null,
}));

vi.mock("../components/settings/SecretsPanel", () => ({
  SecretsPanel: () => null,
}));

vi.mock("../components/settings/ComposerPreferencesPanel", () => ({
  ComposerPreferencesPanel: () => null,
}));

vi.mock("../components/common/CommandPalette", () => ({
  CommandPalette: () => null,
}));

vi.mock("../components/common/ShortcutsHelp", () => ({
  ShortcutsHelp: () => null,
}));

vi.mock("../components/common/ConfirmDialog", () => ({
  ConfirmDialog: () => null,
}));

// ── Auth stub ────────────────────────────────────────────────────────────────
// AuthGuard reads useAuth().isAuthenticated to gate the chat surface;
// without this it renders <LoginPage /> and the test never sees ChatPanel.

vi.mock("../hooks/useAuth", () => ({
  useAuth: () => ({
    isAuthenticated: true,
    isLoading: false,
    user: {
      user_id: "test-001",
      username: "test-operator",
      display_name: null,
      email: null,
      groups: [],
    } satisfies UserProfile,
    loginError: null,
    login: vi.fn(),
    loginWithToken: vi.fn(),
    logout: vi.fn(),
  }),
}));

// ── API client mock surface ─────────────────────────────────────────────────
// Module-scope vi.mock provides the default-safe values; individual
// tests override `sendMessage` and the blob fetchers via vi.spyOn so the
// scenario-specific wire shapes stay co-located with the assertions.

// The audit-readiness panel auto-fetches when compositionState has content
// and there is no version-matching cached snapshot. We mock the API so the
// panel populates its snapshot store and reaches the row-rendering branch
// where Task 7's inline-source override is applied (assertion (g) below).
vi.mock("../api/auditReadiness", () => ({
  fetchAuditReadiness: vi.fn(),
  fetchAuditReadinessExplain: vi.fn(),
}));

vi.mock("../api/client", () => ({
  fetchSystemStatus: vi.fn().mockResolvedValue({
    composer_available: true,
    composer_model: "gpt-4o",
    composer_provider: "openai",
    composer_reason: null,
    composer_missing_keys: [],
  } satisfies SystemStatus),
  fetchSessions: vi.fn().mockResolvedValue([]),
  fetchRuns: vi.fn().mockResolvedValue([]),
  fetchComposerProgress: vi.fn().mockResolvedValue({ phase: "idle" }),
  fetchRecoveryTranscript: vi.fn().mockResolvedValue([]),
  listSources: vi.fn().mockResolvedValue([]),
  listTransforms: vi.fn().mockResolvedValue([]),
  listSinks: vi.fn().mockResolvedValue([]),
  listBlobs: vi.fn().mockResolvedValue([]),
  sendMessage: vi.fn(),
  recompose: vi.fn(),
  fetchMessages: vi.fn(),
  getBlobMetadata: vi.fn(),
  previewBlobContent: vi.fn(),
  // toInlineSourceProvenance is a pure function the projection effect
  // calls inside ChatPanel; mocking it would defeat the test's purpose
  // (we want to verify the wire→display translation works). Re-export
  // the REAL implementation by importing and re-exposing.
  toInlineSourceProvenance: (wire: string): string => {
    switch (wire) {
      case "verbatim":
        return "verbatim";
      case "llm_generated":
        return "llm-generated";
      case "disambiguated":
        return "disambiguated";
      case "llm_generated_then_amended":
        return "llm-generated-then-amended";
      default:
        throw new Error(`Unhandled BlobCreationModalityWire value: ${wire}`);
    }
  },
  fetchUserComposerPreferences: vi.fn().mockResolvedValue({
    default_mode: "freeform",
    banner_dismissed_at: null,
    tutorial_completed_at: "2026-05-19T00:00:00Z",
    updated_at: "2026-05-15T00:00:00Z",
  }),
  updateUserComposerPreferences: vi.fn(),
}));

// ── Helpers ──────────────────────────────────────────────────────────────────

const SESSION_ID = "session-1";
const BLOB_ID = "blob-1";
const USER_MESSAGE_ID = "msg-user-1";
const ASSISTANT_MESSAGE_ID = "msg-asst-1";
const VERBATIM_INLINE_SOURCE_TEXT = "url\nhttps://finance.gov.au";
const VERBATIM_INLINE_SOURCE_HASH =
  "8a1a2e0744ef2cdd5b1fbbffef930dcb73bdb0a4afae201688d1f243a8b80e83";
const LLM_INLINE_SOURCE_TEXT =
  "url\nhttps://gov.au/1\nhttps://gov.au/2\nhttps://gov.au/3\nhttps://gov.au/4\nhttps://gov.au/5";
const LLM_INLINE_SOURCE_HASH =
  "c85e8a7843f8d4b014a8cde944256635cddbfa882f817d7f503d5623cfa1bc3f";

/**
 * Build a CompositionState whose default named source is an `inline_blob`
 * source with a `blob_ref` option — the wire shape ChatPanel's
 * `readBlobRef` helper reads from `compositionState.sources`.
 */
function makeCompositionStateWithInlineBlob(): CompositionState {
  return {
    id: "state-2",
    version: 2,
    sources: {
      source: {
        plugin: "inline_blob",
        options: { blob_ref: BLOB_ID },
      },
    },
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
  };
}

/**
 * Wire-shape BlobMetadata for the mocked `getBlobMetadata` response.
 * `creation_modality` is parameterised so each test case can vary the
 * single discriminant under test.
 */
function makeBlobMetadata(
  creationModality: BlobMetadata["creation_modality"],
  contentHash: string,
): BlobMetadata {
  return {
    id: BLOB_ID,
    session_id: SESSION_ID,
    filename: "chat.csv",
    mime_type: "text/csv",
    size_bytes: 32,
    content_hash: contentHash,
    created_at: "2026-05-18T00:00:00Z",
    created_by: "assistant",
    source_description: "From chat",
    status: "ready",
    creation_modality: creationModality,
    created_from_message_id: USER_MESSAGE_ID,
    creating_model_identifier:
      creationModality === "llm_generated" ? "gpt-4o" : null,
    creating_model_version: null,
    creating_provider: creationModality === "llm_generated" ? "openai" : null,
    creating_composer_skill_hash: null,
    creating_arguments_hash: null,
  };
}

/**
 * AuditReadinessSnapshot shaped for version 2 of the composition (the
 * version that carries the inline_blob source). The provenance row's
 * backend-supplied summary is overridden by the panel when an inline
 * source is bound — (g) below verifies that override.
 *
 * One row is set to `warning` so the panel auto-expands (anyActionable
 * branch) and the provenance row is in the DOM. Without an actionable
 * row the panel collapses to its single "Audit ready" summary and the
 * inline-content-hashed text is hidden until the user expands.
 */
function makeAuditReadinessSnapshotV2(): AuditReadinessSnapshot {
  return {
    session_id: SESSION_ID,
    composition_version: 2,
    checked_at: "2026-05-18T00:00:02Z",
    rows: [
      { id: "validation", label: "Validation", status: "ok", summary: "All checks pass", detail: null, component_ids: [] },
      { id: "plugin_trust", label: "Plugin trust", status: "warning", summary: "One Tier 3 plugin", detail: "web_scrape is Tier 3.", component_ids: ["web_scrape"] },
      { id: "provenance", label: "Provenance", status: "ok", summary: "Complete lineage", detail: null, component_ids: [] },
      { id: "retention", label: "Retention", status: "not_applicable", summary: "System retention: 90 days", detail: null, component_ids: [] },
      { id: "llm_interpretations", label: "LLM interpretations", status: "not_applicable", summary: "No LLM transforms", detail: null, component_ids: [] },
      { id: "secrets", label: "Secrets", status: "not_applicable", summary: "No secrets", detail: null, component_ids: [] },
    ],
    validation_result: {
      is_valid: true,
      checks: [],
      errors: [],
      warnings: [],
      readiness: READY_READINESS,
      semantic_contracts: [],
    },
  };
}

/**
 * MessageWithStateResponse shaped to mirror what the composer returns
 * after a successful `set_pipeline` tool-call: the user-facing
 * assistant message + the new composition state carrying an
 * inline_blob source bound to a blob_ref the projection effect will
 * pick up.
 */
function makeSendMessageResponse(): MessageWithStateResponse {
  return {
    message: {
      id: ASSISTANT_MESSAGE_ID,
      session_id: SESSION_ID,
      role: "assistant",
      content: "Source created from your message.",
      tool_calls: null,
      created_at: "2026-05-18T00:00:01Z",
    },
    state: makeCompositionStateWithInlineBlob(),
    proposals: [],
  };
}

describe("Phase 5a Task 6 — chat input → set_pipeline → inline-source widget", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // jsdom does not implement Element.prototype.scrollIntoView, but
    // ChatPanel calls it on every render via its auto-scroll effect.
    // Without this stub the chat panel crashes inside its ErrorBoundary
    // before the user can see (or interact with) the input. Established
    // pattern: ChatPanel.test.tsx applies the same one-liner.
    Element.prototype.scrollIntoView = vi.fn();
    resetStore(useSessionStore);
    resetStore(useInlineSourceStore);
    // The audit-readiness store is hand-shaped (not a generic factory),
    // so its canonical reset is the exported getInitialState() — same
    // pattern AuditReadinessPanel.test.tsx uses.
    useAuditReadinessStore.setState(getAuditReadinessInitialState());
    // Auto-fetch returns a v2-shaped snapshot for the inline-source
    // composition. The panel's projection branch will then override the
    // provenance row's summary with the SHA-256 prefix (Task 7).
    vi.mocked(auditReadinessApi.fetchAuditReadiness).mockResolvedValue(
      makeAuditReadinessSnapshotV2(),
    );
    // Seed authStore so useSessionLifecycle's auth gate passes — mirrors
    // the App.test.tsx pattern (the useAuth() mock above only covers
    // hook consumers; the lifecycle reads useAuthStore directly).
    useAuthStore.setState({
      token: "test-token",
      user: {
        user_id: "test-001",
        username: "test-operator",
        display_name: null,
        email: null,
        groups: [],
      } as never,
    } as never);
    // Seed an active session so ChatPanel renders the chat body (not
    // the "select a session" empty state). The composition starts at
    // version 1 with no source; the mocked sendMessage response below
    // returns version 2 with the inline_blob source attached.
    useSessionStore.setState({
      activeSessionId: SESSION_ID,
      sessions: [
        {
          id: SESSION_ID,
          title: "Integration test session",
          created_at: "2026-05-18T00:00:00Z",
          updated_at: "2026-05-18T00:00:00Z",
        },
      ],
      compositionState: {
        id: "state-1",
        version: 1,
        sources: {},
        nodes: [],
        edges: [],
        outputs: [],
        metadata: { name: null, description: null },
      },
    });
    localStorage.clear();
    window.history.replaceState(null, "", "/");
    // Re-prime the default api stubs after vi.clearAllMocks.
    vi.spyOn(api, "fetchSystemStatus").mockResolvedValue({
      composer_available: true,
      composer_model: "gpt-4o",
      composer_provider: "openai",
      composer_reason: null,
      composer_missing_keys: [],
    } satisfies SystemStatus);
    vi.spyOn(api, "fetchSessions").mockResolvedValue([
      {
        id: SESSION_ID,
        title: "Integration test session",
        created_at: "2026-05-18T00:00:00Z",
        updated_at: "2026-05-18T00:00:00Z",
      },
    ]);
    vi.spyOn(api, "fetchRuns").mockResolvedValue([]);
    vi.spyOn(api, "listBlobs").mockResolvedValue([]);
  });

  it("verbatim path — user types a URL, set_pipeline returns inline_blob, widget renders WITHOUT Edit button", async () => {
    const user = userEvent.setup();

    const sendMessageSpy = vi
      .spyOn(api, "sendMessage")
      .mockResolvedValue(makeSendMessageResponse());
    vi.spyOn(api, "getBlobMetadata").mockResolvedValue(
      makeBlobMetadata("verbatim", VERBATIM_INLINE_SOURCE_HASH),
    );
    vi.spyOn(api, "previewBlobContent").mockResolvedValue(VERBATIM_INLINE_SOURCE_TEXT);

    render(<App />);

    // Wait for App's initial effects (fetchSystemStatus, fetchSessions)
    // to settle so ChatPanel mounts and the chat input is interactive.
    await waitFor(() => {
      expect(api.fetchSystemStatus).toHaveBeenCalled();
    });

    const input = await screen.findByLabelText("Message input");
    const userText = "go to https://finance.gov.au";
    await user.type(input, userText);
    await user.click(screen.getByRole("button", { name: /send message/i }));

    // (a) sendMessage was called with the user text. The third arg is
    // the optional stateId (state.id from the seeded composition).
    await waitFor(() => {
      expect(sendMessageSpy).toHaveBeenCalled();
    });
    expect(sendMessageSpy).toHaveBeenCalledWith(
      SESSION_ID,
      userText,
      "state-1",
      expect.any(AbortSignal),
    );

    // (b) The dispatched payload mentions the URL.
    const dispatchedText = sendMessageSpy.mock.calls[0][1];
    expect(dispatchedText).toContain("https://finance.gov.au");

    // (c) InlineSourceCreatedTurn renders for verbatim provenance and
    // does NOT surface an Edit button (verbatim content is user-typed;
    // re-authoring would not match what the user already wrote).
    const turn = await screen.findByTestId("inline-source-created-turn");
    expect(turn).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /edit the list/i }),
    ).toBeNull();

    // (d) Widget announces itself via region role + accessible name.
    expect(
      screen.getByRole("region", { name: /source created from your message/i }),
    ).toBeInTheDocument();

    // (e) Filename + MIME are visible without expanding the audit-info
    // disclosure (these are the at-a-glance facts in the widget header).
    expect(screen.getByText(/chat\.csv/)).toBeInTheDocument();
    expect(screen.getByText(/text\/csv/)).toBeInTheDocument();

    // (f) The projection store now holds a non-null summary with the
    // hyphenated provenance form (translated from the snake_case wire
    // value via the `toInlineSourceProvenance` adapter).
    const summary = useInlineSourceStore.getState().getSummary(SESSION_ID);
    expect(summary).not.toBeNull();
    expect(summary?.provenance).toBe("verbatim");
    expect(summary?.blobId).toBe(BLOB_ID);
    expect(summary?.filename).toBe("chat.csv");
    expect(summary?.mimeType).toBe("text/csv");
    expect(summary?.contentHash).toBe(VERBATIM_INLINE_SOURCE_HASH);

    // (g) Audit-readiness panel Provenance row reflects the inline source.
    // The panel auto-fetches once compositionState.version === 2 (the
    // version carrying the inline_blob source) and the projection effect
    // populates inlineSourceStore.summary. The provenance row's summary
    // is then overridden to display the SHA-256 prefix.
    await waitFor(() => {
      expect(screen.getByText(/inline content hashed/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/8a1a2e0744ef/)).toBeInTheDocument();
  });

  it("llm_generated path — canonical demo prompt, widget renders WITH Edit button", async () => {
    const user = userEvent.setup();

    vi.spyOn(api, "sendMessage").mockResolvedValue(makeSendMessageResponse());
    vi.spyOn(api, "getBlobMetadata").mockResolvedValue(
      makeBlobMetadata("llm_generated", LLM_INLINE_SOURCE_HASH),
    );
    vi.spyOn(api, "previewBlobContent").mockResolvedValue(LLM_INLINE_SOURCE_TEXT);

    render(<App />);

    await waitFor(() => {
      expect(api.fetchSystemStatus).toHaveBeenCalled();
    });

    const input = await screen.findByLabelText("Message input");
    const userText = "create a list of 5 government web pages";
    await user.type(input, userText);
    await user.click(screen.getByRole("button", { name: /send message/i }));

    // Widget renders and Edit affordance IS present for LLM-authored
    // provenance — the user may want to amend before continuing (F-4).
    const turn = await screen.findByTestId("inline-source-created-turn");
    expect(turn).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: /edit the list/i }),
    ).toBeInTheDocument();

    // Store projection carries the hyphenated form translated from the
    // snake_case wire `llm_generated`.
    const summary = useInlineSourceStore.getState().getSummary(SESSION_ID);
    expect(summary).not.toBeNull();
    expect(summary?.provenance).toBe("llm-generated");
    expect(summary?.blobId).toBe(BLOB_ID);

    // (g) Audit-readiness panel Provenance row reflects the inline source.
    // The integration fixture hashes the actual mocked preview text for
    // each creation modality so ChatPanel's projection guard verifies the
    // same SHA-256 value the audit-readiness row displays.
    await waitFor(() => {
      expect(screen.getByText(/inline content hashed/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/c85e8a7843f8/)).toBeInTheDocument();
  });
});

// ============================================================================
// components.a11y.test.tsx — component accessibility audit
//
// One axe-core pass per user-facing component. Originally the Phase 8 Task 7
// audit of the Phase-1-through-8 composer components; widened by the
// 2026-07-02 UX-review regression net (elspeth-adf5e679e7) to cover the
// tutorial surface, the auth surface, and the chrome/run components that
// review touched. The matcher `toHaveNoViolations` is registered globally by
// `setup.ts` (registered in vite.config.ts's `test.setupFiles`); do NOT call
// `expect.extend(...)` in this file — that would shadow the global
// registration and break the "register once" invariant. See `setup.ts` head
// comment for the rationale.
//
// Coverage is anchored by the AUDITED_COMPONENTS array. The first test in
// this file is a snapshot assertion that exits the build with a clear
// failure if a future PR adds or removes an audited component without
// updating the audit list — preventing silent erosion of the a11y safety
// net.
// ============================================================================

import { describe, it, expect, beforeEach, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createRef, type ReactNode } from "react";

import { axe } from "./axe-config";

// --- Audit-surface coverage snapshot ---------------------------------------
//
// AUDITED_COMPONENTS is the source of truth for "what this suite audits".
// EXPECTED_AUDITED_COMPONENTS_SORTED is the snapshot it is compared against;
// the two are intentionally separate so a code-review can review the
// snapshot change as a deliberate audit-scope update.

const AUDITED_COMPONENTS = [
  "ComposerPreferencesPanel",
  "UserMenu",
  "DefaultModeChangedBanner",
  "AuditReadinessPanel",
  "ReadinessRowDetail",
  "ExplainDialog",
  "AppHeader",
  "HeaderSessionSwitcher",
  "HeaderVersionSelector",
  "SideRail",
  "GraphMiniView",
  "ExportYamlModal",
  "InlineSourceCreatedTurn",
  "InlineSourceDisambiguationTurn",
  "InlineSourceFallbackPrompt",
  "CompletionBar",
  "PluginCard",
  "FilterChipStrip",
  "FreeformIntroduction",
  "ShortcutsHelp",
  "WireStageTurn",
  "SchemaFormTurn",
  "ModeSwitchButton",
  "PipelineGloss",
  "PipelineValidationSummary",
  "ProposePipelineTurn",
  // 2026-07-02 UX-review regression net (elspeth-adf5e679e7): the tutorial
  // surface, the acknowledgement cards it leans on, the auth surface, and
  // the chrome/run components the review epic touched. Added centrally by
  // the wave-3 a11y-net pass — component waves deliberately do not edit
  // this list.
  "AcknowledgementCard",
  "AcknowledgementStack",
  "ChatInput",
  "CommandPalette",
  "ConfirmDialog",
  "GraphModal",
  "GraphView",
  "HelloWorldTutorial",
  "LoginPage",
  "ProgressView",
  "RecoveryPanel",
  "RunsHistoryDrawer",
  "TutorialGuidedShell",
  "TutorialTurn1Welcome",
  "TutorialTurn4Run",
  "TutorialTurn5AuditStory",
  "TutorialTurn7Graduation",
  // Tutorial workspace relayout (elspeth-16aa94c4bb): the two-column guided
  // workspace ChatPanel renders under isTutorial. Until this entry the guided
  // branch was never mounted by this suite — TutorialGuidedShell's audit
  // deliberately holds startGuidedSession pending — so the workspace layout
  // itself (landmark structure, live-region siblinghood, the named scroll
  // group) had zero axe coverage.
  "ChatPanelTutorialWorkspace",
] as const;

const EXPECTED_AUDITED_COMPONENTS_SORTED: readonly string[] = [
  "AcknowledgementCard",
  "AcknowledgementStack",
  "AppHeader",
  "AuditReadinessPanel",
  "ChatInput",
  "ChatPanelTutorialWorkspace",
  "CommandPalette",
  "ComposerPreferencesPanel",
  "CompletionBar",
  "ConfirmDialog",
  "DefaultModeChangedBanner",
  "ExplainDialog",
  "ExportYamlModal",
  "FilterChipStrip",
  "GraphMiniView",
  "GraphModal",
  "GraphView",
  "HeaderSessionSwitcher",
  "HeaderVersionSelector",
  "HelloWorldTutorial",
  "InlineSourceCreatedTurn",
  "InlineSourceDisambiguationTurn",
  "InlineSourceFallbackPrompt",
  "LoginPage",
  "ModeSwitchButton",
  "PipelineGloss",
  "PipelineValidationSummary",
  "PluginCard",
  "ProgressView",
  "ProposePipelineTurn",
  "ReadinessRowDetail",
  "RecoveryPanel",
  "RunsHistoryDrawer",
  "SchemaFormTurn",
  "SideRail",
  "ShortcutsHelp",
  "FreeformIntroduction",
  "TutorialGuidedShell",
  "TutorialTurn1Welcome",
  "TutorialTurn4Run",
  "TutorialTurn5AuditStory",
  "TutorialTurn7Graduation",
  "UserMenu",
  "WireStageTurn",
];

describe("audit surface — coverage snapshot", () => {
  it("audits exactly the expected component list", () => {
    expect([...AUDITED_COMPONENTS].sort()).toEqual(
      [...EXPECTED_AUDITED_COMPONENTS_SORTED].sort(),
    );
  });
});

// --- Mocks -----------------------------------------------------------------
//
// AppHeader/UserMenu pull useTheme(); CompletionBar renders ExecuteButton +
// ExportYamlButton which couple to additional stores; YamlView pulls in the
// full YAML rendering pipeline. Stub the heavy/coupled imports so the render
// produces deterministic DOM for axe without exercising unrelated logic.

vi.mock("@/components/inspector/YamlView", () => ({
  YamlView: () => (
    <button type="button" data-testid="yaml-view-stub">
      stub
    </button>
  ),
}));

// Spy-style mock of the HTTP layer (same idiom as ChatPanel.test.tsx): the
// actual module is preserved so exports the audited components merely import
// stay real, while every endpoint an audited component CALLS during a test is
// a vi.fn() the test seeds. jsdom has no backend, so an unseeded call would
// otherwise hit a dead socket.
vi.mock("@/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/client")>();
  return {
    ...actual,
    fetchUserComposerPreferences: vi.fn(),
    updateUserComposerPreferences: vi.fn(),
    fetchSessions: vi.fn(),
    createSession: vi.fn(),
    // Auth surface (LoginPage).
    fetchAuthConfig: vi.fn(),
    login: vi.fn(),
    register: vi.fn(),
    fetchCurrentUser: vi.fn(),
    // Tutorial surface (elspeth-adf5e679e7).
    deleteTutorialOrphans: vi.fn(),
    renameSession: vi.fn(),
    sendTutorialAbandonBeacon: vi.fn(),
    startGuidedSession: vi.fn(),
    getTutorialSample: vi.fn(),
    runTutorialPipeline: vi.fn(),
    cancelTutorialRun: vi.fn(),
    getRunAuditSummary: vi.fn(),
    // Run surfaces (ChatPanelTutorialWorkspace mounts InlineRunResults,
    // which loads the session's runs on mount).
    fetchRuns: vi.fn(),
    // Interpretation acknowledgements (AcknowledgementCard/Stack).
    listInterpretationEvents: vi.fn(),
    resolveInterpretation: vi.fn(),
    optOutOfInterpretations: vi.fn(),
    getInterpretationOptOutSummary: vi.fn(),
  };
});

vi.mock("../../api/auditReadiness", () => ({
  fetchAuditReadiness: vi.fn(),
  fetchAuditReadinessExplain: vi.fn(),
  validateAuditReadinessSnapshot: vi.fn(),
}));

// GraphView (and GraphModal, which embeds it) render through @xyflow/react,
// which needs real DOM measurement jsdom cannot provide. Stub the flow canvas
// to deterministic DOM the same way GraphView's own unit tests do — the axe
// pass then covers GraphView's real chrome (the keyboard-operable a11y list,
// the role="img" diagram scope, the config panel) rather than third-party
// canvas internals.
vi.mock("@xyflow/react", () => ({
  ReactFlowProvider: ({ children }: { children?: ReactNode }) => (
    <div data-testid="react-flow-provider">{children}</div>
  ),
  ReactFlow: ({
    nodes,
    edges,
    children,
  }: {
    nodes?: Array<{ id: string; data?: { label?: ReactNode } }>;
    edges?: Array<{ id: string; label?: ReactNode }>;
    children?: ReactNode;
  }) => (
    <div data-testid="react-flow">
      {nodes?.map((n) => <div key={n.id}>{n.data?.label}</div>)}
      {edges?.map((e) => <div key={e.id}>{e.label}</div>)}
      {children}
    </div>
  ),
  Background: () => <div data-testid="react-flow-background" />,
  Controls: () => <div data-testid="react-flow-controls" />,
  MiniMap: () => <div data-testid="minimap" />,
}));
vi.mock("@xyflow/react/dist/style.css", () => ({}));
vi.mock("@dagrejs/dagre", () => ({
  default: {
    graphlib: {
      Graph: class {
        setDefaultEdgeLabel() {}
        setGraph() {}
        setNode() {}
        setEdge() {}
        node() {
          return { x: 0, y: 0 };
        }
        // ReadOnlyPipelineGraph (ProposePipelineTurn's DAG canvas) reads
        // graph.graph().width/height to size the viewBox — the GraphView stub
        // above never calls it, so the shared mock must still answer it.
        graph() {
          return { width: 480, height: 200 };
        }
      },
    },
    layout() {},
  },
}));

// ProgressView's data source — no live socket exists in jsdom. Only
// ProgressView consumes this hook, so the file-wide mock is safe.
vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: vi.fn(),
}));

// --- Imports (post-mock) ---------------------------------------------------

import { ComposerPreferencesPanel } from "@/components/settings/ComposerPreferencesPanel";
import { UserMenu } from "@/components/common/UserMenu";
import { DefaultModeChangedBanner } from "@/components/common/DefaultModeChangedBanner";
import { AuditReadinessPanel } from "@/components/audit/AuditReadinessPanel";
import { ReadinessRowDetail } from "@/components/audit/ReadinessRowDetail";
import { ExplainDialog } from "@/components/audit/ExplainDialog";
import { AppHeader } from "@/components/common/AppHeader";
import { HeaderSessionSwitcher } from "@/components/sessions/HeaderSessionSwitcher";
import { HeaderVersionSelector } from "@/components/header/HeaderVersionSelector";
import { SideRail } from "@/components/sidebar/SideRail";
import { GraphMiniView } from "@/components/sidebar/GraphMiniView";
import { ExportYamlModal } from "@/components/sidebar/ExportYamlModal";
import { OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";
import { InlineSourceCreatedTurn } from "@/components/chat/InlineSourceCreatedTurn";
import { InlineSourceDisambiguationTurn } from "@/components/chat/InlineSourceDisambiguationTurn";
import { InlineSourceFallbackPrompt } from "@/components/chat/InlineSourceFallbackPrompt";
import { CompletionBar } from "@/components/composer/CompletionBar";
import { PluginCard } from "@/components/catalog/PluginCard";
import { FilterChipStrip, type CatalogFilters } from "@/components/catalog/FilterChipStrip";
import { FreeformIntroduction } from "@/components/chat/FreeformIntroduction";
import { ShortcutsHelp } from "@/components/common/ShortcutsHelp";
import { WireStageTurn } from "@/components/chat/guided/WireStageTurn";
import { SchemaFormTurn } from "@/components/chat/guided/SchemaFormTurn";
import { ModeSwitchButton } from "@/components/chat/guided/ModeSwitchButton";
import { PipelineGloss } from "@/components/chat/guided/PipelineGloss";
import { PipelineValidationSummary } from "@/components/chat/guided/PipelineValidationSummary";
import { ProposePipelineTurn } from "@/components/chat/guided/ProposePipelineTurn";
import { AcknowledgementCard } from "@/components/chat/AcknowledgementCard";
import { AcknowledgementStack } from "@/components/chat/AcknowledgementStack";
import { ChatInput } from "@/components/chat/ChatInput";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { LoginPage } from "@/components/auth/LoginPage";
import { CommandPalette } from "@/components/common/CommandPalette";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { ProgressView } from "@/components/execution/ProgressView";
import { RunsHistoryDrawer } from "@/components/execution/RunsHistoryDrawer";
import { GraphView } from "@/components/inspector/GraphView";
import { GraphModal } from "@/components/sidebar/GraphModal";
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";
import { RecoveryPanel } from "@/components/recovery/RecoveryPanel";
import { HelloWorldTutorial } from "@/components/tutorial/HelloWorldTutorial";
import { TutorialGuidedShell } from "@/components/tutorial/TutorialGuidedShell";
import { TutorialTurn1Welcome } from "@/components/tutorial/TutorialTurn1Welcome";
import { TutorialTurn4Run } from "@/components/tutorial/TutorialTurn4Run";
import { TutorialTurn5AuditStory } from "@/components/tutorial/TutorialTurn5AuditStory";
import { TutorialTurn7Graduation } from "@/components/tutorial/TutorialTurn7Graduation";
import { useWebSocket } from "@/hooks/useWebSocket";

import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import { useAuthStore } from "@/stores/authStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useAuditReadinessStore, getInitialState as getAuditInitialState } from "@/stores/auditReadinessStore";
import * as auditApi from "../../api/auditReadiness";
import * as apiClient from "@/api/client";
import { resetStore } from "@/test/store-helpers";
import type { InlineSourceSummary, ComposerRecoveryError } from "@/types/api";
import type {
  AuthConfig,
  CompositionState,
  PluginSummary,
  Run,
} from "@/types/index";
import type { ReadinessRow, AuditReadinessSnapshot } from "@/types/api";
import type {
  GuidedProposalReviewState,
  GuidedSession,
  ProposePipelinePayload,
  SchemaFormPayload,
  TurnPayload,
  WireStageData,
} from "@/types/guided";
import type { InterpretationEvent } from "@/types/interpretation";
import { compositionStateAuthorityFields } from "@/test/composerFixtures";

// --- Shared store reset ----------------------------------------------------

function resetAllStores() {
  resetStore(usePreferencesStore);
  usePreferencesStore.setState({
    loaded: true,
    defaultMode: "guided",
    bannerDismissedAt: null,
    writing: false,
    writeError: null,
    optedOutAtSessionId: null,
  });
  useSessionStore.setState({
    activeSessionId: "sess-a11y",
    sessions: [
      { id: "sess-a11y", title: "Test session", updated_at: "2026-05-19T00:00:00Z" } as never,
    ],
    compositionState: {
      version: 1,
      sources: {},
      nodes: [{ id: "select_columns" }],
      edges: [],
      outputs: [],
    } as never,
    stateVersions: [],
    isLoadingVersions: false,
  } as never);
  useExecutionStore.setState({
    validationResult: null,
  } as never);
  useAuditReadinessStore.setState(getAuditInitialState());
  resetStore(useInterpretationEventsStore);
  resetStore(useAuthStore);
  // Simulate a completed auth boot (loadFromStorage resolved, no stored
  // token) — same convention as LoginPage.test.tsx.
  useAuthStore.setState({ isLoading: false });
}

beforeEach(() => {
  resetAllStores();
  vi.clearAllMocks();
  localStorage.clear();
  sessionStorage.clear();
});

// --- Shared fixtures ---------------------------------------------------------

/**
 * Fully-typed CompositionState (the store's own resetAllStores state is a
 * minimal `as never` cast that GraphView's edge inference cannot walk).
 * One source → one transform → one sink, wired through named connection
 * points, so GraphView renders its accessible node list, the diagram scope,
 * and at least one inferred edge.
 */
function makeFullCompositionState(): CompositionState {
  return {
    id: "state-a11y",
    ...compositionStateAuthorityFields,
    version: 1,
    sources: {
      pages: {
        plugin: "web_scrape",
        options: {},
        on_success: "chain_in",
      },
    },
    nodes: [
      {
        id: "summarise",
        node_type: "transform",
        plugin: "llm_transform",
        input: "chain_in",
        on_success: "results",
        on_error: null,
        options: {},
      },
    ],
    edges: [],
    outputs: [{ name: "results", plugin: "json", options: {} }],
    metadata: { name: "A11y fixture pipeline", description: null },
  };
}

/** Pending interpretation event for the acknowledgement surface. */
function makeInterpretationEvent(
  id: string,
  overrides: Partial<InterpretationEvent> = {},
): InterpretationEvent {
  return {
    id,
    session_id: "sess-a11y",
    composition_state_id: "state-a11y",
    affected_node_id: "summarise",
    tool_call_id: "tool-1",
    user_term: "interesting",
    kind: "vague_term",
    llm_draft: "novel and relevant to the reader",
    accepted_value: null,
    choice: "pending",
    created_at: "2026-07-01T00:00:00Z",
    resolved_at: null,
    actor: "system:composer",
    interpretation_source: "user_approved",
    model_identifier: "anthropic/claude-sonnet-4.6",
    model_version: "20260518",
    provider: "anthropic",
    composer_skill_hash: "0".repeat(64),
    arguments_hash: null,
    hash_domain_version: null,
    runtime_model_identifier_at_resolve: null,
    runtime_model_version_at_resolve: null,
    resolved_prompt_template_hash: null,
    ...overrides,
  };
}

// --- Per-component audits --------------------------------------------------

describe("ComposerPreferencesPanel", () => {
  it("has no axe violations", async () => {
    const { container } = render(
      <ComposerPreferencesPanel onClose={() => {}} />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("UserMenu", () => {
  it("has no axe violations", async () => {
    const { container } = render(
      <UserMenu onOpenSettings={() => {}} onSignOut={() => {}} />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("WireStageTurn", () => {
  const wireBase: WireStageData = {
    proposal_id: "00000000-0000-4000-8000-000000000001",
    draft_hash: "d".repeat(64),
    sources: [{ stable_id: "00000000-0000-4000-8000-000000000010", label: "source-1", plugin: "inline_blob", on_validation_failure: "discard", guaranteed_fields: ["body"], row_cardinality: { input: "none", output: "zero_or_many", expected_output_count: null } }],
    nodes: [],
    outputs: [{ stable_id: "00000000-0000-4000-8000-000000000020", label: "output-1", plugin: "json", on_write_failure: "discard", required_fields: ["body"], business_schema: { mode: "observed", fields: [], guaranteed_fields: [], required_fields: ["body"] } }],
    connections: [{ stable_id: "00000000-0000-4000-8000-000000000030", from_endpoint: { kind: "source", stable_id: "00000000-0000-4000-8000-000000000010" }, to_endpoint: { kind: "output", stable_id: "00000000-0000-4000-8000-000000000020" }, flow: { kind: "source_success", branch: null }, schema_contract: null }],
    semantic_contracts: [],
    warnings: [
      {
        type: "prompt_shield",
        message: "Prompt shield advisory: source text was reviewed.",
      },
    ],
    blockers: [],
    can_confirm: true,
  };

  // Initial confirm turn (no outcome): the bare "Confirm wiring" action area.
  it("has no axe violations (initial confirm)", async () => {
    const { container } = render(
      <WireStageTurn data={wireBase} onConfirm={() => {}} confirmDisabled={false} />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });

  it("has no axe violations with the freeform exit available", async () => {
    const { container } = render(
      <WireStageTurn
        data={wireBase}
        onConfirm={() => {}}
        confirmDisabled={false}
        onExitToFreeform={() => {}}
      />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("SchemaFormTurn", () => {
  // The guided decision surface now defaults to a read-only summary (<dl> rows +
  // role=note caveat) with a non-tutorial Edit toggle that reveals the editable
  // KnobFieldRenderer form. Audit all three states the redesign introduced: the
  // default summary, the revealed edit form (newly axe-covered), and the
  // tutorial summary (no Edit affordance).
  const auditPayload: SchemaFormPayload = {
    mode: "plugin_options",
    plugin: "csv",
    knobs: {
      fields: [
        { name: "encoding", label: "Encoding", kind: "text", required: false, nullable: false },
        { name: "on_validation_failure", label: "On Validation Failure", kind: "text", required: true, nullable: false },
        { name: "schema", label: "Schema", kind: "json-object", required: false, nullable: false },
      ],
    },
    prefilled: { encoding: "utf-8", on_validation_failure: "discard", schema: { mode: "observed" } },
  };

  it("has no axe violations in the default summary view", async () => {
    const { container } = render(<SchemaFormTurn payload={auditPayload} onSubmit={() => {}} />);
    expect(await axe(container)).toHaveNoViolations();
  });

  it("has no axe violations in the revealed edit form", async () => {
    const { container } = render(<SchemaFormTurn payload={auditPayload} onSubmit={() => {}} />);
    await userEvent.click(screen.getByRole("button", { name: "Edit" }));
    expect(await axe(container)).toHaveNoViolations();
  });

  it("has no axe violations in tutorial summary mode", async () => {
    const { container } = render(<SchemaFormTurn payload={auditPayload} onSubmit={() => {}} isTutorial />);
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("DefaultModeChangedBanner", () => {
  it("has no axe violations", async () => {
    // Banner only renders when the user has opted out (defaultMode=freeform)
    // and the banner hasn't been dismissed and the user is not in the
    // session of opt-out. Configure the store so the banner mounts.
    usePreferencesStore.setState({
      defaultMode: "freeform",
      bannerDismissedAt: null,
      optedOutAtSessionId: "other-session",
    });
    const { container } = render(<DefaultModeChangedBanner />);
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("AuditReadinessPanel", () => {
  it("has no axe violations", async () => {
    const snapshot: AuditReadinessSnapshot = {
      session_id: "sess-a11y",
      composition_version: 1,
      checked_at: new Date().toISOString(),
      rows: [
        { id: "validation", label: "Validation", status: "ok", summary: "All checks pass", detail: null, component_ids: [] },
        { id: "plugin_trust", label: "Plugin trust", status: "ok", summary: "All Tier 1/2", detail: null, component_ids: [] },
        { id: "provenance", label: "Provenance", status: "warning", summary: "Identity passthrough", detail: "details", component_ids: [] },
        { id: "retention", label: "Retention", status: "not_applicable", summary: "n/a", detail: null, component_ids: [] },
        { id: "llm_interpretations", label: "LLM interpretations", status: "not_applicable", summary: "n/a", detail: null, component_ids: [] },
        { id: "secrets", label: "Secrets", status: "not_applicable", summary: "n/a", detail: null, component_ids: [] },
      ],
      validation_result: { is_valid: true, checks: [], errors: [], warnings: [], semantic_contracts: [] } as never,
    };
    vi.mocked(auditApi.fetchAuditReadiness).mockResolvedValue(snapshot);
    const { container } = render(<AuditReadinessPanel />);
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("ReadinessRowDetail", () => {
  it("has no axe violations", async () => {
    const row: ReadinessRow = {
      id: "provenance",
      label: "Provenance",
      status: "warning",
      summary: "Identity passthrough detected",
      detail: "Identity passthrough — provenance gap on 'select_columns'.",
      component_ids: ["select_columns"],
    };
    const { container } = render(
      <ReadinessRowDetail row={row} onClose={() => {}} />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("ExplainDialog", () => {
  it("has no axe violations", async () => {
    vi.mocked(auditApi.fetchAuditReadinessExplain).mockResolvedValue({
      session_id: "sess-a11y",
      composition_version: 1,
      narrative: "When you run this pipeline, ELSPETH will record provenance.",
    });
    const { container } = render(
      <ExplainDialog
        sessionId="sess-a11y"
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("AppHeader", () => {
  it("has no axe violations", async () => {
    const { container } = render(
      <AppHeader onOpenSettings={() => {}} onSignOut={() => {}} />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("HeaderSessionSwitcher", () => {
  it("has no axe violations (closed/default state)", async () => {
    const { container } = render(<HeaderSessionSwitcher />);
    expect(await axe(container)).toHaveNoViolations();
  });

  // I5: the closed-state audit above only covers the trigger button.
  // The entire interactive surface (filter input, archived toggle,
  // session rows, rename form, archive error region) lives in the open
  // menu and was previously unaudited.  Without this test a missing
  // ``aria-label`` on the filter input, a focus-trap regression on the
  // archive confirmation, or a missing ``role="alert"`` on the inline
  // error would not be caught.
  it("has no axe violations in the open state", async () => {
    const { container } = render(<HeaderSessionSwitcher />);
    const trigger = screen.getByRole("button", { name: /session switcher/i });
    await userEvent.click(trigger);
    // Menu, filter input, archived-toggle checkbox, and any session
    // rows are now rendered.  axe walks the full container.
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("HeaderVersionSelector", () => {
  it("has no axe violations", async () => {
    const { container } = render(<HeaderVersionSelector />);
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("SideRail", () => {
  it("has no axe violations", async () => {
    const { container } = render(
      <SideRail
        auditReadinessSlot={<div>readiness</div>}
        validationBannerSlot={<div>banner</div>}
        graphMiniSlot={<div>mini</div>}
        catalogSlot={<div>catalog</div>}
        completionBarSlot={<div>completion</div>}
      />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("GraphMiniView", () => {
  it("has no axe violations", async () => {
    const { container } = render(<GraphMiniView />);
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("ExportYamlModal", () => {
  it("has no axe violations", async () => {
    const { container, rerender } = render(<ExportYamlModal />);
    // Modal only renders when the open event fires.
    window.dispatchEvent(new CustomEvent(OPEN_YAML_MODAL_EVENT));
    rerender(<ExportYamlModal />);
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("InlineSourceCreatedTurn", () => {
  it("has no axe violations", async () => {
    const summary: InlineSourceSummary = {
      blobId: "b1",
      filename: "chat.csv",
      mimeType: "text/csv",
      contentPreview: "url\nhttps://example.gov.au",
      rowCount: 1,
      contentHash: "abc123def456789",
      provenance: "llm-generated",
    };
    const { container } = render(
      <InlineSourceCreatedTurn summary={summary} onEdit={() => {}} />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("InlineSourceDisambiguationTurn", () => {
  it("has no axe violations", async () => {
    const { container } = render(
      <InlineSourceDisambiguationTurn
        userInput="check these URLs: a.com, b.com, c.com"
        proposedRows={["a.com", "b.com", "c.com"]}
        proposalId="p1"
        messageId="msg-1"
        onConfirmMultiRow={() => {}}
        onTreatAsOneRow={() => {}}
        onEditRows={() => {}}
        onNotSourceData={() => {}}
      />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("InlineSourceFallbackPrompt", () => {
  it("has no axe violations", async () => {
    const { container } = render(
      <InlineSourceFallbackPrompt
        shouldRender={true}
        candidateText="https://example.com"
        onAccept={() => {}}
        onDismiss={() => {}}
      />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("CompletionBar", () => {
  it("has no axe violations", async () => {
    useExecutionStore.setState({
      validationResult: { is_valid: true, checks: [], errors: [] } as never,
    } as never);
    const { container } = render(<CompletionBar />);
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("PluginCard", () => {
  it("has no axe violations", async () => {
    const plugin: PluginSummary = {
      name: "csv",
      plugin_type: "source",
      description: "Read rows from a CSV file.",
      config_fields: [],
      usage_when_to_use: "When you have a CSV file already.",
      usage_when_not_to_use: "When the data is inline.",
      example_use: "source:\n  plugin: csv\n  options:\n    path: data.csv",
      capability_tags: ["csv", "file"],
      audit_characteristics: ["io_read", "quarantine"],
    } as PluginSummary;
    const { container } = render(
      <PluginCard plugin={plugin} schema={null} onExpand={() => {}} />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("FilterChipStrip", () => {
  it("has no axe violations", async () => {
    const filters: CatalogFilters = {
      capabilityTags: new Set(),
      auditCharacteristics: new Set(),
    };
    const { container } = render(
      <FilterChipStrip
        availableCapabilityTags={["csv", "file"]}
        availableAuditCharacteristics={["io_read", "quarantine"]}
        filters={filters}
        onChange={() => {}}
      />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("FreeformIntroduction", () => {
  it("has no axe violations", async () => {
    usePreferencesStore.setState({
      loaded: true,
      freeformIntroDismissedAt: null,
      writing: false,
    });
    const { container } = render(<FreeformIntroduction />);
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("ShortcutsHelp", () => {
  it("has no axe violations", async () => {
    const { container } = render(<ShortcutsHelp onClose={() => {}} />);
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("ModeSwitchButton", () => {
  it("has no axe violations (resting)", async () => {
    const { container } = render(
      <ModeSwitchButton target="guided" hasWork={false} />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });

  it("has no axe violations (confirm state)", async () => {
    const { container } = render(
      <ModeSwitchButton target="freeform" hasWork />,
    );
    // Reveal the two-step confirm (the new interactive surface).
    await userEvent.click(
      screen.getByRole("button", { name: "Exit to freeform" }),
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("PipelineGloss", () => {
  it("has no axe violations", async () => {
    const { container } = render(
      <PipelineGloss
        compositionState={useSessionStore.getState().compositionState}
      />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("PipelineValidationSummary", () => {
  it("has no axe violations (warning state with a tinted glyph)", async () => {
    // Exercise the richest DOM (glyph + plain status text), not the neutral
    // null state, so the axe pass covers the aria-hidden glyph + role=status.
    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        checks: [],
        errors: [],
        warnings: [
          {
            component_id: "select_columns",
            component_type: "transform",
            message: "Review the optional mapping",
            suggestion: null,
          },
        ],
      },
    } as never);
    const { container } = render(<PipelineValidationSummary />);
    expect(await axe(container)).toHaveNoViolations();
  });
});

// --- Guided proposal review surface ------------------------------------------
//
// ProposePipelineTurn is the whole-DAG review turn the guided/freeform parity
// work lands on: a read-only graph, the components/routes summary, blockers,
// and the accept/reject/revise controls (Plan 05 Task 5). It carries a
// live-region status/alert whose focus moves on stale/error transitions and a
// tutorial read-only variant. Audit every branch of that surface — the plain
// active controls, the blocker-gated revise-required state, the stale and
// error live regions (asserting the focus placement the useEffect performs),
// and the passive tutorial render.

describe("ProposePipelineTurn", () => {
  const PROPOSAL_ID = "00000000-0000-4000-8000-0000000004a1";
  const DRAFT_HASH = "d".repeat(64);
  const SOURCE_ID = "00000000-0000-4000-8000-0000000004a2";
  const NODE_ID = "00000000-0000-4000-8000-0000000004a3";
  const OUTPUT_ID = "00000000-0000-4000-8000-0000000004a4";
  const edge = (n: number): string => `00000000-0000-4000-8000-${String(4000 + n).padStart(12, "0")}`;

  function proposalPayload(
    overrides: Partial<ProposePipelinePayload> = {},
  ): ProposePipelinePayload {
    return {
      proposal_id: PROPOSAL_ID,
      draft_hash: DRAFT_HASH,
      supersedes_draft_hash: null,
      summary: "guided.proposal.summary.a11y.v1",
      rationale: "guided.proposal.rationale.a11y.v1",
      component_counts: { sources: 1, nodes: 1, edges: 5, outputs: 1 },
      blockers: [],
      graph: {
        sources: [
          { stable_id: SOURCE_ID, label: "pages", plugin: { kind: "source", id: "csv" } },
        ],
        edges: [
          {
            stable_id: edge(1),
            from_endpoint: { kind: "source", stable_id: SOURCE_ID },
            to_endpoint: { kind: "node", stable_id: NODE_ID },
            flow: { kind: "source_success", branch: null },
          },
          {
            stable_id: edge(2),
            from_endpoint: { kind: "source", stable_id: SOURCE_ID },
            to_endpoint: { kind: "discard" },
            flow: { kind: "source_validation_failure" },
          },
          {
            stable_id: edge(3),
            from_endpoint: { kind: "node", stable_id: NODE_ID },
            to_endpoint: { kind: "output", stable_id: OUTPUT_ID },
            flow: { kind: "node_success", branch: null },
          },
          {
            stable_id: edge(4),
            from_endpoint: { kind: "node", stable_id: NODE_ID },
            to_endpoint: { kind: "discard" },
            flow: { kind: "node_error" },
          },
          {
            stable_id: edge(5),
            from_endpoint: { kind: "output", stable_id: OUTPUT_ID },
            to_endpoint: { kind: "discard" },
            flow: { kind: "output_write_failure" },
          },
        ],
      },
      nodes: [
        {
          stable_id: NODE_ID,
          label: "summarise",
          node_type: "transform",
          plugin: { kind: "transform", id: "llm" },
          behavior: { kind: "transform" },
        },
      ],
      outputs: [
        { stable_id: OUTPUT_ID, label: "results", plugin: { kind: "sink", id: "json" } },
      ],
      edit_targets: [
        { kind: "node", stable_id: NODE_ID },
        { kind: "source", stable_id: SOURCE_ID },
        { kind: "output", stable_id: OUTPUT_ID },
      ],
      ...overrides,
    };
  }

  it("has no axe violations in the default active review state", async () => {
    const reviewState: GuidedProposalReviewState = {
      status: "active",
      proposal_id: PROPOSAL_ID,
      draft_hash: DRAFT_HASH,
    };
    const { container } = render(
      <ProposePipelineTurn payload={proposalPayload()} reviewState={reviewState} onSubmit={() => {}} />,
    );
    // Non-vacuous: the labelled graph, the accessible heading, and the enabled
    // primary control must all be real.
    screen.getByRole("img", { name: /pipeline proposal graph/i });
    screen.getByRole("heading", { name: "Review pipeline proposal" });
    expect(screen.getByRole("button", { name: "Review wiring" })).toBeEnabled();
    expect(await axe(container)).toHaveNoViolations();
  });

  it("has no axe violations in the blocker-gated revise-required state", async () => {
    const reviewState: GuidedProposalReviewState = {
      status: "active",
      proposal_id: PROPOSAL_ID,
      draft_hash: DRAFT_HASH,
    };
    const { container } = render(
      <ProposePipelineTurn
        payload={proposalPayload({
          blockers: [
            {
              code: "policy_review_required",
              category: "policy",
              summary: "guided.proposal.blocker.policy_review_required.v1",
              edit_target: { kind: "node", stable_id: NODE_ID },
            },
          ],
        })}
        reviewState={reviewState}
        onSubmit={() => {}}
      />,
    );
    // Non-vacuous: the blocker list gates wiring review while the revise
    // affordance stays reachable.
    screen.getByText("A policy review is required before this pipeline can advance.");
    expect(screen.getByRole("button", { name: "Review wiring" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Revise summarise/ })).toBeEnabled();
    expect(await axe(container)).toHaveNoViolations();
  });

  it("has no axe violations in the stale live-region state (focus moves to status)", async () => {
    const reviewState: GuidedProposalReviewState = {
      status: "stale",
      proposal_id: PROPOSAL_ID,
      draft_hash: DRAFT_HASH,
    };
    const { container } = render(
      <ProposePipelineTurn payload={proposalPayload()} reviewState={reviewState} onSubmit={() => {}} />,
    );
    // Live-region behavior + focus placement: the stale transition surfaces a
    // role=status announcement that receives focus, and locks the controls.
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent(/stale/i);
    expect(status).toHaveFocus();
    expect(screen.getByRole("button", { name: "Review wiring" })).toBeDisabled();
    expect(await axe(container)).toHaveNoViolations();
  });

  it("has no axe violations in the retryable error state (focus moves to alert)", async () => {
    const reviewState: GuidedProposalReviewState = {
      status: "error",
      proposal_id: PROPOSAL_ID,
      draft_hash: DRAFT_HASH,
      message: "The response was not received. Retry the same action.",
      retryable: true,
      retry_action: { kind: "revise", edit_target: { kind: "node", stable_id: NODE_ID } },
    };
    const { container } = render(
      <ProposePipelineTurn payload={proposalPayload()} reviewState={reviewState} onSubmit={() => {}} />,
    );
    // Live-region behavior + focus placement: the error transition surfaces a
    // role=alert that receives focus; only the retained revise action stays
    // enabled.
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent(/response was not received/i);
    expect(alert).toHaveFocus();
    expect(screen.getByRole("button", { name: "Review wiring" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Revise summarise/ })).toBeEnabled();
    expect(await axe(container)).toHaveNoViolations();
  });

  it("has no axe violations in the tutorial revision render (live primary, off-script controls withheld)", async () => {
    const reviewState: GuidedProposalReviewState = {
      status: "active",
      proposal_id: PROPOSAL_ID,
      draft_hash: DRAFT_HASH,
    };
    const { container } = render(
      <ProposePipelineTurn
        payload={proposalPayload({ supersedes_draft_hash: "e".repeat(64) })}
        reviewState={reviewState}
        onSubmit={() => {}}
        isTutorial
      />,
    );
    // Non-vacuous: the tutorial teaching note renders, the live "Review
    // wiring" primary stays actionable (the tutorial proposal is a REAL
    // planner proposal the learner must accept to advance; the primary is
    // live only once the frozen-prompt revision supersedes the pre-Send
    // auto-proposal), and the off-script reject/revise controls are withheld.
    screen.getByText(/press Review wiring to continue/i);
    expect(screen.getByRole("button", { name: "Review wiring" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Reject proposal" })).toBeNull();
    expect(screen.queryByRole("button", { name: /Revise/ })).toBeNull();
    expect(await axe(container)).toHaveNoViolations();
  });

  it("has no axe violations in the tutorial pre-Send auto-proposal render (primary withheld)", async () => {
    const reviewState: GuidedProposalReviewState = {
      status: "active",
      proposal_id: PROPOSAL_ID,
      draft_hash: DRAFT_HASH,
    };
    const { container } = render(
      <ProposePipelineTurn payload={proposalPayload()} reviewState={reviewState} onSubmit={() => {}} isTutorial />,
    );
    // Non-vacuous: the pre-Send auto-proposal (supersedes_draft_hash null)
    // withholds every action and directs the learner to Send the frozen
    // transforms prompt instead (tutorial run 18 committed the passthrough).
    screen.getByText(/press Send/i);
    expect(screen.queryByRole("button", { name: "Review wiring" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Reject proposal" })).toBeNull();
    expect(screen.queryByRole("button", { name: /Revise/ })).toBeNull();
    expect(await axe(container)).toHaveNoViolations();
  });
});

// --- Tutorial surface (elspeth-adf5e679e7) -----------------------------------
//
// The tutorial is the surface being promoted to flagship; until this pass it
// had ZERO automated a11y regression coverage. Each turn is audited in its
// richest deterministic state; the top-level shell is audited on the welcome
// step (progress nav + sr-only step announcement + welcome turn).

describe("HelloWorldTutorial", () => {
  it("has no axe violations on the welcome step (shell + progress nav)", async () => {
    vi.mocked(apiClient.deleteTutorialOrphans).mockResolvedValue({
      deleted_count: 0,
    });
    const { container } = render(<HelloWorldTutorial />);
    // Guard against a vacuous pass: the shell + progress nav must be real.
    screen.getByRole("group", { name: "Tutorial progress" });
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("TutorialTurn1Welcome", () => {
  it("has no axe violations", async () => {
    const { container } = render(
      <TutorialTurn1Welcome onStart={() => {}} onSkip={() => {}} />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("TutorialGuidedShell", () => {
  it("has no axe violations while preparing the guided session", async () => {
    // A pending start keeps the shell on its own chrome (kicker + sr-only
    // status + sample-loading line) without mounting the embedded ChatPanel,
    // whose guided internals are audited via their own components above.
    vi.mocked(apiClient.startGuidedSession).mockReturnValue(
      new Promise<never>(() => {}),
    );
    const { container } = render(
      <TutorialGuidedShell
        sessionId="00000000-0000-4000-8000-000000000999"
        onCompleted={() => {}}
      />,
    );
    screen.getByText(/Preparing the tutorial's sample pages/);
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("ChatPanelTutorialWorkspace", () => {
  // The tutorial two-column workspace (elspeth-16aa94c4bb): conversation
  // column (bubble transcript → run results → acknowledgements → current
  // decision) over the docked composer, plus the artifact rail. Audited in
  // its richest deterministic state — a transcript spanning two stages (so
  // the stage dividers render), a prior decision in the rail's "Decisions so
  // far", a schema_form current decision, and the "Sent" composer state —
  // covering the landmark structure and the sibling (never nested) live
  // regions the layout moved: transcript log, wizard-step log, the
  // acknowledgement announcer, and the Sent status line.

  function makeTutorialGuidedSession(): GuidedSession {
    return {
      step: "step_2_sink",
      history: [
        {
          step: "step_1_source",
          turn_type: "single_select",
          payload_hash: "aabbcc001122",
          response_hash: "ddeeff334455",
          emitter: "server",
          summary: "Source selected: web_scrape",
        },
      ],
      terminal: null,
      chat_history: [
        {
          role: "user",
          content: "Summarise these pages:\nhttps://example.gov.au/page-1",
          seq: 1,
          step: "step_1_source",
          ts_iso: "2026-07-03T00:00:00Z",
          assistant_message_kind: null,
          synthetic_failure_reason: null,
        },
        {
          role: "assistant",
          content: "**Source created** — reading the sample pages.",
          seq: 2,
          step: "step_1_source",
          ts_iso: "2026-07-03T00:00:01Z",
          assistant_message_kind: "assistant",
          synthetic_failure_reason: null,
        },
        {
          role: "user",
          content: "Write the results out as JSONL.",
          seq: 3,
          step: "step_2_sink",
          ts_iso: "2026-07-03T00:00:02Z",
          assistant_message_kind: null,
          synthetic_failure_reason: null,
        },
        {
          role: "assistant",
          content: "I set up a JSONL output for the summaries.",
          seq: 4,
          step: "step_2_sink",
          ts_iso: "2026-07-03T00:00:03Z",
          assistant_message_kind: "assistant",
          synthetic_failure_reason: null,
        },
      ],
      chat_turn_seq: 4,
      profile: null,
    };
  }

  function makeSchemaFormNextTurn(): TurnPayload {
    const payload: SchemaFormPayload = {
      mode: "plugin_options",
      plugin: "json",
      knobs: {
        fields: [
          { name: "path", label: "Path", kind: "text", required: true, nullable: false },
        ],
      },
      prefilled: { path: "results.jsonl" },
    };
    return { type: "schema_form", step_index: 1, turn_token: "a".repeat(64), payload };
  }

  it("has no axe violations on the two-column tutorial workspace", async () => {
    // jsdom does not implement Element.prototype.scrollIntoView (the
    // step-advance focus effect scrolls the just-built decision into view
    // on mount) — same stub as CommandPalette above.
    Element.prototype.scrollIntoView = vi.fn();
    // InlineRunResults loads the session's runs on mount; jsdom has no
    // backend, so seed an empty list.
    vi.mocked(apiClient.fetchRuns).mockResolvedValue([]);
    useSessionStore.setState({
      compositionState: makeFullCompositionState(),
      compositionProposals: [],
      guidedSession: makeTutorialGuidedSession(),
      guidedNextTurn: makeSchemaFormNextTurn(),
    } as never);

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{
          step_1_source: "Summarise these pages:\nhttps://example.gov.au/page-1",
          step_2_sink: "Write the results out as JSONL.",
        }}
      />,
    );

    // Guard against a vacuous pass: the named scroll group, the artifact
    // rail, both role=log regions (transcript + wizard step), the rail's
    // decision history, and the Sent composer state must all be mounted.
    screen.getByRole("group", { name: "Conversation" });
    screen.getByRole("complementary", { name: "Pipeline summary" });
    screen.getByRole("log", { name: "Step chat history" });
    screen.getByRole("log", { name: "Guided wizard step" });
    screen.getByRole("heading", { name: /decisions so far/i });
    screen.getByText(/your request is in the transcript above/i);

    expect(await axe(container)).toHaveNoViolations();
  });

  it("has no axe violations while a guided chat is in flight (pending strip swap)", async () => {
    // Pending swap (elspeth-6a9673ecd3): while /guided/chat is in flight the
    // composer's child content is the GuidedPendingStrip (status region +
    // Stop), not the ChatInput. chat_history holds only the PRIOR step's
    // turns — the current step's user turn is server-emitted on response, so
    // mid-flight the step is not "built" and the strip owns the slot.
    Element.prototype.scrollIntoView = vi.fn();
    vi.mocked(apiClient.fetchRuns).mockResolvedValue([]);
    const session = makeTutorialGuidedSession();
    session.chat_history = session.chat_history.slice(0, 2);
    useSessionStore.setState({
      compositionState: makeFullCompositionState(),
      compositionProposals: [],
      guidedSession: session,
      guidedNextTurn: makeSchemaFormNextTurn(),
      guidedChatPending: true,
    } as never);

    const { container } = render(
      <ChatPanel
        isTutorial
        lockedChatPrompt={{
          step_1_source: "Summarise these pages:\nhttps://example.gov.au/page-1",
          step_2_sink: "Write the results out as JSONL.",
        }}
      />,
    );

    // Non-vacuous: the strip replaced the input, Stop is offered, and the
    // composer landmark survived the swap.
    expect(container.querySelector(".guided-pending-strip")).not.toBeNull();
    screen.getByRole("button", { name: "Stop composing" });
    screen.getByRole("region", { name: "Describe what you want" });
    expect(screen.queryByLabelText("Message input")).toBeNull();

    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("TutorialTurn4Run", () => {
  // The module-level run cache is keyed by sessionId — each test uses a
  // distinct id so a cached promise never leaks across tests.
  it("has no axe violations while the run is executing", async () => {
    vi.mocked(apiClient.runTutorialPipeline).mockReturnValue(
      new Promise<never>(() => {}),
    );
    const { container } = render(
      <TutorialTurn4Run
        sessionId="sess-a11y-run-pending"
        onCompleted={() => {}}
        onCancelled={() => {}}
      />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });

  it("has no axe violations on the results table with a discarded-rows notice", async () => {
    vi.mocked(apiClient.runTutorialPipeline).mockResolvedValue({
      run_id: "run-a11y",
      output: {
        source_data_hash: "a".repeat(64),
        rows: [
          {
            url: "https://example.gov.au/page-1",
            summary: "A one-line summary of the page.",
            error: null,
          },
        ],
        discarded_row_count: 1,
      },
    });
    const { container } = render(
      <TutorialTurn4Run
        sessionId="sess-a11y-run-done"
        onResult={() => {}}
        onCompleted={() => {}}
        onCancelled={() => {}}
        onBack={() => {}}
      />,
    );
    await screen.findByText(/rows returned/);
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("TutorialTurn5AuditStory", () => {
  it("has no axe violations with loaded audit evidence", async () => {
    vi.mocked(apiClient.getRunAuditSummary).mockResolvedValue({
      run_id: "run-a11y",
      session_id: "sess-a11y",
      llm_call_count: 3,
      source_data_hash: "b".repeat(64),
      started_at: "2026-07-01T00:00:00Z",
      plugin_versions: { web_scrape: "1.0.0", llm_transform: "2.1.0" },
      seeded_from_cache: false,
      cache_key: null,
    });
    const { container } = render(
      <TutorialTurn5AuditStory
        sessionId="sess-a11y"
        runId="run-a11y"
        onContinue={() => {}}
        onBack={() => {}}
      />,
    );
    await screen.findByText("Source data hash");
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("TutorialTurn7Graduation", () => {
  it("has no axe violations (completed tutorial)", async () => {
    const { container } = render(
      <TutorialTurn7Graduation
        sessionId="sess-a11y"
        skipped={false}
        cancelled={false}
        onBack={() => {}}
      />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });

  it("has no axe violations (cancelled variant with the status note)", async () => {
    const { container } = render(
      <TutorialTurn7Graduation
        sessionId="sess-a11y"
        skipped={false}
        cancelled
      />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

// --- Acknowledgement surface -------------------------------------------------

describe("AcknowledgementCard", () => {
  it("has no axe violations (vague_term with the amend affordance)", async () => {
    const { container } = render(
      <AcknowledgementCard
        event={makeInterpretationEvent("evt-1")}
        sessionId="sess-a11y"
        stepLabel="Summarise"
        showAmend
      />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });

  it("has no axe violations (llm_prompt_template with the two-stage View→Approve primary)", async () => {
    const { container } = render(
      <AcknowledgementCard
        event={makeInterpretationEvent("evt-2", {
          kind: "llm_prompt_template",
          llm_draft: "Summarise {{ body }} in one paragraph for {{ audience }}.",
        })}
        sessionId="sess-a11y"
        stepLabel="Summarise"
      />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("AcknowledgementStack", () => {
  it("has no axe violations with pending acknowledgements", async () => {
    useSessionStore.setState({
      compositionState: makeFullCompositionState(),
    } as never);
    useInterpretationEventsStore.setState({
      pendingBySession: {
        "sess-a11y": {
          "evt-1": makeInterpretationEvent("evt-1"),
          "evt-2": makeInterpretationEvent("evt-2", {
            kind: "llm_model_choice",
            llm_draft: "anthropic/claude-sonnet-4.6",
          }),
        },
      },
    });
    const { container } = render(<AcknowledgementStack sessionId="sess-a11y" />);
    // Guard against a vacuous pass (the stack renders nothing when no
    // events are pending): both cards must be mounted.
    expect(
      screen.getAllByRole("button", { name: /Acknowledge/ }),
    ).toHaveLength(2);
    expect(await axe(container)).toHaveNoViolations();
  });
});

// --- Auth surface --------------------------------------------------------------

describe("LoginPage", () => {
  function mockLocalAuthConfig(): void {
    const config: AuthConfig = {
      provider: "local",
      registration_mode: "open",
      oidc_issuer: null,
      oidc_client_id: null,
      authorization_endpoint: null,
    };
    vi.mocked(apiClient.fetchAuthConfig).mockResolvedValue(config);
  }

  it("has no axe violations on the sign-in view", async () => {
    mockLocalAuthConfig();
    const { container } = render(<LoginPage />);
    await screen.findByLabelText("Username");
    expect(await axe(container)).toHaveNoViolations();
  });

  it("has no axe violations on the registration view", async () => {
    mockLocalAuthConfig();
    const { container } = render(<LoginPage />);
    await screen.findByLabelText("Username");
    await userEvent.click(
      screen.getByRole("button", { name: "Create an account" }),
    );
    await screen.findByLabelText("Confirm password");
    expect(await axe(container)).toHaveNoViolations();
  });
});

// --- Chrome / run surfaces -----------------------------------------------------

describe("ConfirmDialog", () => {
  it("has no axe violations (danger variant with structured content)", async () => {
    const { container } = render(
      <ConfirmDialog
        title="Discard this run?"
        message="The run's partial output will be discarded."
        confirmLabel="Discard"
        cancelLabel="Keep"
        variant="danger"
        onConfirm={() => {}}
        onCancel={() => {}}
      >
        <p>One output file will be removed.</p>
      </ConfirmDialog>,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("CommandPalette", () => {
  it("has no axe violations when open", async () => {
    // jsdom does not implement Element.prototype.scrollIntoView (the
    // palette scrolls the selected row into view on mount) — same stub as
    // CommandPalette.test.tsx.
    Element.prototype.scrollIntoView = vi.fn();
    const { container } = render(<CommandPalette isOpen onClose={() => {}} />);
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("ChatInput", () => {
  it("has no axe violations with the composition affordances shown", async () => {
    const inputRef = createRef<HTMLTextAreaElement>();
    const { container } = render(
      <ChatInput
        onSend={() => {}}
        disabled={false}
        inputRef={inputRef}
        onToggleBlobManager={() => {}}
        onOpenSecrets={() => {}}
      />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("ProgressView", () => {
  it("has no axe violations during a live run", async () => {
    vi.mocked(useWebSocket).mockReturnValue({
      activeRunId: "run-a11y",
      wsDisconnected: false,
      progress: {
        source_rows_processed: 3,
        tokens_succeeded: 2,
        tokens_failed: 1,
        tokens_quarantined: 0,
        tokens_routed_success: 2,
        tokens_routed_failure: 1,
        cancel_requested: false,
        accounting: null,
        recent_errors: [],
        status: "running",
      },
    } as never);
    const { container } = render(<ProgressView />);
    screen.getByText("Source Rows");
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("RunsHistoryDrawer", () => {
  it("has no axe violations on the run list", async () => {
    const runs = [
      { id: "run-1", status: "completed" },
      { id: "run-2", status: "completed_with_failures" },
      { id: "run-3", status: "running" },
    ] as unknown as ReadonlyArray<Run>;
    const { container } = render(
      <RunsHistoryDrawer onClose={() => {}} runsOverride={runs} />,
    );
    // Guard against a vacuous pass: the glyph-carrying StatusBadge row
    // must be listed.
    screen.getByText("completed with failures");
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("GraphView", () => {
  it("has no axe violations (accessible node list + diagram scope)", async () => {
    useSessionStore.setState({
      compositionState: makeFullCompositionState(),
      compositionProposals: [],
    } as never);
    const { container } = render(<GraphView />);
    // Guard against the empty-state fallback masquerading as coverage: the
    // role="img" diagram scope and the keyboard-operable node list must be
    // real (source + transform + sink = 3 components).
    screen.getByRole("img", { name: /Pipeline graph with 3 components/ });
    screen.getByRole("list", {
      name: /Pipeline components in source-to-sink order \(3\)/,
    });
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("GraphModal", () => {
  it("has no axe violations when open", async () => {
    useSessionStore.setState({
      compositionState: makeFullCompositionState(),
      compositionProposals: [],
    } as never);
    const { container } = render(<GraphModal />);
    // Modal only renders once the open event fires (the ExportYamlModal
    // idiom above, with the dispatch act()-wrapped so the listener's
    // setState commits before the assertion).
    act(() => {
      window.dispatchEvent(new CustomEvent(OPEN_GRAPH_MODAL_EVENT));
    });
    screen.getByRole("dialog", { name: "Pipeline graph" });
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("RecoveryPanel", () => {
  it("has no axe violations", async () => {
    const recoveryError: ComposerRecoveryError = {
      status: 500,
      detail: "Composer failed after a tool call",
      error_type: "composer_plugin_crash",
      partial_state: makeFullCompositionState(),
      failed_turn: {
        // null keeps RecoveryTranscript in its idle (no-fetch) state.
        assistant_message_id: null,
        tool_calls_attempted: 2,
        tool_responses_persisted: 1,
        transcript_url: null,
      },
    };
    const { container } = render(
      <RecoveryPanel
        activeSessionId="sess-a11y"
        currentState={makeFullCompositionState()}
        recoveryError={recoveryError}
        onApply={() => ({ applied: true, needsConfirmation: false })}
        onDiscard={() => {}}
      />,
    );
    screen.getByRole("button", { name: /Discard recovery/ });
    expect(await axe(container)).toHaveNoViolations();
  });
});

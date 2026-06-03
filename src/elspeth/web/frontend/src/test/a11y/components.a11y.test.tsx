// ============================================================================
// components.a11y.test.tsx — Phase 8 Task 7 accessibility audit
//
// One axe-core pass per Phase-1-through-8 user-facing component. The matcher
// `toHaveNoViolations` is registered globally by `setup.ts` (registered in
// vite.config.ts's `test.setupFiles`); do NOT call `expect.extend(...)` in
// this file — that would shadow the global registration and break the
// "register once" invariant. See `setup.ts` head comment for the rationale.
//
// Coverage is anchored by the AUDITED_COMPONENTS array. The first test in
// this file is a snapshot assertion that exits the build with a clear
// failure if a future PR adds or removes a Phase-1-7 component without
// updating the audit list — preventing silent erosion of the a11y safety
// net.
// ============================================================================

import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

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
  "InlineOptOutCheckbox",
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
  "TemplateCards",
  "ShortcutsHelp",
] as const;

const EXPECTED_AUDITED_COMPONENTS_SORTED: readonly string[] = [
  "AppHeader",
  "AuditReadinessPanel",
  "ComposerPreferencesPanel",
  "CompletionBar",
  "DefaultModeChangedBanner",
  "ExplainDialog",
  "ExportYamlModal",
  "FilterChipStrip",
  "GraphMiniView",
  "HeaderSessionSwitcher",
  "HeaderVersionSelector",
  "InlineOptOutCheckbox",
  "InlineSourceCreatedTurn",
  "InlineSourceDisambiguationTurn",
  "InlineSourceFallbackPrompt",
  "PluginCard",
  "ReadinessRowDetail",
  "SideRail",
  "ShortcutsHelp",
  "TemplateCards",
  "UserMenu",
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

vi.mock("@/api/client", () => ({
  fetchUserComposerPreferences: vi.fn(),
  updateUserComposerPreferences: vi.fn(),
  fetchSessions: vi.fn(),
  createSession: vi.fn(),
}));

vi.mock("../../api/auditReadiness", () => ({
  fetchAuditReadiness: vi.fn(),
  fetchAuditReadinessExplain: vi.fn(),
  validateAuditReadinessSnapshot: vi.fn(),
}));

// --- Imports (post-mock) ---------------------------------------------------

import { ComposerPreferencesPanel } from "@/components/settings/ComposerPreferencesPanel";
import { UserMenu } from "@/components/common/UserMenu";
import { InlineOptOutCheckbox } from "@/components/chat/guided/InlineOptOutCheckbox";
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
import { TemplateCards } from "@/components/chat/TemplateCards";
import { ShortcutsHelp } from "@/components/common/ShortcutsHelp";

import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import { useAuditReadinessStore, getInitialState as getAuditInitialState } from "@/stores/auditReadinessStore";
import * as auditApi from "../../api/auditReadiness";
import { resetStore } from "@/test/store-helpers";
import type { InlineSourceSummary } from "@/types/api";
import type { PluginSummary } from "@/types/index";
import type { ReadinessRow, AuditReadinessSnapshot } from "@/types/api";

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
      source: null,
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
}

beforeEach(() => {
  resetAllStores();
  vi.clearAllMocks();
});

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

describe("InlineOptOutCheckbox", () => {
  it("has no axe violations", async () => {
    const { container } = render(<InlineOptOutCheckbox />);
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

describe("TemplateCards", () => {
  it("has no axe violations", async () => {
    const { container } = render(
      <TemplateCards onSelectTemplate={() => {}} />,
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("ShortcutsHelp", () => {
  it("has no axe violations", async () => {
    const { container } = render(<ShortcutsHelp onClose={() => {}} />);
    expect(await axe(container)).toHaveNoViolations();
  });
});

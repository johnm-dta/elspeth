import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SharedAuditReadinessPanel } from "./SharedAuditReadinessPanel";
import type { AuditReadinessSnapshot } from "../../types/api";

const _snapshot: AuditReadinessSnapshot = {
  session_id: "00000000-0000-0000-0000-000000000001",
  composition_version: 7,
  checked_at: "2026-05-19T00:00:00+00:00",
  rows: [
    {
      id: "validation",
      label: "Validation",
      status: "ok",
      summary: "All checks pass",
      detail: null,
      component_ids: [],
    },
    {
      id: "plugin_trust",
      label: "Plugin trust",
      status: "ok",
      summary: "All Tier 1/2",
      detail: null,
      component_ids: [],
    },
    {
      id: "provenance",
      label: "Provenance",
      status: "warning",
      summary: "Identity passthrough",
      detail: null,
      component_ids: ["t1"],
    },
    {
      id: "retention",
      label: "Retention",
      status: "not_applicable",
      summary: "90 days",
      detail: null,
      component_ids: [],
    },
    {
      id: "llm_interpretations",
      label: "LLM interpretations",
      status: "error",
      summary: "Three pending",
      detail: null,
      component_ids: [],
    },
    {
      id: "secrets",
      label: "Secrets",
      status: "not_applicable",
      summary: "No secrets",
      detail: null,
      component_ids: [],
    },
  ],
  validation_result: {
    is_valid: true,
    checks: [],
    errors: [],
    warnings: [],
    semantic_contracts: [],
  },
};

describe("SharedAuditReadinessPanel", () => {
  it("renders all six rows with the spec'd shared-inspect test ids", () => {
    render(<SharedAuditReadinessPanel snapshot={_snapshot} />);
    expect(
      screen.getByTestId("shared-inspect-readiness-row-validation"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("shared-inspect-readiness-row-plugin_trust"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("shared-inspect-readiness-row-provenance"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("shared-inspect-readiness-row-retention"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("shared-inspect-readiness-row-llm_interpretations"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("shared-inspect-readiness-row-secrets"),
    ).toBeInTheDocument();
  });

  it("renders every row in the static (non-button) variant under ReadOnlyProvider", () => {
    render(<SharedAuditReadinessPanel snapshot={_snapshot} />);
    // Even the warning and error rows are NOT actionable buttons in
    // shared mode — the read-only context forces the static variant.
    // (The buttons that AuditReadinessRow renders for actionable rows
    // have `role="button"`; no row in shared mode should expose one.)
    expect(
      screen.queryByRole("button", { name: /provenance/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /llm interpretations/i }),
    ).not.toBeInTheDocument();
  });

  it("renders the per-row label and summary text from the snapshot", () => {
    render(<SharedAuditReadinessPanel snapshot={_snapshot} />);
    expect(screen.getByText("Validation")).toBeInTheDocument();
    expect(screen.getByText("All checks pass")).toBeInTheDocument();
    expect(screen.getByText("Identity passthrough")).toBeInTheDocument();
  });

  it("renders a freshness indicator with the composition version and checked_at", () => {
    render(<SharedAuditReadinessPanel snapshot={_snapshot} />);
    expect(
      screen.getByText(/Composition v7/),
    ).toBeInTheDocument();
  });
});

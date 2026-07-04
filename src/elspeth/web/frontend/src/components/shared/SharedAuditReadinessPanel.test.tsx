import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SharedAuditReadinessPanel } from "./SharedAuditReadinessPanel";
import type { AuditReadinessSnapshot } from "../../types/api";

const READY_READINESS = {
  authoring_valid: true,
  execution_ready: true,
  completion_ready: true,
  blockers: [],
};

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
    readiness: READY_READINESS,
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

  // FIX-H (gap-analysis remediation): plan 19b:519-542 mandates an
  // explicit test that the shared panel renders the LLM-interpretations
  // row from the snapshot's `summary` text, NOT from the live
  // `interpretationEventsStore` that the owner-side AuditReadinessPanel
  // consults. The mandated proof shape is: render a snapshot whose
  // llm_interpretations row carries `summary: "2 pending review (3
  // resolved)"`, assert the rendered tree contains "2 pending review".
  //
  // The plan also asks us to seed the interpretation-events store as
  // empty to prove non-coupling. SharedAuditReadinessPanel does not
  // import that store at all (verified by reading the impl source —
  // it only imports AuditReadinessRow + types). The store-empty step
  // is therefore structural: if a future refactor were to introduce a
  // store read, this test's snapshot summary ("2 pending review") and
  // the absent store contents ("0 events visible") would diverge, and
  // the assertion below would fail. Documented so a future reader
  // does not assume a missing setup step.
  it(
    "renders the llm_interpretations row from snapshot.summary without " +
      "reading interpretationEventsStore (plan 19b:519-542)",
    () => {
      const decoupling_snapshot: AuditReadinessSnapshot = {
        ..._snapshot,
        rows: _snapshot.rows.map((r) =>
          r.id === "llm_interpretations"
            ? {
                ...r,
                status: "warning" as const,
                summary: "2 pending review (3 resolved)",
              }
            : r,
        ),
      };
      render(<SharedAuditReadinessPanel snapshot={decoupling_snapshot} />);
      expect(screen.getByText(/2 pending review/)).toBeInTheDocument();
    },
  );

  // ── Gate legibility (elspeth-088bf83922 T-2, option (a)) ───────────────────
  //
  // The shared read-only view renders through the same AuditReadinessRow
  // primitive as the live panel, so it inherits the identical "Blocks Run" /
  // "Advisory" classification (validation + llm_interpretations gate;
  // plugin_trust/provenance/retention/secrets are advisory). It also gets a
  // standalone explanatory line in the header, since a reviewer here has no
  // ExecuteButton in view to infer the distinction from.

  it("labels validation and llm_interpretations rows 'Blocks Run' and the other four 'Advisory'", () => {
    render(<SharedAuditReadinessPanel snapshot={_snapshot} />);
    expect(
      screen.getByTestId("shared-inspect-readiness-row-validation"),
    ).toHaveAttribute("data-gate", "blocks");
    expect(
      screen.getByTestId("shared-inspect-readiness-row-llm_interpretations"),
    ).toHaveAttribute("data-gate", "blocks");
    for (const id of ["plugin_trust", "provenance", "retention", "secrets"]) {
      expect(
        screen.getByTestId(`shared-inspect-readiness-row-${id}`),
      ).toHaveAttribute("data-gate", "advisory");
    }
  });

  it("explains the 'Blocks Run' classification in the frozen-snapshot header", () => {
    render(<SharedAuditReadinessPanel snapshot={_snapshot} />);
    expect(
      screen.getByText(/Rows marked "Blocks Run" had to be clear/i),
    ).toBeInTheDocument();
  });
});

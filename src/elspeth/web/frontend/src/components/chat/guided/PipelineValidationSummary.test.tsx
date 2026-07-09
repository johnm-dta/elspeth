import { beforeEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { PipelineValidationSummary } from "./PipelineValidationSummary";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import { resetStore } from "@/test/store-helpers";
import { makeComposition } from "@/test/composerFixtures";
import type { ValidationReadiness, ValidationResult } from "@/types/index";

// Seed a source→llm→csv composition so component_ids map to plain phrases:
//   "source" → "read your data", "rater" → "rate each row", "out" → "write a CSV".
function seedComposition() {
  useSessionStore.setState({
    compositionState: makeComposition(1, {
      sources: { source: { plugin: "text", options: {} } },
      nodes: [
        {
          id: "rater",
          node_type: "transform",
          plugin: "llm",
          input: "source",
          on_success: null,
          on_error: null,
          options: {},
        },
      ],
      outputs: [{ name: "out", plugin: "csv", options: {} }],
    }),
  } as never);
}

// PipelineValidationSummary does not read `readiness`; supply a constant so the
// fixtures stay focused on errors/warnings.
const READINESS: ValidationReadiness = {
  authoring_valid: true,
  execution_ready: true,
  completion_ready: true,
  blockers: [],
};

function setValidation(result: Omit<ValidationResult, "readiness"> | null) {
  useExecutionStore.setState({
    validationResult: result === null ? null : { ...result, readiness: READINESS },
  } as never);
}

describe("PipelineValidationSummary", () => {
  beforeEach(() => {
    resetStore(useSessionStore);
    resetStore(useExecutionStore);
    seedComposition();
  });

  it("shows a plain 'looks good' status when the validation result is valid", () => {
    setValidation({ is_valid: true, checks: [], errors: [], warnings: [] });
    render(<PipelineValidationSummary />);
    expect(screen.getByText(/looks good/i)).toBeInTheDocument();
  });

  it("renders a warning with the PLAIN node name (mapped from component_id, not the raw id)", () => {
    setValidation({
      is_valid: true,
      checks: [],
      errors: [],
      warnings: [
        {
          component_id: "rater",
          component_type: "transform",
          message: "Review the prompt wording",
          suggestion: null,
        },
      ],
    });
    render(<PipelineValidationSummary />);
    expect(screen.getByText(/rate each row/)).toBeInTheDocument();
    expect(screen.getByText(/review the prompt wording/i)).toBeInTheDocument();
    // The raw component_id must NOT be surfaced.
    expect(screen.queryByText(/rater/)).toBeNull();
  });

  it("renders an error with the plain node name and the message", () => {
    setValidation({
      is_valid: false,
      checks: [],
      errors: [
        {
          component_id: "out",
          component_type: "sink",
          message: "Missing output path",
          suggestion: null,
        },
      ],
      warnings: [],
    });
    render(<PipelineValidationSummary />);
    expect(screen.getByText(/write a CSV/)).toBeInTheDocument();
    expect(screen.getByText(/missing output path/i)).toBeInTheDocument();
  });

  it("falls back to a generic phrase for an unmappable component_id (no crash)", () => {
    setValidation({
      is_valid: false,
      checks: [],
      errors: [
        {
          component_id: "ghost_component",
          component_type: null,
          message: "Detached node detected",
          suggestion: null,
        },
      ],
      warnings: [],
    });
    render(<PipelineValidationSummary />);
    expect(screen.getByText(/this step/i)).toBeInTheDocument();
    expect(screen.getByText(/detached node detected/i)).toBeInTheDocument();
    // raw id absent
    expect(screen.queryByText(/ghost_component/)).toBeNull();
  });

  it("renders a null component_id finding as the message directly, with no 'this step' possessive", () => {
    // A settings-level finding owns no step (component_id === null), so the
    // headline is the message alone — never a bare "'this step':" prefix
    // (elspeth-901a404926).
    setValidation({
      is_valid: false,
      checks: [],
      errors: [
        {
          component_id: null,
          component_type: null,
          message: "Pipeline has no sink",
          suggestion: null,
        },
      ],
      warnings: [],
    });
    render(<PipelineValidationSummary />);
    const status = screen.getByRole("status");
    expect(status.textContent).toMatch(/1 problem to fix — Pipeline has no sink/);
    expect(status.textContent).not.toMatch(/this step/i);
  });

  it("renders a neutral status element when there is no validation result", () => {
    setValidation(null);
    render(<PipelineValidationSummary />);
    // The root must always be present so the mount/D1/parity tests can find it.
    expect(
      screen.getByTestId("pipeline-validation-summary"),
    ).toBeInTheDocument();
  });

  // ── elspeth-3b35abf148 variant 4: the backend suggestion is rendered ────────
  it("renders the backend's actionable suggestion with the error", () => {
    setValidation({
      is_valid: false,
      checks: [],
      errors: [
        {
          component_id: null,
          component_type: null,
          message: "Cannot resolve secret references: OPENROUTER_API_KEY",
          suggestion:
            "Add the missing secrets via the Secrets panel before executing.",
        },
      ],
      warnings: [],
    });
    render(<PipelineValidationSummary />);
    expect(
      screen.getByText(/add the missing secrets via the secrets panel/i),
    ).toBeInTheDocument();
  });

  it("adds an honest availability note for Secrets-panel suggestions in the tutorial", () => {
    setValidation({
      is_valid: false,
      checks: [],
      errors: [
        {
          component_id: null,
          component_type: null,
          message: "Cannot resolve secret references: OPENROUTER_API_KEY",
          suggestion:
            "Add the missing secrets via the Secrets panel before executing.",
        },
      ],
      warnings: [],
    });
    render(<PipelineValidationSummary isTutorial />);
    expect(
      screen.getByText(/part of the full composer, outside this tutorial/i),
    ).toBeInTheDocument();
  });

  // ── elspeth-3b35abf148 error-rendering: contract dumps are humanised ────────
  it("humanises a schema-contract-violation dump and keeps the raw text behind a details expander", () => {
    const rawDump =
      "Schema contract violation: 'rater' -> 'out'. " +
      "Consumer (csv) requires fields: [score]. " +
      "Producer (llm) guarantees: [body]. Missing fields: [score].";
    setValidation({
      is_valid: false,
      checks: [],
      errors: [
        {
          component_id: "node:rater",
          component_type: "transform",
          message: rawDump,
          suggestion: null,
        },
      ],
      warnings: [],
    });
    render(<PipelineValidationSummary />);
    // The role=status headline is the plain-language line, mapped through the
    // gloss ("rate each row" / "write a CSV"), never the raw dump.
    const status = screen.getByRole("status");
    expect(status.textContent).toMatch(/aren't connected correctly/i);
    expect(status.textContent).toMatch(/rate each row/);
    expect(status.textContent).toMatch(/write a CSV/);
    expect(status.textContent).not.toMatch(/Schema contract violation/);
    // The verbatim dump survives behind the expander for the engineer read.
    expect(screen.getByText("Technical details")).toBeInTheDocument();
    expect(screen.getByText(rawDump)).toBeInTheDocument();
  });

  it("humanises the DAG preflight dump format (edge 'X' → 'Y', unicode arrow) — the live-verified variant", () => {
    // core/dag/graph.py format — the exact shape live-verified in the review
    // ("Schema contract violation: edge 'transform_guided_xform_0_…' → …").
    const rawDump =
      "Schema contract violation: edge 'rater' → 'out'\n" +
      "  Consumer (csv) requires fields: ['score']\n" +
      "  Producer (llm) guarantees: ['body']\n" +
      "  Missing fields: ['score']\n" +
      "\n" +
      "Fix: Either:\n" +
      "  1. Add missing fields to producer's schema or guaranteed_fields, or\n" +
      "  2. Remove from consumer's required_input_fields if truly optional";
    setValidation({
      is_valid: false,
      checks: [],
      errors: [
        {
          component_id: "out",
          component_type: "output",
          message: rawDump,
          suggestion: null,
        },
      ],
      warnings: [],
    });
    render(<PipelineValidationSummary />);
    const status = screen.getByRole("status");
    expect(status.textContent).toMatch(/aren't connected correctly/i);
    expect(status.textContent).toMatch(/rate each row/);
    expect(status.textContent).toMatch(/write a CSV/);
    expect(status.textContent).not.toMatch(/Schema contract violation/);
    expect(screen.getByText("Technical details")).toBeInTheDocument();
  });

  it("humanises the edge-contract preflight dump (Edge contract violation between producer node …)", () => {
    // web/execution/validation.py _format_edge_contract_failure format.
    const rawDump =
      "Edge contract violation between producer node 'rater' (schema 'LLMOutput') " +
      "and consumer node 'out' (schema 'CsvInput'):\n" +
      "Missing required fields (consumer requires, producer does not guarantee):\n" +
      "  - 'score'";
    setValidation({
      is_valid: false,
      checks: [],
      errors: [
        {
          component_id: "out",
          component_type: "output",
          message: rawDump,
          suggestion: null,
        },
      ],
      warnings: [],
    });
    render(<PipelineValidationSummary />);
    const status = screen.getByRole("status");
    expect(status.textContent).toMatch(/aren't connected correctly/i);
    expect(status.textContent).toMatch(/rate each row/);
    expect(status.textContent).toMatch(/write a CSV/);
    expect(status.textContent).not.toMatch(/Edge contract violation/);
    expect(screen.getByText("Technical details")).toBeInTheDocument();
  });

  // ── elspeth-016f463ff0: interpretation-review-pending dumps are humanised ──
  it("humanises a pending-review dump with the acknowledge-card step label, raw text behind the expander", () => {
    // web/execution/validation.py _format_interpretation_site format — the
    // exact shape live-captured in the review.
    const rawDump =
      "pipeline_decision review pending for transform 'rater': drop_raw_html_fields";
    setValidation({
      is_valid: false,
      checks: [],
      errors: [
        {
          component_id: "rater",
          component_type: "transform",
          message: rawDump,
          suggestion: "Resolve the pending interpretation review before running.",
          error_code: "interpretation_review_pending",
        },
      ],
      warnings: [],
    });
    render(<PipelineValidationSummary />);
    const status = screen.getByRole("status");
    // "Summarise" is the SAME label the acknowledgement card renders for the
    // llm plugin (stepLabelForPlugin) — the strip and the card must agree.
    expect(status.textContent).toMatch(/The Summarise step is waiting for your review\./);
    expect(status.textContent).not.toMatch(/pipeline_decision/);
    expect(status.textContent).not.toMatch(/rater/);
    // The verbatim dump survives behind the expander for the operator read.
    expect(screen.getByText("Technical details")).toBeInTheDocument();
    expect(screen.getByText(rawDump)).toBeInTheDocument();
  });

  it("pending-review dump with an unmappable component id gets a generic headline, never the raw id", () => {
    const rawDump =
      "pipeline_decision review pending for transform 'guided_xform_9': drop_raw_html_fields";
    setValidation({
      is_valid: false,
      checks: [],
      errors: [
        {
          component_id: "guided_xform_9",
          component_type: "transform",
          message: rawDump,
          suggestion: null,
          error_code: "interpretation_review_pending",
        },
      ],
      warnings: [],
    });
    render(<PipelineValidationSummary />);
    const status = screen.getByRole("status");
    expect(status.textContent).toMatch(/A step is waiting for your review\./);
    expect(status.textContent).not.toMatch(/guided_xform_9/);
    expect(screen.getByText("Technical details")).toBeInTheDocument();
  });

  // ── elspeth-901a404926: reframed settings finding (null component) ─────────
  it("renders a null-component reframed settings finding without the 'this step' possessive or a pydantic leak", () => {
    setValidation({
      is_valid: false,
      checks: [],
      errors: [
        {
          component_id: null,
          component_type: null,
          message: "Add an output step so your pipeline has somewhere to send its results.",
          suggestion:
            "Pick an output like CSV or JSON and connect your last step to it, then validate again.",
          error_code: "missing_sink",
        },
      ],
      warnings: [],
    });
    render(<PipelineValidationSummary />);
    const status = screen.getByRole("status");
    // The headline is the plain reframed message, with no bare "'this step':"
    // possessive for a settings-level (null-component) finding.
    expect(status.textContent).toMatch(/1 problem to fix — Add an output step/);
    expect(status.textContent).not.toMatch(/this step/);
    // The reframed message is clean — no raw pydantic internals reach the
    // announced live region.
    expect(status.textContent).not.toMatch(/ElspethSettings|Field required|pydantic/i);
    // A clean settings finding carries no raw dump, so no expander appears.
    expect(screen.queryByText("Technical details")).toBeNull();
  });
});

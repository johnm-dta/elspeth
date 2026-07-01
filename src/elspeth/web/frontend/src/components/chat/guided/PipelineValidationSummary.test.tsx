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

  it("handles a null component_id finding via the generic fallback", () => {
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
    expect(screen.getByText(/this step/i)).toBeInTheDocument();
    expect(screen.getByText(/pipeline has no sink/i)).toBeInTheDocument();
  });

  it("renders a neutral status element when there is no validation result", () => {
    setValidation(null);
    render(<PipelineValidationSummary />);
    // The root must always be present so the mount/D1/parity tests can find it.
    expect(
      screen.getByTestId("pipeline-validation-summary"),
    ).toBeInTheDocument();
  });
});

// ============================================================================
// SchemaFormTurn -- wire-response and rendering contract regression coverage.
//
// Pins SIX contracts (see implementation header for full rationale):
//   1. Field-type-to-control mapping:
//        string (no enum)     -> <input type="text">
//        string with enum     -> <select> + <option> per enum value
//        integer              -> <input type="number" step="1">
//        number               -> <input type="number" step="any">
//        boolean              -> <input type="checkbox">
//        array / object / $ref / anyOf -> <textarea> JSON fallback
//   2. Required-vs-advanced split:
//        Fields listed in schema_block.required appear unconditionally.
//        Fields absent from required are hidden until "Show advanced" is clicked.
//        Toggle is reversible. No toggle button when there are no optional fields.
//   3. Prefill-vs-default precedence:
//        prefilled[name] beats schema default beats empty.
//   4. Submit wire shape (all-properties inclusion):
//        edited_values contains EVERY property from the schema, using the
//        user's current input value -- or the prefilled/default for untouched
//        fields. Other five GuidedRespondRequest fields are explicitly null.
//   5. JSON-fallback behaviour:
//        Fallback fields show stringified initial value; invalid JSON shows an
//        inline error and disables Continue.
//   6. Focus management on advanced toggle:
//        No auto-focus on initial mount; first advanced field gets focus on
//        Show-advanced click; re-collapsing focuses the toggle button.
//
// Source of truth:
//   - protocol.py:53-56 (SchemaFormPayload wire shape)
//   - schema_block = Pydantic ConfigModel.model_json_schema() output
//   - state_machine.py:_advance_step_1 confirms schema_form is currently an
//     intra-step turn (returns early without consuming edited_values), but
//     the full edited_values dict is pre-wired for when the intra-step flow
//     is connected (Tasks 2.2+).
// ============================================================================

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SchemaFormTurn } from "./SchemaFormTurn";
import { nullResponse } from "@/test/guided-fixtures";
import type { GuidedRespondRequest, SchemaFormPayload } from "@/types/guided";

// ── Fixtures ─────────────────────────────────────────────────────────────────

// Realistic Pydantic model_json_schema() output for a CsvSourceSettings model.
// path is required; delimiter/has_header/encoding are optional with defaults.
const CSV_PAYLOAD: SchemaFormPayload = {
  plugin: "csv",
  schema_block: {
    title: "CsvSourceSettings",
    type: "object",
    properties: {
      path: {
        type: "string",
        title: "Path",
        description: "Filesystem path to the CSV file",
      },
      delimiter: {
        type: "string",
        title: "Delimiter",
        default: ",",
      },
      has_header: {
        type: "boolean",
        title: "Has header",
        default: true,
      },
      encoding: {
        type: "string",
        title: "Encoding",
        default: "utf-8",
      },
    },
    required: ["path"],
  },
  prefilled: {},
};

// Schema with an enum string field (required) and integer / number fields (optional).
const NUMERIC_PAYLOAD: SchemaFormPayload = {
  plugin: "database",
  schema_block: {
    title: "DatabaseSourceSettings",
    type: "object",
    properties: {
      mode: {
        type: "string",
        enum: ["read", "write", "append"],
        title: "Mode",
      },
      batch_size: {
        type: "integer",
        title: "Batch size",
        default: 1000,
      },
      timeout: {
        type: "number",
        title: "Timeout",
        default: 30.0,
      },
    },
    required: ["mode"],
  },
  prefilled: { mode: "read" },
};

// Schema with an array field (JSON fallback) and an object field (JSON fallback).
const FALLBACK_PAYLOAD: SchemaFormPayload = {
  plugin: "transform",
  schema_block: {
    title: "TransformSettings",
    type: "object",
    properties: {
      name: {
        type: "string",
        title: "Name",
      },
      tags: {
        type: "array",
        title: "Tags",
        items: { type: "string" },
      },
      metadata: {
        type: "object",
        title: "Metadata",
      },
    },
    required: ["name"],
  },
  prefilled: { tags: ["alpha", "beta"], metadata: { version: 1 } },
};

// Schema with a $ref field (JSON fallback).
const REF_PAYLOAD: SchemaFormPayload = {
  plugin: "sink",
  schema_block: {
    title: "SinkSettings",
    type: "object",
    properties: {
      target: {
        type: "string",
        title: "Target",
      },
      nested_config: {
        $ref: "#/$defs/NestedConfig",
        title: "Nested config",
      },
    },
    $defs: {
      NestedConfig: {
        type: "object",
        properties: { key: { type: "string" } },
      },
    },
    required: ["target"],
  },
  prefilled: {},
};

// Schema with anyOf field (JSON fallback -- Optional[str] pattern from Pydantic).
const ANY_OF_PAYLOAD: SchemaFormPayload = {
  plugin: "filter",
  schema_block: {
    title: "FilterSettings",
    type: "object",
    properties: {
      name: { type: "string", title: "Name" },
      label: {
        anyOf: [{ type: "string" }, { type: "null" }],
        title: "Label",
        default: null,
      },
    },
    required: ["name"],
  },
  prefilled: {},
};

// Schema with NO optional fields -- no "Show advanced" toggle expected.
const NO_ADVANCED_PAYLOAD: SchemaFormPayload = {
  plugin: "csv",
  schema_block: {
    title: "MinimalSettings",
    type: "object",
    properties: {
      path: { type: "string", title: "Path" },
    },
    required: ["path"],
  },
  prefilled: {},
};

// Schema with NO fields at all -- empty form, Continue always enabled.
const EMPTY_SCHEMA_PAYLOAD: SchemaFormPayload = {
  plugin: "passthrough",
  schema_block: {
    title: "PassthroughSettings",
    type: "object",
    properties: {},
  },
  prefilled: {},
};

// Schema with ALL optional fields (no required array at all).
const ALL_OPTIONAL_PAYLOAD: SchemaFormPayload = {
  plugin: "sink",
  schema_block: {
    title: "SinkSettings",
    type: "object",
    properties: {
      format: { type: "string", title: "Format", default: "json" },
      compress: { type: "boolean", title: "Compress", default: false },
    },
  },
  prefilled: {},
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function renderTurn(
  payload: SchemaFormPayload,
  onSubmit?: (body: GuidedRespondRequest) => void,
) {
  const handler = onSubmit ?? vi.fn();
  render(<SchemaFormTurn payload={payload} onSubmit={handler} />);
  return handler;
}

// ── 1. Field-type rendering ───────────────────────────────────────────────────

describe("field-type rendering", () => {
  it("renders a string field as <input type='text'>", () => {
    renderTurn(CSV_PAYLOAD);
    const input = screen.getByRole("textbox", { name: /path/i });
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute("type", "text");
  });

  it("wires description to hint text via aria-describedby", () => {
    renderTurn(CSV_PAYLOAD);
    const input = screen.getByRole("textbox", { name: /path/i });
    const describedById = input.getAttribute("aria-describedby");
    expect(describedById).toBeTruthy();
    const hint = document.getElementById(describedById!);
    expect(hint).toHaveTextContent("Filesystem path to the CSV file");
  });

  it("does NOT wire aria-describedby when description is absent", async () => {
    const user = userEvent.setup();
    renderTurn(CSV_PAYLOAD);
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    // delimiter has no description
    const input = screen.getByRole("textbox", { name: /delimiter/i });
    expect(input).not.toHaveAttribute("aria-describedby");
  });

  it("renders an enum string field as <select>", () => {
    renderTurn(NUMERIC_PAYLOAD);
    const select = screen.getByRole("combobox", { name: /mode/i });
    expect(select.tagName).toBe("SELECT");
    const options = select.querySelectorAll("option");
    expect(options).toHaveLength(3);
    expect(options[0]).toHaveTextContent("read");
    expect(options[1]).toHaveTextContent("write");
    expect(options[2]).toHaveTextContent("append");
  });

  it("renders an integer field as <input type='number' step='1'>", async () => {
    const user = userEvent.setup();
    renderTurn(NUMERIC_PAYLOAD);
    // Show advanced first since batch_size is optional
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    const input = screen.getByRole("spinbutton", { name: /batch size/i });
    expect(input).toHaveAttribute("type", "number");
    expect(input).toHaveAttribute("step", "1");
  });

  it("renders a number field as <input type='number' step='any'>", async () => {
    const user = userEvent.setup();
    renderTurn(NUMERIC_PAYLOAD);
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    const input = screen.getByRole("spinbutton", { name: /timeout/i });
    expect(input).toHaveAttribute("type", "number");
    expect(input).toHaveAttribute("step", "any");
  });

  it("renders a boolean field as <input type='checkbox'>", async () => {
    const user = userEvent.setup();
    renderTurn(CSV_PAYLOAD);
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    const checkbox = screen.getByRole("checkbox", { name: /has header/i });
    expect(checkbox).toBeInTheDocument();
  });

  it("renders an array field as a <textarea> JSON fallback", async () => {
    const user = userEvent.setup();
    renderTurn(FALLBACK_PAYLOAD);
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    const textarea = screen.getByRole("textbox", { name: /tags/i });
    expect(textarea.tagName).toBe("TEXTAREA");
  });

  it("renders an object field as a <textarea> JSON fallback", async () => {
    const user = userEvent.setup();
    renderTurn(FALLBACK_PAYLOAD);
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    const textarea = screen.getByRole("textbox", { name: /metadata/i });
    expect(textarea.tagName).toBe("TEXTAREA");
  });

  it("renders a $ref field as a <textarea> JSON fallback", async () => {
    const user = userEvent.setup();
    renderTurn(REF_PAYLOAD);
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    const textarea = screen.getByRole("textbox", { name: /nested config/i });
    expect(textarea.tagName).toBe("TEXTAREA");
  });

  it("renders an anyOf field as a <textarea> JSON fallback", async () => {
    const user = userEvent.setup();
    renderTurn(ANY_OF_PAYLOAD);
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    const textarea = screen.getByRole("textbox", { name: /label/i });
    expect(textarea.tagName).toBe("TEXTAREA");
  });
});

// ── 2. Required vs advanced split ─────────────────────────────────────────────

describe("required vs advanced split", () => {
  it("shows required fields without clicking Show advanced", () => {
    renderTurn(CSV_PAYLOAD);
    expect(screen.getByRole("textbox", { name: /path/i })).toBeInTheDocument();
  });

  it("hides optional fields initially", () => {
    renderTurn(CSV_PAYLOAD);
    expect(screen.queryByRole("textbox", { name: /delimiter/i })).toBeNull();
    expect(screen.queryByRole("checkbox", { name: /has header/i })).toBeNull();
    expect(screen.queryByRole("textbox", { name: /encoding/i })).toBeNull();
  });

  it("reveals optional fields after clicking Show advanced", async () => {
    const user = userEvent.setup();
    renderTurn(CSV_PAYLOAD);
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    expect(screen.getByRole("textbox", { name: /delimiter/i })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /has header/i })).toBeInTheDocument();
  });

  it("hides optional fields again when toggle is clicked a second time", async () => {
    const user = userEvent.setup();
    renderTurn(CSV_PAYLOAD);
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    await user.click(screen.getByRole("button", { name: /hide advanced/i }));
    expect(screen.queryByRole("textbox", { name: /delimiter/i })).toBeNull();
  });

  it("does NOT render the Show advanced button when there are no optional fields", () => {
    renderTurn(NO_ADVANCED_PAYLOAD);
    expect(screen.queryByRole("button", { name: /show advanced/i })).toBeNull();
  });

  it("renders Show advanced when all fields are optional (no required array)", async () => {
    const user = userEvent.setup();
    renderTurn(ALL_OPTIONAL_PAYLOAD);
    const btn = screen.getByRole("button", { name: /show advanced/i });
    await user.click(btn);
    expect(screen.getByRole("textbox", { name: /format/i })).toBeInTheDocument();
  });
});

// ── 3. Prefill-vs-default precedence ─────────────────────────────────────────

describe("prefill vs default precedence", () => {
  it("shows the prefilled value for a field present in prefilled", async () => {
    const user = userEvent.setup();
    renderTurn(NUMERIC_PAYLOAD);
    // mode is required and prefilled with "read"
    const select = screen.getByRole("combobox", { name: /mode/i });
    expect((select as HTMLSelectElement).value).toBe("read");
    void user; // prevent unused-variable lint
  });

  it("shows the schema default for an optional field not in prefilled", async () => {
    const user = userEvent.setup();
    renderTurn(CSV_PAYLOAD);
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    // delimiter has default "," and is not in CSV_PAYLOAD.prefilled
    const input = screen.getByRole("textbox", { name: /delimiter/i }) as HTMLInputElement;
    expect(input.value).toBe(",");
  });

  it("shows the schema boolean default for a checkbox", async () => {
    const user = userEvent.setup();
    renderTurn(CSV_PAYLOAD);
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    const checkbox = screen.getByRole("checkbox", { name: /has header/i }) as HTMLInputElement;
    expect(checkbox.checked).toBe(true);
  });

  it("shows empty for a required field with no prefill and no schema default", () => {
    renderTurn(CSV_PAYLOAD);
    const input = screen.getByRole("textbox", { name: /path/i }) as HTMLInputElement;
    expect(input.value).toBe("");
  });

  it("shows JSON-stringified prefilled value in a JSON-fallback textarea", async () => {
    const user = userEvent.setup();
    renderTurn(FALLBACK_PAYLOAD);
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    const textarea = screen.getByRole("textbox", { name: /tags/i }) as HTMLTextAreaElement;
    // Prefilled is ["alpha","beta"] -- should appear as JSON
    expect(JSON.parse(textarea.value)).toEqual(["alpha", "beta"]);
    void user;
  });

  it("shows parseable JSON for a $ref field with no prefill and no default", async () => {
    const user = userEvent.setup();
    // REF_PAYLOAD has nested_config with $ref and no prefill; initialValueFor
    // emits "null" (a valid JSON literal) as the sentinel for object/$ref fields
    renderTurn(REF_PAYLOAD);
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    const textarea = screen.getByRole("textbox", { name: /nested config/i }) as HTMLTextAreaElement;
    // Pin round-trip: whatever value appears, it must be valid JSON
    expect(() => JSON.parse(textarea.value)).not.toThrow();
    void user;
  });
});

// ── 4. Submit wire shape ──────────────────────────────────────────────────────

describe("submit wire shape", () => {
  it("fires onSubmit with all-null fields except edited_values on Continue", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SchemaFormTurn payload={CSV_PAYLOAD} onSubmit={onSubmit} />);

    await user.type(screen.getByRole("textbox", { name: /path/i }), "/data/file.csv");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    expect(onSubmit).toHaveBeenCalledOnce();
    const body = onSubmit.mock.calls[0][0];
    expect(body).toEqual({
      ...nullResponse(),
      chosen: null,
      custom_inputs: null,
      edited_values: expect.objectContaining({ path: "/data/file.csv" }),
    });
  });

  it("includes EVERY schema property in edited_values (touched and untouched)", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SchemaFormTurn payload={CSV_PAYLOAD} onSubmit={onSubmit} />);

    await user.type(screen.getByRole("textbox", { name: /path/i }), "/data/file.csv");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    const { edited_values } = onSubmit.mock.calls[0][0];
    // All four properties must be present; optional ones at their defaults
    expect(edited_values).toHaveProperty("path", "/data/file.csv");
    expect(edited_values).toHaveProperty("delimiter", ",");
    expect(edited_values).toHaveProperty("has_header", true);
    expect(edited_values).toHaveProperty("encoding", "utf-8");
  });

  it("uses prefilled value in edited_values for untouched prefilled fields", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SchemaFormTurn payload={NUMERIC_PAYLOAD} onSubmit={onSubmit} />);

    // mode is required and already prefilled -- just continue
    await user.click(screen.getByRole("button", { name: /continue/i }));

    const { edited_values } = onSubmit.mock.calls[0][0];
    expect(edited_values).toHaveProperty("mode", "read");
    // optional fields at their defaults
    expect(edited_values).toHaveProperty("batch_size", 1000);
    expect(edited_values).toHaveProperty("timeout", 30.0);
  });

  it("emits {} edited_values and enables Continue for a schema with no fields", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SchemaFormTurn payload={EMPTY_SCHEMA_PAYLOAD} onSubmit={onSubmit} />);

    const continueBtn = screen.getByRole("button", { name: /continue/i });
    expect(continueBtn).not.toBeDisabled();
    await user.click(continueBtn);

    expect(onSubmit).toHaveBeenCalledOnce();
    expect(onSubmit.mock.calls[0][0].edited_values).toEqual({});
  });

  it("includes parsed JSON value for a JSON-fallback field in edited_values", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SchemaFormTurn payload={FALLBACK_PAYLOAD} onSubmit={onSubmit} />);

    // name is required
    await user.type(screen.getByRole("textbox", { name: /name/i }), "my-transform");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    const { edited_values } = onSubmit.mock.calls[0][0];
    // tags was prefilled as ["alpha","beta"] -- should appear parsed
    expect(edited_values).toHaveProperty("tags");
    expect((edited_values as Record<string, unknown>)["tags"]).toEqual(["alpha", "beta"]);
  });
});

// ── 5. Continue disabled invariant ────────────────────────────────────────────

describe("Continue disabled invariant", () => {
  it("disables Continue when a required string field is empty", () => {
    renderTurn(CSV_PAYLOAD);
    const continueBtn = screen.getByRole("button", { name: /continue/i });
    expect(continueBtn).toBeDisabled();
  });

  it("enables Continue once the required string field has a value", async () => {
    const user = userEvent.setup();
    renderTurn(CSV_PAYLOAD);
    await user.type(screen.getByRole("textbox", { name: /path/i }), "/data");
    expect(screen.getByRole("button", { name: /continue/i })).not.toBeDisabled();
  });

  it("clicking disabled Continue does NOT fire onSubmit", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SchemaFormTurn payload={CSV_PAYLOAD} onSubmit={onSubmit} />);
    // path is empty, Continue is disabled
    await user.click(screen.getByRole("button", { name: /continue/i }));
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("enables Continue when the required field is a prefilled enum select", () => {
    // NUMERIC_PAYLOAD prefills mode="read" -- Continue should be enabled immediately
    renderTurn(NUMERIC_PAYLOAD);
    expect(screen.getByRole("button", { name: /continue/i })).not.toBeDisabled();
  });

  it("a required boolean checkbox always satisfies the required check", () => {
    const boolRequiredPayload: SchemaFormPayload = {
      plugin: "x",
      schema_block: {
        type: "object",
        properties: { enabled: { type: "boolean", title: "Enabled" } },
        required: ["enabled"],
      },
      prefilled: {},
    };
    renderTurn(boolRequiredPayload);
    // boolean is always present (true or false) -- should be enabled
    expect(screen.getByRole("button", { name: /continue/i })).not.toBeDisabled();
  });

  it("disables Continue when all fields are optional and advanced is collapsed (all-optional schema always enabled)", () => {
    // With no required fields, Continue is always enabled since there's nothing blocking
    renderTurn(ALL_OPTIONAL_PAYLOAD);
    expect(screen.getByRole("button", { name: /continue/i })).not.toBeDisabled();
  });

  it("disables Continue when a JSON-fallback field has a parse error", async () => {
    const user = userEvent.setup();
    renderTurn(FALLBACK_PAYLOAD);

    // name is required -- fill it
    await user.type(screen.getByRole("textbox", { name: /name/i }), "x");

    // tags textarea shows prefilled JSON -- set it to invalid JSON via fireEvent
    // (userEvent.type interprets { and " as special key descriptors; fireEvent.change
    // sets the value directly without key-descriptor parsing)
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    const tagsTextarea = screen.getByRole("textbox", { name: /tags/i });
    fireEvent.change(tagsTextarea, { target: { value: "not valid json {" } });

    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
    // An inline error message should be visible
    expect(screen.getByText(/invalid json/i)).toBeInTheDocument();
  });

  it("re-enables Continue after fixing a JSON parse error", async () => {
    const user = userEvent.setup();
    renderTurn(FALLBACK_PAYLOAD);
    await user.type(screen.getByRole("textbox", { name: /name/i }), "x");
    await user.click(screen.getByRole("button", { name: /show advanced/i }));

    const tagsTextarea = screen.getByRole("textbox", { name: /tags/i });
    fireEvent.change(tagsTextarea, { target: { value: "not valid json {" } });
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();

    fireEvent.change(tagsTextarea, { target: { value: '["fixed"]' } });
    expect(screen.getByRole("button", { name: /continue/i })).not.toBeDisabled();
  });
});

// ── 6. Focus management ───────────────────────────────────────────────────────

describe("focus management", () => {
  it("does NOT auto-focus any input on initial mount", () => {
    renderTurn(CSV_PAYLOAD);
    const path = screen.getByRole("textbox", { name: /path/i });
    expect(document.activeElement).not.toBe(path);
  });

  it("focuses the first newly-revealed advanced field on Show advanced click", async () => {
    const user = userEvent.setup();
    renderTurn(CSV_PAYLOAD);
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    // First advanced field for CSV_PAYLOAD is "delimiter"
    const delimiter = screen.getByRole("textbox", { name: /delimiter/i });
    expect(document.activeElement).toBe(delimiter);
  });

  it("focuses the toggle button after re-collapsing advanced", async () => {
    const user = userEvent.setup();
    renderTurn(CSV_PAYLOAD);
    await user.click(screen.getByRole("button", { name: /show advanced/i }));
    const hideBtn = screen.getByRole("button", { name: /hide advanced/i });
    await user.click(hideBtn);
    // After collapse, focus goes to the toggle button
    const showBtn = screen.getByRole("button", { name: /show advanced/i });
    expect(document.activeElement).toBe(showBtn);
  });
});

// ── 7. Distinctness pin (useId) ───────────────────────────────────────────────

describe("useId distinctness", () => {
  it("two simultaneous instances have distinct DOM nodes for label targets", async () => {
    const user = userEvent.setup();
    const { container: c1 } = render(
      <SchemaFormTurn payload={CSV_PAYLOAD} onSubmit={vi.fn()} />,
    );
    const { container: c2 } = render(
      <SchemaFormTurn payload={CSV_PAYLOAD} onSubmit={vi.fn()} />,
    );

    // Expand advanced in both so all inputs are present
    const showBtns = screen.getAllByRole("button", { name: /show advanced/i });
    for (const btn of showBtns) {
      await user.click(btn);
    }

    // Get path inputs from each container -- they must be distinct DOM nodes
    const pathInputs = screen.getAllByRole("textbox", { name: /path/i });
    expect(pathInputs.length).toBeGreaterThanOrEqual(2);
    expect(pathInputs[0]).not.toBe(pathInputs[1]);

    // IDs must be distinct strings
    const id1 = pathInputs[0].id;
    const id2 = pathInputs[1].id;
    expect(id1).toBeTruthy();
    expect(id2).toBeTruthy();
    expect(id1).not.toBe(id2);

    void c1;
    void c2;
  });
});

// ── 8. Edge cases ─────────────────────────────────────────────────────────────

describe("edge cases", () => {
  it("renders gracefully with no properties (empty form, Continue enabled)", () => {
    renderTurn(EMPTY_SCHEMA_PAYLOAD);
    expect(screen.getByRole("button", { name: /continue/i })).not.toBeDisabled();
    expect(screen.queryByRole("button", { name: /show advanced/i })).toBeNull();
  });

  it("title falls back to property name when title is absent", () => {
    const noTitlePayload: SchemaFormPayload = {
      plugin: "x",
      schema_block: {
        type: "object",
        properties: {
          my_field: { type: "string" },
        },
        required: ["my_field"],
      },
      prefilled: {},
    };
    renderTurn(noTitlePayload);
    // Should render with label derived from property name
    expect(screen.getByRole("textbox", { name: /my_field/i })).toBeInTheDocument();
  });

  it("renders Show advanced (N) with count of optional fields", () => {
    renderTurn(CSV_PAYLOAD); // 3 optional fields
    const btn = screen.getByRole("button", { name: /show advanced/i });
    expect(btn.textContent).toMatch(/3/);
  });

  it("parses valid JSON in a fallback textarea and includes it in submit", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SchemaFormTurn payload={FALLBACK_PAYLOAD} onSubmit={onSubmit} />);
    await user.type(screen.getByRole("textbox", { name: /name/i }), "t");
    await user.click(screen.getByRole("button", { name: /show advanced/i }));

    // Use fireEvent.change for JSON content to avoid userEvent key-descriptor
    // parsing of special characters like "[", '"', and "]".
    const tagsTextarea = screen.getByRole("textbox", { name: /tags/i });
    fireEvent.change(tagsTextarea, { target: { value: '["x","y"]' } });
    await user.click(screen.getByRole("button", { name: /continue/i }));

    const { edited_values } = onSubmit.mock.calls[0][0];
    expect((edited_values as Record<string, unknown>)["tags"]).toEqual(["x", "y"]);
  });
});

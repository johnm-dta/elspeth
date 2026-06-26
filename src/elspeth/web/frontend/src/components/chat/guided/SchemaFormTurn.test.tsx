import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SchemaFormTurn } from "./SchemaFormTurn";
import type { FieldKind, KnobField, SchemaFormPayload } from "@/types/guided";

function pluginPayload(fields: KnobField[], prefilled: Record<string, unknown> = {}): SchemaFormPayload {
  return {
    mode: "plugin_options",
    plugin: "example",
    knobs: { fields },
    prefilled,
  };
}

function field(overrides: Partial<KnobField> & Pick<KnobField, "name" | "kind">): KnobField {
  return {
    label: overrides.name,
    required: false,
    nullable: false,
    ...overrides,
  };
}

function recipeDecisionPayload(
  fields: KnobField[],
  prefilled: Record<string, unknown> = {},
): SchemaFormPayload {
  return {
    mode: "recipe_decision",
    knobs: { fields },
    prefilled,
    recipe_context: {
      recipe_name: "web-scrape-llm-rate-jsonl",
      description: "Scrape URLs, rate with an LLM, write JSONL.",
      alternatives: ["build_manually"],
    },
  };
}

describe("SchemaFormTurn", () => {
  it("renders an enabled Apply recipe and submits the prefilled slots when knobs are empty", async () => {
    // The passive tutorial web-scrape offer prefills ALL required slots, so the
    // emitted offer has knobs.fields == []. Pin that the recipe_decision widget
    // still renders a usable, enabled "Apply recipe" and resubmits the prefilled
    // slots verbatim (the accept-seam binding check requires every prefilled
    // slot be echoed) — the state the passive learner hits at STEP_2.5.
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    const prefilled = {
      source_blob_id: "blob-123",
      source_plugin: "json",
      output_path: "outputs/ratings.jsonl",
      model: "anthropic/claude-sonnet-4.6",
      api_key_secret: "OPENROUTER_API_KEY",
      abuse_contact: "noreply@demo.com",
      scraping_reason: "demo",
    };
    render(
      <SchemaFormTurn
        payload={recipeDecisionPayload([], prefilled)}
        onSubmit={onSubmit}
      />,
    );
    const button = screen.getByRole("button", { name: "Apply recipe" });
    expect(button).toBeEnabled();
    await user.click(button);
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        chosen: ["accept"],
        edited_values: {
          recipe_name: "web-scrape-llm-rate-jsonl",
          slots: prefilled,
        },
      }),
    );
  });

  it.each([
    ["text", "textbox"],
    ["number-int", "spinbutton"],
    ["number-float", "spinbutton"],
    ["blob-ref", "textbox"],
  ] satisfies Array<[FieldKind, string]>)("renders %s as an editable control", (kind, role) => {
    render(<SchemaFormTurn payload={pluginPayload([field({ name: kind, label: kind, kind })])} onSubmit={vi.fn()} />);

    expect(screen.getByRole(role, { name: kind })).toBeInTheDocument();
  });

  it("renders checkbox, enum, string-list, and JSON kinds", () => {
    render(
      <SchemaFormTurn
        payload={pluginPayload([
          field({ name: "enabled", label: "Enabled", kind: "checkbox" }),
          field({ name: "provider", label: "Provider", kind: "enum", enum: ["azure", "openrouter"] }),
          field({ name: "columns", label: "Columns", kind: "string-list" }),
          field({ name: "obj", label: "Object", kind: "json-object" }),
          field({ name: "arr", label: "Array", kind: "json-array" }),
          field({ name: "val", label: "Value", kind: "json-value" }),
        ])}
        onSubmit={vi.fn()}
      />,
    );

    expect(screen.getByRole("checkbox", { name: "Enabled" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Provider" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Columns" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Object" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Array" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Value" })).toBeInTheDocument();
  });

  it("uses aria-describedby for field descriptions", () => {
    render(
      <SchemaFormTurn
        payload={pluginPayload([
          field({ name: "path", label: "Path", kind: "text", description: "Filesystem path" }),
        ])}
        onSubmit={vi.fn()}
      />,
    );

    const input = screen.getByRole("textbox", { name: "Path" });
    const hint = screen.getByText("Filesystem path");
    expect(input).toHaveAttribute("aria-describedby", hint.id);
  });

  it("submits only visible fields and drops variant state on discriminator change", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <SchemaFormTurn
        payload={pluginPayload(
          [
            field({ name: "provider", label: "Provider", kind: "enum", enum: ["azure", "openrouter"], required: true }),
            field({
              name: "deployment_name",
              label: "Deployment",
              kind: "text",
              visible_when: { field: "provider", equals: "azure" },
            }),
            field({
              name: "base_url",
              label: "Base URL",
              kind: "text",
              visible_when: { field: "provider", equals: "openrouter" },
            }),
          ],
          { provider: "azure" },
        )}
        onSubmit={onSubmit}
      />,
    );

    await user.type(screen.getByRole("textbox", { name: "Deployment" }), "gpt-4");
    await user.selectOptions(screen.getByRole("combobox", { name: "Provider" }), "openrouter");
    expect(screen.queryByRole("textbox", { name: "Deployment" })).not.toBeInTheDocument();
    await user.type(screen.getByRole("textbox", { name: "Base URL" }), "https://openrouter.ai/api/v1");
    await user.click(screen.getByRole("button", { name: "Continue" }));

    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        edited_values: expect.objectContaining({
          options: {
            provider: "openrouter",
            base_url: "https://openrouter.ai/api/v1",
          },
        }),
      }),
    );
  });

  it("disables continue when a required text field is cleared", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <SchemaFormTurn
        payload={pluginPayload([field({ name: "template", label: "Template", kind: "text", required: true })], {
          template: "hello",
        })}
        onSubmit={onSubmit}
      />,
    );

    const input = screen.getByRole("textbox", { name: "Template" });
    await user.clear(input);
    const button = screen.getByRole("button", { name: "Continue" });
    expect(button).toBeDisabled();
    await user.click(button);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("submits nullable cleared text as null", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <SchemaFormTurn
        payload={pluginPayload([field({ name: "system_prompt", label: "System Prompt", kind: "text", nullable: true })], {
          system_prompt: "initial",
        })}
        onSubmit={onSubmit}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Clear System Prompt" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));

    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        edited_values: expect.objectContaining({
          options: { system_prompt: null },
        }),
      }),
    );
  });

  it("supports keyboard editing of string-list fields", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <SchemaFormTurn
        payload={pluginPayload([field({ name: "columns", label: "Columns", kind: "string-list" })])}
        onSubmit={onSubmit}
      />,
    );

    const textarea = screen.getByRole("textbox", { name: "Columns" });
    await user.click(textarea);
    await user.keyboard("name{enter}age");
    await user.click(screen.getByRole("button", { name: "Continue" }));

    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        edited_values: expect.objectContaining({
          options: { columns: ["name", "age"] },
        }),
      }),
    );
  });

  it("parses number and JSON field values on submit", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <SchemaFormTurn
        payload={pluginPayload([
          field({ name: "count", label: "Count", kind: "number-int" }),
          field({ name: "temperature", label: "Temperature", kind: "number-float" }),
          field({ name: "obj", label: "Object", kind: "json-object" }),
          field({ name: "arr", label: "Array", kind: "json-array" }),
          field({ name: "val", label: "Value", kind: "json-value" }),
        ])}
        onSubmit={onSubmit}
      />,
    );

    await user.type(screen.getByRole("spinbutton", { name: "Count" }), "7");
    await user.type(screen.getByRole("spinbutton", { name: "Temperature" }), "0.5");
    fireEvent.change(screen.getByRole("textbox", { name: "Object" }), { target: { value: '{"ok":true}' } });
    fireEvent.change(screen.getByRole("textbox", { name: "Array" }), { target: { value: '["a","b"]' } });
    fireEvent.change(screen.getByRole("textbox", { name: "Value" }), { target: { value: '"x"' } });
    await user.click(screen.getByRole("button", { name: "Continue" }));

    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        edited_values: expect.objectContaining({
          options: {
            count: 7,
            temperature: 0.5,
            obj: { ok: true },
            arr: ["a", "b"],
            val: "x",
          },
        }),
      }),
    );
  });

  it("renders recipe context and submits recipe slot decisions", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <SchemaFormTurn
        payload={{
          mode: "recipe_decision",
          knobs: {
            fields: [field({ name: "threshold", label: "Threshold", kind: "number-float", required: true })],
          },
          prefilled: { source_blob_id: "blob-1" },
          recipe_context: {
            recipe_name: "split-by-score",
            description: "Split rows by score",
            alternatives: ["build_manually"],
          },
        }}
        onSubmit={onSubmit}
      />,
    );

    expect(screen.getByRole("heading", { level: 3, name: "split-by-score" })).toBeInTheDocument();
    await user.type(screen.getByRole("spinbutton", { name: "Threshold" }), "0.9");
    await user.click(screen.getByRole("button", { name: "Apply recipe" }));

    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        chosen: ["accept"],
        edited_values: {
          recipe_name: "split-by-score",
          slots: { source_blob_id: "blob-1", threshold: 0.9 },
        },
      }),
    );
  });

  it("submits build_manually for recipe alternatives", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <SchemaFormTurn
        payload={{
          mode: "recipe_decision",
          knobs: { fields: [] },
          prefilled: {},
          recipe_context: {
            recipe_name: "split-by-score",
            description: "Split rows by score",
            alternatives: ["build_manually"],
          },
        }}
        onSubmit={onSubmit}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Build manually" }));

    expect(onSubmit).toHaveBeenCalledWith({
      chosen: ["build_manually"],
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  });

  describe("tutorial validation-failure teaching copy", () => {
    const ovfField = field({ name: "on_validation_failure", kind: "text", required: true });

    it("shows the quarantine-sink teaching line on a tutorial source form with on_validation_failure", () => {
      render(
        <SchemaFormTurn
          payload={pluginPayload([ovfField], { on_validation_failure: "discard" })}
          onSubmit={vi.fn()}
          isTutorial
        />,
      );
      expect(screen.getByText(/quarantine sink/i)).toBeInTheDocument();
    });

    it("does not show the teaching line when not in tutorial mode", () => {
      render(
        <SchemaFormTurn
          payload={pluginPayload([ovfField], { on_validation_failure: "discard" })}
          onSubmit={vi.fn()}
        />,
      );
      expect(screen.queryByText(/quarantine sink/i)).not.toBeInTheDocument();
    });

    it("does not show the teaching line in tutorial mode when the form has no on_validation_failure knob", () => {
      render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "path", kind: "text", required: true })])}
          onSubmit={vi.fn()}
          isTutorial
        />,
      );
      expect(screen.queryByText(/quarantine sink/i)).not.toBeInTheDocument();
    });
  });
});

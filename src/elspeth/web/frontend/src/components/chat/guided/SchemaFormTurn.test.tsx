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
  ] satisfies Array<[FieldKind, string]>)("renders %s as an editable control", async (kind, role) => {
    const user = userEvent.setup();
    render(<SchemaFormTurn payload={pluginPayload([field({ name: kind, label: kind, kind })])} onSubmit={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Edit" }));
    expect(screen.getByRole(role, { name: kind })).toBeInTheDocument();
  });

  it("renders checkbox, enum, string-list, and JSON kinds", async () => {
    const user = userEvent.setup();
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

    await user.click(screen.getByRole("button", { name: "Edit" }));
    expect(screen.getByRole("checkbox", { name: "Enabled" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Provider" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Columns" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Object" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Array" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Value" })).toBeInTheDocument();
  });

  it("uses aria-describedby for field descriptions", async () => {
    const user = userEvent.setup();
    render(
      <SchemaFormTurn
        payload={pluginPayload([
          field({ name: "path", label: "Path", kind: "text", description: "Filesystem path" }),
        ])}
        onSubmit={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Edit" }));
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

    await user.click(screen.getByRole("button", { name: "Edit" }));
    await user.type(screen.getByRole("textbox", { name: "Deployment" }), "gpt-4");
    // `Provider` is required, so its accessible name now carries the "(required)"
    // cue — match on the base label.
    await user.selectOptions(screen.getByRole("combobox", { name: /Provider/ }), "openrouter");
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

    await user.click(screen.getByRole("button", { name: "Edit" }));
    // `Template` is required, so its accessible name now carries the "(required)"
    // cue — match on the base label.
    const input = screen.getByRole("textbox", { name: /Template/ });
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

    await user.click(screen.getByRole("button", { name: "Edit" }));
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

    await user.click(screen.getByRole("button", { name: "Edit" }));
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

    await user.click(screen.getByRole("button", { name: "Edit" }));
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
    await user.click(screen.getByRole("button", { name: "Edit" }));
    // `Threshold` is required, so its accessible name now carries the
    // "(required)" cue — match on the base label.
    await user.type(screen.getByRole("spinbutton", { name: /Threshold/ }), "0.9");
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

  // M14 (WCAG 3.3.1 / 3.3.2): required fields must announce themselves
  // visibly + programmatically, and the cases the form already knows are bad
  // (broken JSON, non-numeric numbers) must surface an inline error instead of
  // only disabling Continue.
  describe("required marking and inline validation", () => {
    it("marks required non-checkbox fields visibly and programmatically", async () => {
      const user = userEvent.setup();
      const { container } = render(
        <SchemaFormTurn
          payload={pluginPayload([
            field({ name: "token", label: "Token", kind: "text", required: true }),
            field({ name: "provider", label: "Provider", kind: "enum", enum: ["a", "b"], required: true }),
            // A checkbox always carries a boolean value, so the form never gates
            // it as required (canSubmit skips it) — it must NOT be marked.
            field({ name: "flag", label: "Flag", kind: "checkbox", required: true }),
          ])}
          onSubmit={vi.fn()}
        />,
      );

      await user.click(screen.getByRole("button", { name: "Edit" }));
      const token = screen.getByRole("textbox", { name: /Token/ });
      expect(token).toBeRequired();
      expect(token).toHaveAttribute("aria-required", "true");
      expect(token).toHaveAccessibleName(/\(required\)/);

      const provider = screen.getByRole("combobox", { name: /Provider/ });
      expect(provider).toBeRequired();
      expect(provider).toHaveAttribute("aria-required", "true");
      expect(provider).toHaveAccessibleName(/\(required\)/);

      const flag = screen.getByRole("checkbox", { name: "Flag" });
      expect(flag).not.toBeRequired();
      expect(flag).not.toHaveAttribute("aria-required");
      expect(flag).toHaveAccessibleName("Flag");

      // One visible (aria-hidden) asterisk per marked field — and none for the
      // checkbox.
      const markers = container.querySelectorAll(".guided-schema-required-marker");
      expect(markers).toHaveLength(2);
      markers.forEach((marker) => {
        expect(marker).toHaveTextContent("*");
        expect(marker).toHaveAttribute("aria-hidden", "true");
      });
    });

    it("does not mark optional fields as required", async () => {
      const user = userEvent.setup();
      const { container } = render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "note", label: "Note", kind: "text" })])}
          onSubmit={vi.fn()}
        />,
      );

      await user.click(screen.getByRole("button", { name: "Edit" }));
      const note = screen.getByRole("textbox", { name: "Note" });
      expect(note).not.toBeRequired();
      expect(note).not.toHaveAttribute("aria-required");
      expect(note).toHaveAccessibleName("Note");
      expect(container.querySelector(".guided-schema-required-marker")).toBeNull();
    });

    it("flags invalid JSON inline and blocks Continue until it parses", async () => {
      const user = userEvent.setup();
      const onSubmit = vi.fn();
      render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "cfg", label: "Config", kind: "json-object" })])}
          onSubmit={onSubmit}
        />,
      );

      await user.click(screen.getByRole("button", { name: "Edit" }));
      const textarea = screen.getByRole("textbox", { name: "Config" });
      fireEvent.change(textarea, { target: { value: "{not valid" } });

      expect(textarea).toHaveAttribute("aria-invalid", "true");
      const error = screen.getByRole("alert");
      expect(error).toHaveTextContent(/invalid json/i);
      expect(textarea).toHaveAttribute("aria-describedby", error.id);
      expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();

      // Correcting the text clears the error and re-enables submit.
      fireEvent.change(textarea, { target: { value: '{"ok":true}' } });
      expect(textarea).not.toHaveAttribute("aria-invalid");
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
    });

    it("flags a non-integer in an integer field inline and blocks Continue, then submits the corrected number", async () => {
      const user = userEvent.setup();
      const onSubmit = vi.fn();
      render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "n", label: "Count", kind: "number-int" })])}
          onSubmit={onSubmit}
        />,
      );

      await user.click(screen.getByRole("button", { name: "Edit" }));
      const input = screen.getByRole("spinbutton", { name: "Count" });
      // "1.5" was silently truncated to 1 before this fix — now it's surfaced.
      fireEvent.change(input, { target: { value: "1.5" } });

      expect(input).toHaveAttribute("aria-invalid", "true");
      const error = screen.getByRole("alert");
      expect(error).toHaveTextContent(/whole number/i);
      expect(input).toHaveAttribute("aria-describedby", error.id);
      expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();

      // A valid integer clears the error and submits the real number (proving
      // the raw-text-on-invalid path does not corrupt a subsequent good value).
      fireEvent.change(input, { target: { value: "2" } });
      expect(input).not.toHaveAttribute("aria-invalid");
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      const submit = screen.getByRole("button", { name: "Continue" });
      expect(submit).toBeEnabled();
      fireEvent.click(submit);
      expect(onSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          edited_values: expect.objectContaining({ options: { n: 2 } }),
        }),
      );
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

    it("attaches the caveat to the on_validation_failure summary row in tutorial mode", () => {
      const { container } = render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "on_validation_failure", label: "On Validation Failure", kind: "text", required: true })], {
            on_validation_failure: "discard",
          })}
          onSubmit={vi.fn()}
          isTutorial
        />,
      );
      const caveat = container.querySelector(".guided-schema-summary-caveat");
      expect(caveat).not.toBeNull();
      expect(caveat?.textContent ?? "").toMatch(/quarantine sink/i);
    });

    it("masks an absolute blob storage_path to its friendly basename in tutorial summary mode", () => {
      // Path-leak guard (retargeted to the summary-first surface): a blob-backed
      // source commits the server's absolute storage_path. The tutorial renders
      // summary-only (no Edit affordance), so the friendly basename appears as
      // read-only summary text and no editable input exists at all.
      const { container } = render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "path", kind: "text", required: true })], {
            path: "/home/john/elspeth/data/blobs/sess/cb7f1f46-b724-4472-9acb-1680cefef45e_project_pages.json",
          })}
          onSubmit={vi.fn()}
          isTutorial
        />,
      );
      expect(screen.getByText("project_pages.json")).toBeInTheDocument();
      expect(screen.queryByText(/\/home\/john\/elspeth\/data\/blobs/)).not.toBeInTheDocument();
      expect(container.querySelector(".guided-schema-input")).toBeNull();
    });

    it("does NOT mask the path outside tutorial mode (operator sees the real value)", async () => {
      const user = userEvent.setup();
      render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "path", kind: "text", required: true })], {
            path: "/home/john/elspeth/data/blobs/sess/cb7f1f46-b724-4472-9acb-1680cefef45e_project_pages.json",
          })}
          onSubmit={vi.fn()}
        />,
      );
      await user.click(screen.getByRole("button", { name: "Edit" }));
      // `path` is required, so its label now carries the "(required)" cue —
      // match on the base label.
      const input = screen.getByLabelText(/path/) as HTMLInputElement;
      expect(input.value).toBe(
        "/home/john/elspeth/data/blobs/sess/cb7f1f46-b724-4472-9acb-1680cefef45e_project_pages.json",
      );
      expect(input).not.toHaveAttribute("readonly");
    });

    it("submits the REAL absolute path (not the mask) on Continue in tutorial mode", async () => {
      // Load-bearing: the mask is display-only. The committed pipeline must still
      // receive the real storage_path so the run can read the blob.
      const onSubmit = vi.fn();
      const realPath =
        "/home/john/elspeth/data/blobs/sess/cb7f1f46-b724-4472-9acb-1680cefef45e_project_pages.json";
      render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "path", kind: "text", required: true })], {
            path: realPath,
          })}
          onSubmit={onSubmit}
          isTutorial
        />,
      );
      await userEvent.click(screen.getByRole("button", { name: /continue/i }));
      expect(onSubmit).toHaveBeenCalledTimes(1);
      const body = onSubmit.mock.calls[0][0] as {
        edited_values: { options: Record<string, unknown> };
      };
      expect(body.edited_values.options.path).toBe(realPath);
    });
  });

  describe("blob:<ref> sentinel masking", () => {
    // The guided emitter masks a blob-backed source's absolute storage_path as
    // a stable `blob:<blob_ref>` wire sentinel (BLOB_REF_PATH_PREFIX,
    // web/composer/guided/protocol.py). A raw UUID means nothing to a learner,
    // so both views render a friendly label instead — the sentinel stays in
    // form state and flows to submit unchanged, mirroring the /-path mask.
    const blobSentinel = "blob:cb7f1f46-b724-4472-9acb-1680cefef45e";

    it("masks a blob:<ref> path to the friendly label in the summary view", () => {
      render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "path", kind: "text", required: true })], {
            path: blobSentinel,
          })}
          onSubmit={vi.fn()}
        />,
      );
      expect(screen.getByText("Uploaded sample data")).toBeInTheDocument();
      expect(screen.queryByText(blobSentinel)).not.toBeInTheDocument();
    });

    it("masks a blob:<ref> path in the edit view (read-only), outside tutorial mode too", async () => {
      const user = userEvent.setup();
      render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "path", kind: "text", required: true })], {
            path: blobSentinel,
          })}
          onSubmit={vi.fn()}
        />,
      );
      await user.click(screen.getByRole("button", { name: "Edit" }));
      const input = screen.getByLabelText(/path/) as HTMLInputElement;
      expect(input.value).toBe("Uploaded sample data");
      expect(input).toHaveAttribute("readonly");
    });

    it("submits the REAL blob:<ref> sentinel (not the label) on Continue", async () => {
      // Load-bearing: the mask is display-only. The commit handler re-resolves
      // the sentinel server-side, so the submitted value must stay `blob:<ref>`.
      const onSubmit = vi.fn();
      render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "path", kind: "text", required: true })], {
            path: blobSentinel,
          })}
          onSubmit={onSubmit}
        />,
      );
      await userEvent.click(screen.getByRole("button", { name: /continue/i }));
      expect(onSubmit).toHaveBeenCalledTimes(1);
      const body = onSubmit.mock.calls[0][0] as {
        edited_values: { options: Record<string, unknown> };
      };
      expect(body.edited_values.options.path).toBe(blobSentinel);
    });

    it("masks a blob:<ref> path in tutorial summary mode", () => {
      render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "path", kind: "text", required: true })], {
            path: blobSentinel,
          })}
          onSubmit={vi.fn()}
          isTutorial
        />,
      );
      expect(screen.getByText("Uploaded sample data")).toBeInTheDocument();
      expect(screen.queryByText(blobSentinel)).not.toBeInTheDocument();
    });
  });

  describe("read-only summary view", () => {
    it("renders prefilled scalar knobs as read-only text, not editable controls", () => {
      render(
        <SchemaFormTurn
          payload={pluginPayload(
            [
              field({ name: "encoding", label: "Encoding", kind: "text" }),
              field({ name: "skip_rows", label: "Skip Rows", kind: "number-int" }),
              field({ name: "enabled", label: "Enabled", kind: "checkbox" }),
            ],
            { encoding: "utf-8", skip_rows: 0, enabled: true },
          )}
          onSubmit={vi.fn()}
        />,
      );
      // Summary is the default — no schema-form inputs are rendered up front.
      expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
      expect(screen.queryByRole("spinbutton")).not.toBeInTheDocument();
      expect(screen.getByText("Encoding")).toBeInTheDocument();
      expect(screen.getByText("utf-8")).toBeInTheDocument();
      expect(screen.getByText("Skip Rows")).toBeInTheDocument();
      expect(screen.getByText("0")).toBeInTheDocument();
      expect(screen.getByText("Yes")).toBeInTheDocument(); // checkbox -> Yes/No
    });

    // Empty-row treatment (elspeth-eba8820005): never a literal "null" or
    // "(none)" — optional-empty rows are elided, required-empty rows render a
    // muted "Not set".
    it("elides an optional field with an empty value from the summary", () => {
      render(
        <SchemaFormTurn
          payload={pluginPayload(
            [
              field({ name: "columns", label: "Columns", kind: "string-list" }),
              field({ name: "encoding", label: "Encoding", kind: "text" }),
            ],
            { columns: [], encoding: "utf-8" },
          )}
          onSubmit={vi.fn()}
        />,
      );
      expect(screen.queryByText("Columns")).not.toBeInTheDocument();
      expect(screen.queryByText("(none)")).not.toBeInTheDocument();
      expect(screen.getByText("Encoding")).toBeInTheDocument();
    });

    it("renders a required-but-empty field as a muted 'Not set', and names it in the banner", () => {
      render(
        <SchemaFormTurn
          payload={pluginPayload([
            field({ name: "format", label: "Format", kind: "text", required: true }),
          ])}
          onSubmit={vi.fn()}
        />,
      );
      const notSet = screen.getByText("Not set");
      expect(notSet).toHaveClass("guided-schema-summary-not-set");
      expect(
        screen.getByText("1 value needs attention: Format — open Edit to review."),
      ).toBeInTheDocument();
    });

    it("never renders a literal null for a JSON field holding null (elided when optional, Not set when required)", () => {
      const { rerender } = render(
        <SchemaFormTurn
          payload={pluginPayload(
            [field({ name: "mapping", label: "Field Mapping", kind: "json-value" })],
            { mapping: null },
          )}
          onSubmit={vi.fn()}
        />,
      );
      // Optional + null → the row is elided entirely.
      expect(screen.queryByText("Field Mapping")).not.toBeInTheDocument();
      expect(screen.queryByText("null")).not.toBeInTheDocument();

      rerender(
        <SchemaFormTurn
          payload={pluginPayload(
            [field({ name: "mapping", label: "Field Mapping", kind: "json-value", required: true })],
            { mapping: null },
          )}
          onSubmit={vi.fn()}
        />,
      );
      // Required + null → the row stays, with a muted "Not set", never "null".
      expect(screen.getByText("Field Mapping")).toBeInTheDocument();
      expect(screen.getByText("Not set")).toBeInTheDocument();
      expect(screen.queryByText("null")).not.toBeInTheDocument();
    });

    it("shows a plain 'no settings need review' line when every row is optional-and-empty", () => {
      render(
        <SchemaFormTurn
          payload={pluginPayload(
            [
              field({ name: "columns", label: "Columns", kind: "string-list" }),
              field({ name: "data_key", label: "Data key", kind: "text" }),
            ],
            { columns: [] },
          )}
          onSubmit={vi.fn()}
        />,
      );
      expect(
        screen.getByText("No settings need review for this step."),
      ).toBeInTheDocument();
    });

    it("renders a JSON-shaped knob value through CodeBlock (pretty/highlighted)", () => {
      const { container } = render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "schema", label: "Schema", kind: "json-object" })], {
            schema: { mode: "observed", guaranteed_fields: ["url"] },
          })}
          onSubmit={vi.fn()}
        />,
      );
      expect(container.querySelector("[data-codeblock-format]")).not.toBeNull();
    });

    it("masks an absolute blob path to its friendly basename in the summary", () => {
      render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "path", label: "Path", kind: "text" })], {
            path: "/home/u/data/blobs/sess/cb7f1f46-b724-4472-9acb-1680cefef45e_project_pages.json",
          })}
          onSubmit={vi.fn()}
        />,
      );
      expect(screen.getByText("project_pages.json")).toBeInTheDocument();
      expect(screen.queryByText(/\/home\/u\/data\/blobs/)).not.toBeInTheDocument();
    });

    it("hides a visible_when-gated field from the summary when its predicate is unmet", () => {
      render(
        <SchemaFormTurn
          payload={pluginPayload(
            [
              field({ name: "provider", label: "Provider", kind: "enum", enum: ["azure", "openrouter"] }),
              field({ name: "deployment_name", label: "Deployment", kind: "text", visible_when: { field: "provider", equals: "azure" } }),
            ],
            { provider: "openrouter" },
          )}
          onSubmit={vi.fn()}
        />,
      );
      expect(screen.queryByText("Deployment")).not.toBeInTheDocument();
    });

    // FORWARD GUARD (not a genuine RED — today's form already submits prefilled
    // values on Continue): pins that the summary-default path keeps submit parity.
    it("submits the prefilled values verbatim from the summary (no edit)", async () => {
      const user = userEvent.setup();
      const onSubmit = vi.fn();
      render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "encoding", label: "Encoding", kind: "text" })], { encoding: "utf-8" })}
          onSubmit={onSubmit}
        />,
      );
      await user.click(screen.getByRole("button", { name: "Continue" }));
      expect(onSubmit).toHaveBeenCalledWith(
        expect.objectContaining({ edited_values: expect.objectContaining({ options: { encoding: "utf-8" } }) }),
      );
    });
  });

  describe("edit toggle", () => {
    it("reveals the editable form on Edit, and returns on Done editing (non-tutorial)", async () => {
      const user = userEvent.setup();
      render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "encoding", label: "Encoding", kind: "text" })], { encoding: "utf-8" })}
          onSubmit={vi.fn()}
        />,
      );
      expect(screen.queryByRole("textbox", { name: "Encoding" })).not.toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: "Edit" }));
      expect(screen.getByRole("textbox", { name: "Encoding" })).toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: "Done editing" }));
      expect(screen.queryByRole("textbox", { name: "Encoding" })).not.toBeInTheDocument();
    });

    it("does NOT render an Edit button in tutorial mode", () => {
      render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "encoding", label: "Encoding", kind: "text" })], { encoding: "utf-8" })}
          onSubmit={vi.fn()}
          isTutorial
        />,
      );
      expect(screen.queryByRole("button", { name: "Edit" })).not.toBeInTheDocument();
    });

    it("submits the edited value after editing via the form", async () => {
      const user = userEvent.setup();
      const onSubmit = vi.fn();
      render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "encoding", label: "Encoding", kind: "text" })], { encoding: "utf-8" })}
          onSubmit={onSubmit}
        />,
      );
      await user.click(screen.getByRole("button", { name: "Edit" }));
      const input = screen.getByRole("textbox", { name: "Encoding" });
      await user.clear(input);
      await user.type(input, "latin-1");
      await user.click(screen.getByRole("button", { name: "Continue" }));
      expect(onSubmit).toHaveBeenCalledWith(
        expect.objectContaining({ edited_values: expect.objectContaining({ options: { encoding: "latin-1" } }) }),
      );
    });

    it("blocks Done editing while a field holds invalid JSON, then allows it once corrected", async () => {
      const user = userEvent.setup();
      render(
        <SchemaFormTurn
          payload={pluginPayload([field({ name: "cfg", label: "Config", kind: "json-object" })], { cfg: { ok: true } })}
          onSubmit={vi.fn()}
        />,
      );
      await user.click(screen.getByRole("button", { name: "Edit" }));
      fireEvent.change(screen.getByRole("textbox", { name: "Config" }), { target: { value: "{bad" } });
      expect(screen.getByRole("button", { name: "Done editing" })).toBeDisabled();
      fireEvent.change(screen.getByRole("textbox", { name: "Config" }), { target: { value: '{"ok":false}' } });
      expect(screen.getByRole("button", { name: "Done editing" })).toBeEnabled();
    });

    it("shows a needs-edit banner NAMING the blocking fields when unfilled required fields block Continue", () => {
      render(
        <SchemaFormTurn
          payload={pluginPayload([
            field({ name: "token", label: "Token", kind: "text", required: true }),
            field({ name: "data_key", label: "Data key", kind: "text", required: true }),
          ])}
          onSubmit={vi.fn()}
        />,
      );
      // Names the fields (elspeth-eba8820005) instead of "Some values need
      // attention" — the user should not have to open Edit and hunt.
      expect(
        screen.getByText(
          "2 values need attention: Token, Data key — open Edit to review.",
        ),
      ).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
    });
  });
});

// ============================================================================
// PluginCard — rendering regression coverage for bug elspeth-dcf12c061b.
//
// Pins the two JSON-Schema shapes the card renders:
//   1. Flat ``{properties, required}`` from single-model plugins.
//   2. Pydantic discriminated union (``oneOf`` + ``$defs`` + ``discriminator``)
//      from plugins like the LLM transform — rendered as one section per
//      variant, labelled via the discriminator mapping.
// ============================================================================

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PluginCard } from "./PluginCard";
import type { PluginSummary, PluginSchemaInfo } from "@/types/index";

function makePlugin(overrides: Partial<PluginSummary> = {}): PluginSummary {
  return {
    name: "example",
    plugin_type: "transform",
    description: "An example plugin",
    config_fields: [],
    usage_when_to_use: null,
    usage_when_not_to_use: null,
    example_use: null,
    capability_tags: [],
    audit_characteristics: [],
    ...overrides,
  };
}

const FLAT_SCHEMA: PluginSchemaInfo = {
  name: "csv",
  plugin_type: "source",
  description: "CSV source",
  json_schema: {
    properties: {
      path: { type: "string", description: "File path" },
      encoding: { type: "string" },
    },
    required: ["path"],
  },
};

// Minimal Pydantic-shaped discriminated union (real LLM transform shape,
// trimmed). ``$defs`` entry names match the ``discriminator.mapping`` values
// so the label-resolution path exercises the real production contract.
const DISCRIMINATED_SCHEMA: PluginSchemaInfo = {
  name: "llm",
  plugin_type: "transform",
  description: "LLM transform",
  json_schema: {
    oneOf: [
      { $ref: "#/$defs/AzureOpenAIConfig" },
      { $ref: "#/$defs/OpenRouterConfig" },
    ],
    discriminator: {
      propertyName: "provider",
      mapping: {
        azure: "#/$defs/AzureOpenAIConfig",
        openrouter: "#/$defs/OpenRouterConfig",
      },
    },
    $defs: {
      AzureOpenAIConfig: {
        properties: {
          deployment_name: { type: "string", description: "Azure deployment" },
          endpoint: { type: "string" },
          api_key: { type: "string", description: "Azure API key" },
        },
        required: ["deployment_name", "endpoint", "api_key"],
      },
      OpenRouterConfig: {
        properties: {
          model: { type: "string", description: "OpenRouter model id" },
          api_key: { type: "string", description: "OpenRouter API key" },
        },
        required: ["model", "api_key"],
      },
    },
  },
};

describe("PluginCard — collapsed header", () => {
  it("renders plugin name and description without expanding", () => {
    render(
      <PluginCard
        plugin={makePlugin({ name: "csv", description: "CSV source" })}
        schema={null}
        onExpand={vi.fn()}
      />,
    );
    expect(screen.getByText("csv")).toBeInTheDocument();
    expect(screen.getByText("CSV source")).toHaveAttribute("title", "CSV source");
    // Expanded content does NOT render while collapsed.
    expect(screen.queryByText(/Loading/)).not.toBeInTheDocument();
  });
});

describe("PluginCard — flat single-model schema", () => {
  it("links the schema disclosure to the expanded schema panel", async () => {
    const user = userEvent.setup();
    render(
      <PluginCard
        plugin={makePlugin({ name: "csv loader", plugin_type: "source" })}
        schema={FLAT_SCHEMA}
        onExpand={vi.fn()}
      />,
    );

    const disclosure = screen.getByRole("button", { name: /schema for csv loader/i });
    expect(disclosure).toHaveAttribute(
      "aria-controls",
      "plugin-card-schema-panel-source-csv-loader",
    );

    await user.click(disclosure);
    const panel = screen.getByText("path").closest(".plugin-card-expanded");
    expect(panel).toHaveAttribute("id", "plugin-card-schema-panel-source-csv-loader");
  });

  it("renders each property and marks the required ones", async () => {
    const user = userEvent.setup();
    render(
      <PluginCard
        plugin={makePlugin({ name: "csv" })}
        schema={FLAT_SCHEMA}
        onExpand={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: /schema for csv/i }));
    expect(screen.getByText("path")).toBeInTheDocument();
    expect(screen.getByText("encoding")).toBeInTheDocument();
    // Required badge appears for ``path`` but not ``encoding``.
    const badges = screen.getAllByText("required");
    expect(badges).toHaveLength(1);
    expect(screen.getByText("File path")).toBeInTheDocument();
  });
});

describe("PluginCard — discriminated union", () => {
  it("renders one section per variant labelled by discriminator value", async () => {
    const user = userEvent.setup();
    render(
      <PluginCard
        plugin={makePlugin({ name: "llm" })}
        schema={DISCRIMINATED_SCHEMA}
        onExpand={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: /schema for llm/i }));
    // Labels come from the discriminator mapping (provider: azure / openrouter),
    // NOT the raw $defs class names (AzureOpenAIConfig / OpenRouterConfig).
    expect(screen.getByText("provider: azure")).toBeInTheDocument();
    expect(screen.getByText("provider: openrouter")).toBeInTheDocument();
    expect(screen.queryByText("provider: AzureOpenAIConfig")).not.toBeInTheDocument();
  });

  it("marks a field required only within variants whose required list names it", async () => {
    const user = userEvent.setup();
    render(
      <PluginCard
        plugin={makePlugin({ name: "llm" })}
        schema={DISCRIMINATED_SCHEMA}
        onExpand={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: /schema for llm/i }));
    // Azure variant: deployment_name, endpoint, api_key required (3 badges).
    // OpenRouter variant: model, api_key required (2 badges).
    // Total: 5 required badges across the rendered card.
    const badges = screen.getAllByText("required");
    expect(badges).toHaveLength(5);
    // Both Azure-only and OpenRouter-only fields are rendered.
    expect(screen.getByText("deployment_name")).toBeInTheDocument();
    expect(screen.getByText("endpoint")).toBeInTheDocument();
    expect(screen.getByText("model")).toBeInTheDocument();
    // ``api_key`` appears in both variants — there should be two nodes.
    expect(screen.getAllByText("api_key")).toHaveLength(2);
  });

  it("falls back to def name when discriminator mapping is absent", async () => {
    const user = userEvent.setup();
    const schemaWithoutMapping: PluginSchemaInfo = {
      ...DISCRIMINATED_SCHEMA,
      json_schema: {
        ...(DISCRIMINATED_SCHEMA.json_schema as Record<string, unknown>),
        discriminator: { propertyName: "provider" },
      },
    };
    render(
      <PluginCard
        plugin={makePlugin()}
        schema={schemaWithoutMapping}
        onExpand={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: /schema for example/i }));
    // Without mapping, the $defs class name is the fallback label.
    expect(screen.getByText("provider: AzureOpenAIConfig")).toBeInTheDocument();
    expect(screen.getByText("provider: OpenRouterConfig")).toBeInTheDocument();
  });

  it("defaults discriminator label prefix to 'variant' when propertyName missing", async () => {
    const user = userEvent.setup();
    const schemaWithoutDiscProp: PluginSchemaInfo = {
      ...DISCRIMINATED_SCHEMA,
      json_schema: {
        oneOf: [{ $ref: "#/$defs/A" }],
        $defs: { A: { properties: {}, required: [] } },
      },
    };
    render(
      <PluginCard
        plugin={makePlugin()}
        schema={schemaWithoutDiscProp}
        onExpand={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: /schema for example/i }));
    expect(screen.getByText("variant: A")).toBeInTheDocument();
  });

  it("skips oneOf entries whose $ref does not target local $defs", async () => {
    const user = userEvent.setup();
    const mixedRefSchema: PluginSchemaInfo = {
      ...DISCRIMINATED_SCHEMA,
      json_schema: {
        oneOf: [
          { $ref: "#/components/schemas/External" },
          { $ref: "#/$defs/Local" },
        ],
        discriminator: {
          propertyName: "provider",
          mapping: { local: "#/$defs/Local" },
        },
        $defs: {
          Local: {
            properties: { field: { type: "string" } },
            required: ["field"],
          },
        },
      },
    };
    render(
      <PluginCard
        plugin={makePlugin()}
        schema={mixedRefSchema}
        onExpand={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: /schema for example/i }));
    expect(screen.getByText("provider: local")).toBeInTheDocument();
    expect(screen.queryByText(/External/)).not.toBeInTheDocument();
  });

  it("silently drops oneOf refs whose $defs entry is missing", async () => {
    // Mirror of the backend policy at the boundary: a dangling $ref in the
    // frontend should not crash the render — the variant is simply absent.
    const user = userEvent.setup();
    const danglingSchema: PluginSchemaInfo = {
      ...DISCRIMINATED_SCHEMA,
      json_schema: {
        oneOf: [
          { $ref: "#/$defs/Missing" },
          { $ref: "#/$defs/Present" },
        ],
        discriminator: {
          propertyName: "provider",
          mapping: {
            missing: "#/$defs/Missing",
            present: "#/$defs/Present",
          },
        },
        $defs: {
          Present: {
            properties: { field: { type: "string" } },
            required: [],
          },
        },
      },
    };
    render(
      <PluginCard
        plugin={makePlugin()}
        schema={danglingSchema}
        onExpand={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: /schema for example/i }));
    expect(screen.getByText("provider: present")).toBeInTheDocument();
    expect(screen.queryByText("provider: missing")).not.toBeInTheDocument();
  });

  it("renders 'No configuration fields' when a variant has no properties", async () => {
    const user = userEvent.setup();
    const emptyVariantSchema: PluginSchemaInfo = {
      ...DISCRIMINATED_SCHEMA,
      json_schema: {
        oneOf: [{ $ref: "#/$defs/Empty" }],
        discriminator: {
          propertyName: "provider",
          mapping: { empty: "#/$defs/Empty" },
        },
        $defs: { Empty: {} },
      },
    };
    render(
      <PluginCard
        plugin={makePlugin()}
        schema={emptyVariantSchema}
        onExpand={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: /schema for example/i }));
    expect(screen.getByText("No configuration fields.")).toBeInTheDocument();
  });
});

describe("PluginCard — error and loading states", () => {
  it("shows schema error message and suppresses content", async () => {
    const user = userEvent.setup();
    const onRetrySchema = vi.fn();
    render(
      <PluginCard
        plugin={makePlugin()}
        schema={null}
        schemaError
        onExpand={vi.fn()}
        onRetrySchema={onRetrySchema}
      />,
    );
    await user.click(screen.getByRole("button", { name: /schema for example/i }));
    expect(
      screen.getByText(/Failed to load schema/),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Retry loading schema" }));
    expect(onRetrySchema).toHaveBeenCalledTimes(1);
  });

  it("announces schema loading with a spinner when schema is null and no error", async () => {
    const user = userEvent.setup();
    render(
      <PluginCard plugin={makePlugin()} schema={null} onExpand={vi.fn()} />,
    );
    await user.click(screen.getByRole("button", { name: /schema for example/i }));
    const status = screen.getByRole("status");

    expect(status).toHaveAttribute("aria-live", "polite");
    expect(status).toHaveTextContent("Loading schema");
    expect(status.querySelector(".spinner")).toHaveAttribute("aria-hidden", "true");
  });

  it("calls onExpand exactly once per expand toggle", async () => {
    const user = userEvent.setup();
    const onExpand = vi.fn();
    render(
      <PluginCard plugin={makePlugin()} schema={null} onExpand={onExpand} />,
    );
    await user.click(screen.getByRole("button", { name: /schema for example/i })); // expand
    await user.click(screen.getByRole("button", { name: /schema for example/i })); // collapse
    await user.click(screen.getByRole("button", { name: /schema for example/i })); // expand again
    // onExpand fires on transitions from collapsed → expanded only (2 times).
    expect(onExpand).toHaveBeenCalledTimes(2);
  });
});

describe("PluginCard — Phase 7B reshape", () => {
  it("renders the plugin name and one-line description", () => {
    render(<PluginCard plugin={makePlugin({ name: "csv", description: "Read rows from a CSV file." })} schema={null} onExpand={() => {}} />);
    expect(screen.getByText("csv")).toBeInTheDocument();
    expect(screen.getByText(/Read rows from a CSV file/i)).toBeInTheDocument();
  });

  it("renders one audit-characteristic icon per flag", () => {
    render(
      <PluginCard
        plugin={makePlugin({ audit_characteristics: ["io_read", "quarantine"] })}
        schema={null}
        onExpand={() => {}}
      />,
    );
    expect(screen.getByText(/reads i\/?o/i)).toBeInTheDocument();
    expect(screen.getByText(/quarantines/i)).toBeInTheDocument();
  });

  it("renders the 'Use when' prose in the details disclosure", async () => {
    render(<PluginCard plugin={makePlugin({ usage_when_to_use: "When you have a CSV file already." })} schema={null} onExpand={() => {}} />);
    expect(screen.queryByText(/use when/i)).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /reference details for example/i }));
    expect(screen.getByText(/use when/i)).toBeInTheDocument();
    expect(screen.getByText(/when you have a csv file/i)).toBeInTheDocument();
  });

  it("links the details disclosure to the details panel", async () => {
    render(
      <PluginCard
        plugin={makePlugin({ name: "csv loader", usage_when_to_use: "When you have a CSV file already." })}
        schema={null}
        onExpand={() => {}}
      />,
    );
    const detailsButton = screen.getByRole("button", { name: /reference details for csv loader/i });
    expect(detailsButton).toHaveAttribute(
      "aria-controls",
      "plugin-card-details-panel-transform-csv-loader",
    );

    await userEvent.click(detailsButton);
    const panel = screen.getByText(/when you have a csv file/i).closest(".plugin-card-details");
    expect(panel).toHaveAttribute("id", "plugin-card-details-panel-transform-csv-loader");
  });

  it("renders the 'Avoid when' prose in the details disclosure", async () => {
    render(<PluginCard plugin={makePlugin({ usage_when_not_to_use: "When the data is inline." })} schema={null} onExpand={() => {}} />);
    await userEvent.click(screen.getByRole("button", { name: /reference details for example/i }));
    expect(screen.getByText(/avoid when/i)).toBeInTheDocument();
  });

  it("renders the example use as a code block preserving whitespace in details", async () => {
    render(<PluginCard plugin={makePlugin({ example_use: "source:\n  plugin: csv" })} schema={null} onExpand={() => {}} />);
    await userEvent.click(screen.getByRole("button", { name: /reference details for example/i }));
    const codeBlock = screen.getByText(/plugin: csv/);
    expect(codeBlock.tagName.toLowerCase()).toBe("pre");
  });

  it("falls back to a generic message when prose fields are null in details", async () => {
    render(
      <PluginCard
        plugin={makePlugin({ usage_when_to_use: null, usage_when_not_to_use: null, example_use: null })}
        schema={null}
        onExpand={() => {}}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: /reference details for example/i }));
    // Per design doc 08-§Risks: "Empty entries fall back to a generic
    // 'see the technical description' message rather than blocking display."
    expect(screen.getByText(/see the technical description/i)).toBeInTheDocument();
  });

  it("does NOT render a 'Use in pipeline' button (toolkit affordance removed)", () => {
    render(<PluginCard plugin={makePlugin()} schema={null} onExpand={() => {}} />);
    expect(screen.queryByRole("button", { name: /use in pipeline/i })).not.toBeInTheDocument();
  });

  it("calls onExpand when the 'Schema →' disclosure is activated", async () => {
    const onExpand = vi.fn();
    render(<PluginCard plugin={makePlugin({ name: "csv" })} schema={null} onExpand={onExpand} />);
    const disclosure = screen.getByRole("button", { name: /schema for csv/i });
    await userEvent.click(disclosure);
    expect(onExpand).toHaveBeenCalled();
  });

  it("renders the expanded schema after onExpand resolves and schema arrives", () => {
    const schema: PluginSchemaInfo = {
      name: "csv",
      plugin_type: "source",
      description: "Read CSV files.",
      json_schema: {
        properties: { path: { type: "string", description: "Path to file" } },
        required: ["path"],
      },
    } as unknown as PluginSchemaInfo;
    render(<PluginCard plugin={makePlugin({ name: "csv" })} schema={schema} onExpand={() => {}} initialExpanded />);
    expect(screen.getByText("path")).toBeInTheDocument();
  });
});

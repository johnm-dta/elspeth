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
    expect(screen.getByText("CSV source")).toBeInTheDocument();
    // Expanded content does NOT render while collapsed.
    expect(screen.queryByText(/Loading/)).not.toBeInTheDocument();
  });
});

describe("PluginCard — flat single-model schema", () => {
  it("renders each property and marks the required ones", async () => {
    const user = userEvent.setup();
    render(
      <PluginCard
        plugin={makePlugin({ name: "csv" })}
        schema={FLAT_SCHEMA}
        onExpand={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button"));
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
    await user.click(screen.getByRole("button"));
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
    await user.click(screen.getByRole("button"));
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
    await user.click(screen.getByRole("button"));
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
    await user.click(screen.getByRole("button"));
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
    await user.click(screen.getByRole("button"));
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
    await user.click(screen.getByRole("button"));
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
    await user.click(screen.getByRole("button"));
    expect(screen.getByText("No configuration fields.")).toBeInTheDocument();
  });
});

describe("PluginCard — error and loading states", () => {
  it("shows schema error message and suppresses content", async () => {
    const user = userEvent.setup();
    render(
      <PluginCard
        plugin={makePlugin()}
        schema={null}
        schemaError
        onExpand={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button"));
    expect(
      screen.getByText(/Failed to load schema/),
    ).toBeInTheDocument();
  });

  it("shows Loading when schema is null and no error", async () => {
    const user = userEvent.setup();
    render(
      <PluginCard plugin={makePlugin()} schema={null} onExpand={vi.fn()} />,
    );
    await user.click(screen.getByRole("button"));
    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });

  it("calls onExpand exactly once per expand toggle", async () => {
    const user = userEvent.setup();
    const onExpand = vi.fn();
    render(
      <PluginCard plugin={makePlugin()} schema={null} onExpand={onExpand} />,
    );
    await user.click(screen.getByRole("button")); // expand
    await user.click(screen.getByRole("button")); // collapse
    await user.click(screen.getByRole("button")); // expand again
    // onExpand fires on transitions from collapsed → expanded only (2 times).
    expect(onExpand).toHaveBeenCalledTimes(2);
  });
});

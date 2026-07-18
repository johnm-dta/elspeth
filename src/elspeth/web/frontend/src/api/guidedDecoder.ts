import type { CompositionState } from "@/types/index";
import type {
  ChatTurn,
  GetGuidedResponse,
  GuidedChatResponse,
  GuidedRespondResponse,
  GuidedSession,
  GuidedStep,
  TerminalState,
  TurnPayload,
  TurnType,
} from "@/types/guided";

const TURN_TYPES = new Set<TurnType>([
  "inspect_and_confirm",
  "single_select",
  "multi_select_with_custom",
  "schema_form",
  "propose_pipeline",
  "confirm_wiring",
]);
const STEPS = new Set<GuidedStep>([
  "step_1_source",
  "step_2_sink",
  "step_3_transforms",
  "step_4_wire",
]);
const STEP_INDEX: Record<GuidedStep, number> = {
  step_1_source: 0,
  step_2_sink: 1,
  step_3_transforms: 2,
  step_4_wire: 3,
};
const LEGAL_TURNS: Record<GuidedStep, ReadonlySet<TurnType>> = {
  step_1_source: new Set(["inspect_and_confirm", "single_select", "schema_form"]),
  step_2_sink: new Set(["single_select", "multi_select_with_custom", "schema_form"]),
  step_3_transforms: new Set(["propose_pipeline", "single_select", "schema_form"]),
  step_4_wire: new Set(["confirm_wiring"]),
};
const SHA256 = /^[0-9a-f]{64}$/;
const CANONICAL_UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const BLOCKER_CATEGORY = {
  pipeline_invalid: "validation",
  policy_review_required: "policy",
  plugin_unavailable: "availability",
  interpretation_required: "interpretation",
} as const;
const BLOCKER_SUMMARY = {
  pipeline_invalid: "guided.proposal.blocker.pipeline_invalid.v1",
  policy_review_required: "guided.proposal.blocker.policy_review_required.v1",
  plugin_unavailable: "guided.proposal.blocker.plugin_unavailable.v1",
  interpretation_required: "guided.proposal.blocker.interpretation_required.v1",
} as const;
const PROPOSAL_SUMMARY_TEMPLATE = "guided.proposal.summary.full_graph.v1";
const PROPOSAL_RATIONALE_TEMPLATE = "guided.proposal.rationale.review_required.v1";
const NODE_TYPES = new Set(["transform", "gate", "aggregation", "queue", "coalesce"]);
const FLOW_KINDS = new Set([
  "source_success", "source_validation_failure", "node_success", "node_error",
  "gate_route", "gate_fork", "queue_continue", "coalesce_success", "output_write_failure",
]);
const TRIGGER_KINDS = ["count", "timeout", "condition"] as const;
const COALESCE_POLICIES = new Set(["require_all", "quorum", "best_effort", "first"]);
const COALESCE_MERGES = new Set(["union", "nested", "select"]);
const FIELD_KINDS = new Set([
  "text", "number-int", "number-float", "checkbox", "enum", "string-list",
  "blob-ref", "json-object", "json-array", "json-value",
]);
const FIELD_TIERS = new Set(["essential", "common", "advanced"]);
const ITEM_KINDS = new Set(["text", "number-int", "number-float"]);
const COMPOSITION_NODE_TYPES = new Set(["transform", "gate", "aggregation", "coalesce", "queue"]);
const COMPOSITION_EDGE_TYPES = new Set(["on_success", "on_error", "route_true", "route_false", "fork"]);
const POLICY_REASONS = new Set([
  "plugin_not_enabled", "plugin_not_installed", "plugin_unavailable",
  "credential_unavailable", "profile_unavailable",
]);
const CATALOG_PLUGIN_ID = /^[a-z][a-z0-9_]{0,63}$/;
const MAX_PROPOSAL_COMPONENTS = 256;
const MAX_PROPOSAL_EDGES = 1_024;
const MAX_PROPOSAL_ALIASES = 64;
type ComponentKind = "source" | "node" | "edge" | "output";
interface DecodedTarget {
  kind: ComponentKind;
  stableId: string;
}

function invalid(path: string, detail: string): never {
  throw new Error(`Invalid guided response at ${path}: ${detail}`);
}

function record(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    invalid(path, "expected object");
  }
  return value as Record<string, unknown>;
}

function exactRecord(
  value: unknown,
  path: string,
  required: readonly string[],
  optional: readonly string[] = [],
): Record<string, unknown> {
  const result = record(value, path);
  const allowed = new Set([...required, ...optional]);
  for (const key of required) {
    if (!Object.prototype.hasOwnProperty.call(result, key)) invalid(path, `missing ${key}`);
  }
  for (const key of Object.keys(result)) {
    if (!allowed.has(key)) invalid(path, `unexpected ${key}`);
  }
  return result;
}

function stringValue(value: unknown, path: string): string {
  if (typeof value !== "string") invalid(path, "expected string");
  return value;
}

function canonicalUuid(value: unknown, path: string): string {
  const decoded = stringValue(value, path);
  if (!CANONICAL_UUID.test(decoded)) invalid(path, "expected canonical UUID");
  return decoded;
}

function nullableString(value: unknown, path: string): string | null {
  return value === null ? null : stringValue(value, path);
}

function booleanValue(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") invalid(path, "expected boolean");
  return value;
}

function integerValue(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isInteger(value)) invalid(path, "expected integer");
  return value;
}

function arrayValue(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) invalid(path, "expected array");
  return value;
}

function stringArray(value: unknown, path: string): string[] {
  return arrayValue(value, path).map((item, index) => stringValue(item, `${path}[${index}]`));
}

function jsonValue(value: unknown, path: string): unknown {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) invalid(path, "expected finite JSON number");
    return value;
  }
  if (Array.isArray(value)) return value.map((item, index) => jsonValue(item, `${path}[${index}]`));
  const decoded = record(value, path);
  return Object.fromEntries(
    Object.entries(decoded).map(([key, item]) => [key, jsonValue(item, `${path}.${key}`)]),
  );
}

function jsonRecord(value: unknown, path: string): Record<string, unknown> {
  const decoded = jsonValue(value, path);
  if (typeof decoded !== "object" || decoded === null || Array.isArray(decoded)) {
    invalid(path, "expected JSON object");
  }
  return decoded as Record<string, unknown>;
}

function validateOptions(value: unknown, path: string): void {
  arrayValue(value, path).forEach((item, index) => {
    const option = exactRecord(item, `${path}[${index}]`, ["id", "label", "hint"]);
    stringValue(option.id, `${path}[${index}].id`);
    stringValue(option.label, `${path}[${index}].label`);
    nullableString(option.hint, `${path}[${index}].hint`);
  });
}

function validateKnobs(value: unknown, path: string): void {
  const knobs = exactRecord(value, path, ["fields"]);
  arrayValue(knobs.fields, `${path}.fields`).forEach((item, index) => {
    const fieldPath = `${path}.fields[${index}]`;
    const field = exactRecord(
      item,
      fieldPath,
      ["name", "label", "kind", "required", "nullable"],
      ["description", "tier", "default", "enum", "item_kind", "visible_when"],
    );
    stringValue(field.name, `${fieldPath}.name`);
    stringValue(field.label, `${fieldPath}.label`);
    const kind = stringValue(field.kind, `${fieldPath}.kind`);
    if (!FIELD_KINDS.has(kind)) invalid(`${fieldPath}.kind`, "unknown field kind");
    booleanValue(field.required, `${fieldPath}.required`);
    booleanValue(field.nullable, `${fieldPath}.nullable`);
    if (field.description !== undefined) stringValue(field.description, `${fieldPath}.description`);
    if (field.tier !== undefined) {
      const tier = stringValue(field.tier, `${fieldPath}.tier`);
      if (!FIELD_TIERS.has(tier)) invalid(`${fieldPath}.tier`, "unknown field tier");
    }
    if (field.enum !== undefined) stringArray(field.enum, `${fieldPath}.enum`);
    if (field.item_kind !== undefined) {
      const itemKind = stringValue(field.item_kind, `${fieldPath}.item_kind`);
      if (!ITEM_KINDS.has(itemKind)) invalid(`${fieldPath}.item_kind`, "unknown item kind");
    }
    if (field.visible_when !== undefined) {
      const predicate = exactRecord(field.visible_when, `${fieldPath}.visible_when`, ["field", "equals"]);
      stringValue(predicate.field, `${fieldPath}.visible_when.field`);
    }
  });
}

function validateEditTarget(value: unknown, path: string): DecodedTarget {
  const target = exactRecord(value, path, ["kind", "stable_id"]);
  const kind = stringValue(target.kind, `${path}.kind`);
  if (!["source", "node", "edge", "output"].includes(kind)) {
    invalid(`${path}.kind`, "unknown component kind");
  }
  return {
    kind: kind as ComponentKind,
    stableId: canonicalUuid(target.stable_id, `${path}.stable_id`),
  };
}

type ProposalEndpointKind = "source" | "node" | "output" | "discard";
interface DecodedProposalEndpoint { kind: ProposalEndpointKind; stableId: string | null }
interface DecodedProposalFlow { kind: string; route?: string; routes?: string[]; branch?: string | null }
interface DecodedProposalEdge {
  path: string;
  stableId: string;
  from: DecodedProposalEndpoint;
  to: DecodedProposalEndpoint;
  flow: DecodedProposalFlow;
}
interface DecodedProposalBehavior {
  kind: string;
  routeAliases: string[];
  forkBranches: Array<{ routes: string[]; branch: string }>;
  branchAliases: string[];
}
interface DecodedProposalNode {
  stableId: string;
  nodeType: string;
  behavior: DecodedProposalBehavior;
}

function validateProposalPlugin(value: unknown, path: string, expectedKind: "source" | "transform" | "sink"): void {
  const plugin = exactRecord(value, path, ["kind", "id"]);
  if (stringValue(plugin.kind, `${path}.kind`) !== expectedKind) invalid(`${path}.kind`, `expected ${expectedKind}`);
  if (!CATALOG_PLUGIN_ID.test(stringValue(plugin.id, `${path}.id`))) invalid(`${path}.id`, "invalid catalog plugin identifier");
}

function structuralAlias(value: unknown, kind: "route" | "branch", path: string): string {
  const alias = stringValue(value, path);
  const match = new RegExp(`^${kind}-([1-9][0-9]*)$`).exec(alias);
  if (match === null || Number(match[1]) > MAX_PROPOSAL_ALIASES) invalid(path, "expected exact server ordinal alias");
  return alias;
}

function aliasArray(value: unknown, kind: "route" | "branch", path: string, minimum = 0): string[] {
  const aliases = arrayValue(value, path).map((item, index) => structuralAlias(item, kind, `${path}[${index}]`));
  if (aliases.length < minimum || aliases.length > MAX_PROPOSAL_ALIASES) invalid(path, "outside bounded alias count");
  if (new Set(aliases).size !== aliases.length) invalid(path, "duplicate aliases");
  return aliases;
}

function canonicalIntegerString(value: unknown, path: string, positive: boolean): string {
  const decoded = stringValue(value, path);
  const pattern = positive ? /^[1-9][0-9]*$/ : /^(?:0|-?[1-9][0-9]*)$/;
  if (!pattern.test(decoded)) invalid(path, "expected canonical decimal integer string");
  return decoded;
}

function finitePositiveNumber(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    invalid(path, "expected finite positive number");
  }
  return value;
}

function validateProposalBehavior(value: unknown, nodeType: string, path: string): DecodedProposalBehavior {
  const behaviorPath = `${path}.behavior`;
  const behavior = record(value, behaviorPath);
  if (!Object.prototype.hasOwnProperty.call(behavior, "kind")) invalid(behaviorPath, "missing kind");
  if (stringValue(behavior.kind, `${behaviorPath}.kind`) !== nodeType) invalid(`${behaviorPath}.kind`, "does not match node_type");
  if (nodeType === "transform" || nodeType === "queue") {
    exactRecord(behavior, behaviorPath, ["kind"]);
    return { kind: nodeType, routeAliases: [], forkBranches: [], branchAliases: [] };
  }
  if (nodeType === "gate") {
    const exact = exactRecord(behavior, behaviorPath, ["kind", "route_aliases", "fork_branches"]);
    const routeAliases = aliasArray(exact.route_aliases, "route", `${behaviorPath}.route_aliases`, 1);
    const forkBranches = arrayValue(exact.fork_branches, `${behaviorPath}.fork_branches`).map((item, index) => {
      const itemPath = `${behaviorPath}.fork_branches[${index}]`;
      const pair = exactRecord(item, itemPath, ["routes", "branch"]);
      const routes = aliasArray(pair.routes, "route", `${itemPath}.routes`, 1);
      const branch = structuralAlias(pair.branch, "branch", `${itemPath}.branch`);
      if (routes.some((route) => !routeAliases.includes(route))) invalid(`${itemPath}.routes`, "not declared by this gate");
      return { routes, branch };
    });
    if (forkBranches.length > MAX_PROPOSAL_ALIASES) invalid(`${behaviorPath}.fork_branches`, "outside bounded count");
    const branches = forkBranches.map(({ branch }) => branch);
    if (new Set(branches).size !== branches.length) invalid(`${behaviorPath}.fork_branches`, "duplicate physical branch");
    const firstRoutes = forkBranches[0]?.routes;
    if (firstRoutes !== undefined && forkBranches.some(({ routes }) => JSON.stringify(routes) !== JSON.stringify(firstRoutes))) {
      invalid(`${behaviorPath}.fork_branches`, "branches do not share one ordered route sequence");
    }
    return { kind: nodeType, routeAliases, forkBranches, branchAliases: [] };
  }
  if (nodeType === "aggregation") {
    const exact = exactRecord(behavior, behaviorPath, [
      "kind", "trigger_kinds", "count", "timeout_seconds", "output_mode", "expected_output_count",
    ]);
    const triggers = stringArray(exact.trigger_kinds, `${behaviorPath}.trigger_kinds`);
    const canonicalTriggers = TRIGGER_KINDS.filter((kind) => triggers.includes(kind));
    if (new Set(triggers).size !== triggers.length || canonicalTriggers.length !== triggers.length || canonicalTriggers.some((kind, index) => kind !== triggers[index])) {
      invalid(`${behaviorPath}.trigger_kinds`, "not a unique canonical trigger subsequence");
    }
    if (exact.count !== null) canonicalIntegerString(exact.count, `${behaviorPath}.count`, true);
    if ((exact.count !== null) !== triggers.includes("count")) invalid(`${behaviorPath}.count`, "does not match count trigger kind");
    if (exact.timeout_seconds !== null) finitePositiveNumber(exact.timeout_seconds, `${behaviorPath}.timeout_seconds`);
    if ((exact.timeout_seconds !== null) !== triggers.includes("timeout")) invalid(`${behaviorPath}.timeout_seconds`, "does not match timeout trigger kind");
    const outputMode = stringValue(exact.output_mode, `${behaviorPath}.output_mode`);
    if (!["default", "passthrough", "transform"].includes(outputMode)) invalid(`${behaviorPath}.output_mode`, "unknown mode");
    if (exact.expected_output_count !== null) canonicalIntegerString(exact.expected_output_count, `${behaviorPath}.expected_output_count`, false);
    return { kind: nodeType, routeAliases: [], forkBranches: [], branchAliases: [] };
  }
  const exact = exactRecord(behavior, behaviorPath, ["kind", "branch_aliases", "policy", "merge"]);
  const branchAliases = aliasArray(exact.branch_aliases, "branch", `${behaviorPath}.branch_aliases`, 2);
  if (!COALESCE_POLICIES.has(stringValue(exact.policy, `${behaviorPath}.policy`))) invalid(`${behaviorPath}.policy`, "unknown policy");
  if (!COALESCE_MERGES.has(stringValue(exact.merge, `${behaviorPath}.merge`))) invalid(`${behaviorPath}.merge`, "unknown merge");
  return { kind: nodeType, routeAliases: [], forkBranches: [], branchAliases };
}

function validateProposalEndpoint(value: unknown, path: string, allowDiscard: boolean): DecodedProposalEndpoint {
  const endpoint = record(value, path);
  if (endpoint.kind === "discard") {
    if (!allowDiscard) invalid(path, "discard is not a legal source endpoint");
    exactRecord(endpoint, path, ["kind"]);
    return { kind: "discard", stableId: null };
  }
  const exact = exactRecord(endpoint, path, ["kind", "stable_id"]);
  const kind = stringValue(exact.kind, `${path}.kind`);
  if (kind !== "source" && kind !== "node" && kind !== "output") invalid(`${path}.kind`, "unknown endpoint kind");
  return { kind, stableId: canonicalUuid(exact.stable_id, `${path}.stable_id`) };
}

function validateProposalFlow(value: unknown, path: string): DecodedProposalFlow {
  const flow = record(value, path);
  const kind = stringValue(flow.kind, `${path}.kind`);
  if (!FLOW_KINDS.has(kind)) invalid(`${path}.kind`, "unknown flow kind");
  if (["source_success", "node_success", "queue_continue", "coalesce_success"].includes(kind)) {
    const exact = exactRecord(flow, path, ["kind", "branch"]);
    const branch = exact.branch === null ? null : structuralAlias(exact.branch, "branch", `${path}.branch`);
    return { kind, branch };
  }
  if (["source_validation_failure", "node_error", "output_write_failure"].includes(kind)) {
    exactRecord(flow, path, ["kind"]);
    return { kind };
  }
  if (kind === "gate_route") {
    const exact = exactRecord(flow, path, ["kind", "route", "branch"]);
    return {
      kind,
      route: structuralAlias(exact.route, "route", `${path}.route`),
      branch: exact.branch === null ? null : structuralAlias(exact.branch, "branch", `${path}.branch`),
    };
  }
  const exact = exactRecord(flow, path, ["kind", "routes", "branch"]);
  return {
    kind,
    routes: aliasArray(exact.routes, "route", `${path}.routes`, 1),
    branch: structuralAlias(exact.branch, "branch", `${path}.branch`),
  };
}

function validateProposalPayload(value: unknown, path: string): void {
  const payload = exactRecord(value, path, [
    "proposal_id", "draft_hash", "summary", "rationale", "component_counts",
    "blockers", "graph", "nodes", "outputs", "edit_targets",
  ]);
  canonicalUuid(payload.proposal_id, `${path}.proposal_id`);
  if (!SHA256.test(stringValue(payload.draft_hash, `${path}.draft_hash`))) invalid(`${path}.draft_hash`, "expected sha256");
  if (stringValue(payload.summary, `${path}.summary`) !== PROPOSAL_SUMMARY_TEMPLATE) invalid(`${path}.summary`, "unknown template id");
  if (stringValue(payload.rationale, `${path}.rationale`) !== PROPOSAL_RATIONALE_TEMPLATE) invalid(`${path}.rationale`, "unknown template id");

  const countsRecord = exactRecord(payload.component_counts, `${path}.component_counts`, ["sources", "nodes", "edges", "outputs"]);
  const counts = {
    sources: integerValue(countsRecord.sources, `${path}.component_counts.sources`),
    nodes: integerValue(countsRecord.nodes, `${path}.component_counts.nodes`),
    edges: integerValue(countsRecord.edges, `${path}.component_counts.edges`),
    outputs: integerValue(countsRecord.outputs, `${path}.component_counts.outputs`),
  };
  if (counts.sources < 1 || counts.sources > MAX_PROPOSAL_COMPONENTS || counts.outputs < 1 || counts.outputs > MAX_PROPOSAL_COMPONENTS ||
      counts.nodes < 0 || counts.nodes > MAX_PROPOSAL_COMPONENTS || counts.edges < 0 || counts.edges > MAX_PROPOSAL_EDGES) {
    invalid(`${path}.component_counts`, "outside bounded range");
  }

  const blockerTargets: DecodedTarget[] = [];
  const blockers = arrayValue(payload.blockers, `${path}.blockers`);
  if (blockers.length > MAX_PROPOSAL_ALIASES) invalid(`${path}.blockers`, "outside bounded count");
  blockers.forEach((item, index) => {
    const blockerPath = `${path}.blockers[${index}]`;
    const blocker = exactRecord(item, blockerPath, ["code", "category", "summary", "edit_target"]);
    const code = stringValue(blocker.code, `${blockerPath}.code`);
    if (!(code in BLOCKER_CATEGORY)) invalid(`${blockerPath}.code`, "unknown blocker code");
    const typedCode = code as keyof typeof BLOCKER_CATEGORY;
    if (stringValue(blocker.category, `${blockerPath}.category`) !== BLOCKER_CATEGORY[typedCode]) invalid(`${blockerPath}.category`, "does not match blocker code");
    if (stringValue(blocker.summary, `${blockerPath}.summary`) !== BLOCKER_SUMMARY[typedCode]) invalid(`${blockerPath}.summary`, "unknown blocker template");
    if (blocker.edit_target !== null) blockerTargets.push(validateEditTarget(blocker.edit_target, `${blockerPath}.edit_target`));
  });

  const graph = exactRecord(payload.graph, `${path}.graph`, ["sources", "edges"]);
  const componentKinds = new Map<string, ComponentKind>();
  const addComponent = (stableId: string, kind: ComponentKind, componentPath: string) => {
    if (componentKinds.has(stableId)) invalid(componentPath, "component stable IDs must be globally unique");
    componentKinds.set(stableId, kind);
  };
  const sources = arrayValue(graph.sources, `${path}.graph.sources`).map((item, index) => {
    const sourcePath = `${path}.graph.sources[${index}]`;
    const source = exactRecord(item, sourcePath, ["stable_id", "label", "plugin"]);
    const stableId = canonicalUuid(source.stable_id, `${sourcePath}.stable_id`);
    addComponent(stableId, "source", `${sourcePath}.stable_id`);
    if (stringValue(source.label, `${sourcePath}.label`) !== `source-${index + 1}`) invalid(`${sourcePath}.label`, "not exact server ordinal");
    validateProposalPlugin(source.plugin, `${sourcePath}.plugin`, "source");
    return stableId;
  });
  const nodes = arrayValue(payload.nodes, `${path}.nodes`).map((item, index): DecodedProposalNode => {
    const nodePath = `${path}.nodes[${index}]`;
    const node = exactRecord(item, nodePath, ["stable_id", "label", "node_type", "plugin", "behavior"]);
    const stableId = canonicalUuid(node.stable_id, `${nodePath}.stable_id`);
    addComponent(stableId, "node", `${nodePath}.stable_id`);
    if (stringValue(node.label, `${nodePath}.label`) !== `node-${index + 1}`) invalid(`${nodePath}.label`, "not exact server ordinal");
    const nodeType = stringValue(node.node_type, `${nodePath}.node_type`);
    if (!NODE_TYPES.has(nodeType)) invalid(`${nodePath}.node_type`, "unknown node type");
    if (nodeType === "transform" || nodeType === "aggregation") validateProposalPlugin(node.plugin, `${nodePath}.plugin`, "transform");
    else if (node.plugin !== null) invalid(`${nodePath}.plugin`, "structural node plugin must be null");
    return { stableId, nodeType, behavior: validateProposalBehavior(node.behavior, nodeType, nodePath) };
  });
  const outputs = arrayValue(payload.outputs, `${path}.outputs`).map((item, index) => {
    const outputPath = `${path}.outputs[${index}]`;
    const output = exactRecord(item, outputPath, ["stable_id", "label", "plugin"]);
    const stableId = canonicalUuid(output.stable_id, `${outputPath}.stable_id`);
    addComponent(stableId, "output", `${outputPath}.stable_id`);
    if (stringValue(output.label, `${outputPath}.label`) !== `output-${index + 1}`) invalid(`${outputPath}.label`, "not exact server ordinal");
    validateProposalPlugin(output.plugin, `${outputPath}.plugin`, "sink");
    return stableId;
  });
  const nodeById = new Map(nodes.map((node) => [node.stableId, node]));
  const decodedEdges = arrayValue(graph.edges, `${path}.graph.edges`).map((item, index): DecodedProposalEdge => {
    const edgePath = `${path}.graph.edges[${index}]`;
    const edge = exactRecord(item, edgePath, ["stable_id", "from_endpoint", "to_endpoint", "flow"]);
    const stableId = canonicalUuid(edge.stable_id, `${edgePath}.stable_id`);
    addComponent(stableId, "edge", `${edgePath}.stable_id`);
    return {
      path: edgePath,
      stableId,
      from: validateProposalEndpoint(edge.from_endpoint, `${edgePath}.from_endpoint`, false),
      to: validateProposalEndpoint(edge.to_endpoint, `${edgePath}.to_endpoint`, true),
      flow: validateProposalFlow(edge.flow, `${edgePath}.flow`),
    };
  });
  if (counts.sources !== sources.length || counts.nodes !== nodes.length || counts.edges !== decodedEdges.length || counts.outputs !== outputs.length) {
    invalid(`${path}.component_counts`, "does not match exact structural projection");
  }

  const editTargets = arrayValue(payload.edit_targets, `${path}.edit_targets`).map((item, index) =>
    validateEditTarget(item, `${path}.edit_targets[${index}]`),
  );
  const discardId = "__proposal_discard__";
  const adjacency = new Map<string, Set<string>>();
  const reverseAdjacency = new Map<string, Set<string>>();
  for (const stableId of [...sources, ...nodes.map((node) => node.stableId), ...outputs, discardId]) {
    adjacency.set(stableId, new Set());
    reverseAdjacency.set(stableId, new Set());
  }
  const outgoingFlows = new Map<string, DecodedProposalFlow[]>();
  const incomingEdges = new Map<string, Array<{ from: string; flow: DecodedProposalFlow }>>();
  const gateRoutes = new Map<string, string[]>();
  const gateForks = new Map<string, Array<{ routes: string[]; branch: string }>>();
  const branchOrigins = new Map<string, string[]>();
  const branchAdjacency = new Map<string, Map<string, Set<string>>>();
  const branchUses: Array<{ branch: string; from: string; flowKind: string; path: string }> = [];
  const legalNodeFlows: Record<string, ReadonlySet<string>> = {
    transform: new Set(["node_success", "node_error"]),
    aggregation: new Set(["node_success", "node_error"]),
    gate: new Set(["gate_route", "gate_fork"]),
    queue: new Set(["queue_continue"]),
    coalesce: new Set(["coalesce_success"]),
  };
  const legalTargets: Record<string, ReadonlySet<ProposalEndpointKind>> = {
    source_success: new Set(["node", "output"]),
    source_validation_failure: new Set(["output", "discard"]),
    node_success: new Set(["node", "output"]),
    node_error: new Set(["node", "output", "discard"]),
    gate_route: new Set(["node", "output", "discard"]),
    gate_fork: new Set(["node", "output"]),
    queue_continue: new Set(["node", "output"]),
    coalesce_success: new Set(["node", "output"]),
    output_write_failure: new Set(["output", "discard"]),
  };
  for (const edge of decodedEdges) {
    if (edge.from.stableId === null || componentKinds.get(edge.from.stableId) !== edge.from.kind) invalid(`${edge.path}.from_endpoint`, "kind and stable_id do not resolve together");
    if (edge.to.kind !== "discard" && (edge.to.stableId === null || componentKinds.get(edge.to.stableId) !== edge.to.kind)) invalid(`${edge.path}.to_endpoint`, "kind and stable_id do not resolve together");
    const fromId = edge.from.stableId;
    const toId = edge.to.kind === "discard" ? discardId : edge.to.stableId;
    if (fromId === null || toId === null) invalid(edge.path, "unresolved endpoint");
    const expectedFrom = edge.flow.kind.startsWith("source_") ? "source" : edge.flow.kind === "output_write_failure" ? "output" : "node";
    if (edge.from.kind !== expectedFrom) invalid(`${edge.path}.flow`, "illegal for source endpoint kind");
    if (edge.from.kind === "node" && !legalNodeFlows[nodeById.get(fromId)!.nodeType].has(edge.flow.kind)) invalid(`${edge.path}.flow`, "illegal for node_type");
    if (!legalTargets[edge.flow.kind].has(edge.to.kind)) invalid(`${edge.path}.flow`, "illegal for target endpoint kind");
    if (fromId === toId) invalid(edge.path, "self-loop");
    if (edge.to.kind === "node" && nodeById.get(toId)!.nodeType === "coalesce" && edge.flow.branch == null) invalid(`${edge.path}.flow`, "coalesce input requires branch alias");
    adjacency.get(fromId)!.add(toId);
    reverseAdjacency.get(toId)!.add(fromId);
    outgoingFlows.set(fromId, [...(outgoingFlows.get(fromId) ?? []), edge.flow]);
    incomingEdges.set(toId, [...(incomingEdges.get(toId) ?? []), { from: fromId, flow: edge.flow }]);
    if (edge.flow.kind === "gate_route") gateRoutes.set(fromId, [...(gateRoutes.get(fromId) ?? []), edge.flow.route!]);
    if (edge.flow.kind === "gate_fork") {
      gateForks.set(fromId, [...(gateForks.get(fromId) ?? []), { routes: edge.flow.routes!, branch: edge.flow.branch! }]);
      branchOrigins.set(edge.flow.branch!, [...(branchOrigins.get(edge.flow.branch!) ?? []), toId]);
    }
    if (edge.flow.branch != null) {
      const branchGraph = branchAdjacency.get(edge.flow.branch) ?? new Map<string, Set<string>>();
      branchGraph.set(fromId, new Set([...(branchGraph.get(fromId) ?? []), toId]));
      if (!branchGraph.has(toId)) branchGraph.set(toId, new Set());
      branchAdjacency.set(edge.flow.branch, branchGraph);
      branchUses.push({ branch: edge.flow.branch, from: fromId, flowKind: edge.flow.kind, path: edge.path });
    }
  }

  for (const node of nodes) {
    const flows = outgoingFlows.get(node.stableId) ?? [];
    const kinds = flows.map((flow) => flow.kind);
    if (node.nodeType === "transform" || node.nodeType === "aggregation") {
      if (kinds.length !== 2 || kinds.filter((kind) => kind === "node_success").length !== 1 || kinds.filter((kind) => kind === "node_error").length !== 1) invalid(path, "node lacks exact success/error flows");
    } else if (node.nodeType === "gate") {
      const directRoutes = gateRoutes.get(node.stableId) ?? [];
      if (new Set(directRoutes).size !== directRoutes.length) invalid(path, "gate direct route alias resolves more than once");
      const forkRoutes = (gateForks.get(node.stableId) ?? []).flatMap((branch) => branch.routes);
      if (directRoutes.some((route) => forkRoutes.includes(route))) invalid(path, "gate route selects direct target and fork fanout");
      const projectedRoutes = [...new Set([...directRoutes, ...forkRoutes])];
      if (JSON.stringify(projectedRoutes) !== JSON.stringify(node.behavior.routeAliases)) invalid(path, "gate route aliases disagree with flows");
      if (JSON.stringify(gateForks.get(node.stableId) ?? []) !== JSON.stringify(node.behavior.forkBranches)) invalid(path, "gate fork branches disagree with flows");
    } else if (node.nodeType === "queue") {
      if (kinds.length !== 1 || kinds[0] !== "queue_continue" || !(incomingEdges.get(node.stableId)?.length)) invalid(path, "queue lacks exact producer/successor flow");
      const queueTargets = [...adjacency.get(node.stableId)!];
      const queueTarget = queueTargets[0];
      if (queueTargets.length !== 1 || !nodeById.has(queueTarget) || nodeById.get(queueTarget)!.nodeType === "queue") {
        invalid(path, "queue continuation does not target one ordinary non-queue node");
      }
    } else {
      if (kinds.length !== 1 || kinds[0] !== "coalesce_success") invalid(path, "coalesce lacks exact success flow");
      const incomingBranches = (incomingEdges.get(node.stableId) ?? []).flatMap(({ flow }) => flow.branch == null ? [] : [flow.branch]);
      if (JSON.stringify(incomingBranches) !== JSON.stringify(node.behavior.branchAliases)) invalid(path, "coalesce branches disagree with incoming flows");
    }
  }
  for (const sourceId of sources) {
    const kinds = (outgoingFlows.get(sourceId) ?? []).map((flow) => flow.kind);
    if (kinds.length !== 2 || kinds.filter((kind) => kind === "source_success").length !== 1 || kinds.filter((kind) => kind === "source_validation_failure").length !== 1) invalid(path, "source lacks exact success/validation flows");
  }
  const outputFailureTargets = new Map<string, string>();
  for (const outputId of outputs) {
    const flows = outgoingFlows.get(outputId) ?? [];
    if (flows.length !== 1 || flows[0].kind !== "output_write_failure") invalid(path, "output lacks exact write-failure flow");
    const targets = [...adjacency.get(outputId)!];
    if (targets.length !== 1) invalid(path, "output has ambiguous write-failure target");
    outputFailureTargets.set(outputId, targets[0]);
  }
  for (const targetId of outputFailureTargets.values()) {
    if (outputs.includes(targetId) && outputFailureTargets.get(targetId) !== discardId) invalid(path, "output write-failure chain");
  }

  const allRouteAliases = nodes.flatMap((node) => node.behavior.routeAliases);
  if (new Set(allRouteAliases).size !== allRouteAliases.length) {
    invalid(path, "route aliases are not globally unique across gate nodes");
  }
  const routeAliases = new Set(allRouteAliases);
  const expectedRoutes = new Set(Array.from({ length: routeAliases.size }, (_, index) => `route-${index + 1}`));
  if (routeAliases.size !== expectedRoutes.size || [...routeAliases].some((alias) => !expectedRoutes.has(alias))) invalid(path, "route aliases are not one global ordinal sequence");
  const branchAliases = new Set(branchOrigins.keys());
  const expectedBranches = new Set(Array.from({ length: branchAliases.size }, (_, index) => `branch-${index + 1}`));
  if (branchAliases.size !== expectedBranches.size || [...branchAliases].some((alias) => !expectedBranches.has(alias)) || [...branchOrigins.values()].some((origins) => origins.length !== 1)) invalid(path, "fork branch aliases are not unique global ordinals");
  const branchCoalesceOwner = new Map<string, string>();
  for (const node of nodes.filter((item) => item.nodeType === "coalesce")) {
    for (const branch of node.behavior.branchAliases) {
      const existingOwner = branchCoalesceOwner.get(branch);
      if (existingOwner !== undefined && existingOwner !== node.stableId) {
        invalid(path, "fork branch alias is consumed by more than one coalesce node");
      }
      branchCoalesceOwner.set(branch, node.stableId);
    }
  }
  for (const use of branchUses) {
    const origins = branchOrigins.get(use.branch);
    if (origins === undefined || origins.length !== 1) invalid(`${use.path}.flow`, "branch has no unique gate_fork origin");
    if (use.flowKind === "gate_fork") continue;
    const branchGraph = branchAdjacency.get(use.branch)!;
    const seen = new Set(origins);
    const branchFrontier = [...origins];
    while (branchFrontier.length > 0) {
      const current = branchFrontier.pop()!;
      for (const target of branchGraph.get(current) ?? []) {
        if (!seen.has(target)) { seen.add(target); branchFrontier.push(target); }
      }
    }
    if (!seen.has(use.from)) invalid(`${use.path}.flow`, "branch use is not downstream of gate_fork origin");
  }
  for (const node of nodes.filter((item) => item.nodeType === "coalesce")) {
    for (const branch of node.behavior.branchAliases) {
      const origins = branchOrigins.get(branch);
      if (origins === undefined) invalid(path, "coalesce branch has no fork origin");
      const branchGraph = branchAdjacency.get(branch)!;
      const seen = new Set(origins);
      const frontier = [...origins];
      while (frontier.length > 0) {
        const current = frontier.pop()!;
        for (const target of branchGraph.get(current) ?? []) {
          if (!seen.has(target)) { seen.add(target); frontier.push(target); }
        }
      }
      if (!seen.has(node.stableId)) invalid(path, "coalesce branch disconnected from fork origin");
    }
  }

  const vertices = [...sources, ...nodes.map((node) => node.stableId), ...outputs];
  const indegree = new Map(vertices.map((stableId) => [stableId, 0]));
  for (const fromId of vertices) {
    for (const targetId of adjacency.get(fromId) ?? []) {
      if (targetId !== discardId) indegree.set(targetId, indegree.get(targetId)! + 1);
    }
  }
  const zeroIndegree = vertices.filter((stableId) => indegree.get(stableId) === 0);
  let visitedCount = 0;
  while (zeroIndegree.length > 0) {
    const current = zeroIndegree.pop()!;
    visitedCount += 1;
    for (const targetId of adjacency.get(current) ?? []) {
      if (targetId === discardId) continue;
      indegree.set(targetId, indegree.get(targetId)! - 1);
      if (indegree.get(targetId) === 0) zeroIndegree.push(targetId);
    }
  }
  if (visitedCount !== vertices.length) invalid(path, "graph contains cycle");
  const reachable = new Set(sources);
  const frontier = [...sources];
  while (frontier.length > 0) {
    const current = frontier.pop()!;
    for (const target of adjacency.get(current) ?? []) {
      if (!reachable.has(target)) { reachable.add(target); frontier.push(target); }
    }
  }
  if ([...nodes.map((node) => node.stableId), ...outputs].some((stableId) => !reachable.has(stableId))) invalid(path, "component unreachable from sources");
  const reachesTerminal = new Set([...outputs, discardId]);
  const reverseFrontier = [...reachesTerminal];
  while (reverseFrontier.length > 0) {
    const current = reverseFrontier.pop()!;
    for (const source of reverseAdjacency.get(current) ?? []) {
      if (!reachesTerminal.has(source)) { reachesTerminal.add(source); reverseFrontier.push(source); }
    }
  }
  if ([...sources, ...nodes.map((node) => node.stableId)].some((stableId) => !reachesTerminal.has(stableId))) invalid(path, "component does not reach output or discard");

  for (const [owner, targets] of [[`${path}.blockers`, blockerTargets], [`${path}.edit_targets`, editTargets]] as const) {
    const seen = new Set<string>();
    for (const target of targets) {
      const key = `${target.kind}:${target.stableId}`;
      if (seen.has(key)) invalid(owner, "contains duplicate target");
      seen.add(key);
      if (componentKinds.get(target.stableId) !== target.kind) invalid(owner, "target kind and stable_id do not resolve together");
    }
  }
}

function validateWirePayload(value: unknown, path: string): void {
  const payload = exactRecord(
    value,
    path,
    ["topology", "edge_contracts", "semantic_contracts", "warnings"],
    ["advisor_findings", "signoff_outcome", "passes_remaining"],
  );
  const topology = exactRecord(payload.topology, `${path}.topology`, ["sources", "nodes", "outputs"]);
  for (const [name, item] of Object.entries(record(topology.sources, `${path}.topology.sources`))) {
    const source = exactRecord(item, `${path}.topology.sources.${name}`, [
      "id", "plugin", "on_success", "on_validation_failure",
    ]);
    stringValue(source.id, `${path}.topology.sources.${name}.id`);
    stringValue(source.plugin, `${path}.topology.sources.${name}.plugin`);
    nullableString(source.on_success, `${path}.topology.sources.${name}.on_success`);
    stringValue(source.on_validation_failure, `${path}.topology.sources.${name}.on_validation_failure`);
  }
  arrayValue(topology.nodes, `${path}.topology.nodes`).forEach((item, index) => {
    const nodePath = `${path}.topology.nodes[${index}]`;
    const node = exactRecord(item, nodePath, [
      "id", "node_type", "plugin", "input", "on_success", "on_error", "routes", "fork_to", "branches",
    ]);
    stringValue(node.id, `${nodePath}.id`);
    stringValue(node.node_type, `${nodePath}.node_type`);
    nullableString(node.plugin, `${nodePath}.plugin`);
    nullableString(node.input, `${nodePath}.input`);
    nullableString(node.on_success, `${nodePath}.on_success`);
    nullableString(node.on_error, `${nodePath}.on_error`);
    if (node.routes !== null) {
      Object.entries(record(node.routes, `${nodePath}.routes`)).forEach(([key, itemValue]) => stringValue(itemValue, `${nodePath}.routes.${key}`));
    }
    if (node.fork_to !== null) stringArray(node.fork_to, `${nodePath}.fork_to`);
    if (node.branches !== null) {
      if (Array.isArray(node.branches)) stringArray(node.branches, `${nodePath}.branches`);
      else Object.entries(record(node.branches, `${nodePath}.branches`)).forEach(([key, itemValue]) => stringValue(itemValue, `${nodePath}.branches.${key}`));
    }
  });
  arrayValue(topology.outputs, `${path}.topology.outputs`).forEach((item, index) => {
    const output = exactRecord(item, `${path}.topology.outputs[${index}]`, ["id", "sink_name", "plugin", "on_write_failure"]);
    Object.entries(output).forEach(([key, itemValue]) => stringValue(itemValue, `${path}.topology.outputs[${index}].${key}`));
  });
  arrayValue(payload.edge_contracts, `${path}.edge_contracts`).forEach((item, index) => {
    const edgePath = `${path}.edge_contracts[${index}]`;
    const edge = exactRecord(item, edgePath, [
      "from", "to", "producer_guarantees", "consumer_requires", "missing_fields", "satisfied",
    ]);
    stringValue(edge.from, `${edgePath}.from`);
    stringValue(edge.to, `${edgePath}.to`);
    stringArray(edge.producer_guarantees, `${edgePath}.producer_guarantees`);
    stringArray(edge.consumer_requires, `${edgePath}.consumer_requires`);
    stringArray(edge.missing_fields, `${edgePath}.missing_fields`);
    booleanValue(edge.satisfied, `${edgePath}.satisfied`);
  });
  arrayValue(payload.semantic_contracts, `${path}.semantic_contracts`).forEach((item, index) => record(item, `${path}.semantic_contracts[${index}]`));
  arrayValue(payload.warnings, `${path}.warnings`).forEach((item, index) => record(item, `${path}.warnings[${index}]`));
  if (payload.advisor_findings !== undefined) stringValue(payload.advisor_findings, `${path}.advisor_findings`);
  if (payload.signoff_outcome !== undefined) stringValue(payload.signoff_outcome, `${path}.signoff_outcome`);
  if (payload.passes_remaining !== undefined) integerValue(payload.passes_remaining, `${path}.passes_remaining`);
}

function decodeTurn(value: unknown, step: GuidedStep, path: string): TurnPayload {
  const turn = exactRecord(value, path, ["type", "step_index", "turn_token", "payload"]);
  const type = stringValue(turn.type, `${path}.type`);
  if (!TURN_TYPES.has(type as TurnType)) invalid(`${path}.type`, "unknown discriminator");
  const turnType = type as TurnType;
  if (!LEGAL_TURNS[step].has(turnType)) invalid(`${path}.type`, "illegal for guided step");
  if (integerValue(turn.step_index, `${path}.step_index`) !== STEP_INDEX[step]) invalid(`${path}.step_index`, "does not match guided step");
  if (!SHA256.test(stringValue(turn.turn_token, `${path}.turn_token`))) invalid(`${path}.turn_token`, "expected sha256");
  const payloadPath = `${path}.payload`;
  switch (turnType) {
    case "inspect_and_confirm": {
      const payload = exactRecord(turn.payload, payloadPath, ["observed"]);
      const observed = exactRecord(payload.observed, `${payloadPath}.observed`, ["columns", "samples", "warnings"]);
      stringArray(observed.columns, `${payloadPath}.observed.columns`);
      arrayValue(observed.samples, `${payloadPath}.observed.samples`).forEach((item, index) => record(item, `${payloadPath}.observed.samples[${index}]`));
      stringArray(observed.warnings, `${payloadPath}.observed.warnings`);
      break;
    }
    case "single_select": {
      const payload = exactRecord(turn.payload, payloadPath, ["question", "options", "allow_custom"]);
      stringValue(payload.question, `${payloadPath}.question`);
      validateOptions(payload.options, `${payloadPath}.options`);
      booleanValue(payload.allow_custom, `${payloadPath}.allow_custom`);
      break;
    }
    case "multi_select_with_custom": {
      const payload = exactRecord(turn.payload, payloadPath, ["question", "options", "default_chosen", "escape_label"]);
      stringValue(payload.question, `${payloadPath}.question`);
      validateOptions(payload.options, `${payloadPath}.options`);
      stringArray(payload.default_chosen, `${payloadPath}.default_chosen`);
      nullableString(payload.escape_label, `${payloadPath}.escape_label`);
      break;
    }
    case "schema_form": {
      const payload = exactRecord(turn.payload, payloadPath, ["mode", "plugin", "knobs", "prefilled"]);
      const mode = stringValue(payload.mode, `${payloadPath}.mode`);
      if (mode !== "plugin_options") invalid(`${payloadPath}.mode`, "unknown schema form mode");
      stringValue(payload.plugin, `${payloadPath}.plugin`);
      validateKnobs(payload.knobs, `${payloadPath}.knobs`);
      record(payload.prefilled, `${payloadPath}.prefilled`);
      break;
    }
    case "propose_pipeline":
      validateProposalPayload(turn.payload, payloadPath);
      break;
    case "confirm_wiring":
      validateWirePayload(turn.payload, payloadPath);
      break;
  }
  return turn as unknown as TurnPayload;
}

function decodeTerminal(value: unknown, path: string): TerminalState | null {
  if (value === null) return null;
  const terminal = exactRecord(value, path, ["kind", "reason", "pipeline_yaml"]);
  const kind = stringValue(terminal.kind, `${path}.kind`);
  if (kind !== "completed" && kind !== "exited_to_freeform") invalid(`${path}.kind`, "unknown terminal kind");
  const reason = nullableString(terminal.reason, `${path}.reason`);
  if (reason !== null && reason !== "user_pressed_exit") invalid(`${path}.reason`, "unknown terminal reason");
  const pipelineYaml = nullableString(terminal.pipeline_yaml, `${path}.pipeline_yaml`);
  if (kind === "completed") {
    if (reason !== null) invalid(`${path}.reason`, "completed terminal must not carry a reason");
    if (pipelineYaml === null || pipelineYaml === "") {
      invalid(`${path}.pipeline_yaml`, "completed terminal requires non-empty YAML");
    }
    return { kind, reason: null, pipeline_yaml: pipelineYaml };
  }
  if (reason !== "user_pressed_exit") invalid(`${path}.reason`, "exited terminal requires a reason");
  if (pipelineYaml !== null) invalid(`${path}.pipeline_yaml`, "exited terminal must not carry YAML");
  return { kind, reason, pipeline_yaml: null };
}

function decodeChatTurn(value: unknown, path: string): ChatTurn {
  const turn = exactRecord(value, path, [
    "role", "content", "seq", "step", "ts_iso", "assistant_message_kind", "synthetic_failure_reason",
  ]);
  const role = stringValue(turn.role, `${path}.role`);
  if (role !== "user" && role !== "assistant") invalid(`${path}.role`, "unknown chat role");
  stringValue(turn.content, `${path}.content`);
  integerValue(turn.seq, `${path}.seq`);
  const step = stringValue(turn.step, `${path}.step`);
  if (!STEPS.has(step as GuidedStep)) invalid(`${path}.step`, "unknown guided step");
  stringValue(turn.ts_iso, `${path}.ts_iso`);
  const kind = turn.assistant_message_kind === null ? null : stringValue(turn.assistant_message_kind, `${path}.assistant_message_kind`);
  const reason = turn.synthetic_failure_reason === null ? null : stringValue(turn.synthetic_failure_reason, `${path}.synthetic_failure_reason`);
  if (role === "user" && (kind !== null || reason !== null)) invalid(path, "user chat turn carries assistant discriminator");
  if (role === "assistant" && kind !== "assistant" && kind !== "synthetic_failure") invalid(path, "assistant chat turn lacks closed discriminator");
  if ((kind === "synthetic_failure") !== (reason !== null)) invalid(path, "synthetic failure discriminator is inconsistent");
  if (reason !== null && !["quality_guard", "unavailable", "not_applied"].includes(reason)) invalid(`${path}.synthetic_failure_reason`, "unknown reason");
  return turn as unknown as ChatTurn;
}

function decodeSession(value: unknown, path: string): GuidedSession {
  const session = exactRecord(value, path, ["step", "history", "terminal", "chat_history", "chat_turn_seq", "profile"]);
  const stepValue = stringValue(session.step, `${path}.step`);
  if (!STEPS.has(stepValue as GuidedStep)) invalid(`${path}.step`, "unknown guided step");
  arrayValue(session.history, `${path}.history`).forEach((item, index) => {
    const historyPath = `${path}.history[${index}]`;
    const history = exactRecord(item, historyPath, ["step", "turn_type", "payload_hash", "response_hash", "summary", "emitter"]);
    if (!STEPS.has(stringValue(history.step, `${historyPath}.step`) as GuidedStep)) invalid(`${historyPath}.step`, "unknown guided step");
    if (!TURN_TYPES.has(stringValue(history.turn_type, `${historyPath}.turn_type`) as TurnType)) invalid(`${historyPath}.turn_type`, "unknown turn type");
    stringValue(history.payload_hash, `${historyPath}.payload_hash`);
    nullableString(history.response_hash, `${historyPath}.response_hash`);
    nullableString(history.summary, `${historyPath}.summary`);
    const emitter = stringValue(history.emitter, `${historyPath}.emitter`);
    if (emitter !== "server" && emitter !== "llm") invalid(`${historyPath}.emitter`, "unknown emitter");
  });
  decodeTerminal(session.terminal, `${path}.terminal`);
  arrayValue(session.chat_history, `${path}.chat_history`).forEach((item, index) => decodeChatTurn(item, `${path}.chat_history[${index}]`));
  integerValue(session.chat_turn_seq, `${path}.chat_turn_seq`);
  if (session.profile !== null) {
    const profile = exactRecord(session.profile, `${path}.profile`, ["coaching", "bookends", "advisor_checkpoints"]);
    Object.entries(profile).forEach(([key, itemValue]) => booleanValue(itemValue, `${path}.profile.${key}`));
  }
  return session as unknown as GuidedSession;
}

function decodeCompositionState(value: unknown, path: string): CompositionState | null {
  if (value === null) return null;
  const state = exactRecord(
    value,
    path,
    [
      "id", "session_id", "version", "sources", "nodes", "edges", "outputs", "metadata",
      "is_valid", "validation_errors", "validation_warnings",
      "validation_suggestions", "derived_from_state_id", "created_at", "composer_meta",
      "plugin_policy_findings",
    ],
  );
  const id = stringValue(state.id, `${path}.id`);
  const sessionId = stringValue(state.session_id, `${path}.session_id`);
  const version = integerValue(state.version, `${path}.version`);
  const isValid = booleanValue(state.is_valid, `${path}.is_valid`);
  const derivedFromStateId = nullableString(state.derived_from_state_id, `${path}.derived_from_state_id`);
  const createdAt = stringValue(state.created_at, `${path}.created_at`);
  const composerMeta = state.composer_meta === null
    ? null
    : jsonRecord(state.composer_meta, `${path}.composer_meta`);

  const sources: CompositionState["sources"] = {};
  const wireSources = state.sources === null ? {} : record(state.sources, `${path}.sources`);
  for (const [name, item] of Object.entries(wireSources)) {
    const sourcePath = `${path}.sources.${name}`;
    const source = exactRecord(item, sourcePath, ["plugin", "options", "on_success", "on_validation_failure"]);
    sources[name] = {
      plugin: stringValue(source.plugin, `${sourcePath}.plugin`),
      options: jsonRecord(source.options, `${sourcePath}.options`),
      on_success: stringValue(source.on_success, `${sourcePath}.on_success`),
      on_validation_failure: stringValue(source.on_validation_failure, `${sourcePath}.on_validation_failure`),
    };
  }

  const wireNodes = state.nodes === null ? [] : arrayValue(state.nodes, `${path}.nodes`);
  const nodes: CompositionState["nodes"] = wireNodes.map((item, index) => {
    const nodePath = `${path}.nodes[${index}]`;
    const node = exactRecord(
      item,
      nodePath,
      ["id", "node_type", "plugin", "input", "on_success", "on_error", "options"],
      ["condition", "routes", "fork_to", "branches", "policy", "merge", "trigger", "output_mode", "expected_output_count"],
    );
    const nodeType = stringValue(node.node_type, `${nodePath}.node_type`);
    if (!COMPOSITION_NODE_TYPES.has(nodeType)) invalid(`${nodePath}.node_type`, "unknown node type");
    const decoded: CompositionState["nodes"][number] = {
      id: stringValue(node.id, `${nodePath}.id`),
      node_type: nodeType as CompositionState["nodes"][number]["node_type"],
      plugin: nullableString(node.plugin, `${nodePath}.plugin`),
      input: stringValue(node.input, `${nodePath}.input`),
      on_success: nullableString(node.on_success, `${nodePath}.on_success`),
      on_error: nullableString(node.on_error, `${nodePath}.on_error`),
      options: jsonRecord(node.options, `${nodePath}.options`),
    };
    if (node.condition !== undefined) decoded.condition = nullableString(node.condition, `${nodePath}.condition`);
    if (node.routes !== undefined) {
      decoded.routes = node.routes === null
        ? null
        : Object.fromEntries(
          Object.entries(record(node.routes, `${nodePath}.routes`)).map(([key, target]) => [key, stringValue(target, `${nodePath}.routes.${key}`)]),
        );
    }
    if (node.fork_to !== undefined) decoded.fork_to = node.fork_to === null ? null : stringArray(node.fork_to, `${nodePath}.fork_to`);
    if (node.branches !== undefined) {
      decoded.branches = node.branches === null
        ? null
        : Array.isArray(node.branches)
          ? stringArray(node.branches, `${nodePath}.branches`)
          : Object.fromEntries(
            Object.entries(record(node.branches, `${nodePath}.branches`)).map(([key, target]) => [key, stringValue(target, `${nodePath}.branches.${key}`)]),
          );
    }
    if (node.policy !== undefined) decoded.policy = nullableString(node.policy, `${nodePath}.policy`);
    if (node.merge !== undefined) decoded.merge = nullableString(node.merge, `${nodePath}.merge`);
    if (node.trigger !== undefined) decoded.trigger = node.trigger === null ? null : jsonRecord(node.trigger, `${nodePath}.trigger`);
    if (node.output_mode !== undefined) {
      const outputMode = nullableString(node.output_mode, `${nodePath}.output_mode`);
      if (outputMode !== null && !["default", "passthrough", "transform"].includes(outputMode)) {
        invalid(`${nodePath}.output_mode`, "unknown aggregation output mode");
      }
      decoded.output_mode = outputMode as CompositionState["nodes"][number]["output_mode"];
    }
    if (node.expected_output_count !== undefined) {
      decoded.expected_output_count = node.expected_output_count === null
        ? null
        : integerValue(node.expected_output_count, `${nodePath}.expected_output_count`);
    }
    return decoded;
  });

  const wireEdges = state.edges === null ? [] : arrayValue(state.edges, `${path}.edges`);
  const edges: CompositionState["edges"] = wireEdges.map((item, index) => {
    const edgePath = `${path}.edges[${index}]`;
    const edge = exactRecord(item, edgePath, ["id", "from_node", "to_node", "edge_type", "label"]);
    const edgeType = stringValue(edge.edge_type, `${edgePath}.edge_type`);
    if (!COMPOSITION_EDGE_TYPES.has(edgeType)) invalid(`${edgePath}.edge_type`, "unknown edge type");
    return {
      id: stringValue(edge.id, `${edgePath}.id`),
      from_node: stringValue(edge.from_node, `${edgePath}.from_node`),
      to_node: stringValue(edge.to_node, `${edgePath}.to_node`),
      edge_type: edgeType as CompositionState["edges"][number]["edge_type"],
      label: nullableString(edge.label, `${edgePath}.label`),
    };
  });

  const wireOutputs = state.outputs === null ? [] : arrayValue(state.outputs, `${path}.outputs`);
  const outputs: CompositionState["outputs"] = wireOutputs.map((item, index) => {
    const outputPath = `${path}.outputs[${index}]`;
    const output = exactRecord(item, outputPath, ["name", "plugin", "options", "on_write_failure"]);
    return {
      name: stringValue(output.name, `${outputPath}.name`),
      plugin: stringValue(output.plugin, `${outputPath}.plugin`),
      options: jsonRecord(output.options, `${outputPath}.options`),
      on_write_failure: stringValue(output.on_write_failure, `${outputPath}.on_write_failure`),
    };
  });

  const metadata = state.metadata === null
    ? { name: null, description: null }
    : (() => {
      const metadataValue = exactRecord(state.metadata, `${path}.metadata`, ["name", "description"]);
      return {
        name: nullableString(metadataValue.name, `${path}.metadata.name`),
        description: nullableString(metadataValue.description, `${path}.metadata.description`),
      };
    })();
  const decodeValidationEntries = (field: "validation_warnings" | "validation_suggestions") => {
    if (state[field] === null) return null;
    return arrayValue(state[field], `${path}.${field}`).map((item, index) => {
      const itemPath = `${path}.${field}[${index}]`;
      const entry = exactRecord(item, itemPath, ["component", "message", "severity"], ["error_code"]);
      return {
        component: stringValue(entry.component, `${itemPath}.component`),
        message: stringValue(entry.message, `${itemPath}.message`),
        severity: stringValue(entry.severity, `${itemPath}.severity`),
        ...(entry.error_code === undefined
          ? {}
          : { error_code: nullableString(entry.error_code, `${itemPath}.error_code`) }),
      };
    });
  };
  const validationErrors = state.validation_errors === null
    ? null
    : stringArray(state.validation_errors, `${path}.validation_errors`);
  const validationWarnings = decodeValidationEntries("validation_warnings");
  const validationSuggestions = decodeValidationEntries("validation_suggestions");
  const policyFindings = arrayValue(state.plugin_policy_findings, `${path}.plugin_policy_findings`).map((item, index) => {
      const itemPath = `${path}.plugin_policy_findings[${index}]`;
      const finding = exactRecord(item, itemPath, ["component_id", "plugin_id", "reason_code", "snapshot_fingerprint"]);
      const reasonCode = stringValue(finding.reason_code, `${itemPath}.reason_code`);
      if (!POLICY_REASONS.has(reasonCode)) invalid(`${itemPath}.reason_code`, "unknown policy reason");
      return {
        component_id: stringValue(finding.component_id, `${itemPath}.component_id`),
        plugin_id: stringValue(finding.plugin_id, `${itemPath}.plugin_id`),
        reason_code: reasonCode as NonNullable<CompositionState["plugin_policy_findings"]>[number]["reason_code"],
        snapshot_fingerprint: stringValue(finding.snapshot_fingerprint, `${itemPath}.snapshot_fingerprint`),
      };
    });
  return {
    id,
    session_id: sessionId,
    version,
    sources,
    nodes,
    edges,
    outputs,
    metadata,
    is_valid: isValid,
    validation_errors: validationErrors,
    validation_warnings: validationWarnings,
    validation_suggestions: validationSuggestions,
    derived_from_state_id: derivedFromStateId,
    created_at: createdAt,
    composer_meta: composerMeta,
    plugin_policy_findings: policyFindings,
  };
}

function decodeStateEnvelope(value: unknown, path: string): GetGuidedResponse {
  const envelope = exactRecord(value, path, ["guided_session", "next_turn", "terminal", "composition_state"]);
  const guidedSession = decodeSession(envelope.guided_session, `${path}.guided_session`);
  const terminal = decodeTerminal(envelope.terminal, `${path}.terminal`);
  if (JSON.stringify(terminal) !== JSON.stringify(guidedSession.terminal)) invalid(path, "terminal projections disagree");
  const nextTurn = envelope.next_turn === null
    ? null
    : decodeTurn(envelope.next_turn, guidedSession.step, `${path}.next_turn`);
  const compositionState = decodeCompositionState(envelope.composition_state, `${path}.composition_state`);
  return {
    guided_session: guidedSession,
    next_turn: nextTurn,
    terminal,
    composition_state: compositionState,
  };
}

export function decodeGetGuidedResponse(value: unknown): GetGuidedResponse {
  return decodeStateEnvelope(value, "response");
}

export function decodeGuidedRespondResponse(value: unknown): GuidedRespondResponse {
  return decodeStateEnvelope(value, "response");
}

export function decodeGuidedChatResponse(value: unknown): GuidedChatResponse {
  const envelope = exactRecord(value, "response", [
    "assistant_message", "assistant_message_kind", "guided_session", "next_turn", "terminal", "composition_state",
  ]);
  const state = decodeStateEnvelope(
    {
      guided_session: envelope.guided_session,
      next_turn: envelope.next_turn,
      terminal: envelope.terminal,
      composition_state: envelope.composition_state,
    },
    "response",
  );
  const assistantMessage = stringValue(envelope.assistant_message, "response.assistant_message");
  const assistantKind = stringValue(envelope.assistant_message_kind, "response.assistant_message_kind");
  if (assistantKind !== "assistant" && assistantKind !== "synthetic_failure") invalid("response.assistant_message_kind", "unknown discriminator");
  return {
    assistant_message: assistantMessage,
    assistant_message_kind: assistantKind,
    ...state,
  };
}

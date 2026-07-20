import type { CompositionState } from "@/types/index";
import type {
  ChatTurn,
  ComponentReviewPayload,
  GetGuidedResponse,
  GuidedChatResponse,
  GuidedRespondResponse,
  GuidedSession,
  GuidedStartOperationReconciliation,
  GuidedStep,
  GuidedEditTarget,
  InspectAndConfirmPayload,
  KnobField,
  MultiSelectWithCustomPayload,
  Option,
  ProposalBlocker,
  ProposalEndpoint,
  ProposalFlow,
  ProposalNodeBehavior,
  ProposalPluginRef,
  ProposePipelinePayload,
  SchemaFormPayload,
  SingleSelectPayload,
  TerminalState,
  TurnPayload,
  TurnType,
  WireStageData,
} from "@/types/guided";

const STEP_INDEX: Record<GuidedStep, number> = {
  step_1_source: 0,
  step_2_sink: 1,
  step_3_transforms: 2,
  step_4_wire: 3,
};
const LEGAL_TURNS: Record<GuidedStep, ReadonlySet<TurnType>> = {
  step_1_source: new Set(["inspect_and_confirm", "single_select", "schema_form", "review_components"]),
  step_2_sink: new Set(["single_select", "multi_select_with_custom", "schema_form", "review_components"]),
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
const COMPOSITION_NODE_TYPES = new Set(["transform", "gate", "aggregation", "coalesce", "queue"]);
const COMPOSITION_EDGE_TYPES = new Set(["on_success", "on_error", "route_true", "route_false", "fork"]);
const POLICY_REASONS = new Set([
  "plugin_not_enabled", "plugin_not_installed", "plugin_unavailable",
  "credential_unavailable", "profile_unavailable",
]);
const CATALOG_PLUGIN_ID = /^[a-z][a-z0-9_]{0,63}$/;
const MAX_PROPOSAL_COMPONENTS = 256;
const MAX_REVIEWED_COMPONENTS_PER_KIND = 256;
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

function decodeProposalNodeType(
  value: unknown,
  path: string,
): ProposePipelinePayload["nodes"][number]["node_type"] {
  const decoded = stringValue(value, path);
  switch (decoded) {
    case "transform":
    case "gate":
    case "aggregation":
    case "queue":
    case "coalesce":
      return decoded;
    default:
      return invalid(path, "unknown node type");
  }
}

function decodeTurnEmitter(value: unknown, path: string): "server" | "llm" {
  const decoded = stringValue(value, path);
  switch (decoded) {
    case "server":
    case "llm":
      return decoded;
    default:
      return invalid(path, "unknown emitter");
  }
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
    ["proposal_id", "draft_hash", "topology", "edge_contracts", "semantic_contracts", "warnings"],
    ["advisor_findings", "signoff_outcome", "passes_remaining"],
  );
  canonicalUuid(payload.proposal_id, `${path}.proposal_id`);
  const draftHash = stringValue(payload.draft_hash, `${path}.draft_hash`);
  if (!SHA256.test(draftHash)) invalid(`${path}.draft_hash`, "expected lowercase SHA-256");
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

function decodeTurnType(value: unknown, path: string): TurnType {
  const type = stringValue(value, path);
  switch (type) {
    case "inspect_and_confirm":
    case "single_select":
    case "multi_select_with_custom":
    case "schema_form":
    case "review_components":
    case "propose_pipeline":
    case "confirm_wiring":
      return type;
    default:
      return invalid(path, "unknown discriminator");
  }
}

function decodeGuidedStep(value: unknown, path: string): GuidedStep {
  const step = stringValue(value, path);
  switch (step) {
    case "step_1_source":
    case "step_2_sink":
    case "step_3_transforms":
    case "step_4_wire":
      return step;
    default:
      return invalid(path, "unknown guided step");
  }
}

function decodeOptions(value: unknown, path: string): Option[] {
  return arrayValue(value, path).map((item, index) => {
    const optionPath = `${path}[${index}]`;
    const option = exactRecord(item, optionPath, ["id", "label", "hint"]);
    return {
      id: stringValue(option.id, `${optionPath}.id`),
      label: stringValue(option.label, `${optionPath}.label`),
      hint: nullableString(option.hint, `${optionPath}.hint`),
    };
  });
}

function decodeInspectPayload(value: unknown, path: string): InspectAndConfirmPayload {
  const payload = exactRecord(value, path, ["observed"]);
  const observed = exactRecord(payload.observed, `${path}.observed`, ["columns", "samples", "warnings"]);
  return {
    observed: {
      columns: stringArray(observed.columns, `${path}.observed.columns`),
      samples: arrayValue(observed.samples, `${path}.observed.samples`).map(
        (item, index) => jsonRecord(item, `${path}.observed.samples[${index}]`),
      ),
      warnings: stringArray(observed.warnings, `${path}.observed.warnings`),
    },
  };
}

function decodeSingleSelectPayload(value: unknown, path: string): SingleSelectPayload {
  const payload = exactRecord(value, path, ["question", "options", "allow_custom"]);
  return {
    question: stringValue(payload.question, `${path}.question`),
    options: decodeOptions(payload.options, `${path}.options`),
    allow_custom: booleanValue(payload.allow_custom, `${path}.allow_custom`),
  };
}

function decodeMultiSelectPayload(value: unknown, path: string): MultiSelectWithCustomPayload {
  const payload = exactRecord(value, path, ["question", "options", "default_chosen", "escape_label"]);
  return {
    question: stringValue(payload.question, `${path}.question`),
    options: decodeOptions(payload.options, `${path}.options`),
    default_chosen: stringArray(payload.default_chosen, `${path}.default_chosen`),
    escape_label: nullableString(payload.escape_label, `${path}.escape_label`),
  };
}

function decodeFieldKind(value: unknown, path: string): KnobField["kind"] {
  const kind = stringValue(value, path);
  switch (kind) {
    case "text":
    case "number-int":
    case "number-float":
    case "checkbox":
    case "enum":
    case "string-list":
    case "blob-ref":
    case "json-object":
    case "json-array":
    case "json-value":
      return kind;
    default:
      return invalid(path, "unknown field kind");
  }
}

function decodeSchemaPayload(value: unknown, path: string): SchemaFormPayload {
  const payload = exactRecord(value, path, ["mode", "plugin", "knobs", "prefilled"]);
  if (stringValue(payload.mode, `${path}.mode`) !== "plugin_options") {
    invalid(`${path}.mode`, "unknown schema form mode");
  }
  const knobs = exactRecord(payload.knobs, `${path}.knobs`, ["fields"]);
  const fields = arrayValue(knobs.fields, `${path}.knobs.fields`).map((item, index): KnobField => {
    const fieldPath = `${path}.knobs.fields[${index}]`;
    const field = exactRecord(
      item,
      fieldPath,
      ["name", "label", "kind", "required", "nullable"],
      ["description", "tier", "default", "enum", "item_kind", "visible_when"],
    );
    const tier = field.tier === undefined ? undefined : stringValue(field.tier, `${fieldPath}.tier`);
    if (tier !== undefined && tier !== "essential" && tier !== "common" && tier !== "advanced") {
      invalid(`${fieldPath}.tier`, "unknown field tier");
    }
    const itemKind = field.item_kind === undefined
      ? undefined
      : stringValue(field.item_kind, `${fieldPath}.item_kind`);
    if (itemKind !== undefined && itemKind !== "text" && itemKind !== "number-int" && itemKind !== "number-float") {
      invalid(`${fieldPath}.item_kind`, "unknown item kind");
    }
    const visibleWhen = field.visible_when === undefined
      ? undefined
      : exactRecord(field.visible_when, `${fieldPath}.visible_when`, ["field", "equals"]);
    return {
      name: stringValue(field.name, `${fieldPath}.name`),
      label: stringValue(field.label, `${fieldPath}.label`),
      kind: decodeFieldKind(field.kind, `${fieldPath}.kind`),
      required: booleanValue(field.required, `${fieldPath}.required`),
      nullable: booleanValue(field.nullable, `${fieldPath}.nullable`),
      ...(field.description === undefined
        ? {}
        : { description: stringValue(field.description, `${fieldPath}.description`) }),
      ...(tier === undefined ? {} : { tier }),
      ...(field.default === undefined ? {} : { default: jsonValue(field.default, `${fieldPath}.default`) }),
      ...(field.enum === undefined ? {} : { enum: stringArray(field.enum, `${fieldPath}.enum`) }),
      ...(itemKind === undefined ? {} : { item_kind: itemKind }),
      ...(visibleWhen === undefined
        ? {}
        : {
            visible_when: {
              field: stringValue(visibleWhen.field, `${fieldPath}.visible_when.field`),
              equals: jsonValue(visibleWhen.equals, `${fieldPath}.visible_when.equals`),
            },
          }),
    };
  });
  return {
    mode: "plugin_options",
    plugin: stringValue(payload.plugin, `${path}.plugin`),
    knobs: { fields },
    prefilled: jsonRecord(payload.prefilled, `${path}.prefilled`),
  };
}

function decodeComponentReviewPayload(
  value: unknown,
  path: string,
): ComponentReviewPayload {
  const payload = exactRecord(value, path, [
    "component_kind",
    "items",
    "allowed_actions",
  ]);
  const componentKind = stringValue(payload.component_kind, `${path}.component_kind`);
  if (componentKind !== "source" && componentKind !== "output") {
    invalid(`${path}.component_kind`, "unknown reviewed component kind");
  }
  const rawItems = arrayValue(payload.items, `${path}.items`);
  if (rawItems.length === 0 || rawItems.length > MAX_REVIEWED_COMPONENTS_PER_KIND) {
    invalid(`${path}.items`, "expected 1 to 256 reviewed components");
  }
  const stableIds = new Set<string>();
  const names = new Set<string>();
  const items = rawItems.map((value, index) => {
    const itemPath = `${path}.items[${index}]`;
    const item = exactRecord(value, itemPath, ["stable_id", "name", "plugin", "status"]);
    const stableId = canonicalUuid(item.stable_id, `${itemPath}.stable_id`);
    const name = stringValue(item.name, `${itemPath}.name`);
    const plugin = stringValue(item.plugin, `${itemPath}.plugin`);
    const status = stringValue(item.status, `${itemPath}.status`);
    if (name.trim() === "") invalid(`${itemPath}.name`, "expected non-empty name");
    if (plugin.trim() === "") invalid(`${itemPath}.plugin`, "expected non-empty plugin");
    if (status !== "reviewed") invalid(`${itemPath}.status`, "expected reviewed");
    if (stableIds.has(stableId)) invalid(`${path}.items`, "duplicate stable id");
    if (names.has(name)) invalid(`${path}.items`, "duplicate component name");
    stableIds.add(stableId);
    names.add(name);
    return { stable_id: stableId, name, plugin, status: "reviewed" as const };
  });

  const actions = stringArray(payload.allowed_actions, `${path}.allowed_actions`).map(
    (action, index): ComponentReviewPayload["allowed_actions"][number] => {
      switch (action) {
        case "add":
        case "edit":
        case "remove":
        case "reorder":
        case "finish":
          return action;
        default:
          return invalid(`${path}.allowed_actions[${index}]`, "unknown component action");
      }
    },
  );
  if (new Set(actions).size !== actions.length) {
    invalid(`${path}.allowed_actions`, "duplicate component action");
  }
  const expectedActions = new Set(["add", "edit", "reorder", "finish"]);
  if (items.length > 1) expectedActions.add("remove");
  if (
    actions.length !== expectedActions.size ||
    actions.some((action) => !expectedActions.has(action))
  ) {
    invalid(`${path}.allowed_actions`, "does not match the closed reviewed collection actions");
  }
  return { component_kind: componentKind, items, allowed_actions: actions };
}

function decodeEditTarget(value: unknown, path: string): GuidedEditTarget {
  const target = exactRecord(value, path, ["kind", "stable_id"]);
  const stableId = canonicalUuid(target.stable_id, `${path}.stable_id`);
  const kind = stringValue(target.kind, `${path}.kind`);
  switch (kind) {
    case "source":
    case "node":
    case "edge":
    case "output":
      return { kind, stable_id: stableId };
    default:
      return invalid(`${path}.kind`, "unknown component kind");
  }
}

function decodeProposalPlugin(
  value: unknown,
  expectedKind: ProposalPluginRef["kind"],
  path: string,
): ProposalPluginRef {
  const plugin = exactRecord(value, path, ["kind", "id"]);
  const kind = stringValue(plugin.kind, `${path}.kind`);
  if (kind !== expectedKind) invalid(`${path}.kind`, `expected ${expectedKind}`);
  return { kind: expectedKind, id: stringValue(plugin.id, `${path}.id`) };
}

function decodeProposalEndpoint(value: unknown, path: string): ProposalEndpoint {
  const endpoint = exactRecord(value, path, ["kind", "stable_id"]);
  const stableId = canonicalUuid(endpoint.stable_id, `${path}.stable_id`);
  const kind = stringValue(endpoint.kind, `${path}.kind`);
  switch (kind) {
    case "source":
    case "node":
    case "output":
      return { kind, stable_id: stableId };
    default:
      return invalid(`${path}.kind`, "unknown endpoint kind");
  }
}

function decodeProposalFlow(value: unknown, path: string): ProposalFlow {
  const flow = record(value, path);
  const kind = stringValue(flow.kind, `${path}.kind`);
  switch (kind) {
    case "source_success":
    case "node_success":
    case "queue_continue":
    case "coalesce_success": {
      const exact = exactRecord(flow, path, ["kind", "branch"]);
      return {
        kind,
        branch: exact.branch === null ? null : structuralAlias(exact.branch, "branch", `${path}.branch`),
      };
    }
    case "source_validation_failure":
    case "node_error":
    case "output_write_failure":
      exactRecord(flow, path, ["kind"]);
      return { kind };
    case "gate_route": {
      const exact = exactRecord(flow, path, ["kind", "route", "branch"]);
      return {
        kind,
        route: structuralAlias(exact.route, "route", `${path}.route`),
        branch: exact.branch === null ? null : structuralAlias(exact.branch, "branch", `${path}.branch`),
      };
    }
    case "gate_fork": {
      const exact = exactRecord(flow, path, ["kind", "routes", "branch"]);
      return {
        kind,
        routes: aliasArray(exact.routes, "route", `${path}.routes`, 1),
        branch: structuralAlias(exact.branch, "branch", `${path}.branch`),
      };
    }
    default:
      return invalid(`${path}.kind`, "unknown flow kind");
  }
}

function decodeProposalBehavior(
  value: unknown,
  nodeType: ProposePipelinePayload["nodes"][number]["node_type"],
  path: string,
): ProposalNodeBehavior {
  const behaviorPath = `${path}.behavior`;
  const behavior = record(value, behaviorPath);
  if (stringValue(behavior.kind, `${behaviorPath}.kind`) !== nodeType) {
    invalid(`${behaviorPath}.kind`, "does not match node_type");
  }
  switch (nodeType) {
    case "transform":
    case "queue":
      exactRecord(behavior, behaviorPath, ["kind"]);
      return { kind: nodeType };
    case "gate": {
      const exact = exactRecord(behavior, behaviorPath, ["kind", "route_aliases", "fork_branches"]);
      return {
        kind: "gate",
        route_aliases: aliasArray(exact.route_aliases, "route", `${behaviorPath}.route_aliases`, 1),
        fork_branches: arrayValue(exact.fork_branches, `${behaviorPath}.fork_branches`).map((item, index) => {
          const itemPath = `${behaviorPath}.fork_branches[${index}]`;
          const branch = exactRecord(item, itemPath, ["routes", "branch"]);
          return {
            routes: aliasArray(branch.routes, "route", `${itemPath}.routes`, 1),
            branch: structuralAlias(branch.branch, "branch", `${itemPath}.branch`),
          };
        }),
      };
    }
    case "aggregation": {
      const exact = exactRecord(behavior, behaviorPath, [
        "kind", "trigger_kinds", "count", "timeout_seconds", "output_mode", "expected_output_count",
      ]);
      const rawTriggers = stringArray(exact.trigger_kinds, `${behaviorPath}.trigger_kinds`);
      const trigger_kinds = TRIGGER_KINDS.filter((trigger) => rawTriggers.includes(trigger));
      const outputMode = stringValue(exact.output_mode, `${behaviorPath}.output_mode`);
      if (outputMode !== "default" && outputMode !== "passthrough" && outputMode !== "transform") {
        invalid(`${behaviorPath}.output_mode`, "unknown mode");
      }
      return {
        kind: "aggregation",
        trigger_kinds,
        count: exact.count === null ? null : canonicalIntegerString(exact.count, `${behaviorPath}.count`, true),
        timeout_seconds: exact.timeout_seconds === null
          ? null
          : finitePositiveNumber(exact.timeout_seconds, `${behaviorPath}.timeout_seconds`),
        output_mode: outputMode,
        expected_output_count: exact.expected_output_count === null
          ? null
          : canonicalIntegerString(exact.expected_output_count, `${behaviorPath}.expected_output_count`, false),
      };
    }
    case "coalesce": {
      const exact = exactRecord(behavior, behaviorPath, ["kind", "branch_aliases", "policy", "merge"]);
      const policy = stringValue(exact.policy, `${behaviorPath}.policy`);
      if (policy !== "require_all" && policy !== "quorum" && policy !== "best_effort" && policy !== "first") {
        invalid(`${behaviorPath}.policy`, "unknown policy");
      }
      const merge = stringValue(exact.merge, `${behaviorPath}.merge`);
      if (merge !== "union" && merge !== "nested" && merge !== "select") {
        invalid(`${behaviorPath}.merge`, "unknown merge");
      }
      return {
        kind: "coalesce",
        branch_aliases: aliasArray(exact.branch_aliases, "branch", `${behaviorPath}.branch_aliases`, 2),
        policy,
        merge,
      };
    }
  }
}

function decodeProposalPayload(value: unknown, path: string): ProposePipelinePayload {
  validateProposalPayload(value, path);
  const payload = exactRecord(value, path, [
    "proposal_id", "draft_hash", "summary", "rationale", "component_counts",
    "blockers", "graph", "nodes", "outputs", "edit_targets",
  ]);
  const counts = exactRecord(payload.component_counts, `${path}.component_counts`, ["sources", "nodes", "edges", "outputs"]);
  const graph = exactRecord(payload.graph, `${path}.graph`, ["sources", "edges"]);
  const blockers = arrayValue(payload.blockers, `${path}.blockers`).map((item, index): ProposalBlocker => {
    const blockerPath = `${path}.blockers[${index}]`;
    const blocker = exactRecord(item, blockerPath, ["code", "category", "summary", "edit_target"]);
    const code = stringValue(blocker.code, `${blockerPath}.code`);
    if (code !== "pipeline_invalid" && code !== "policy_review_required" && code !== "plugin_unavailable" && code !== "interpretation_required") {
      invalid(`${blockerPath}.code`, "unknown blocker code");
    }
    const category = stringValue(blocker.category, `${blockerPath}.category`);
    if (category !== "validation" && category !== "policy" && category !== "availability" && category !== "interpretation") {
      invalid(`${blockerPath}.category`, "unknown blocker category");
    }
    return {
      code,
      category,
      summary: stringValue(blocker.summary, `${blockerPath}.summary`),
      edit_target: blocker.edit_target === null ? null : decodeEditTarget(blocker.edit_target, `${blockerPath}.edit_target`),
    };
  });
  const nodes = arrayValue(payload.nodes, `${path}.nodes`).map((item, index) => {
    const nodePath = `${path}.nodes[${index}]`;
    const node = exactRecord(item, nodePath, ["stable_id", "label", "node_type", "plugin", "behavior"]);
    const rawType = decodeProposalNodeType(node.node_type, `${nodePath}.node_type`);
    const plugin = node.plugin === null
      ? null
      : decodeProposalPlugin(node.plugin, "transform", `${nodePath}.plugin`);
    return {
      stable_id: canonicalUuid(node.stable_id, `${nodePath}.stable_id`),
      label: stringValue(node.label, `${nodePath}.label`),
      node_type: rawType,
      plugin,
      behavior: decodeProposalBehavior(node.behavior, rawType, nodePath),
    };
  });
  return {
    proposal_id: canonicalUuid(payload.proposal_id, `${path}.proposal_id`),
    draft_hash: stringValue(payload.draft_hash, `${path}.draft_hash`),
    summary: stringValue(payload.summary, `${path}.summary`),
    rationale: stringValue(payload.rationale, `${path}.rationale`),
    component_counts: {
      sources: integerValue(counts.sources, `${path}.component_counts.sources`),
      nodes: integerValue(counts.nodes, `${path}.component_counts.nodes`),
      edges: integerValue(counts.edges, `${path}.component_counts.edges`),
      outputs: integerValue(counts.outputs, `${path}.component_counts.outputs`),
    },
    blockers,
    graph: {
      sources: arrayValue(graph.sources, `${path}.graph.sources`).map((item, index) => {
        const sourcePath = `${path}.graph.sources[${index}]`;
        const source = exactRecord(item, sourcePath, ["stable_id", "label", "plugin"]);
        return {
          stable_id: canonicalUuid(source.stable_id, `${sourcePath}.stable_id`),
          label: stringValue(source.label, `${sourcePath}.label`),
          plugin: decodeProposalPlugin(source.plugin, "source", `${sourcePath}.plugin`),
        };
      }),
      edges: arrayValue(graph.edges, `${path}.graph.edges`).map((item, index) => {
        const edgePath = `${path}.graph.edges[${index}]`;
        const edge = exactRecord(item, edgePath, ["stable_id", "from_endpoint", "to_endpoint", "flow"]);
        const targetRecord = record(edge.to_endpoint, `${edgePath}.to_endpoint`);
        return {
          stable_id: canonicalUuid(edge.stable_id, `${edgePath}.stable_id`),
          from_endpoint: decodeProposalEndpoint(edge.from_endpoint, `${edgePath}.from_endpoint`),
          to_endpoint: targetRecord.kind === "discard"
            ? (() => {
                exactRecord(targetRecord, `${edgePath}.to_endpoint`, ["kind"]);
                return { kind: "discard" as const };
              })()
            : decodeProposalEndpoint(edge.to_endpoint, `${edgePath}.to_endpoint`),
          flow: decodeProposalFlow(edge.flow, `${edgePath}.flow`),
        };
      }),
    },
    nodes,
    outputs: arrayValue(payload.outputs, `${path}.outputs`).map((item, index) => {
      const outputPath = `${path}.outputs[${index}]`;
      const output = exactRecord(item, outputPath, ["stable_id", "label", "plugin"]);
      return {
        stable_id: canonicalUuid(output.stable_id, `${outputPath}.stable_id`),
        label: stringValue(output.label, `${outputPath}.label`),
        plugin: decodeProposalPlugin(output.plugin, "sink", `${outputPath}.plugin`),
      };
    }),
    edit_targets: arrayValue(payload.edit_targets, `${path}.edit_targets`).map(
      (item, index) => decodeEditTarget(item, `${path}.edit_targets[${index}]`),
    ),
  };
}

function decodeWirePayload(value: unknown, path: string): WireStageData {
  validateWirePayload(value, path);
  const payload = exactRecord(value, path, ["proposal_id", "draft_hash", "topology", "edge_contracts", "semantic_contracts", "warnings"], [
    "advisor_findings", "signoff_outcome", "passes_remaining",
  ]);
  const topology = exactRecord(payload.topology, `${path}.topology`, ["sources", "nodes", "outputs"]);
  return {
    proposal_id: canonicalUuid(payload.proposal_id, `${path}.proposal_id`),
    draft_hash: stringValue(payload.draft_hash, `${path}.draft_hash`),
    topology: {
      sources: Object.fromEntries(
        Object.entries(record(topology.sources, `${path}.topology.sources`)).map(([name, item]) => {
          const sourcePath = `${path}.topology.sources.${name}`;
          const source = exactRecord(item, sourcePath, ["id", "plugin", "on_success", "on_validation_failure"]);
          return [name, {
            id: stringValue(source.id, `${sourcePath}.id`),
            plugin: stringValue(source.plugin, `${sourcePath}.plugin`),
            on_success: nullableString(source.on_success, `${sourcePath}.on_success`),
            on_validation_failure: stringValue(source.on_validation_failure, `${sourcePath}.on_validation_failure`),
          }];
        }),
      ),
      nodes: arrayValue(topology.nodes, `${path}.topology.nodes`).map((item, index) => {
        const nodePath = `${path}.topology.nodes[${index}]`;
        const node = exactRecord(item, nodePath, [
          "id", "node_type", "plugin", "input", "on_success", "on_error", "routes", "fork_to", "branches",
        ]);
        return {
          id: stringValue(node.id, `${nodePath}.id`),
          node_type: stringValue(node.node_type, `${nodePath}.node_type`),
          plugin: nullableString(node.plugin, `${nodePath}.plugin`),
          input: nullableString(node.input, `${nodePath}.input`),
          on_success: nullableString(node.on_success, `${nodePath}.on_success`),
          on_error: nullableString(node.on_error, `${nodePath}.on_error`),
          routes: node.routes === null ? null : Object.fromEntries(
            Object.entries(record(node.routes, `${nodePath}.routes`)).map(([key, target]) => [key, stringValue(target, `${nodePath}.routes.${key}`)]),
          ),
          fork_to: node.fork_to === null ? null : stringArray(node.fork_to, `${nodePath}.fork_to`),
          branches: node.branches === null
            ? null
            : Array.isArray(node.branches)
              ? stringArray(node.branches, `${nodePath}.branches`)
              : Object.fromEntries(
                  Object.entries(record(node.branches, `${nodePath}.branches`)).map(([key, target]) => [key, stringValue(target, `${nodePath}.branches.${key}`)]),
                ),
        };
      }),
      outputs: arrayValue(topology.outputs, `${path}.topology.outputs`).map((item, index) => {
        const outputPath = `${path}.topology.outputs[${index}]`;
        const output = exactRecord(item, outputPath, ["id", "sink_name", "plugin", "on_write_failure"]);
        return {
          id: stringValue(output.id, `${outputPath}.id`),
          sink_name: stringValue(output.sink_name, `${outputPath}.sink_name`),
          plugin: stringValue(output.plugin, `${outputPath}.plugin`),
          on_write_failure: stringValue(output.on_write_failure, `${outputPath}.on_write_failure`),
        };
      }),
    },
    edge_contracts: arrayValue(payload.edge_contracts, `${path}.edge_contracts`).map((item, index) => {
      const edgePath = `${path}.edge_contracts[${index}]`;
      const edge = exactRecord(item, edgePath, [
        "from", "to", "producer_guarantees", "consumer_requires", "missing_fields", "satisfied",
      ]);
      return {
        from: stringValue(edge.from, `${edgePath}.from`),
        to: stringValue(edge.to, `${edgePath}.to`),
        producer_guarantees: stringArray(edge.producer_guarantees, `${edgePath}.producer_guarantees`),
        consumer_requires: stringArray(edge.consumer_requires, `${edgePath}.consumer_requires`),
        missing_fields: stringArray(edge.missing_fields, `${edgePath}.missing_fields`),
        satisfied: booleanValue(edge.satisfied, `${edgePath}.satisfied`),
      };
    }),
    semantic_contracts: arrayValue(payload.semantic_contracts, `${path}.semantic_contracts`).map(
      (item, index) => jsonRecord(item, `${path}.semantic_contracts[${index}]`),
    ),
    warnings: arrayValue(payload.warnings, `${path}.warnings`).map(
      (item, index) => jsonRecord(item, `${path}.warnings[${index}]`),
    ),
    ...(payload.advisor_findings === undefined
      ? {}
      : { advisor_findings: stringValue(payload.advisor_findings, `${path}.advisor_findings`) }),
    ...(payload.signoff_outcome === undefined
      ? {}
      : { signoff_outcome: stringValue(payload.signoff_outcome, `${path}.signoff_outcome`) }),
    ...(payload.passes_remaining === undefined
      ? {}
      : { passes_remaining: integerValue(payload.passes_remaining, `${path}.passes_remaining`) }),
  };
}

function decodeTurn(value: unknown, step: GuidedStep, path: string): TurnPayload {
  const turn = exactRecord(value, path, ["type", "step_index", "turn_token", "payload"]);
  const turnType = decodeTurnType(turn.type, `${path}.type`);
  if (!LEGAL_TURNS[step].has(turnType)) invalid(`${path}.type`, "illegal for guided step");
  const stepIndex = integerValue(turn.step_index, `${path}.step_index`);
  if (stepIndex !== STEP_INDEX[step]) invalid(`${path}.step_index`, "does not match guided step");
  const turnToken = stringValue(turn.turn_token, `${path}.turn_token`);
  if (!SHA256.test(turnToken)) invalid(`${path}.turn_token`, "expected sha256");
  const payloadPath = `${path}.payload`;
  switch (turnType) {
    case "inspect_and_confirm":
      return { type: turnType, step_index: stepIndex, turn_token: turnToken, payload: decodeInspectPayload(turn.payload, payloadPath) };
    case "single_select":
      return { type: turnType, step_index: stepIndex, turn_token: turnToken, payload: decodeSingleSelectPayload(turn.payload, payloadPath) };
    case "multi_select_with_custom":
      return { type: turnType, step_index: stepIndex, turn_token: turnToken, payload: decodeMultiSelectPayload(turn.payload, payloadPath) };
    case "schema_form":
      return { type: turnType, step_index: stepIndex, turn_token: turnToken, payload: decodeSchemaPayload(turn.payload, payloadPath) };
    case "review_components":
      return { type: turnType, step_index: stepIndex, turn_token: turnToken, payload: decodeComponentReviewPayload(turn.payload, payloadPath) };
    case "propose_pipeline":
      return { type: turnType, step_index: stepIndex, turn_token: turnToken, payload: decodeProposalPayload(turn.payload, payloadPath) };
    case "confirm_wiring":
      return { type: turnType, step_index: stepIndex, turn_token: turnToken, payload: decodeWirePayload(turn.payload, payloadPath) };
  }
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
  const step = decodeGuidedStep(turn.step, `${path}.step`);
  const content = stringValue(turn.content, `${path}.content`);
  const seq = integerValue(turn.seq, `${path}.seq`);
  const tsIso = stringValue(turn.ts_iso, `${path}.ts_iso`);
  const kind = turn.assistant_message_kind === null ? null : stringValue(turn.assistant_message_kind, `${path}.assistant_message_kind`);
  const reason = turn.synthetic_failure_reason === null ? null : stringValue(turn.synthetic_failure_reason, `${path}.synthetic_failure_reason`);
  if (role === "user" && (kind !== null || reason !== null)) invalid(path, "user chat turn carries assistant discriminator");
  if (role === "assistant" && kind !== "assistant" && kind !== "synthetic_failure") invalid(path, "assistant chat turn lacks closed discriminator");
  if ((kind === "synthetic_failure") !== (reason !== null)) invalid(path, "synthetic failure discriminator is inconsistent");
  if (reason !== null && !["quality_guard", "unavailable", "not_applied"].includes(reason)) invalid(`${path}.synthetic_failure_reason`, "unknown reason");
  if (role === "user") {
    return {
      role,
      content,
      seq,
      step,
      ts_iso: tsIso,
      assistant_message_kind: null,
      synthetic_failure_reason: null,
    };
  }
  if (kind === "assistant") {
    return {
      role,
      content,
      seq,
      step,
      ts_iso: tsIso,
      assistant_message_kind: kind,
      synthetic_failure_reason: null,
    };
  }
  if (kind !== "synthetic_failure") return invalid(path, "assistant chat turn lacks closed discriminator");
  if (reason !== "quality_guard" && reason !== "unavailable" && reason !== "not_applied") {
    return invalid(`${path}.synthetic_failure_reason`, "unknown reason");
  }
  return {
    role,
    content,
    seq,
    step,
    ts_iso: tsIso,
    assistant_message_kind: kind,
    synthetic_failure_reason: reason,
  };
}

function decodeSession(value: unknown, path: string): GuidedSession {
  const session = exactRecord(value, path, ["step", "history", "terminal", "chat_history", "chat_turn_seq", "profile"]);
  const step = decodeGuidedStep(session.step, `${path}.step`);
  const history = arrayValue(session.history, `${path}.history`).map((item, index) => {
    const historyPath = `${path}.history[${index}]`;
    const recordValue = exactRecord(item, historyPath, ["step", "turn_type", "payload_hash", "response_hash", "summary", "emitter"]);
    return {
      step: decodeGuidedStep(recordValue.step, `${historyPath}.step`),
      turn_type: decodeTurnType(recordValue.turn_type, `${historyPath}.turn_type`),
      payload_hash: stringValue(recordValue.payload_hash, `${historyPath}.payload_hash`),
      response_hash: nullableString(recordValue.response_hash, `${historyPath}.response_hash`),
      summary: nullableString(recordValue.summary, `${historyPath}.summary`),
      emitter: decodeTurnEmitter(recordValue.emitter, `${historyPath}.emitter`),
    };
  });
  const terminal = decodeTerminal(session.terminal, `${path}.terminal`);
  const chatHistory = arrayValue(session.chat_history, `${path}.chat_history`).map(
    (item, index) => decodeChatTurn(item, `${path}.chat_history[${index}]`),
  );
  const chatTurnSeq = integerValue(session.chat_turn_seq, `${path}.chat_turn_seq`);
  const profile = session.profile === null
    ? null
    : (() => {
        const profile = exactRecord(session.profile, `${path}.profile`, ["coaching", "bookends", "advisor_checkpoints"]);
        return {
          coaching: booleanValue(profile.coaching, `${path}.profile.coaching`),
          bookends: booleanValue(profile.bookends, `${path}.profile.bookends`),
          advisor_checkpoints: booleanValue(profile.advisor_checkpoints, `${path}.profile.advisor_checkpoints`),
        };
      })();
  return {
    step,
    history,
    terminal,
    chat_history: chatHistory,
    chat_turn_seq: chatTurnSeq,
    profile,
  };
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

export function decodeGuidedStartOperationReconciliation(
  value: unknown,
): GuidedStartOperationReconciliation {
  const base = record(value, "response");
  const status = stringValue(base.status, "response.status");
  switch (status) {
    case "in_progress":
      exactRecord(value, "response", ["status"]);
      return { status };
    case "failed": {
      const envelope = exactRecord(value, "response", ["status", "failure_code"]);
      const failureCode = stringValue(envelope.failure_code, "response.failure_code");
      switch (failureCode) {
        case "provider_unavailable":
        case "provider_timeout":
        case "invalid_provider_response":
        case "stale_conflict":
        case "integrity_error":
        case "custody_error":
        case "quota_exceeded":
        case "operation_failed":
        case "request_cancelled":
          return { status, failure_code: failureCode };
        default:
          return invalid("response.failure_code", "unknown guided operation failure code");
      }
    }
    case "completed": {
      const envelope = exactRecord(value, "response", ["status", "composition_state_id"]);
      return {
        status,
        composition_state_id: canonicalUuid(
          envelope.composition_state_id,
          "response.composition_state_id",
        ),
      };
    }
    default:
      return invalid("response.status", "unknown guided-start reconciliation status");
  }
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

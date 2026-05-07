# Web UX Sub-Spec 4: Composer

**Status:** Draft
**Date:** 2026-03-28
**Parent Spec:** `docs/superpowers/specs/2026-03-28-web-ux-composer-mvp-design.md`
**Phase:** 4
**Depends On:** Sub-Specs 2 (Auth & Sessions), 3 (Catalog)
**Blocks:** Sub-Specs 5 (Execution), 6 (Frontend)

---

## Scope

This sub-spec covers the ComposerService module: the data models that represent
a pipeline under construction, the LLM tool-use loop that mutates those models,
the validation that runs on every mutation, and the YAML generator that produces
executable ELSPETH pipeline configuration from the composition state. It also
defines how the chat API endpoint triggers the composer and returns state updates
to the frontend.

Out of scope: dry-run validation (Sub-Spec 5), pipeline execution (Sub-Spec 5),
frontend rendering of composition state (Sub-Spec 6), catalog discovery
implementation (Sub-Spec 3 -- this spec consumes the CatalogService protocol).

---

## CompositionState Model

CompositionState is the central data structure of the composer. It is a frozen
dataclass representing an immutable, versioned snapshot of a pipeline under
construction. Every edit produces a new instance with an incremented version
number. The previous version is never modified.

**Authoritative definition:** CompositionState (frozen dataclass in
`composer/state.py`) is the single authoritative domain model. Sub-Spec 2's
database stores it as serialised JSON. API responses use a Pydantic schema in
`sessions/schemas.py` for serialisation. The dataclass does NOT contain
`is_valid` or `validation_errors` -- those travel alongside it (in
ValidationSummary and ToolResult), not inside it.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| source | SourceSpec or None | The pipeline's single data source. None until set. |
| nodes | tuple[NodeSpec, ...] | Ordered list of transform, gate, aggregation, and coalesce nodes. |
| edges | tuple[EdgeSpec, ...] | Connections between nodes (on_success, on_error, route destinations). |
| outputs | tuple[OutputSpec, ...] | Sink configurations. |
| metadata | PipelineMetadata | Pipeline name, description, and landscape URL. |
| version | int | Monotonically increasing per session, starting at 1. |

### Immutability Contract

CompositionState is declared with `frozen=True` and `slots=True`. All container
fields (nodes, edges, outputs) are stored as tuples, not lists. The `__post_init__`
method calls `freeze_fields()` from `elspeth.contracts.freeze` on every container
field and on the metadata field. This enforces deep immutability -- nested dicts
within SourceSpec options, NodeSpec options, and OutputSpec options are converted to
MappingProxyType recursively.

Mutation methods (with_source, with_node, without_node, with_edge, without_edge,
with_metadata) are instance methods that return a new CompositionState with the
relevant field changed and the version incremented by one. They do not modify the
current instance.

### Reconstruction from Dict (Seam Contract: Sub-2 ↔ Sub-4)

CompositionState and all nested `*Spec` types define a `from_dict(d: dict)` class
method that reconstructs a frozen dataclass instance from a plain Python dict (as
produced by `to_dict()` or deserialised from JSON). This is the inverse of
`to_dict()` and is **load-bearing for every seam that reads state from the
database**: the composer route handler, the execution service, and the validation
endpoint.

**Factory methods:**

- `SourceSpec.from_dict(d: dict) -> SourceSpec` — constructs from `{"plugin": ..., "on_success": ..., "options": ..., "on_validation_failure": ...}`.
- `NodeSpec.from_dict(d: dict) -> NodeSpec` — constructs from the full NodeSpec field dict. Optional fields (`condition`, `routes`, `fork_to`, `branches`, `policy`, `merge`) default to `None` when absent from the dict.
- `EdgeSpec.from_dict(d: dict) -> EdgeSpec` — constructs from `{"id": ..., "from_node": ..., "to_node": ..., "edge_type": ..., "label": ...}`.
- `OutputSpec.from_dict(d: dict) -> OutputSpec` — constructs from `{"name": ..., "plugin": ..., "options": ..., "on_write_failure": ...}`.
- `PipelineMetadata.from_dict(d: dict) -> PipelineMetadata` — constructs from `{"name": ..., "description": ..., "landscape_url": ...}`. Missing fields use the dataclass defaults.
- `CompositionState.from_dict(d: dict) -> CompositionState` — reconstructs the full hierarchy: calls `SourceSpec.from_dict()` on `d["source"]` (if not None), `NodeSpec.from_dict()` on each entry in `d["nodes"]`, etc. Sets `version` from `d["version"]`.

**Round-trip invariant:** For any `CompositionState` instance `s`, `CompositionState.from_dict(s.to_dict())` must produce a value-equal instance (same fields, same deep-frozen containers). This invariant is enforced by a dedicated test in `test_state.py`.

**Relationship to Sub-2:** Sub-2's `from_record()` deserialises DB row JSON columns into plain dicts, then calls `CompositionState.from_dict()` to reconstruct the domain model. Sub-2's `to_record()` calls `CompositionState.to_dict()` to serialise to JSON-ready dicts. The `_version` envelope (Sub-2) wraps the dict produced by `to_dict()` and is stripped before calling `from_dict()`.

### Validation

CompositionState exposes a `validate()` method that returns a tuple of
`(is_valid: bool, errors: tuple[str, ...])`. This performs Stage 1 validation
(composition-time checks -- see the Stage 1 Validation section below). The
validate method is a pure function of the state; it does not consult the catalog
or any external service.

---

## Data Models

### SourceSpec

Frozen dataclass representing the pipeline source configuration.

| Field | Type | Description |
|-------|------|-------------|
| plugin | str | Source plugin name (e.g. "csv", "json", "dataverse"). |
| on_success | str | Named connection point for the first downstream node. |
| options | Mapping[str, Any] | Plugin-specific configuration (path, schema, etc.). |
| on_validation_failure | str | How to handle rows that fail schema validation ("discard" or "quarantine"). |

The options field is deep-frozen in `__post_init__` via `freeze_fields()`.

### NodeSpec

Frozen dataclass representing a transform, gate, aggregation, or coalesce node.

| Field | Type | Description |
|-------|------|-------------|
| id | str | Unique node identifier within the pipeline. |
| node_type | str | One of "transform", "gate", "aggregation", "coalesce". |
| plugin | str or None | Plugin name. None for gates and coalesces (which are config-driven). |
| input | str | Named connection point this node reads from. |
| on_success | str or None | Named connection point for successful output. None for gates (which use routes). |
| on_error | str or None | Named connection point for error output. None means errors are not diverted. |
| options | Mapping[str, Any] | Plugin-specific configuration. |
| condition | str or None | Gate expression (Python expression evaluated against row). None for non-gates. |
| routes | Mapping[str, str] or None | Gate route mapping (e.g. {"true": "sink_a", "false": "sink_b"}). None for non-gates. |
| fork_to | tuple[str, ...] or None | Fork destinations for fork gates. None for non-fork nodes. |
| branches | tuple[str, ...] or None | Branch inputs for coalesce nodes. None for non-coalesce nodes. |
| policy | str or None | Coalesce policy (e.g. "require_all"). None for non-coalesce nodes. |
| merge | str or None | Coalesce merge strategy (e.g. "nested"). None for non-coalesce nodes. |

All container fields (options, routes, fork_to, branches) are deep-frozen in
`__post_init__`. Fields that are None for a given node_type are enforced as None
by the node_type -- for example, a "transform" node with a non-None condition
field is a validation error caught in Stage 1.

### EdgeSpec

Frozen dataclass representing a connection between two nodes.

| Field | Type | Description |
|-------|------|-------------|
| id | str | Unique edge identifier. |
| from_node | str | Source node ID (or "source" for the pipeline source). |
| to_node | str | Destination node ID or sink name. |
| edge_type | str | One of "on_success", "on_error", "route_true", "route_false", "fork". |
| label | str or None | Display label (e.g. the route key for gate edges). |

EdgeSpec has no container fields requiring freeze guards beyond what frozen=True
provides. If label is made into a container field in future, add a freeze guard
at that time.

### OutputSpec

Frozen dataclass representing a sink configuration.

| Field | Type | Description |
|-------|------|-------------|
| name | str | Sink name (used as the named connection point in edges and routes). |
| plugin | str | Sink plugin name (e.g. "csv", "json", "database"). |
| options | Mapping[str, Any] | Plugin-specific configuration. |
| on_write_failure | str | How to handle write failures ("discard" or "quarantine"). |

The options field is deep-frozen in `__post_init__` via `freeze_fields()`.

### PipelineMetadata

Frozen dataclass for pipeline-level metadata.

| Field | Type | Description |
|-------|------|-------------|
| name | str | Pipeline name. Defaults to "Untitled Pipeline". |
| description | str | Pipeline description. Defaults to empty string. |

PipelineMetadata has no container fields. frozen=True is sufficient.

**Removed field (security fix S1):** `landscape_url` was originally included as a
metadata override. It has been removed because the LLM's `set_metadata` tool could
be tricked via prompt injection into setting `landscape_url` to an
attacker-controlled database, diverting the audit trail. The Landscape database URL
must come exclusively from `WebSettings.get_landscape_url()` — it is infrastructure
configuration, not pipeline metadata.

---

## Composition Tools

Composition tools are the interface between the LLM and the CompositionState.
They are divided into discovery tools (read-only) and mutation tools
(state-changing). Every tool is defined with a JSON Schema description that the
LLM receives as part of its tool definitions.

### Discovery Tools

Discovery tools do not modify state. They return information the LLM needs to
make composition decisions.

**list_sources()** -- Returns a list of available source plugins with name and
one-line summary for each. Delegates to CatalogService.list_sources(). The LLM
uses this to know what source plugins exist before calling set_source.

**list_transforms()** -- Returns a list of available transform plugins with name
and one-line summary for each. Delegates to CatalogService.list_transforms().
Includes built-in pseudo-plugins (gates, aggregations, coalesces) with their
configuration shapes.

**list_sinks()** -- Returns a list of available sink plugins with name and
one-line summary for each. Delegates to CatalogService.list_sinks().

**get_plugin_schema(plugin_type, name)** -- Returns the full Pydantic config
schema for a specific plugin, identified by plugin_type ("source", "transform",
or "sink") and name (e.g. "csv"). The schema includes field names, types,
defaults, and descriptions. Delegates to
CatalogService.get_schema(plugin_type, name). The LLM uses this to construct
valid option dicts for set_source, upsert_node, and set_output.

**get_expression_grammar()** -- Returns the gate expression syntax reference:
what variables are available (row, the row dict), what operators are supported,
and example expressions. This is a static string, not a catalog lookup.

**get_current_state()** -- Returns the full current CompositionState serialized
as a JSON-compatible dict. Includes source, all nodes, all edges, all outputs,
metadata, version, and current validation status. The LLM uses this to
understand what has already been configured before deciding what to do next.

### Mutation Tools

Mutation tools modify the CompositionState and return a ToolResult. Every
mutation tool follows the same contract:

1. Accept the current CompositionState and the mutation parameters.
2. Apply the mutation, producing a new CompositionState with incremented version.
3. Run Stage 1 validation on the new state.
4. Return a ToolResult containing the new state, validation summary, and the
   list of affected node IDs.

If the mutation itself is invalid (e.g. removing a node that does not exist),
the tool returns a ToolResult with success=False and the original state
unchanged. It does not raise an exception -- the error is returned to the LLM
as a tool result so it can self-correct.

**set_source(source_spec)** -- Sets or replaces the pipeline source. Accepts a
SourceSpec (plugin name, on_success connection, options, validation failure
mode). Validates that the plugin exists in the catalog. Affected nodes:
["source"].

**Source path allowlist (security fix S2):** If the source options contain a
`path` or `file` key (used by CSV, JSON, and other file-based source plugins),
the value must resolve to a path under `{WebSettings.data_dir}/uploads/`. The
tool validates this constraint and returns `success=False` with an error message
if the path is outside the allowed directory. This prevents the LLM from being
tricked via prompt injection into configuring a source that reads arbitrary
server-side files (e.g., `/etc/passwd`, environment files, other users' data).
The allowlist is also enforced at the execution boundary in Sub-Spec 5's
`validate_pipeline()` (defense in depth).

**upsert_node(id, node_spec)** -- Adds a new node or updates an existing node.
If a node with the given ID already exists, it is replaced. If it does not
exist, it is appended. Validates that the plugin exists (for transform and
aggregation types) and that configuration fields match the plugin schema.
Affected nodes: [id], plus any nodes whose edges reference this node.

**upsert_edge(edge_spec)** -- Adds a new edge or updates an existing edge (matched
by id). Validates that from_node and to_node reference valid node IDs or sink
names in the current state. Affected nodes: [edge_spec.from_node,
edge_spec.to_node].

**remove_node(id)** -- Removes the node with the given ID. Also removes all
edges that reference this node (either as from_node or to_node). Returns
success=False if no node with this ID exists. Affected nodes: [id] plus all
nodes that had edges to/from the removed node.

**remove_edge(id)** -- Removes the edge with the given ID. Returns success=False
if no edge with this ID exists. Affected nodes: the from_node and to_node of
the removed edge.

**set_metadata(patch)** -- Updates pipeline metadata fields. Accepts a partial
dict -- only the fields present in the patch are updated; others are preserved.
For example, `set_metadata({"name": "My Pipeline"})` sets the name without
changing the description. Affected nodes: [] (metadata changes do not affect
node validity).

### ToolResult

ToolResult is a frozen dataclass returned by every tool (discovery and mutation).

| Field | Type | Description |
|-------|------|-------------|
| success | bool | Whether the mutation was applied. False if the mutation was invalid. |
| updated_state | CompositionState | The full state after mutation (or the original state if success=False). |
| validation | ValidationSummary | Stage 1 validation result: is_valid and errors. |
| affected_nodes | tuple[str, ...] | Node IDs that were changed or had their edges changed. |
| data | Any or None | Optional data payload for discovery tools. None for mutation tools. |

ToolResult's `__post_init__` calls `freeze_fields()` on affected_nodes. The
updated_state and validation fields are themselves frozen dataclasses, so
frozen=True on ToolResult is sufficient for those fields.

### ValidationSummary

Frozen dataclass for the Stage 1 validation result.

| Field | Type | Description |
|-------|------|-------------|
| is_valid | bool | True if no validation errors. |
| errors | tuple[str, ...] | Human-readable error messages. |

No container fields beyond the errors tuple, which is already immutable.
frozen=True is sufficient.

---

## Stage 1 Validation

Stage 1 validation runs at composition time, on every mutation tool call. It
checks structural correctness of the CompositionState without instantiating any
plugins or consulting the engine. Its purpose is to catch errors early so the
LLM can self-correct within the tool-use loop.

### Checks

1. **Source exists.** If source is None, emit "No source configured."
2. **At least one output exists.** If outputs is empty, emit "No sinks configured."
3. **Edge references are valid.** Every edge's from_node must be either "source"
   or a node ID that exists in nodes. Every edge's to_node must be either a node
   ID that exists in nodes or a sink name that exists in outputs.
4. **Node IDs are unique.** No two nodes may share the same ID.
5. **Output names are unique.** No two outputs may share the same name.
6. **Edge IDs are unique.** No two edges may share the same ID.
7. **Node type field consistency.** Gates must have a condition and routes.
   Transforms must not have condition or routes. Coalesces must have branches
   and policy. Aggregations must have a plugin.
8. **Connection completeness.** Every node's input must be reachable -- there must
   be at least one edge whose to_node matches the node's input, or the node's
   input must match the source's on_success.

Stage 1 does NOT check: plugin existence in catalog (the LLM already checked via
get_plugin_schema), config field types against Pydantic schemas (deferred to
Stage 2 dry-run), schema compatibility between nodes (Stage 2), or route
destination validity beyond existence (Stage 2).

Validation errors are returned as a tuple of human-readable strings. Each error
identifies the offending element (e.g. "Edge 'e1' references unknown node
'transform_3' as to_node").

---

## ComposerService

### Protocol

The ComposerService protocol defines a single primary method:

**compose(message, messages, state) -> ComposerResult** -- Accepts the user's
chat message, the pre-fetched conversation history, and the current
CompositionState. Runs the LLM tool-use loop. Returns a ComposerResult
containing the assistant's text response and the (possibly updated)
CompositionState.

**Parameter types (seam contract B):**

| Parameter | Type | Source |
|-----------|------|--------|
| `message` | `str` | The user's current chat message |
| `messages` | `list[ChatMessageRecord]` | Pre-fetched via `session_service.get_messages(session_id)` by the route handler |
| `state` | `CompositionState` | Reconstructed from DB via `CompositionState.from_dict()` by the route handler |

The route handler mediates between SessionService and ComposerService.
ComposerService does NOT depend on SessionService — the route handler pre-fetches
the chat history and passes it in. `_build_messages()` uses the `messages`
parameter to construct the LLM message list.

ComposerResult is a frozen dataclass with two fields: message (str) and state
(CompositionState). The state may be the same instance as the input state if
the LLM did not make any tool calls.

### Dependencies

ComposerService depends on:

- **CatalogService** (protocol) -- for discovery tool delegation.
- **LLM client** -- LiteLLM completion interface, configured with the model
  name from WebSettings.composer_model.
- **WebSettings** -- for max_turns, timeout, and rate limit configuration.

These are injected at construction time, not resolved per-call.

---

## LLM Tool-Use Loop

The compose method drives a bounded tool-use loop. The loop runs for at most
`max_turns` iterations (default 20, configured via WebSettings.composer_max_turns).
Each iteration sends the message history to the LLM, processes any tool calls in
the response, and appends results back to the message list.

### Loop Structure

1. Build the message list via `_build_messages(session, state, message)`.
2. Collect tool definitions via `_get_tool_definitions()`.
3. For each turn up to max_turns:
   a. Call the LLM with the current message list and tool definitions.
   b. If the response contains tool calls, execute each one against the current
      CompositionState. Each tool call may produce a new CompositionState (from
      ToolResult.updated_state). The state variable is updated after each tool
      call.
   c. Append the tool results as tool-role messages to the message list.
   d. Append the assistant's response (including tool_calls metadata) to the
      message list.
   e. If the response contains no tool calls (just a text response), the loop
      terminates. Return the text and the current state as a ComposerResult.
4. If the loop exhausts max_turns without the LLM producing a text-only response,
   raise ComposerConvergenceError.

### State Reflection

After each tool call, the ToolResult includes the full updated CompositionState
and its validation status. This means the LLM sees the consequences of its
mutations immediately -- including any validation errors -- and can self-correct
on the next tool call within the same turn.

### LiteLLM Provider Abstraction

The LLM client uses LiteLLM for provider abstraction. The model string
(e.g. "gpt-4o", "claude-sonnet-4-20250514", "anthropic/claude-sonnet-4-20250514")
is configured at deployment time via WebSettings.composer_model. The
ComposerService does not hardcode any provider. LiteLLM handles API key
resolution, request formatting, and response normalization.

### Error Handling

If the LLM returns a malformed tool call (unknown tool name, invalid arguments),
the tool executor returns an error message as the tool result. The LLM sees this
error and can retry. This is not a crash -- it is a normal part of the tool-use
conversation.

If the LLM client itself fails (network error, rate limit, authentication
error), the exception propagates up to the route handler. The route handler
translates it to an appropriate HTTP error response. The ComposerService does
not retry LLM calls -- that is LiteLLM's responsibility (LiteLLM has built-in
retry logic for transient failures).

### HTTP Error Shapes

The route handler translates composer failures to structured HTTP error
responses:

| Exception | HTTP Status | Response Body |
|-----------|-------------|---------------|
| `ComposerConvergenceError` | 422 Unprocessable Entity | `{error_type: "convergence", message: "...", turns_used: int}` |
| LLM client failure (network, rate limit) | 502 Bad Gateway | `{error_type: "llm_unavailable", message: "..."}` |
| LLM authentication failure | 502 Bad Gateway | `{error_type: "llm_auth_error", message: "..."}` |

All error responses use `Content-Type: application/json`. The `message` field
contains a human-readable description suitable for display in the frontend chat
pane. The `error_type` field is a stable machine-readable discriminator that the
frontend uses to select the appropriate error UI treatment.

---

## System Prompt

The LLM receives a four-part message sequence on every call within the
tool-use loop.

### 1. System Message

A static system message defining the LLM's role, constraints, and tool usage
instructions. Key content:

- You are an ELSPETH pipeline composer. Your job is to translate the user's
  natural-language description into a valid pipeline configuration using the
  provided tools.
- Always check the current state (get_current_state) before making changes.
- Always check plugin schemas (get_plugin_schema) before configuring a plugin.
- Use list_sources/list_transforms/list_sinks to discover available plugins.
- After making changes, review the validation result in the tool response. If
  there are errors, fix them before responding to the user.
- When the pipeline is complete and valid, respond with a summary of what was
  built.
- Do not fabricate plugin names or configuration fields. Only use plugins and
  fields that appear in the catalog.

### 2. Injected Context

A system-role or developer-role message injected at the start of every turn
containing:

- The current CompositionState serialized as JSON.
- The current validation status (is_valid and any errors).
- A cached summary of available plugins (names only, not full schemas -- to
  save tokens).

This context is rebuilt on every call to the LLM, reflecting the latest state
after any tool calls in the previous turn.

### 3. Chat History

The full conversation history for the session, in chronological order. Each
message includes its role (user, assistant, tool) and content. Tool call
messages include the tool name, arguments, and result. This gives the LLM
continuity across multiple user messages within the same session.

### 4. User Message

The current user message that triggered this compose call.

### _build_messages()

The `_build_messages(session, state, message)` method constructs the full
message list from these four parts. It returns a new list on every call -- it
does not cache or reuse a previous list. This is important because the tool-use
loop appends to the list during iteration; returning a cached reference would
cause cross-turn contamination.

---

## YAML Generator

The YAML generator is a pure function that converts a CompositionState into
a valid ELSPETH pipeline YAML string. The mapping is deterministic: the same
CompositionState always produces the same YAML output.

### Function Signature

`generate_yaml(state: CompositionState) -> str`

### Mapping Rules

The generator produces YAML matching the structure used by ELSPETH's
`load_settings()` parser. The top-level keys are:

**source:** Maps from SourceSpec. Emits `plugin`, `on_success`, and `options`
as direct YAML keys. The `on_validation_failure` field is nested under options.

**transforms:** Maps from NodeSpec entries where node_type is "transform".
Emits as a YAML list under the `transforms` key. Each entry includes `name`
(from NodeSpec.id), `plugin`, `input`, `on_success`, `on_error` (if set),
and `options`.

**gates:** Maps from NodeSpec entries where node_type is "gate". Emits as a
YAML list under the `gates` key. Each entry includes `name` (from NodeSpec.id),
`input`, `condition`, and `routes`. If fork_to is set, it is emitted as
`fork_to`.

**coalesce:** Maps from NodeSpec entries where node_type is "coalesce". Emits as
a YAML list under the `coalesce` key. Each entry includes `name` (from
NodeSpec.id), `branches`, `policy`, and `merge`.

**sinks:** Maps from OutputSpec entries. Emits as a YAML dict under the `sinks`
key, keyed by OutputSpec.name. Each entry includes `plugin`, `on_write_failure`,
and `options`.

**landscape:** The landscape key is always omitted from generated YAML. The
Landscape database URL is resolved at execution time from
`WebSettings.get_landscape_url()`, not from pipeline configuration. This
prevents the LLM from controlling audit infrastructure (security fix S1).

### Determinism

The generator uses `yaml.dump()` with `default_flow_style=False` and
`sort_keys=True` to ensure deterministic output. This means two
CompositionState instances with the same logical content produce byte-identical
YAML strings. This property is important for the future PipelineArtifact
hashing mechanism.

### Aggregation Nodes

NodeSpec entries where node_type is "aggregation" are emitted under the
`aggregations` key as a YAML list. Each entry includes `name`, `plugin`,
`input`, `on_success`, `on_error` (if set), and `options`.

### Frozen Field Unwrapping (R4)

CompositionState and all nested *Spec types use `freeze_fields()` in their
`__post_init__` methods. This converts `dict` to `MappingProxyType` and `list`
to `tuple` recursively (see the Immutability Contract section above). PyYAML's
default Dumper does not know how to represent `MappingProxyType` and will raise
`RepresenterError` if it encounters one.

`generate_yaml()` must unwrap all frozen dataclass fields to plain Python types
before passing them to `yaml.dump()`. Specifically:

- **CompositionState.to_dict()**: Add a `to_dict()` method to CompositionState
  that performs recursive unwrapping of frozen containers back to plain Python
  types: `MappingProxyType` to `dict`, `tuple` to `list`. This can delegate to
  a `deep_unfreeze()` utility or perform manual recursion. The method returns a
  plain `dict` suitable for both `yaml.dump()` and JSON serialisation.

- **generate_yaml() calls to_dict()**: The generator calls
  `state.to_dict()` and passes the resulting plain dict to `yaml.dump()`, not
  the frozen CompositionState directly.

- **Alternative (not recommended)**: A custom YAML representer could be
  registered: `yaml.add_representer(MappingProxyType, lambda dumper, data:
  dumper.represent_dict(data))`. This solves the YAML case but does not help
  with API serialisation, so it is not the recommended approach.

The `to_dict()` approach is recommended because it is also needed at the API
serialisation boundary -- Sub-Spec 2's `CompositionStateResponse` Pydantic
schema (in `sessions/schemas.py`) must convert CompositionState to a JSON-safe
dict, and `to_dict()` serves both use cases.

---

## API Integration

### POST /api/sessions/{id}/messages

This is the primary interaction endpoint. When a user sends a chat message,
the route handler:

1. **Authenticates** the request via the auth middleware (Sub-Spec 2).
2. **Loads the session** from SessionService, verifying the session belongs to
   the authenticated user.
3. **Persists the user message** as a ChatMessage with role="user".
4. **Loads the current CompositionState** for this session (latest version, or
   an empty initial state if no state exists yet).
5. **Calls ComposerService.compose()** with the user message, session, and
   current state. This runs the LLM tool-use loop and may take several seconds.
6. **Persists the assistant message** as a ChatMessage with role="assistant".
   If the LLM made tool calls, they are stored in the tool_calls JSON field.
7. **If the state changed** (new version number), persists the new
   CompositionState as a new version record in the session.
8. **Returns** a JSON response containing the assistant's ChatMessage and the
   current CompositionState (or null if no state exists yet).

The response shape is `{message: ChatMessage, state: CompositionState | null}`.

While the composer is running, the frontend shows a "composing" indicator. The
request blocks until the composer completes or times out (controlled by
WebSettings.composer_timeout_seconds). There is no streaming in v1.

### GET /api/sessions/{id}/state/yaml

Returns the generated YAML for the current composition state. The route handler
loads the session's active CompositionState and calls `generate_yaml(state)`.

**Response:** `{yaml: str}` -- the YAML string ready for display in the
frontend's YAML tab.

If the session has no CompositionState yet, returns HTTP 404. Authentication
and session ownership checks are identical to the messages endpoint.

### Initial State

When a session has no CompositionState yet (first message), the route handler
creates an initial empty state: source=None, nodes=(), edges=(), outputs=(),
metadata=PipelineMetadata(), version=1. This is passed to the composer as the
starting point. If the composer produces tool calls that mutate the state, the
resulting state is persisted as version 1.

### State Revert and Chat History

When the user reverts to a prior composition version (via Sub-Spec 2's
`set_active_state`), the route handler must inject a system message into the
chat history: "Pipeline reverted to version N." This gives the LLM context that
the state has been rolled back, preventing it from making decisions based on
stale assumptions about what the pipeline currently contains. The injected
message uses role="system" and is persisted as a ChatMessage so it appears in
the conversation history on subsequent turns.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/composer/__init__.py` | Module init |
| Create | `src/elspeth/web/composer/state.py` | CompositionState, SourceSpec, NodeSpec, EdgeSpec, OutputSpec, PipelineMetadata, ValidationSummary |
| Create | `src/elspeth/web/composer/tools.py` | Discovery and mutation tool definitions, ToolResult, tool executor |
| Create | `src/elspeth/web/composer/prompts.py` | System prompt template, context injection, _build_messages() |
| Create | `src/elspeth/web/composer/protocol.py` | ComposerService protocol, ComposerResult, ComposerConvergenceError |
| Create | `src/elspeth/web/composer/service.py` | ComposerServiceImpl -- LLM tool-use loop |
| Create | `src/elspeth/web/composer/yaml_generator.py` | generate_yaml(state) -> str |
| Create | `tests/unit/web/composer/__init__.py` | Test package init |
| Create | `tests/unit/web/composer/test_state.py` | CompositionState immutability, versioning, mutation methods, freeze guards |
| Create | `tests/unit/web/composer/test_tools.py` | Tool execution: discovery delegation, mutation + validation, error paths |
| Create | `tests/unit/web/composer/test_yaml_generator.py` | YAML generation: linear pipeline, gate routing, fork/coalesce, round-trip through load_settings() |
| Create | `tests/unit/web/composer/test_service.py` | Composer loop: single turn, multi-turn, validation self-correction, convergence error, text-only response |
| Modify | `src/elspeth/web/sessions/routes.py` | Wire POST /api/sessions/{id}/messages to ComposerService.compose() |

---

## Acceptance Criteria

1. CompositionState is a frozen dataclass. All container fields are deep-frozen
   via freeze_fields() in __post_init__. Attempting to mutate a field after
   construction raises TypeError or AttributeError.

2. Mutation methods (with_source, with_node, without_node, with_edge,
   without_edge, with_metadata) return new CompositionState instances with
   version incremented by one. The original instance is unchanged.

3. ToolResult is a frozen dataclass. The affected_nodes field is deep-frozen
   via freeze_fields() in __post_init__.

4. Every mutation tool returns a ToolResult with success=True/False, the full
   updated state, a ValidationSummary, and the list of affected node IDs.
   Mutation tools never raise exceptions for invalid input -- they return
   success=False with the original state.

5. Stage 1 validation catches: missing source, missing sinks, dangling edge
   references, duplicate IDs, node-type field inconsistencies, and unreachable
   node inputs. Errors are human-readable strings identifying the offending
   element.

6. The LLM tool-use loop is bounded at max_turns (default 20). Exceeding the
   limit raises ComposerConvergenceError.

7. Discovery tools (list_sources, list_transforms, list_sinks, get_plugin_schema,
   get_expression_grammar, get_current_state) return data without modifying
   state.

8. The YAML generator produces valid ELSPETH pipeline YAML that can be parsed
   by load_settings(). A CompositionState representing a linear pipeline
   (source -> transform -> sink) round-trips through generate_yaml() and
   load_settings() without error.

9. The YAML generator is deterministic: the same CompositionState produces
   byte-identical YAML on repeated calls.

10. POST /api/sessions/{id}/messages persists the user message, calls the
    composer, persists the assistant response, persists any new state version,
    and returns {message, state}.

11. _build_messages() returns a new list on every call. It does not return a
    cached or shared reference.

12. The ComposerService uses LiteLLM for provider abstraction. The model is
    configured via WebSettings.composer_model, not hardcoded.

13. All async test functions carry @pytest.mark.asyncio.

14. All frozen dataclasses with container fields pass the enforce_freeze_guards.py
    CI check.

15. `generate_yaml()` correctly serialises CompositionState objects that have
    been through `freeze_fields()` -- no `RepresenterError` from PyYAML on
    `MappingProxyType` or `tuple` fields.

16. Tests that mock CatalogService must use real PluginSummary and
    PluginSchemaInfo instances, not plain dicts. Mock return types must match
    the CatalogService protocol.

17. Every `*Spec` type and `CompositionState` defines a `from_dict()` class method
    that reconstructs a frozen dataclass instance from a plain Python dict. The
    round-trip invariant `CompositionState.from_dict(s.to_dict()) == s` holds for
    any valid CompositionState instance, including states with nested
    MappingProxyType fields and None optional fields.

18. `from_dict()` is the **only** way to construct domain objects from deserialised
    JSON. Direct dataclass construction from raw dicts is forbidden at seam
    boundaries (Sub-2's `from_record()`, the validation route handler, etc.).

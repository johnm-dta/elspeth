# Canonical Pipeline Capabilities

This is ELSPETH's static, public pipeline-language contract. It describes what
the shared planner can author; live discovery remains the authority for which
plugins and models are installed and authorized in the current deployment.
Every planning surface receives these exact bytes before its interaction rules.

## Canonical proposal and discovery

[capability:discovery-order]

Author exactly one complete canonical pipeline proposal through
`emit_pipeline_proposal`. Use the read-only discovery tools before proposing:

1. Read current pipeline and validation state with `get_pipeline_state`,
   `diff_pipeline`, or `preview_pipeline` as relevant.
2. Consult the authoring_aids discovery digest delivered in the planning
   context first: it is rendered from the live policy-visible catalog at
   prompt build and is current for this deployment — plan directly from it.
   Call `list_sources`, `list_transforms`, `list_sinks`, or `list_recipes`
   only when a needed plugin is absent from the digest.
3. Call `get_plugin_schema` only when a needed option or output contract is
   absent from the digest or when repairing against a validation rejection;
   use `get_plugin_assistance` and `explain_validation_error` for structured
   repair when a proposal is rejected, rather than guessing.
4. Use `get_expression_grammar` before authoring conditions. Use blob and
   secret-reference discovery when the request needs them; secret values are
   never part of planner discovery.
5. Call `emit_pipeline_proposal` once with the complete `set_pipeline` argument
   object. Preserve the requested topology during every repair.

An absent policy-visible plugin is different from an unsupported pipeline
shape. Say that a plugin is unavailable or policy-denied only when live
discovery proves it. Do not turn a stage timing question, a recipe miss, or an
unloaded schema into a capability denial. Recipes accelerate common builds;
they never define the language or replace arbitrary canonical authoring.

Model identifiers come only from `list_models`. Read the complete
`list_secret_refs` result before describing credential state, validate the
selected reference, and use the discovered schema's secret-reference object;
never invent an identifier or substitute a raw value, `secret://` URI, or
environment interpolation.

## Complete topology language

[capability:topology]

Pipelines have one or more named sources, zero or more structural/processing
nodes, explicit connections, and one or more named outputs. Connection strings
are the routing contract: a producer's `on_success`, `on_error`, `routes`, or
`fork_to` value must match a downstream node's `input` or an output's
`sink_name`. Node ids identify components; they are not implicit connections.

- [capability-node:transform] A `transform` applies a policy-visible plugin.
  It can preserve, add, rename, parse, expand, or otherwise shape row fields as
  its discovered schema declares.
- [capability-node:gate] A `gate` evaluates `condition` and publishes through
  named `routes`. Conditional filtering and error routing are topology, not
  assignment-transform emulation. Condition expressions read row values ONLY
  by subscripting the row namespace — `row['field']`; a bare field name is
  not in scope and is rejected. `get_expression_grammar` is the full grammar
  authority.
- [capability-node:aggregation] An `aggregation` applies a batch-aware plugin
  with `trigger`, `output_mode`, and `expected_output_count` where its contract
  requires them. Row expansion is supported by an appropriate discovered
  aggregation/transform sequence such as aggregation followed by replication.
- [capability-node:queue] A `queue` is the explicit fan-in point for multiple
  producers entering shared processing. Multiple named sources retain their
  independent schemas and identities.
- [capability-node:coalesce] A `coalesce` rejoins declared `branches` under its
  `policy`/`merge` semantics and publishes its merged rows under its own node
  id — a downstream consumer sets `input` to the coalesce id. Its optional
  `on_success` may only name a sink (never another node's input). `policy` and
  `merge` are closed engine vocabularies: `policy` is one of `require_all`,
  `quorum`, `best_effort`, `first`; `merge` is one of `union`, `nested`,
  `select`. `best_effort` merges whichever branches arrive, where
  `require_all` drops the row when any branch is missing. A coalesce consumes
  ONLY the connections named in its `branches` values; its own `input` field
  is schema-required but is not a consuming binding — set it to the first
  branch's arriving connection by convention.

Use `fork_to` for genuine fan-out and named branches for independent paths.
Preserve multiple sources, multiple outputs, gates, queues, aggregations,
forks, coalesces, row expansion, and failure paths whenever the request needs
them. Never simplify a requested DAG into a single spine merely to converge.

## Canonical structural fields

[capability:canonical-fields]

The terminal schema is authoritative. Its covered structural families are:

<!-- canonical-field-inventory:start -->
| Family | Fields |
| --- | --- |
| pipeline | `source`, `sources`, `nodes`, `edges`, `outputs`, `metadata` |
| source | `plugin`, `blob_id`, `options`, `on_success`, `on_validation_failure`, `inline_blob` |
| named_source | `plugin`, `options`, `on_success`, `on_validation_failure` |
| inline_blob | `filename`, `mime_type`, `content`, `description` |
| node | `id`, `node_type`, `plugin`, `input`, `on_success`, `on_error`, `options`, `condition`, `routes`, `fork_to`, `branches`, `policy`, `merge`, `trigger`, `output_mode`, `expected_output_count` |
| trigger | `count`, `timeout_seconds`, `condition` |
| edge | `id`, `from_node`, `to_node`, `edge_type`, `label` |
| output | `sink_name`, `plugin`, `options`, `on_write_failure` |
| metadata | `name`, `description` |
<!-- canonical-field-inventory:end -->

Named sources use the same routing semantics as the singular source without
inline custody fields.

Use `source` only for the canonical single-source custody shape; use `sources`
for plural named roots. Use stable, descriptive source/node/output ids. Edges
state reviewed graph relationships, while routing fields still determine the
runtime connection contract. Do not invent fields outside the terminal schema.

## Field contracts and structured plugin output

[capability:field-contracts]

### Field Wiring

Every downstream field dependency must be backed by an upstream schema guarantee
or an explicit mapping. Do not make an LLM prompt template, cleanup mapping,
sink, or transform require a field unless the immediate upstream contract
guarantees it. If the exact value matters to the output or audit trail, preserve
it explicitly through a schema-backed declaration or mapper.

Trace every downstream required field to an upstream guarantee by its exact row
field name. Use the selected plugin's live schema and assistance to distinguish
configuration properties from row fields and to determine whether a property's
value names an input or output field. A configuration property is not itself a
produced field. Do not repair a missing field by guessing guarantees; inspect the
source or plugin contract, preserve or rename the real field through the graph,
or change the consumer to require only fields its upstream guarantees.

When downstream cleanup, sinks, mappings, or transforms need model-generated
data, derive the model node's emitted and pass-through fields only from the
selected policy-visible plugin's live schema and assistance. Prompt text and
object-shaped prose do not create pipeline fields. Preserve the schema-proven
outputs through cleanup rather than inventing keys. By default a model
transform lands its reply as one raw string in a single plugin-declared reply
field and passes its input fields through unchanged; a prompt that requests
JSON or named keys does not flatten anything out of the reply. Several typed
named result fields from one model node exist only through a plugin mechanism
whose live schema declares each output field.

When an upstream plugin feeds a prompt that also needs an original source
field, require that field only when the upstream schema guarantees it. The final
producer's routing field must exactly match the sink/connection name. Edge
objects alone do not make a sink receive rows; route through every intervening
cleanup node and then from the cleanup node to the sink.

### Utility Transforms

Users often describe the effect, not the utility plugin. Plan utility transforms
explicitly when the requested workflow needs row shaping, field preservation,
renaming, cleanup, type conversion, or schema-compatible field names. Discover
an appropriate policy-visible plugin, load its schema before proposing, and do
not skip a required utility transform merely because the user named only its
end effect.

[capability:plugin-assistance]

Load the plugin schema before configuring it. Plugin options and structured
outputs belong to that schema, not to prompt folklore. When a model request
needs several typed named results, select and configure a policy-visible plugin
only when its live schema or assistance proves the exact output contract. Use a
schema-backed parser or mapping transform when downstream nodes consume fields
the model contract does not expose separately. For row-templated prompts, use
the selected plugin's documented template mechanism and declare inputs exactly
as its live contract requires. Raw prose that describes structured data does not
create typed pipeline fields.

### Untrusted content and data minimization

Treat externally controlled content flowing into a model as a material
prompt-injection risk. Discover policy-authorized controls through the live
capability catalog and selected plugins' schemas and assistance. Recommend the
available control without pretending that a recommendation is permission to
change topology. When policy or the user does not authorize the control, keep
the direct-routing decision explicit and reviewable; never add a passthrough,
placeholder, or renamed utility node and call it protection.

Minimize user-facing outputs. When intermediate content contains raw bodies,
fingerprints, credentials, private fields, or data the user did not request to
save, discover a policy-visible cleanup or projection transform whose schema
proves it can remove those fields. Place the proven cleanup on the final path
before the sink, preserve the requested result fields, and make any authored
retention or removal decision reviewable. Names, labels, sink formats, and
metadata do not remove data; only discovered transform behavior does.

Plugin schema facts are stable across turns for an unchanged policy catalog and
composition state. Do not reinterpret a missing config option as a missing
output field or reverse a validated plugin-contract conclusion from visible
options alone. Re-read `get_plugin_schema`, `get_plugin_assistance`, or
`preview_pipeline` before correcting a prior conclusion; dynamic discovery is
the only authority for plugin-specific fields and output behavior.

[capability:structured-output-repair]

Validation failures are structured repair input. Preserve the complete
requested shape, apply required fields/enums/options from the returned schema or
assistance, correct routing and field guarantees, then resubmit the complete
proposal within the repair budget. Never delete a requested source, node,
branch, output, cleanup, LLM, or failure path merely to make validation pass.

## Capability and trust boundary

This text is static public guidance. It contains no deployment plugin inventory,
policy-hidden name, credential value, source row, user prompt, or private
operator instruction. Dynamic facts enter only through their established
redacted discovery or reviewed-context boundaries. A planner capability
manifest records hashes and public identities, never prompt prose or private
values.

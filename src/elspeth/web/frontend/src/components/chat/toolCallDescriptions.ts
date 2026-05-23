/**
 * Generic, audience-facing descriptions of what each composer tool call does.
 *
 * These are deliberately **kind-of-operation** descriptions, not instance
 * descriptions — they tell a user what a tool call of this type does in
 * general, without referring to the specific arguments the LLM passed.
 *
 * If a tool name is not in this map, ToolCallCard falls back to a generic
 * "Composer tool call." string so the tooltip always has content.
 */
export const TOOL_CALL_DESCRIPTIONS: Record<string, string> = {
  // Read-only lookups (rendered as the "i" ribbon, no proposal).
  list_sources:
    "Browses the catalog of available source plugins on this deployment.",
  list_transforms:
    "Browses the catalog of available transform and gate plugins.",
  list_sinks: "Browses the catalog of available sink plugins.",
  list_models:
    "Looks up the LLM models available on this deployment's credentials.",
  list_recipes: "Browses pre-built pipeline recipes that can be applied.",
  list_sessions: "Looks up your previous composer sessions.",
  get_plugin_schema:
    "Reads a plugin's configuration schema to understand its options.",
  get_plugin_assistance:
    "Looks up usage guidance and worked examples for a specific plugin.",
  get_pipeline_state:
    "Reads the current pipeline state being composed in this session.",
  get_audit_info:
    "Looks up audit metadata for the current session (run IDs, lineage anchors).",
  get_expression_grammar:
    "Looks up the expression-language grammar used by routing and templating.",
  diff_pipeline:
    "Compares two pipeline configurations to show what changed between them.",
  preview_pipeline:
    "Builds the pipeline in memory to check for errors before applying.",
  explain_validation_error:
    "Looks up a plain-language explanation for a validation error code.",
  generate_yaml: "Renders the current pipeline as a YAML document.",

  // Mutating tool calls (rendered as a proposal card with Accept / Reject).
  set_source:
    "Sets the pipeline's data source — what records the pipeline starts from.",
  clear_source: "Clears the pipeline's data source.",
  set_pipeline:
    "Replaces the entire pipeline configuration in a single operation.",
  set_output: "Adds or replaces an output sink — where results are written.",
  set_metadata:
    "Sets the pipeline's metadata (name, description, tags).",
  upsert_node:
    "Adds a new transform or gate node, or replaces an existing one with the same id.",
  upsert_edge:
    "Adds or replaces a connection between two nodes in the pipeline.",
  remove_node: "Removes a transform or gate node from the pipeline.",
  remove_edge: "Removes a connection between two nodes.",
  remove_output: "Removes an output sink from the pipeline.",
  patch_source_options:
    "Updates one or more configuration options on the source without replacing it.",
  patch_node_options:
    "Updates one or more configuration options on a transform or gate node.",
  patch_output_options:
    "Updates one or more configuration options on an output sink.",
  new_session: "Starts a fresh composer session.",
  save_session: "Saves the current composer session so it can be resumed later.",
  load_session: "Resumes a previously-saved composer session.",
  delete_session: "Removes a saved composer session.",
};

export function describeToolCall(name: string): string {
  return TOOL_CALL_DESCRIPTIONS[name] ?? "Composer tool call.";
}

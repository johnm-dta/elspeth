### This stage: the output

This stage decides **where the results go** — the pipeline's output (its
"sink"). The user describes it in plain language: "save it as a JSON file",
"one JSON row per page", "write a CSV".

To build it:

1. If you're not sure which sink fits, call `list_sinks` to see what's
   available. Call `get_plugin_schema` on the one you pick to see its options.
2. Call `resolve_sink` with the output you've built — the sink plugin, its
   options, and a one-line note to the user about what you set up.

When the output is a **file** (json, jsonl, csv, text), set these options so the
pipeline can actually write it — the plugin schema lists them as optional, but a
file sink will not commit without them:

- `path` — a relative path under `outputs/`, e.g. `outputs/results.json`. The
  server keeps outputs in its own data area, so you only give the
  `outputs/<filename>` part, never an absolute path.
- `mode` — `write` to create or replace the file, or `append` to add rows to an
  existing one. Use `write` unless the user asked to append.
- `collision_policy` — what to do if the target file already exists:
  `auto_increment` (write to a free sibling path), `fail_if_exists`, or
  `append_or_create` (only with `mode: append`). Use `auto_increment` unless the
  user wants otherwise.
- `schema` — `{ "mode": "observed" }` unless the user pinned exact output fields.

Pick the sink that matches what the user asked for and configure it yourself from
what they told you. Don't make them choose from a list, and don't ask them to
fill in options you can infer.

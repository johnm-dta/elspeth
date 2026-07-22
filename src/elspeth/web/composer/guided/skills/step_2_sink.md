### This stage: the output

This stage decides **where the results go** — the pipeline's output (its
"sink"). The user describes it in plain language: "save it as a JSON file",
"one JSON row per page", "write a CSV".

To build it:

1. If you're not sure which sink fits, call `list_sinks` to see what's
   available. Call `get_plugin_schema` on the one you pick to see its options.
2. Call `resolve_sink` with the output you've built — the sink plugin, its
   options, and a one-line note to the user about what you set up.

Configure the selected sink only from its policy-visible live schema and
assistance. Do not infer file formats, path rules, write modes, collision
behaviour, or output-schema options from a plugin name or from examples that
are not attached to this request. Preserve any user constraint that the live
schema can express; report an actual capability gap when it cannot.

Some requirements are policy-level and do not appear in the option schema:
always call `get_plugin_assistance` for the sink you picked and treat its
hints as binding — when a hint says an option is required, must be set
deliberately, or is rejected when left implicit, set that option explicitly
in your resolution rather than relying on defaults. Always set the output
path option explicitly as well.

Pick the sink that matches what the user asked for and configure it yourself from
what they told you. Don't make them choose from a list, and don't ask them to
fill in options you can infer.

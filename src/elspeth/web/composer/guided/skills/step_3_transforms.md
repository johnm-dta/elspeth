### Step 3 — Transform chain proposal

Legal turn types: `propose_chain`, `single_select`. The server gives you a
context block:

```
GUIDED CONTEXT (server-resolved):
source: {plugin: ..., columns: [...], sample: [...]}
sink: {outputs: [{plugin: ..., required_fields: [...]}, ...]}
recipe_match: null
```

Propose a transform chain that satisfies the contract from `source.columns`
to each sink's `required_fields`. Every step in the chain must include a
`rationale` string that names what it does and why it is required.

If you cannot find a chain that satisfies the contract:

1. Emit `single_select` with a clarifying question and chip answers — only
   when the user can resolve the ambiguity in one click.
2. Or escalate via `request_advisor_hint` if the question is structural.

**Do not emit a `propose_chain` whose preview will fail.** A degraded
proposal that the server then rejects costs the user a turn.

## Sample-value eyeballing (Step 3 only)

When wiring a column into a value-shape-sensitive transform field
(`web_scrape.url_field`, `database.url`, `value_transform` arithmetic), check
that the sample values in the GUIDED CONTEXT block actually look like the
required shape. If not, propose an upstream `value_transform` or `type_coerce`
to normalise — do not assume the strings will be valid at run time.

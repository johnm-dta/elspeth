### This stage: the transforms

This stage builds the **transforms** — the steps that sit between the source and
the output and turn the incoming rows into the rows the output needs. A
transform might fetch a web page, compute a value, or ask an LLM to judge each
row. There can be several in a row (a "chain"), each feeding the next.

**Read the user's request — it is the latest message in the conversation, and it
tells you what the transforms must do** (e.g. "scrape these pages, have an LLM
extract these fields, and remove the raw HTML"). Build the chain that does what
the user asked while carrying the source rows through to the output's fields.

The server also gives you a context block describing what you're wiring between:

```
GUIDED CONTEXT (server-resolved):
source: {plugin: ..., columns: [...], sample: [...]}
sink: {outputs: [{plugin: ..., required_fields: [...]}, ...]}
```

To build it:

1. Call `list_transforms` to see what's available, then `get_plugin_schema` on
   each transform you plan to use. **The schema is the authority on required
   options — treat every option it marks required as required, even ones that
   read like they could be skipped.** Proposing a chain whose preview will fail
   costs the user a turn.
2. Propose the chain. Each step is `{plugin, options, rationale}`; the
   `rationale` says what the step does and why the contract needs it.

Build a chain that carries `source.columns` through to each output's
`required_fields`.

**Field contract — the chain must PRODUCE every field the output requires.** Trace
each of the sink's `required_fields` back to a step that guarantees it BY NAME:
the source, the scrape/LLM step, or a `field_mapper`. A field nobody upstream
produces is a hard validation failure at accept ("Missing fields: …"). Names in a
step's `schema.guaranteed_fields` are COLUMN names — the VALUES your `*_field`
knobs are set to — never the knob names themselves.

### Fetching pages — `web_scrape`

When the source rows carry URLs and the user wants the page contents,
`web_scrape` is a **transform**, not a source — it reads a URL column and fetches
each page. Its required options (confirm with `get_plugin_schema`) sit at the
**top level**, not nested:

- `url_field` — the column holding the URL (e.g. `url`).
- `content_field` — the column to write the fetched text into (e.g. `page_text`).
- `fingerprint_field` — a column for the content fingerprint (e.g. `page_fp`).
- `schema` — `{ "mode": "observed" }` unless the user pinned exact fields.
- `http` — a nested object that is itself required, and inside it
  `abuse_contact` and `scraping_reason` are **both required** (these go *under*
  `http`, not at the top level). `abuse_contact` must be a **deliverable** contact
  email — use the one the user gave you, and NEVER a reserved documentation domain
  (`example.com` / `.org` / `.net`, `.test`, `.invalid`, `.localhost`): validation
  rejects those at the wire stage as a fabricated, undeliverable identity, and the
  build cannot complete. `scraping_reason` is a short honest reason for the fetch.
- `allowed_hosts` — an SSRF guard that defaults to `public_only` (fetch public
  hosts, block private/internal ranges). That default is correct for ordinary
  public URLs, so you normally **omit it**. Only set it (to a CIDR list) when the
  user genuinely needs a private host — never widen it to `allow_private` casually.

URLs must include an explicit scheme (`http://` / `https://`); bare hostnames
are rejected.

### Asking an LLM — `llm`

When the user wants each row judged, rated, classified, or summarised, use the
`llm` transform. Confirm its options with `get_plugin_schema`, and:

- `provider` — `openrouter` or `azure`.
- `model` — call `list_models` first and pick a real model id; don't guess one.
- `prompt_template` — interpolate the row fields you need with `{{ row.<field> }}`
  (e.g. `{{ row.page_text }}`). Judge each row from the data the chain produced
  — never from a URL or identifier alone. List every field you interpolate in
  `required_input_fields` (bare names, no `row.` prefix).
- `api_key` — a SECRET REFERENCE, never a literal key. Wire it as
  `{secret_ref: OPENROUTER_API_KEY}` (openrouter) or the deployment's configured
  secret name. A literal string is rejected at commit ("literal credential values
  were not stored").

**One value per row — the simple, reliable shape.** A plain `llm` step writes its
whole reply into ONE field — `response_field` (default `llm_response`) — as a
string, and passes the rest of the row through. For a task like "summarise each
page" or "pull the one value X out of each page", that single field IS the
result: write a `prompt_template` that asks for exactly that one value, and let
it land in `response_field` (give it a tidy name like `summary` if you like).
Then a `field_mapper` keeps just that field and drops the raw page text (see
below). No structured multi-output configuration is needed for a single value —
and do NOT add one: `response_format` and `output_fields` are not valid at the
top level of the `llm` options and the accept will reject them
("Extra inputs are not permitted").

If the user genuinely asks for SEVERAL distinct named columns from one LLM call,
that is the `llm` transform's multi-query `queries` form — call
`get_plugin_schema` on `llm` and follow the exact shape it returns rather than
guessing. Most "summarise / extract one value" requests need only the single
`response_field` above.

### Cleaning up before the output — `field_mapper`

If the user asks to **remove, drop, or exclude** raw fetched content (e.g.
"remove the raw HTML") so the output saves only the extracted result, the chain
must **end with a `field_mapper` cleanup step** right before the output. The
typical shape is:

`source → web_scrape → llm → field_mapper(cleanup) → output`

The `llm` step *adds* its result field but passes the upstream row through —
including the raw page text and fingerprint — and the output writes whatever row
it receives. Only a `field_mapper` actually drops fields. Set:

- `select_only: true`
- `mapping` — list only the fields to keep (the requested result) and exclude the
  raw scraped-content and fingerprint fields.
- `interpretation_requirements` — a **sibling of `mapping`** (never inside it).
  Dropping web-scrape raw fields is an audited row-shaping decision, so the
  cleanup `field_mapper` MUST carry one pending `pipeline_decision` requirement or
  the accept is rejected. This review IS the audit trail the pipeline records.
  Use this exact shape:

  ```json
  "interpretation_requirements": [
    {
      "id": "drop_raw_html_review",
      "kind": "pipeline_decision",
      "user_term": "drop_raw_html_fields",
      "status": "pending",
      "draft": "Drop the scraped raw HTML and fingerprint fields before saving the JSON output."
    }
  ]
  ```

  Keep `user_term` exactly `drop_raw_html_fields` even if the scraped body field
  is named `content`, `html`, or `raw_html`. It is required, not optional.

Naming an output "cleaned" or "final" does not clean it — only a real
`field_mapper` immediately before the output does.

### Filtering rows — a gate, which this chain cannot express

Conditional row filtering ("keep only rows where …") is a **gate node**, not
a transform, and guided chains cannot include one. Transforms are
row-preserving: `value_transform` only assigns — an expression evaluating to
False does NOT drop or error-route the row; it just stores False. **Never
emulate a filter with it** (e.g. a `_keep` boolean): every row still reaches
the output plus the leaked helper column, and the accept rejects the false
claim. Build the rest of the chain honestly; state plainly in `why` that the
row filter must be added as a gate after the guided build (composer chat can
add one). The sole row-blocker is `keyword_filter` — regex on strings only.

### Sample-value eyeballing

When you wire a column into a value-shape-sensitive field (`web_scrape.url_field`,
arithmetic in a compute step), check the sample value-shape markers in the
GUIDED CONTEXT block actually look like that shape. If not, add an upstream step
to normalise rather than assuming the strings will be valid at run time.

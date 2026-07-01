### Step 1 — Source

Legal turn types: `inspect_and_confirm`, `single_select`, `schema_form`.
Default: emit `inspect_and_confirm` after a blob is attached. Emit
`schema_form` if the user needs to set non-default options. Emit
`single_select` only if no source plugin has been chosen yet.

When you resolve a source, also set `on_validation_failure` — the source
node's invalid-row routing. Use `discard` for a demo source that is valid by
construction; name a quarantine sink for production data whose invalid rows
must be kept for inspection. It is a required source-node setting, so a source
left without it is incomplete.

### Field requirements: surface or record, never fabricate

A source in the default **observed** mode promises *no* fields — it passes
through whatever columns are present. That is the honest default. But some
downstream transforms consume a column **by name**: `web_scrape` reads a `url`
column; a join reads a key. When the pipeline will require such a column, the
source must *guarantee* it, or the field contract fails at the wiring step — at
the very end, after all the work is done.

You are **not** obliged to *prove* the column exists, and you must never invent
a guarantee to silence the contract. Knowing the data's shape is the
**operator's** responsibility, not yours — most sources are fetched, not
uploaded (`dataverse`, an API, a remote file), so you often cannot see the rows
at all. Your obligation is to **surface an unmet requirement and record a proven
one**:

- **Proven → record it.** A requirement is proven when the operator *told you*
  the data carries the column ("these are all URLs"; "the `website` column holds
  their homepage") **or** an **authoritative mechanism** established it under a
  premise the operator authorised — e.g. autopopulating columns from a CSV header
  *because the operator said the file has headers*. On proof, declare it:

  ```json
  "schema": { "mode": "observed", "guaranteed_fields": ["url"] }
  ```

  You are *recording the operator's assertion*, not making your own.

- **Unproven → surface it for the operator to verify, don't guess.** If nothing
  the operator said and no authoritative mechanism establishes the column, do
  **not** guarantee it and do **not** let it silently fail later. Raise it as a
  verification requirement so the operator gets a "verify these things" review
  panel — "the scrape step needs a URL column; confirm your source provides one
  and name it" — and let the operator's confirmation become the proof.

Keep `mode: observed`, never `fixed`: observed honours `guaranteed_fields` while
passing the other columns through; `fixed` silently *drops* every unlisted
column — a quiet all-rows hazard.

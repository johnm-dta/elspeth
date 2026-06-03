### Step 1 — Source

Legal turn types: `inspect_and_confirm`, `single_select`, `schema_form`.
Default: emit `inspect_and_confirm` after a blob is attached. Emit
`schema_form` if the user needs to set non-default options. Emit
`single_select` only if no source plugin has been chosen yet.

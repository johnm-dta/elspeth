# Convergence Suite

The convergence bar for the simple-pipeline-convergence program
(filigree epic `elspeth-783c9dede8`).

These three scenarios encode the failure modes the composer was hitting
on staging: schema-blind CSV classifiers, URL-as-path mistakes on text
sources, and numeric gates over string-typed fields. Each scenario
expresses the convergence bar mechanically — the proof step, repair loop,
and recipes (Steps 3, 4, 5 of the program) must produce pipelines that
score GREEN here without bouncing back to the user.

| Scenario | Catches |
|----------|---------|
| `csv-classifier/` | Fixed schemas that omit observed columns and silently discard every row when paired with `on_validation_failure=discard`. |
| `url-text-smoke/` | URL strings fed into a `text`/`csv` source's `path` field instead of being wrapped as a blob and processed via `web_scrape`. |
| `numeric-gate/` | Gates comparing string-typed CSV fields to numeric literals without prior `type_coerce` or a numeric source schema. |
| `deep-routing-cascade/` | Uploaded CSV loan-triage prompts that require blob discovery, source inspection, three transforms, five chained gates, and seven outputs without exhausting the repair budget. |

## Convergence-bar GREEN criteria

In addition to the legacy criteria (`must_be_valid`,
`must_have_node_kinds_substring_any_of`, `must_have_outputs_min`,
`must_not_contain_passivity`), these scenarios use:

- `must_have_node_chain_in_order` — substrings that must appear in the
  pipeline's node plugins in the listed relative order.
- `must_include_observed_columns` — the source schema must cover these
  columns (observed/flexible mode passes; fixed mode must list them).
- `must_handle_field_as_numeric` — a `type_coerce` node converting the
  field to int/float, or a source schema declaring the numeric type.
- `max_repair_turns` — `state.composer_meta.repair_turns_used` must be
  at most this value. Plumbed end-to-end:
  `service.py::ComposerServiceImpl._compose_loop` threads
  `repair_turns_used` onto `ComposerResult`, which `web/sessions/routes.py`
  persists into the `composition_states.composer_meta` JSON column. The
  `GET /api/sessions/{id}/state` response surfaces it under
  `composer_meta.repair_turns_used`. The criterion still ambers when the
  field is absent (e.g. for revert/fork-derived states that did not run a
  compose to produce them).

## Running

Cohort runs use the standard harness:

```bash
ELSPETH_RGR_SCENARIO=evals/composer-rgr/scenarios/convergence-suite/csv-classifier/scenario.json \
  ELSPETH_EVAL_BASE_URL=https://elspeth.foundryside.dev \
  ELSPETH_EVAL_USER=... ELSPETH_EVAL_PASS=... \
  ./evals/composer-rgr/run_scenario.sh convergence
```

Closure of the parent epic depends on each scenario reaching hard-GREEN
≥ 4/6 across a fresh cohort.

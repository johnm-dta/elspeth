# Composer Capability Parity — canonical fixture corpus

This directory holds the deterministic fixture corpus that drives the composer
capability-parity verification (Plan 05). It is authoring-surface agnostic: the
same fixtures are replayed through the freeform, guided-full, and guided-staged
adapters in later tasks and must derive equivalent committed graphs.

## What is here

- `fixtures/*.json` — nine canonical topology fixtures, one per capability
  class (linear transform, conditional gate, multi-output, fork/coalesce,
  multi-source queue, aggregation, row expansion, error routing, structured
  LLM). Each fixture records:
  - `intent` — a plain-language outcome request (no composer tool-call order).
  - `canonical_arguments` — a VALID `set_pipeline` payload that validates against
    `SetPipelineArgumentsModel` (`src/elspeth/web/composer/redaction.py`) and
    references only plugins available to a trained operator.
  - `semantic_expectations` — the node/edge/output shape each surface must
    derive (preserve-vs-canonicalize split per design §8.1).
  - `runtime_assertions` — behavioural claims the executed graph must satisfy.
- `fixtures/two_llm_colour.csv` — the exact ten-row colour palette input for the
  two-LLM split/merge live-acceptance scenario.
- `fixtures/two_llm_colour_request.txt` — the outcome-only request for that
  scenario (the in-prose upload filename is normalized to `two_llm_colour.csv`
  per Ruling C; accepting the request is therefore no longer byte-verbatim with
  the run sheet).

## Validation scope

`tests/unit/evals/composer_parity/test_fixtures.py` loads every fixture and
validates each `canonical_arguments` payload structurally with
`SetPipelineArgumentsModel.model_validate` **plus** plugin availability against
`PolicyCatalogView.for_trained_operator` and its availability lookups. Full
committed-graph validation (`validate_composition_state`) is the costlier
alternative and is deferred to Task 3's real-path matrix, where the
args -> `CompositionState` conversion inside the session-bound
`_execute_set_pipeline` handler is exercised end to end.

## Real-path assumptions (Task 3)

The Task 3 matrix (`tests/integration/web/composer/parity/`) drives these
fixtures through the real freeform + guided-full production paths (web plugin
policy, operator-profile lowering, audited `set_pipeline`), which imposes two
authoring-form requirements beyond the Task 2 argument shape:

- **Imperative intents.** Every `intent` is phrased as a build request so the
  freeform empty-pipeline gate (`_user_request_expects_pipeline_mutation`)
  routes it through `plan_pipeline` rather than the ordinary compose loop.
- **`structured_llm` uses the web operator-profile form.** Its `llm` node is
  authored with `profile: "task-role"` plus public safe options (`queries`,
  `prompt_template`, `schema`, `required_input_fields`, `temperature`) — the
  private `provider` / `model` / `api_key` / retry knobs are supplied by the
  operator profile at lowering. The real-path harness (and any live run, Task 5/6)
  must configure an LLM profile aliased `task-role`; the multi-query retry budget
  is injected by profile lowering (`_LLMProfileResolver.lower_options`). The
  source declares a fixed schema so the field contract to the LLM's declared
  `required_input_fields` is satisfiable at config time.

## Byte canonicalization (Ruling C)

Both colour files are pinned to **LF newlines, UTF-8 with no BOM, and exactly
one trailing newline**. The SHA-256 values below are computed against that
pinned byte form so the hash gate is not a platform-dependent flake. The hashes
protect test-input integrity, not a plan version. Regenerate them only if the
pinned byte content of the file itself changes.

| File | SHA-256 |
| --- | --- |
| `two_llm_colour.csv` | `067f0ffeb6a349fc33c1ce2f65cac65dcb37eb3bdd30ef8ca2670439238ba702` |
| `two_llm_colour_request.txt` | `11167a51a653a5851c82c85f4b73011c29cd375d91924feeae12e1a349191785` |

Verify locally:

```bash
sha256sum evals/composer-parity/fixtures/two_llm_colour.csv \
          evals/composer-parity/fixtures/two_llm_colour_request.txt
```

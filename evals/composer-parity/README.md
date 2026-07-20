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

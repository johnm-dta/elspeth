# Two-LLM Colour Hybrid Pipeline Design

## Purpose

Demonstrate that ELSPETH's composer can turn an outcome-oriented request into a pipeline that sends every source row through two independent LLM branches, waits for both results, and produces one flat hybrid row. The output must already expose typed numeric fields suitable for a later statistical-aggregation demonstration.

The authoring request may explicitly ask for two LLMs, JSON output, and removal of unwanted fields. It must not prescribe composer tool calls or tell the composer how to configure individual plugins.

## Input

Upload a CSV with these required fields:

- `color_name`: human-readable colour name.
- `hex`: six-digit hexadecimal RGB value including the leading `#`.

Use ten rows that cover obvious and ambiguous mixtures:

| color_name | hex |
| --- | --- |
| Pure Red | `#FF0000` |
| Pure Blue | `#0000FF` |
| Purple | `#800080` |
| Magenta | `#FF00FF` |
| Cyan | `#00FFFF` |
| Navy | `#000080` |
| Orange | `#FF7F00` |
| Teal | `#008080` |
| Grey | `#808080` |
| White | `#FFFFFF` |

## Topology

```text
10-colour CSV
      |
      v
fork every source row
   +--+----------------+
   |                   |
   v                   v
blue assessment LLM   red assessment LLM
   |                   |
   +---------+---------+
             v
 require-all union merge
             |
             v
retain only approved hybrid fields
             |
             v
      successful JSON output

Any branch failure ----------------> failure JSON output
```

The fork must create two auditable child paths for each source row. The coalesce point must require both named branches and use union semantics so the branch-exclusive blue and red fields appear together in one flat row. Shared source fields have identical values on both branches; collision handling must preserve one identical copy while retaining audit lineage.

This design deliberately does not use a single multi-query LLM node. The experiment is intended to prove independently routed LLM branches and a real split/coalesce lifecycle.

## LLM Assessments

Both LLMs receive `color_name` and `hex` as required inputs. Use deterministic sampling where supported and a small, economical model available through the configured provider.

### Blue assessment

Prompt intent:

> Act as a colour-composition assessor. Given the colour name and hex value, estimate the perceptible amount of blue in the colour. Return only a JSON object with `amount`, `confidence`, and `reason`. `amount` must be an integer from 0 to 100, where 0 means no perceptible blue, 50 means blue is a substantial but balanced component, and 100 means the colour is overwhelmingly blue. `confidence` must be a number from 0 to 1. `reason` must be one concise sentence grounded in the supplied colour.

The branch must extract typed output fields:

- `blue_amount`: integer.
- `blue_confidence`: number.
- `blue_reason`: string.

### Red assessment

Prompt intent:

> Act as a colour-composition assessor. Given the colour name and hex value, estimate the perceptible amount of red in the colour. Return only a JSON object with `amount`, `confidence`, and `reason`. `amount` must be an integer from 0 to 100, where 0 means no perceptible red, 50 means red is a substantial but balanced component, and 100 means the colour is overwhelmingly red. `confidence` must be a number from 0 to 1. `reason` must be one concise sentence grounded in the supplied colour.

The branch must extract typed output fields:

- `red_amount`: integer.
- `red_confidence`: number.
- `red_reason`: string.

Invalid JSON, missing declared fields, or wrong field types are branch failures rather than silently accepted text.

## Hybrid Output Contract

After both branches merge, retain exactly:

- `color_name`
- `hex`
- `blue_amount`
- `blue_confidence`
- `blue_reason`
- `red_amount`
- `red_confidence`
- `red_reason`

Remove raw LLM response text, token-usage fields, model metadata, branch bookkeeping, and other intermediate fields from the successful business output. Write successful rows as one JSON array of objects. Route failures to a separate JSON output with enough framework-generated evidence to identify the failed branch without copying secrets into the business output.

## Acceptance Criteria

1. The composer receives an outcome-oriented request that names two LLMs but contains no composer tool-call instructions.
2. The generated graph contains one source, a two-way fork, two independently configured LLM branches, a require-all coalesce point, a deterministic field-cleanup step, one successful JSON sink, and one failure JSON sink.
3. Validation confirms that both LLMs require `color_name` and `hex`, branch-exclusive typed fields survive the merge, and the final JSON sink accepts the hybrid schema.
4. Execution reads exactly ten source rows and writes exactly ten successful hybrid rows when both branches succeed.
5. Every successful row has all eight approved fields, integer amounts in the inclusive range 0-100, numeric confidences in the inclusive range 0-1, and non-empty reasons.
6. The run records twenty LLM assessments, ten completed coalesces, zero pending tokens, and closed audit accounting.
7. The downloaded JSON output parses successfully and contains no raw-response, usage, or model-metadata fields.

## Follow-on Demonstration

The next iteration may add aggregators over `blue_amount`, `red_amount`, `blue_confidence`, and `red_confidence` to calculate descriptive statistics and compare colour groups. That statistical stage is intentionally out of scope here; this run establishes that the split LLM outputs can be merged into typed, aggregation-ready rows first.

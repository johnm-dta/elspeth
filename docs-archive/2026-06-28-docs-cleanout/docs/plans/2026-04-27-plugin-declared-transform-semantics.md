# Plugin-Declared Transform Semantics and Assistance Specification

> For implementors: load the Elspeth core standards skills before executing this plan. In particular, this work crosses plugin contracts, tier boundaries, composer validation, MCP surfaces, and user-facing diagnostics.

**Status:** Proposed design specification

**Date:** 2026-04-27

**Goal:** Replace the current hardcoded transform-framing validation with a general, plugin-owned semantic contract system. The generic composer, validation, execution, and MCP layers should compare structured producer facts with structured consumer requirements. They must not learn one-off plugin pairings such as `web_scrape` followed by `line_explode`.

**Non-goals:**

- Do not "fix" a particular Wardline YAML pipeline in this work.
- Do not add new hardcoded plugin-pair exceptions to the composer, orchestrator, or execution services.
- Do not make enforcement depend on plugin docstrings, composer skill prose, or LLM-parsed guidance.
- Do not require every existing plugin to declare semantics before this can ship.

---

## Current Repo Reality

The immediate regression comes from a real semantic mismatch that the current schema contract cannot express:

1. `web_scrape` owns the configured output field and content formatting.
   - `WebScrapeConfig` declares `content_field`, `format`, and `text_separator`; `format` is `markdown`, `text`, or `raw`, and `text_separator` defaults to a single space ([web_scrape.py](/home/john/elspeth/src/elspeth/plugins/transforms/web_scrape.py:118)).
   - The transform stores the configured content field, format, and separator on the instance ([web_scrape.py](/home/john/elspeth/src/elspeth/plugins/transforms/web_scrape.py:291)).
   - Runtime extraction passes `format=self._format` and `text_separator=self._text_separator` into `extract_content(...)`, then emits the result under `self._content_field` ([web_scrape.py](/home/john/elspeth/src/elspeth/plugins/transforms/web_scrape.py:479), [web_scrape.py](/home/john/elspeth/src/elspeth/plugins/transforms/web_scrape.py:517)).
   - It already builds a configured output schema contract for the content/fingerprint fields ([web_scrape.py](/home/john/elspeth/src/elspeth/plugins/transforms/web_scrape.py:162), [web_scrape.py](/home/john/elspeth/src/elspeth/plugins/transforms/web_scrape.py:346)).

2. `line_explode` owns the input interpretation.
   - The module docstring defines the transform as line-oriented deaggregation ([line_explode.py](/home/john/elspeth/src/elspeth/plugins/transforms/line_explode.py:1)).
   - `LineExplodeConfig` declares `source_field`, and the instance stores it as `self._source_field` ([line_explode.py](/home/john/elspeth/src/elspeth/plugins/transforms/line_explode.py:33), [line_explode.py](/home/john/elspeth/src/elspeth/plugins/transforms/line_explode.py:147)).
   - Runtime calls `source_value.splitlines()`; compact text containing no newline is valid input but emits one giant row ([line_explode.py](/home/john/elspeth/src/elspeth/plugins/transforms/line_explode.py:174)).

3. The current guard is intentionally too specific for long-term use.
   - `validate_transform_framing_contracts(...)` in composer state builds a local producer map, looks only for `line_explode`, walks only to `web_scrape`, compares `content_field` with `source_field`, and inspects `format` plus `text_separator` ([state.py](/home/john/elspeth/src/elspeth/web/composer/state.py:351)).
   - `/validate` calls that hardcoded guard before YAML generation and uses a hardcoded suggestion ([validation.py](/home/john/elspeth/src/elspeth/web/execution/validation.py:274)).
   - `/execute` independently calls the same hardcoded guard before run creation ([service.py](/home/john/elspeth/src/elspeth/web/execution/service.py:320)).

4. There is already a generic schema-contract preview seam to reuse.
   - `ValidationSummary` already carries structured `edge_contracts` for producer-guarantee versus consumer-required field checks ([state.py](/home/john/elspeth/src/elspeth/web/composer/state.py:281)).
   - `_check_schema_contracts(...)` already instantiates transforms to obtain computed `_output_schema_config`, preserves pass-through propagation semantics, handles expected draft/config probe failures, and scrubs exception details before surfacing warnings ([state.py](/home/john/elspeth/src/elspeth/web/composer/state.py:809), [state.py](/home/john/elspeth/src/elspeth/web/composer/state.py:838)).
   - `ExecutionGraph.validate_edge_compatibility()` already runs schema compatibility after plugin instantiation in `/validate` ([graph.py](/home/john/elspeth/src/elspeth/core/dag/graph.py:1046)).

5. Catalog and MCP responses are typed surfaces, not free-form bags.
   - Catalog response models are strict Pydantic responses with `extra="forbid"` ([schemas.py](/home/john/elspeth/src/elspeth/web/catalog/schemas.py:19)).
   - `CatalogServiceImpl.get_schema()` delegates plugin schema emission to plugin classes ([service.py](/home/john/elspeth/src/elspeth/web/catalog/service.py:61)).
   - MCP validation payloads already include serialized `edge_contracts` ([server.py](/home/john/elspeth/src/elspeth/composer_mcp/server.py:64), [server.py](/home/john/elspeth/src/elspeth/composer_mcp/server.py:282)).

6. Existing prose guidance is useful but must stop being the enforcement source.
   - The pipeline composer skill and web scrape reference already warn that compact `format: text` output needs `text_separator: "\n"` before `line_explode`.
   - That prose should become assistance exposed by plugin declarations, not logic embedded in generic validators or prompts.

---

## Design Decision

Add a plugin-owned semantic contract layer with three distinct surfaces:

1. **Output semantics:** structured, configured facts about fields a plugin emits.
2. **Input semantic requirements:** structured, configured requirements for fields a plugin consumes.
3. **Agent/user assistance:** deterministic, side-effect-free guidance keyed to plugin names and semantic issue codes.

The validator should be generic:

- It asks producers for field facts.
- It asks consumers for field requirements.
- It resolves configured field names through the actual producer/consumer graph.
- It compares machine-readable facts and requirements.
- It emits structured validation entries plus structured semantic edge contracts.

The validator must not parse assistance prose and must not contain `web_scrape -> line_explode` knowledge.

---

## Proposed Contract Types

Create a low-level contract module, for example `src/elspeth/contracts/plugin_semantics.py`. Keep it independent of plugin infrastructure so it can be imported by web, DAG, catalog, tests, and plugin implementations without cycles.

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ContentKind(StrEnum):
    UNKNOWN = "unknown"
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML_RAW = "html_raw"
    JSON_STRUCTURED = "json_structured"
    BINARY = "binary"


class TextFraming(StrEnum):
    UNKNOWN = "unknown"
    NOT_TEXT = "not_text"
    COMPACT = "compact"
    NEWLINE_FRAMED = "newline_framed"
    LINE_COMPATIBLE = "line_compatible"


class UnknownSemanticPolicy(StrEnum):
    ALLOW = "allow"
    WARN = "warn"
    FAIL = "fail"


class SemanticOutcome(StrEnum):
    SATISFIED = "satisfied"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FieldSemanticFacts:
    field_name: str
    content_kind: ContentKind
    text_framing: TextFraming = TextFraming.UNKNOWN
    fact_code: str = "field_semantics"
    configured_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OutputSemanticDeclaration:
    fields: tuple[FieldSemanticFacts, ...] = ()


@dataclass(frozen=True, slots=True)
class FieldSemanticRequirement:
    field_name: str
    accepted_content_kinds: frozenset[ContentKind]
    accepted_text_framings: frozenset[TextFraming]
    requirement_code: str
    severity: str = "high"
    unknown_policy: UnknownSemanticPolicy = UnknownSemanticPolicy.WARN
    configured_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InputSemanticRequirements:
    fields: tuple[FieldSemanticRequirement, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticEdgeContract:
    from_id: str
    to_id: str
    producer_field: str
    consumer_field: str
    producer_facts: FieldSemanticFacts | None
    requirement: FieldSemanticRequirement
    outcome: SemanticOutcome
```

Rules:

- These enums are a closed vocabulary at first. Additions require design review so the project does not rebuild ad hoc runtime validation as expanding prose.
- `UNKNOWN` is an explicit value, not a fabricated guarantee.
- `configured_by` names option paths that influenced the fact or requirement. It must not store secret values, URLs, headers, prompts, row values, or raw exception text.
- `FieldSemanticRequirement` is also a field dependency. A consumer that semantically requires `source_field="content"` requires that field to be available; implementors should not force users to duplicate that truth in `required_input_fields`.

---

## Plugin API Shape

Add defaults on the plugin base classes and protocols. The first implementation can scope enforcement to transforms, but the type surface should be source/transform/sink capable so future producers and consumers do not need a second design.

```python
class BaseTransform:
    def output_semantics(self) -> OutputSemanticDeclaration:
        return OutputSemanticDeclaration()

    def input_semantic_requirements(self) -> InputSemanticRequirements:
        return InputSemanticRequirements()

    @classmethod
    def get_agent_assistance(
        cls,
        *,
        issue_code: str | None = None,
    ) -> PluginAssistance:
        return PluginAssistance()
```

Recommended placement:

- The public methods live on `BaseTransform`, and later mirror onto `BaseSource` and `BaseSink` as needed.
- Plugin-specific helper functions live next to the plugin config/output-schema helper. For example:
  - `web_scrape`: `_build_web_scrape_output_semantics(cfg)` beside `_build_web_scrape_output_schema_config(...)`.
  - `line_explode`: `_build_line_explode_input_requirements(cfg)` beside `_build_line_explode_output_schema_config(...)`.
- Configured truth should be instance-level. Generic validators should instantiate the plugin through the same `PluginManager.create_transform(...)` path already used by composer preview and runtime validation, then call the semantic methods on the validated instance.
- Static catalog/discovery guidance may be class-level, but configured enforcement must come from validated config, not from raw option dict parsing in generic layers.

Method constraints:

- Pure and deterministic: no network, file I/O, secret lookup, environment reads, random values, time, or plugin lifecycle hooks.
- Secret-safe: may return enum values, field names, plugin names, and issue codes; must not return raw config values unless the plugin can prove the option is non-sensitive and useful.
- No `hasattr` probing. Base classes supply the methods, and absence is a framework bug.

---

## Concrete Declarations for the Wardline Regression

### `web_scrape` output semantics

`web_scrape` should declare facts for its configured `content_field`.

Recommended mapping:

| Config | `content_kind` | `text_framing` | Rationale |
| --- | --- | --- | --- |
| `format: text`, `"\n" not in text_separator` | `PLAIN_TEXT` | `COMPACT` | DOM text nodes are joined into one compact string by the configured separator. |
| `format: text`, `"\n" in text_separator` | `PLAIN_TEXT` | `NEWLINE_FRAMED` | Downstream `splitlines()` can recover line/segment boundaries. |
| `format: markdown` | `MARKDOWN` | `LINE_COMPATIBLE` | Markdown extraction preserves line-oriented structure enough for line deaggregation. |
| `format: raw` | `HTML_RAW` | `NOT_TEXT` | Raw HTML is content, but not a line-framed text payload. |

The fact should use the configured field:

```python
FieldSemanticFacts(
    field_name=cfg.content_field,
    content_kind=ContentKind.PLAIN_TEXT,
    text_framing=TextFraming.COMPACT,
    fact_code="web_scrape.content.compact_text",
    configured_by=("format", "text_separator"),
)
```

`web_scrape` may also declare semantics for `fingerprint_field` and fetch metadata later, but they are not needed for this regression.

### `line_explode` input requirements

`line_explode` should declare a requirement for its configured `source_field`.

```python
FieldSemanticRequirement(
    field_name=cfg.source_field,
    accepted_content_kinds=frozenset(
        {ContentKind.PLAIN_TEXT, ContentKind.MARKDOWN},
    ),
    accepted_text_framings=frozenset(
        {TextFraming.NEWLINE_FRAMED, TextFraming.LINE_COMPATIBLE},
    ),
    requirement_code="line_explode.source_field.line_framed_text",
    severity="high",
    unknown_policy=UnknownSemanticPolicy.WARN,
    configured_by=("source_field",),
)
```

`WARN` for unknown producers is the recommended initial compatibility policy. It preserves existing pipelines whose upstream producers have not yet declared semantics, while still failing declared incompatible producers such as compact `web_scrape` text.

---

## Field Resolution

The resolver should use configured plugin field names after plugin config validation:

- Producer field names come from producer semantic facts, for example `web_scrape.content_field`.
- Consumer field names come from consumer semantic requirements, for example `line_explode.source_field`.
- A requirement applies when the downstream consumer's configured field name matches a semantic fact from the effective upstream producer for that same row field.

Resolution rules:

1. Build or reuse the same producer map the composer already uses for schema contracts.
2. Walk through structural gates the same way `_walk_to_real_producer(...)` does today.
3. Preserve pass-through semantics across transforms that declare `passes_through_input=True` only when their semantic API explicitly preserves or transforms the field facts.
4. For the first rollout, shape-changing transforms that rename, drop, or synthesize fields should not inherit upstream semantic facts by default. They either declare their own output semantics or the downstream sees unknown semantics.
5. Coalesce and fork semantics should be conservative:
   - If all contributing branches provide the same compatible fact for the field, preserve it.
   - If branches disagree, report `UNKNOWN` or a warning rather than choosing a branch.
   - Do not fabricate compatibility across branch-exclusive fields.
6. If the field cannot be proven present by schema guarantees or semantic producer facts, report the normal missing-field schema contract when available. If the producer is dynamic/unknown, report the semantic requirement as unknown according to `unknown_policy`.

This keeps the Wardline case simple: `web_scrape` declares facts for the configured `content_field`, `line_explode` declares a requirement for the configured `source_field`, and the generic edge validator compares those declarations when the field names match.

---

## Generic Validator Algorithm

Add a shared semantic validation helper, for example `validate_semantic_contracts(...)`, that returns both `ValidationEntry` values and `SemanticEdgeContract` records.

High-level algorithm:

1. Normalize topology into producer and consumer records.
   - The composer path can build this from `CompositionState`.
   - The engine path can build this from `ExecutionGraph` / `NodeInfo` once semantic declarations are carried into graph construction.
2. Instantiate configured plugins through the real plugin manager.
   - Expected draft config errors should be reported through the existing invalid-options path and should not leak raw exception text.
   - Unexpected exceptions should propagate as framework bugs.
3. For each consumer node, read `input_semantic_requirements()`.
4. For each requirement:
   - Resolve the effective upstream producer for the consumer input.
   - Resolve the producer's `output_semantics()` for the required field.
   - If producer facts are absent or `UNKNOWN`, emit a semantic edge contract with `outcome=UNKNOWN`.
   - Apply the requirement's `unknown_policy`.
   - If producer facts conflict with accepted kinds/framing, emit `outcome=CONFLICT` and a high validation error.
   - If they satisfy the requirement, emit `outcome=SATISFIED`.
5. Deduplicate repeated direct routes to the same consumer the same way schema edge contracts already do.
6. Produce structured diagnostics. A conflict message should name:
   - producer id and plugin
   - consumer id and plugin
   - field name
   - declared producer facts
   - required consumer semantics
   - requirement code

Example conflict wording:

```text
Semantic contract violation: 'scrape' -> 'explode'.
Consumer (line_explode) requires field 'content' to be newline-framed or line-compatible text.
Producer (web_scrape) declares field 'content' as plain_text/compact.
```

The generic validator should not mention `text_separator` or `format: markdown` unless that text is supplied by plugin assistance for the `line_explode.source_field.line_framed_text` or `web_scrape.content.compact_text` issue code.

---

## Integration Points

### Composer preview and mutation validation

Extend `ValidationSummary` with:

```python
semantic_contracts: tuple[SemanticEdgeContract, ...] = ()
```

Then wire semantic validation into `CompositionState.validate()` beside `_check_schema_contracts(...)`.

`preview_pipeline` should include `semantic_contracts` alongside `edge_contracts`, just as it now serializes schema edge contracts ([tools.py](/home/john/elspeth/src/elspeth/web/composer/tools.py:3326)).

`ToolResult.to_dict()` should include `semantic_contracts` in its validation payload, not only in `preview_pipeline`, so mutation tools and agent workflows see the same evidence.

### `/validate`

Replace `_CHECK_TRANSFORM_FRAMING` with a generic semantic check name, for example:

```python
_CHECK_SEMANTIC_CONTRACTS = "semantic_contracts"
```

The check should run at the same early point where the current hardcoded guard runs, before YAML generation, so invalid pipelines do not proceed to settings/plugin/graph construction when the state already carries a known semantic conflict.

The check's suggestions should come from plugin assistance keyed by requirement/fact code, not from a hardcoded web scrape string.

### `/execute`

Replace the direct `validate_transform_framing_contracts(...)` call with the same shared semantic validation helper used by `/validate` and composer preview.

`/execute` must keep this pre-run check because users can execute without calling `/validate` first. Semantic conflicts should still fail before `create_run(...)`, preserving the current safety property.

### Runtime graph validation

Recommended implementation order:

1. Ship the shared `CompositionState` semantic validator first because composer preview, `/validate`, `/execute`, and MCP all start from composition state today.
2. Carry semantic declarations into `NodeInfo` during graph construction once the web paths are stable.
3. Add `ExecutionGraph.validate_semantic_compatibility()` and call it after `validate_edge_compatibility()` in the dry-run validation path.

The graph-level check is important for future non-web YAML/CLI parity, but it does not need to block the first Wardline regression fix if all web execution entrypoints use the shared pre-run helper.

### MCP orchestration

Update the MCP validation payload:

- Add a typed `_SemanticEdgeContractPayload`.
- Add `semantic_contracts` to `_ValidationPayload`.
- Serialize them in `_validation_to_dict(...)`.

MCP `generate_yaml` already refuses invalid `state.validate()` results, so it will inherit semantic failures once `CompositionState.validate()` is wired.

### Catalog and composer tools

Extend strict catalog models deliberately:

- `PluginSummary` can expose a small static `semantic_capabilities` summary if useful.
- `PluginSchemaInfo` should expose structured `assistance` and any static declaration metadata.
- Prefer adding a focused discovery tool such as `get_plugin_assistance` or `get_plugin_contracts` rather than bloating every `list_transforms` response with examples.

The existing `get_plugin_schema` path is a good fit for static metadata because catalog service already delegates schema emission to plugin classes. Configured semantics should still come from instantiated plugins during validation.

---

## Agent and User Guidance

Add a strict assistance model, for example in the same contract module or a web/catalog schema wrapper:

```python
@dataclass(frozen=True, slots=True)
class PluginAssistanceExample:
    title: str
    before: dict[str, object] | None = None
    after: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class PluginAssistance:
    plugin_name: str
    issue_code: str | None
    summary: str
    suggested_fixes: tuple[str, ...] = ()
    examples: tuple[PluginAssistanceExample, ...] = ()
    composer_hints: tuple[str, ...] = ()
```

Guidance rules:

- Guidance is deterministic and side-effect free.
- Guidance may explain why compact text does not satisfy line deaggregation.
- Guidance may suggest `web_scrape` `text_separator: "\n"` or `format: markdown`.
- Guidance must not expose secrets, raw URLs, request headers, row values, credentials, raw provider errors, file paths, or exception strings.
- Validators may attach `issue_code` and `assistance_ref`; they must not parse `summary`, `suggested_fixes`, or examples.

The composer LLM should receive assistance through tools/catalog responses. The skill prose can then say "inspect plugin contracts/assistance for configured producers and consumers" instead of embedding the exact `web_scrape` gotcha as permanent prompt text.

---

## Backward Compatibility and Migration

Default behavior:

- Plugins with no output declarations produce unknown semantics.
- Plugins with no input semantic requirements impose no semantic checks.
- Unknown semantics are never treated as satisfying a requirement.
- Requirements decide how unknown is surfaced with `unknown_policy`.

Recommended first-rollout policy:

- `line_explode` uses `unknown_policy=WARN`.
- Declared incompatible facts fail validation.
- Unknown producers produce a medium warning and a semantic edge record with `outcome=UNKNOWN`.
- Existing pipelines backed by undeclared producers continue to run, but users and agents get a visible prompt to improve or choose a known line-framed producer.

Migration steps:

1. Add contract types, base-class defaults, serialization, and empty validation wiring. No behavior change.
2. Add generic semantic validator and payloads. With no declarations, behavior is still unchanged except optional empty `semantic_contracts` arrays.
3. Implement `web_scrape` output facts and assistance.
4. Implement `line_explode` input requirements and assistance.
5. Delete `validate_transform_framing_contracts(...)` and update imports/tests.
6. Replace hardcoded composer skill/reference gotchas with tool-discoverable assistance references.
7. Expand declarations to other text producers/consumers.
8. Revisit `unknown_policy` after coverage improves; consider moving selected consumers from `WARN` to `FAIL` when plugin coverage makes that safe.

---

## Test Strategy

### Contract model tests

Add tests for:

- Frozen dataclasses and strict enum values.
- `UNKNOWN` not satisfying any explicit requirement.
- Conflict, satisfied, and unknown comparison outcomes.
- Serialization shape for web, MCP, and catalog payloads.

### Plugin declaration tests

`tests/unit/plugins/transforms/test_web_scrape.py`:

- `format: text` with default separator declares `PLAIN_TEXT` plus `COMPACT`.
- `format: text` with `text_separator: "\n"` declares `PLAIN_TEXT` plus `NEWLINE_FRAMED`.
- `format: markdown` declares `MARKDOWN` plus `LINE_COMPATIBLE`.
- `format: raw` declares `HTML_RAW` plus `NOT_TEXT`.
- Custom `content_field` changes the semantic field name.

`tests/unit/plugins/transforms/test_line_explode.py`:

- Configured `source_field` produces a semantic requirement for that exact field.
- The requirement accepts `NEWLINE_FRAMED` and `LINE_COMPATIBLE`.
- The requirement rejects `COMPACT` when producer facts are declared.
- The assistance for the requirement is deterministic and contains no config values beyond safe option names.

### Composer state validation tests

Replace the current hardcoded framing tests in `tests/unit/web/composer/test_state.py` with generic semantic contract tests:

- `web_scrape format: text` with default separator followed by `line_explode` fails validation.
- `web_scrape format: text` with `text_separator: "\n"` followed by `line_explode` passes.
- `web_scrape format: markdown` followed by `line_explode` passes.
- A producer with unknown semantics feeding `line_explode` emits the chosen warning/unknown edge outcome, not a hard error in phase one.
- A producer field whose name does not match `line_explode.source_field` does not trigger a false semantic conflict; normal field-contract validation handles missing fields.
- A gate between producer and consumer preserves the same semantic outcome.
- Conflicting coalesce branch facts produce unknown/warning rather than fabricated satisfaction.

### `/validate` tests

Update `tests/unit/web/execution/test_validation.py`:

- Compact `web_scrape` text fails before YAML generation.
- The failed check name is `semantic_contracts`.
- The returned error uses structured semantic facts/requirements.
- Newline-framed text reaches YAML generation.
- Markdown reaches YAML generation.
- Unknown producer behavior matches the migration policy.

### `/execute` tests

Update `tests/unit/web/execution/test_service.py`:

- Compact `web_scrape` text fails before `create_run(...)`.
- Newline-framed text continues to run creation.
- The execution path calls the same helper as `/validate`, avoiding a second drift-prone implementation.

### MCP and catalog tests

Add or update tests for:

- MCP validation payload includes `semantic_contracts`.
- MCP `generate_yaml` rejects semantic conflicts through `state.validate()`.
- `get_plugin_schema` or the new assistance tool returns strict, serializable plugin assistance.
- Catalog response models reject extra undeclared assistance fields.

### Suggested targeted verification

```bash
PYTHONPATH=src uv run pytest -q \
  tests/unit/plugins/transforms/test_web_scrape.py \
  tests/unit/plugins/transforms/test_line_explode.py \
  tests/unit/web/composer/test_state.py \
  tests/unit/web/execution/test_validation.py \
  tests/unit/web/execution/test_service.py
```

Then run the MCP/catalog tests touched by the implementation and the repo's standard lint/type gates for the final branch.

---

## Risks and Open Questions

1. **Unknown semantics policy.** `WARN` is the least disruptive phase-one choice. It still allows a one-row surprise from undeclared producers. Moving selected consumers to `FAIL` should wait until common producers declare semantics.

2. **Vocabulary drift.** The enum set must stay small. If every plugin invents new text states, the system becomes prose with extra steps.

3. **Rename/pass-through semantics.** Field-renaming transforms need explicit semantic transformations. The first rollout should preserve facts only through structural gates and explicitly declared pass-through preservation.

4. **Coalesce semantics.** Branch disagreement is easy to mishandle. The safe initial answer is unknown/warning unless all participating branches agree.

5. **Runtime parity.** Web pre-run validation is necessary but not sufficient forever. Carrying semantics into `NodeInfo` and adding graph-level validation is the parity endpoint.

6. **Secret leakage.** Assistance and validation errors must use field names, enum values, and issue codes only. They must not include raw URLs, headers, row data, request bodies, provider messages, or exception text.

7. **Plugin construction cost.** Composer preview already instantiates configured transforms to compute output schema contracts. Semantic validation should share or cache those per-node probes rather than construct the same plugin repeatedly.

8. **Check-name compatibility.** Changing `transform_framing` to `semantic_contracts` is the right model, but any frontend or tests that key on the old name must be updated in the same change.

---

## Recommended Rollout Order

1. Land the model/API skeleton with base-class defaults and empty serialization.
2. Add the generic semantic validator and wire it into `CompositionState.validate()`, `/validate`, `/execute`, and MCP payloads.
3. Add `web_scrape` output semantics and assistance.
4. Add `line_explode` input requirements and assistance.
5. Replace and delete the hardcoded `validate_transform_framing_contracts(...)` path.
6. Update docs and composer skills to point agents at plugin assistance rather than hardcoded pair guidance.
7. Add graph-level semantic validation for non-web parity.
8. Expand declarations to other line/text/HTML/JSON producers and consumers, then reconsider stricter unknown policies.

This gives the Wardline regression a mechanical fix while keeping ownership where it belongs: producers declare what they produce, consumers declare what they require, and the generic layer only compares contracts.

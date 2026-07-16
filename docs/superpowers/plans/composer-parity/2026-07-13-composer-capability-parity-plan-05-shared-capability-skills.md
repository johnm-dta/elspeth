# Composer Capability Parity Plan 05: Shared Capability Skills Implementation Plan

> **RETIRED (2026-07-17): DO NOT EXECUTE.** See
> [the current disposition](2026-07-17-current-plan-disposition.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give freeform, guided-full, guided-staged, and tutorial-profile the same accurate capability knowledge, including typed multi-query LLM configuration and stage-aware guidance.

**Architecture:** Extract topology and plugin capability facts into one shared core loaded by every composer surface. Surface overlays describe interaction policy only. Runtime-owned typed configuration models and plugin assistance remain the authority for exact options; prompt prose explains when and why to use them without duplicating schemas.

**Tech Stack:** Python, Pydantic v2, Markdown prompt packs, pytest.

---

## File structure

**Create:**
- `src/elspeth/web/composer/skills/capability_core.md` — shared topology and plugin-use guidance.
- `src/elspeth/plugins/transforms/llm/query_config.py` — typed authoring models for one or many LLM queries.
- `tests/unit/web/composer/test_capability_skill_surfaces.py`

**Modify:**
- `src/elspeth/web/composer/prompts.py`
- `src/elspeth/web/composer/skills/pipeline_composer.md`
- `src/elspeth/web/composer/guided/prompts.py`
- `src/elspeth/web/composer/guided/skills/base.md`
- `src/elspeth/web/composer/guided/skills/step_1_source.md`
- `src/elspeth/web/composer/guided/skills/step_2_sink.md`
- `src/elspeth/web/composer/guided/skills/step_3_transforms.md`
- `src/elspeth/web/composer/guided/skills/step_4_wire.md`
- `src/elspeth/plugins/transforms/llm/base.py`
- `src/elspeth/plugins/transforms/llm/multi_query.py`
- `src/elspeth/plugins/transforms/llm/transform.py`
- `tests/unit/plugins/llm/test_config_schema.py`
- `tests/unit/plugins/llm/test_multi_query.py`
- `tests/unit/web/composer/guided/test_stage_guidance.py` — extend the Plan 04 behavioral contract with shared-core identity.

### Task 1: Extract one shared capability core

**Files:**
- Create: `src/elspeth/web/composer/skills/capability_core.md`
- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md`
- Modify: `src/elspeth/web/composer/guided/skills/base.md`
- Modify: `src/elspeth/web/composer/guided/skills/step_3_transforms.md`
- Modify: `src/elspeth/web/composer/guided/skills/step_4_wire.md`
- Test: `tests/unit/web/composer/test_capability_skill_surfaces.py`

- [ ] **Step 1: Write failing shared-core coverage tests**

Assert the core documents all canonical topology constructs without embedding a
second JSON schema: plural named sources/outputs, transform and aggregation
nodes, queue nodes, gates, forks, `fork_to`, coalesces, merge strategies,
error routing, required/guaranteed fields, row expansion, and structured LLM
queries.

Assert the obsolete guided claims that gates cannot be expressed and wiring is
a single linear spine are absent from every rendered prompt.

- [ ] **Step 2: Extract factual capability guidance**

Move only capability facts into `capability_core.md`. Keep exact field names
where they teach relationships, but direct the model to the effective
`set_pipeline` schema and catalog/plugin assistance for option details.

- [ ] **Step 3: Reduce existing packs to surface overlays**

`pipeline_composer.md` retains freeform interaction/approval policy. Guided
`base.md` and step packs retain stage dialogue and review policy. None may
claim a reduced topology language.

- [ ] **Step 4: Run tests and commit**

```bash
uv run pytest tests/unit/web/composer/test_capability_skill_surfaces.py -q
git add src/elspeth/web/composer/skills/capability_core.md src/elspeth/web/composer/skills/pipeline_composer.md src/elspeth/web/composer/guided/skills/base.md src/elspeth/web/composer/guided/skills/step_3_transforms.md src/elspeth/web/composer/guided/skills/step_4_wire.md tests/unit/web/composer/test_capability_skill_surfaces.py
git commit -m "docs(composer): share canonical capability guidance"
```

### Task 2: Render the same core into every effective prompt

**Files:**
- Modify: `src/elspeth/web/composer/prompts.py`
- Modify: `src/elspeth/web/composer/guided/prompts.py`
- Test: `tests/unit/web/composer/test_capability_skill_surfaces.py`

- [ ] **Step 1: Test actual rendered prompts, not source-file membership**

Capture the effective system prompt for freeform, guided-full, every guided
stage, and tutorial-profile. Assert each contains the exact same shared-core
digest and no duplicate core section.

- [ ] **Step 2: Add one renderer and manifest**

```python
@dataclass(frozen=True, slots=True)
class EffectiveSkillManifest:
    surface: ComposerSurface
    profile: str | None
    capability_core_sha256: str
    rendered_prompt_sha256: str
    planner_implementation_id: str
    terminal_schema_sha256: str
    canonical_schema_sha256: str
    discovery_tools_sha256: str
    plugin_assistance_sha256: str


def render_capability_core() -> str: ...
```

Have `build_system_prompt()`, `load_guided_skill()`, and
`load_step_chat_skill()` call this renderer. Change `guided_staged_skill_hash()`
to hash the effective rendered content used by the call, rather than a list of
source files that may not match the request.

- [ ] **Step 3: Prove tutorial is a profile, not a capability surface**

Assert tutorial and guided-staged report identical core, planner, canonical
schema, and effective tool-schema ids. Their rendered-prompt hashes may differ
only because the tutorial overlay contains lesson instructions.

Wrap the actual completion seam and hash the exact messages, terminal tool,
discovery tool names/schemas, model list, and full installed-catalog assistance
passed to freeform, guided-full, guided-staged topology, and tutorial calls.
Compare those request-bound hashes with the manifest and audit record. Add a
negative control removing one plugin schema/assistance entry from one surface;
the identity test must name that plugin and fail.

- [ ] **Step 4: Run tests and commit**

```bash
uv run pytest tests/unit/web/composer/test_capability_skill_surfaces.py tests/unit/web/composer/guided/test_prompts.py -q
git add src/elspeth/web/composer/prompts.py src/elspeth/web/composer/guided/prompts.py tests/unit/web/composer/test_capability_skill_surfaces.py tests/unit/web/composer/guided/test_prompts.py
git commit -m "refactor(composer): render one capability core on every surface"
```

### Task 3: Add typed single- and multi-query LLM authoring models

**Files:**
- Create: `src/elspeth/plugins/transforms/llm/query_config.py`
- Modify: `src/elspeth/plugins/transforms/llm/base.py`
- Modify: `src/elspeth/plugins/transforms/llm/multi_query.py`
- Test: `tests/unit/plugins/llm/test_config_schema.py`
- Test: `tests/unit/plugins/llm/test_multi_query.py`

- [ ] **Step 1: Pin accepted mapping and ordered-list forms**

Add tests for the existing single-query mapping and named multi-query list,
including structured `output_fields`, unique names, suffix behavior, response
format, and rejection of extra fields. Preserve current runtime input forms.

- [ ] **Step 2: Define runtime-owned authoring models**

```python
class QueryDefinition(PluginConfig):
    input_fields: dict[str, str]
    response_format: ResponseFormat = ResponseFormat.STANDARD
    output_fields: tuple[OutputFieldConfig, ...] | None = None
    template: str | None = None
    max_tokens: int | None = Field(default=None, gt=0)


class NamedQueryDefinition(QueryDefinition):
    name: str
```

Type `LLMConfig.queries` as the exact union of the accepted mapping form and
ordered named-query form. Convert validated models to existing
`QuerySpec` runtime values in one adapter; do not duplicate parsing logic.

- [ ] **Step 3: Verify emitted JSON schema is actionable**

Assert `get_plugin_schema(transform, llm)` and catalog discovery expose the two
accepted shapes, structured output-field contract, and required names. The
generic canonical `set_pipeline.nodes[*].options` remains plugin-agnostic; test
separately that it accepts an options payload already validated by the LLM
plugin schema. Verify existing YAML configuration round-trips unchanged.

- [ ] **Step 4: Run tests and commit**

```bash
uv run pytest tests/unit/plugins/llm/test_config_schema.py tests/unit/plugins/llm/test_multi_query.py tests/property/plugins/llm/test_multi_query_properties.py -q
git add src/elspeth/plugins/transforms/llm/query_config.py src/elspeth/plugins/transforms/llm/base.py src/elspeth/plugins/transforms/llm/multi_query.py tests/unit/plugins/llm/test_config_schema.py tests/unit/plugins/llm/test_multi_query.py
git commit -m "feat(llm): type multi-query composer configuration"
```

### Task 4: Teach plugin assistance the complete LLM contract

**Files:**
- Modify: `src/elspeth/plugins/transforms/llm/transform.py`
- Test: `tests/unit/plugins/llm/test_config_schema.py`
- Test: `tests/unit/web/composer/test_capability_skill_surfaces.py`

- [ ] **Step 1: Write assistance tests for two independent structured queries**

Assert `get_agent_assistance()` describes named multiple queries, JSON output,
field typing, output suffixes, and secret references without containing a
literal credential or prescribing composer tool-call sequences.

- [ ] **Step 2: Replace the single-query-only example**

Provide one compact mapping example and one two-query structured example. Keep
the assistance advisory and point exact validation to the typed schema.

- [ ] **Step 3: Verify all surfaces receive assistance through discovery**

Run catalog discovery for each surface and assert the same assistance digest is
present in planner context.

- [ ] **Step 4: Run tests and commit**

```bash
uv run pytest tests/unit/plugins/llm/test_config_schema.py tests/unit/web/composer/test_capability_skill_surfaces.py tests/integration/web/test_catalog_discovery.py -q
git add src/elspeth/plugins/transforms/llm/transform.py tests/unit/plugins/llm/test_config_schema.py tests/unit/web/composer/test_capability_skill_surfaces.py
git commit -m "docs(llm): expose multi-query composer assistance"
```

### Task 5: Fold stage timing into the shared capability matrix

**Files:**
- Modify: `src/elspeth/web/composer/guided/skills/step_1_source.md`
- Modify: `src/elspeth/web/composer/guided/skills/step_2_sink.md`
- Modify: `src/elspeth/web/composer/guided/skills/step_3_transforms.md`
- Modify: `src/elspeth/web/composer/guided/skills/step_4_wire.md`
- Test: `tests/unit/web/composer/guided/test_stage_guidance.py`

- [ ] **Step 1: Write wrong-stage guidance contract tests**

For each source, sink, transform, aggregation, queue, and topology request,
assert the responsible stage is correct. Before that stage, the prompt must
require the solver to identify timing, preserve the request, name the target
stage, and continue the current review. After that stage, it must use stable-id
back/edit rather than creating an orphaned deferral.

- [ ] **Step 2: Add the stage-responsibility matrix**

Consolidate the behavior guidance landed in Plan 04 into the shared matrix and
remove any duplicated local plugin-name tables. Document only sequencing responsibility. Explicitly prohibit configuring a
plugin in the wrong component, claiming it is unsupported, discarding it,
silently advancing, or treating deferral as a repair.

- [ ] **Step 3: Assert prompts match the catalog-backed implementation**

The prose may suggest “hang on, we need to wait,” but the target stage must be
derived by Plan 04's catalog lookup. Test that no hard-coded plugin-name table
exists in the skill packs.

- [ ] **Step 4: Run plan gate and commit**

```bash
uv run pytest tests/unit/web/composer/test_capability_skill_surfaces.py tests/unit/web/composer/guided/test_stage_guidance.py tests/unit/plugins/llm/test_config_schema.py -q
git add src/elspeth/web/composer/guided/skills/step_1_source.md src/elspeth/web/composer/guided/skills/step_2_sink.md src/elspeth/web/composer/guided/skills/step_3_transforms.md src/elspeth/web/composer/guided/skills/step_4_wire.md tests/unit/web/composer/guided/test_stage_guidance.py
git commit -m "docs(guided): explain catalog-backed stage deferral"
```

## Plan 05 completion gate

- [ ] Every composer surface renders the same capability-core digest.
- [ ] Guided packs contain no linear-only or gate-impossibility claims.
- [ ] LLM configuration exposes typed single- and multi-query shapes.
- [ ] Plugin assistance teaches structured multi-query output without tool-call choreography.
- [ ] Wrong-stage guidance preserves capability and matches catalog-derived stage ownership.

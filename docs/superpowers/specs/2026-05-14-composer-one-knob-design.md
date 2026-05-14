# Composer One-Knob Design

**Date:** 2026-05-14
**Status:** Draft repaired after implementation-readiness and plan-review feedback
**Tracking epic:** `elspeth-a5fbc1ed4a` — "Composer one-knob configuration — single guided knob schema across plugin options and recipes"
**Builds on:** `2026-05-11-composer-guided-mode-design.md` (parent guided-mode design)
**Investigation report:** `.scratch/composer-knob-wiring-report.md`
**Branch:** RC5.2
**Reviewers (completed):** axiom-solution-architect (CHANGES-REQUESTED → folded), axiom-python-engineering (Critical x2 → folded), yzmir-systems-thinking (validated), yzmir-llm-specialist (Medium x2 → folded)

---

## 1. Problem

The guided composer carries **three parallel typing systems** for the same conceptual datum — an operator-facing configuration value (a "knob"):

| System | Defined in | Consumed by | Values |
|--------|------------|-------------|--------|
| Pydantic JSON Schema | per-plugin `config_model` | `SchemaFormTurn.inferFieldType` | full JSON-Schema |
| Recipe `SlotSpec` | `src/elspeth/web/composer/recipes.py:25-52` | `_coerce_slot`, `RecipeOfferTurn.tsx` | `{blob_id, str, float, int, str_list}` |
| Frontend `FieldType` | `SchemaFormTurn.tsx:101` | local switch | `{text, number-int, number-float, checkbox, enum, json-fallback}` |

This causes:

- **Information loss at layer boundaries.** `Optional[X]` Pydantic fields (the most common shape in real plugin contracts) collapse to a raw JSON textarea in the frontend because the renderer doesn't recognise `anyOf: [T, null]`. Inspection facts collected before the schema form are discarded by the emitter.
- **Drift between renderers.** `RecipeOfferTurn` and `SchemaFormTurn` render the same kinds of fields with separate code paths.
- **Renderer-as-content-author.** A hardcoded `commonLabels` dictionary lives in the React renderer because Pydantic-emitted `title` metadata isn't reliably authored at the plugin level.

The operator's directive is unambiguous: **"standardise on one knob unless there's a very compelling reason not to."**

## 2. Design Choices

Each choice cites the alternative considered and the finding it traces to.

### Choice 1 — JSON Schema is the canonical authoring system

**Rejected:** Recipe `SlotSpec` (strict subset, only five types); a new fourth representation (unjustified expansion).

**Why:** JSON Schema is already the broadest of the three; Pydantic emits it natively from `Field(...)`; off-the-shelf tooling exists; it already powers half the guided flow. Recipe slots become a constrained subset expressible as JSON Schema with `format`/`enum` annotations.

**Traces to:** Report §3.2 (typing-systems inventory); operator directive §6.1.

### Choice 2 — Authoring lives inside each plugin's Pydantic `config_model`

**Rejected:** A new `contracts/knobs/` module separate from plugins.

**Why:** Pydantic configuration models are already the plugin contract and are exposed through each plugin's live `config_model`. Non-plugin-backed recipes get a thin adapter (`KnobSchema.from_slot_specs`) until those recipes are re-authored against plugin-backed shapes. `SlotType` and `SlotSpec` live in `contracts/` so catalog lowering can depend on the slot contract without importing the composer recipe registry.

**Traces to:** Report Finding 5 (recipe-slot/schema-form unification).

### Choice 3 — Lowering is server-side, before the schema crosses the wire

**Rejected:** Renderer-side rules only (forces a TypeScript implementation to mirror the Python contract; doubles the test surface).

**Why:** The backend has the live Pydantic model and the catalog. Doing lowering server-side gives a single Python implementation, testable in Python, and lets the renderer collapse to a discriminator dispatch.

**Traces to:** Report Finding 1 (Optional/anyOf shape lost in renderer).

**Implementation-readiness repair:** Lowering must preserve the live plugin catalog's current expressiveness. The readiness review found current schemas with `$ref`, nullable arrays, nullable objects, object maps, array-of-model fields, complex `anyOf`, and the LLM top-level `oneOf`. These are not plugin bugs. The lowered subset therefore includes explicit JSON fallback knob kinds (`json-object`, `json-array`, `json-value`) plus first-class `string-list`; unsupported live contracts must not make `CatalogServiceImpl` fail to start merely because they are richer than the first-pass UI.

### Choice 4 — Recipe rendering folded into `SCHEMA_FORM` via a `mode` discriminator on `SchemaFormPayload`

**Rejected:** (a) Delete `TurnType.RECIPE_OFFER` outright (silently absorbs a non-trivial state-machine refactor — solution-architect H1). (b) Keep two separate turn types renderers (defeats the directive).

**Resolution:** `SchemaFormPayload` carries a required `mode: "plugin_options" | "recipe_decision"` discriminator. When `mode == "recipe_decision"`, the payload additionally carries `recipe_context: RecipeContext`. The renderer dispatches on `mode`; for `recipe_decision` it composes the form fields with a `RecipeContextHeader` peer that shows recipe name, description, and the "build manually" alternative. `TurnType.RECIPE_OFFER` is **retained** as a distinct wire discriminator (for state-machine routing); only the rendering is unified.

This preserves four integration points without rewriting them:

- `state_machine.py:773-800` `_advance_step_2_5` — gates on `RECIPE_OFFER`, unchanged.
- `routes.py:2584-2815` recipe_offer dispatch handles `chosen=['accept'|'build_manually']` and `edited_values={recipe_name, slots}` — unchanged.
- `protocol.py:220, 249, 273` `STEP_2_5_RECIPE_MATCH → RECIPE_OFFER` binding — unchanged.
- Renderer dispatch — gains a `mode` switch; reuses field-rendering primitives.

**Traces to:** Operator directive (unification); solution-architect H1 (scope reality check).

### Choice 5 — Inspection-fact prefill threaded via a new `GuidedSession.step_1_inspection_facts` field

**Rejected:** Pass facts inline through emitter call chains without persisting them (breaks the rebuild path at `routes.py:5215` which has no facts in scope).

**Resolution:** Add `step_1_inspection_facts: SourceInspectionFacts | None` to `GuidedSession`. Written when the initial `INSPECT_AND_CONFIRM` turn is emitted, read by `build_step_1_schema_form_turn` after lifting its signature to `(plugin, catalog, *, inspection_facts)`. Same field is read at the GET rebuild path so refresh produces an identical prefilled form.

**Traces to:** Report Finding 3; solution-architect M2.

**Implementation-readiness repair:** the live `SourceInspectionFacts` shape currently carries source kind, redacted identity, byte range, sample row count, observed headers, inferred types, URL candidates, and warnings. It does not carry delimiter or encoding. Step 1 prefill may use observed schema/header facts immediately; delimiter/encoding prefill requires extending `SourceInspectionFacts` first.

### Choice 6 — `composer_tier` is wire-optional; partitioning UI deferred

**Rejected:** Materialise the default `"common"` on the wire when unannotated (creates an "explicit common" vs "unannotated" ambiguity — llm-diagnostician Finding 5).

**Resolution:** `tier` is **omitted from the wire** when no annotation exists. Frontend treats absent as `"common"` for partitioning purposes today. The three-section partitioning UI itself is **deferred** until a second plugin demonstrates the need (solution-architect M1). Per-plugin annotation pass is also deferred. The wire shape is future-proofed; the UI work is not paid for now.

**Replaces with right-sized enforcement:** A CI lint rule (`scripts/cicd/enforce_options_metadata.py`) asserts every metadata-bearing plugin configuration field carries `title` and `description`, including provider-specific fields on discriminated variant models. This is the systems-thinking review's #1 governance floor — independent of tier. It is in scope for this design.

**Traces to:** Report Finding 4; systems-thinking #1 second-order effect; solution-arch M1; llm-diagnostician Finding 5.

## 3. Architecture

```
Plugin config_model (Pydantic v2)
        │
        │  Annotated[T, Field(title=, description=, default=,
        │              json_schema_extra={"composer_tier": ...})]
        ▼
CatalogServiceImpl._catalog_schema  ── extended to also produce knob_schema
        │  Existing: emits json_schema dict (preserved for /catalog routes)
        │  NEW: also lowers to KnobSchema; rich live shapes use explicit JSON fallback knobs
        │  Single-model plugins:  KnobSchema.from_model(plugin_cls.config_model)
        │  Discriminated unions:  KnobSchema.from_discriminated_model(plugin_cls)
        ▼
PluginSchemaInfo (catalog response)
        │  Existing: json_schema: dict[str, Any]
        │  NEW field: knob_schema: KnobSchema
        ▼
emitters.build_step_*_schema_form_turn
        │  Read PluginSchemaInfo.knob_schema directly; no lowering at request time
        ▼
Turn { type: SCHEMA_FORM, payload: SchemaFormPayload }
        │  Payload includes required mode + recipe_context for Choice 4
        ▼
SchemaFormTurn.tsx  ◄── rewrite: closed kind dispatch
        │  - text / number-int / number-float / checkbox / enum
        │  - string-list (first-class)
        │  - blob-ref (uuid + inline-uploader)
        │  - json-object / json-array / json-value (explicit fallback kinds)
        │  - nullable wrapper (clear button)
        │  - mode-based composition (plugin_options vs recipe_decision)
        │  - visible_when predicates for discriminated-union variant fields
        ▼
GuidedRespondRequest.edited_values
        │  Shape unchanged from current design:
        │     plugin_options mode: { plugin, options, observed_columns, sample_rows }
        │     recipe_decision mode: { recipe_name, slots, chosen }
        ▼
handle_step_1/2/3 → _execute_set_*  ── unchanged
```

**Catalog API extension (in scope for this design):**

`PluginSchemaInfo` (`catalog/schemas.py:44`, a Pydantic `_StrictResponse` subclass — not a `@dataclass(frozen=True)`, so the `freeze_fields` discipline does not apply) gains a `knob_schema: KnobSchema` field. The lowering happens in `CatalogServiceImpl.__init__` at startup, not in `_catalog_schema` per-request: the constructor walks every registered source/transform/sink, calls the appropriate `KnobSchema.from_*` constructor, and stores the result in a cache keyed by `(plugin_kind, name)`. `get_schema` reads the cache. `_catalog_schema` is unchanged in shape but its result is built once.

This is what makes the "halts startup on plugin bug" disposition (§6) mechanically enforceable rather than aspirational: if any plugin's schema is malformed or violates the one-knob invariants, `__init__` raises `KnobSchemaLoweringError` and the FastAPI dependency-injection bootstrap halts before the first request is served. Valid-but-rich schemas lower to `json-object`, `json-array`, or `json-value` fallback knobs. Under per-request lowering a true invariant error would surface as a 500 on first catalog hit — silent in CI, loud only at production traffic time.

`knob_schema` is computed once per `CatalogServiceImpl` instance; plugin hot-reload is out of scope for this design. If a plugin is later added that supports hot-reload, the cache invalidation contract is for that work to specify.

The existing `json_schema: dict` field is preserved for the public `/catalog/schemas/...` HTTP route (external consumers and the auto-generated catalog page); the guided composer reads only `knob_schema`. This split avoids forcing external consumers onto the lowered subset.

## 4. Wire Shape: `KnobSchema`

```python
class KnobField(TypedDict):
    name: str
    label: str             # title or name (no fallback dictionary)
    description: NotRequired[str]      # absent when unset
    kind: Literal[
        "text", "number-int", "number-float", "checkbox",
        "enum", "string-list", "blob-ref",
        "json-object", "json-array", "json-value",
    ]
    tier: NotRequired[Literal["essential", "common", "advanced"]]
                          # ABSENT when unannotated; renderer treats absent as common
    required: bool
    default: NotRequired[object]
                          # ABSENT when no default declared
                          # PRESENT (possibly null) when an explicit default exists
                          # — disambiguates from "required, no default" for LLM consumers
    nullable: bool         # was anyOf [T, null]; renderer uses for clear-button affordance
    enum: NotRequired[list[str]]              # only when kind == "enum"
    item_kind: NotRequired[Literal[...]]      # only when kind == "string-list"

class KnobSchema(TypedDict):
    fields: list[KnobField]   # preserves declaration order from model_fields

class RecipeContext(TypedDict):
    recipe_name: str
    description: str
    alternatives: list[str]   # currently {"build_manually"}

class _PluginOptionsPayload(TypedDict):
    mode: Literal["plugin_options"]
    plugin: str
    knobs: KnobSchema
    prefilled: dict[str, object]

class _RecipeDecisionPayload(TypedDict):
    mode: Literal["recipe_decision"]
    knobs: KnobSchema
    prefilled: dict[str, object]
    recipe_context: RecipeContext

SchemaFormPayload = _PluginOptionsPayload | _RecipeDecisionPayload
```

The tagged-union form (rather than a flat TypedDict with `NotRequired` correlated fields) lets TypeScript's discriminated-union narrowing on `payload.mode` make `recipe_context`'s presence known to the type system without runtime guards. The emitter constructs the correct variant on the Python side; the renderer narrows on `mode` on the TypeScript side. `mode` is required in both variants — the legacy "absent → plugin_options" fallback is not preserved (this is a new wire contract; there is no legacy to maintain per CLAUDE.md no-legacy-code policy).

No raw `anyOf`, `$ref`, `oneOf`, or nested JSON Schema objects cross the wire. They are lowered into first-class knob kinds when possible, or into explicit fallback kinds when the shape is valid but richer than the guided UI can model:

| Fallback kind | Used for |
|---------------|----------|
| `json-object` | Object maps, nested object models, model `$ref`, nullable object fields |
| `json-array` | Arrays that are not exactly `list[str]`, including array-of-model fields and nullable arrays |
| `json-value` | Complex unions or values that cannot be narrowed to a safer control without changing semantics |

Only malformed schemas or violations of the one-knob invariants raise `KnobSchemaLoweringError`.

## 4.1 Discriminated-union plugins

Some plugins (notably `LLMTransform` at `_catalog_schema:165-170, 182`) carry **discriminated unions** in their configuration model — a top-level discriminator field (e.g., `provider: Literal["azure", "openrouter"]`) gates which other fields are valid. These are real, in active use, and rejecting them with `KnobSchemaLoweringError` would break a flagship plugin.

**Handling:** `KnobSchema.from_discriminated_model` flattens variants into a single `fields` list with `visible_when` annotations.

```python
class KnobField(TypedDict):
    # ... fields from §4 ...
    visible_when: NotRequired[VisibilityPredicate]
        # Renderer hides this field unless the predicate matches the current
        # form state. Absent → always visible (the common case).

class VisibilityPredicate(TypedDict):
    field: str          # name of an EARLIER-DECLARED KnobField in the same KnobSchema
    equals: object      # exact value the other field must hold
```

The discriminator itself is rendered as a regular `kind="enum"` knob and is emitted **first** in declaration order. Variant fields receive `visible_when={"field": "<discriminator>", "equals": "<variant-value>"}`. The renderer evaluates predicates against current form state; only fields whose predicate matches are visible.

**Evaluation semantics (committed):**

- **Field reference:** `field` must name a KnobField declared *earlier* in `KnobSchema.fields`. Forward references are rejected at catalog load. This makes evaluation order independent of operator interaction order.
- **No nesting:** a field with `visible_when` whose target field also has `visible_when` is rejected at catalog load. Visibility chains are forbidden.
- **Form state for evaluation:** `prefilled ∪ operator-edits`. Schema defaults are not consulted at predicate-evaluation time — only what the operator has been shown or has changed.
- **Submission contract:** `edited_values.options` must contain values *only* for fields visible under the currently-selected discriminator value. The backend rejects (HTTP 400, `code="hidden_field_submitted"`) any submission containing values for fields whose predicate does not match form state. This is the same auditability discipline the rest of ELSPETH applies — a hidden field that silently drops a submitted value is evidence tampering.
- **Discriminator change:** when the operator changes the discriminator value mid-edit, variant-specific form state is **discarded**. Switching back re-prefills the variant's fields from schema defaults, not from prior typed values. This is committed UX, not a frontend choice — the audit trail records only one variant per submission and shadow state would silently re-introduce dropped fields on the next discriminator flip.

**Why this approach:** It preserves the one-knob wire shape (no separate `discriminated_form` payload), it matches operator mental model (the discriminator looks like a normal dropdown), and `visible_when` becomes a small documented capability rather than a discriminated-union special case.

**Scope guard (mechanically enforced):** `visible_when` supports **exactly** the keys `{field, equals}` — no more, no less. A predicate with any other key or missing either key raises `KnobSchemaLoweringError` at catalog load. This converts the original "redesign signal" policy into a build gate; a future plugin that needs AND/OR predicates produces a startup failure, not a quiet accretion. AND/OR/`present`/`not_equals` extensions are explicitly out of scope; adding them requires a new design.

**Forward-looking note:** when the second discriminated-union plugin is registered, treat it as a redesign forcing function. Two distinct plugins relying on flat single-predicate visibility is the point at which the limitation begins to bind — either a richer predicate language lands then, or both plugins are refactored to avoid the need.

## 5. Lowering function design

```python
# src/elspeth/web/catalog/knob_schema.py

class KnobSchemaLoweringError(Exception):
    """Raised at catalog-load time when a plugin's Pydantic schema is malformed
    or violates a one-knob invariant. Valid-but-rich live schemas lower to
    explicit JSON fallback knobs instead of halting startup."""


class KnobSchema(TypedDict):
    fields: list[KnobField]

    @classmethod
    def from_model(
        cls,
        model_cls: type[BaseModel],
        *,
        composer_tier_default: str = "common",
    ) -> "KnobSchema":
        """Lower a plugin config_model to KnobSchema.

        Takes the live model class (not a serialised dict) so $defs is
        available via model_cls.model_fields metadata and $ref resolution is
        bounded by definition rather than by ad-hoc dict navigation.

        Raises KnobSchemaLoweringError on:
            - malformed oneOf or unsupported discriminator metadata
            - discriminated unions
            - $ref pointing outside model_cls's own bounded schema scope
            - visibility/invariant violations

        Valid rich fields lower conservatively:
            - nested object/$ref shapes -> kind="json-object"
            - non-string arrays -> kind="json-array"
            - complex unions -> kind="json-value"
        """

    @classmethod
    def from_discriminated_model(
        cls,
        plugin_cls: type,
        *,
        composer_tier_default: str = "common",
    ) -> "KnobSchema":
        """Lower a discriminated-union plugin (see §4.1).

        Requires the plugin class to expose:

            @classmethod
            def discriminated_variants(cls) -> tuple[str, dict[str, type[BaseModel]]]:
                '''Return (discriminator_field_name, {literal_value: variant_cls}).'''

        For LLMTransform this is a trivial wrapper over _PROVIDERS. The contract
        is a structural Protocol declared in contracts/; catalog dispatch checks
        for a callable discriminated_variants() method and raises
        KnobSchemaLoweringError at catalog load if it is absent.

        Mandates the Pydantic `Annotated[Union[...], Field(discriminator="...")]`
        authoring form. The `Annotated[..., Discriminator(...)]` (pydantic v2
        Discriminator class) form produces a different __pydantic_core_schema__
        shape and is not supported; a plugin using that form raises
        KnobSchemaLoweringError at catalog load with a remediation message
        pointing at this paragraph.

        Walks the discriminator field first (emitted as kind="enum"), then
        each variant's model_fields, attaching visible_when predicates."""

    @classmethod
    def from_slot_specs(cls, slots: Mapping[str, SlotSpec]) -> "KnobSchema":
        """Adapter constructor for non-plugin-backed recipes.

        Maps each SlotSpec to a KnobField:
            blob_id  → kind="blob-ref"
            str      → kind="text"
            int      → kind="number-int"
            float    → kind="number-float"
            str_list → kind="string-list", item_kind="text"

        Test is parametrised over typing.get_args(SlotType) so a new SlotType
        member fails test collection rather than runtime."""
```

**Why the class-method entry point:** Python-engineering review Critical 2 — accepting a serialised `dict` cannot resolve `$ref` because `$defs` lives at the schema root, not the property site. `model_cls.model_fields` carries the resolved `FieldInfo` metadata directly; `model_cls.model_json_schema()` can be invoked with a controlled `ref_template` when needed.

## 6. Failure modes

| Mode | Disposition |
|------|------------|
| Plugin config schema contains `oneOf` (non-discriminated, multiple structural variants) | Lowers to `kind="json-value"` unless the lowering can prove a narrower safe control. No startup failure for valid live richness. |
| Plugin config schema is a discriminated union (e.g. `LLMTransform`) | Lowered via `KnobSchema.from_discriminated_model` (see §4.1). Discriminator becomes an enum knob; variant fields get `visible_when` predicates. No error. |
| Discriminated-union plugin missing `discriminated_variants()` classmethod, or using `Annotated[..., Discriminator(...)]` instead of `Field(discriminator=...)` | `KnobSchemaLoweringError` at catalog load with a remediation message. |
| `VisibilityPredicate` has keys outside `{field, equals}`, or `field` is a forward reference, or `field` is itself `visible_when`-gated | `KnobSchemaLoweringError` at catalog load. Mechanically enforces the §4.1 scope guard. |
| Operator submission contains values for currently-hidden fields | HTTP 400 with `detail.code="hidden_field_submitted"` and per-field predicate detail. Backend never silently drops submitted values. |
| Plugin config schema contains recursive `$ref` | Lowers conservatively to `kind="json-object"` when bounded metadata can be retained. Raises only if the schema cannot be bounded safely. |
| Catalog load fails for true invariant violations | `__init__` raises before FastAPI serves the first request. Error message carries: plugin kind, plugin name, field path within the model, violated constraint, suggested remediation. |
| `SourceInspectionFacts` absent at Step 1 emit | Prefill falls back to the existing constant `{"schema": {"mode": "observed"}}`. No special case needed. |
| `SlotSpec.slot_type` not in adapter map | `KnobSchemaLoweringError` at catalog load (recipes are loaded eagerly). Parametrised test over `typing.get_args(SlotType)` catches this in CI. |
| Frontend receives unknown `kind` value | TypeScript exhaustiveness check fails build. Production fallback is "render as text, log warning" — defence in depth only; the type system prevents the case from reaching runtime. |
| `nullable=true` field cleared by operator | Submitted as JSON `null`. Pydantic `Optional[T]` accepts `null`; backend `_execute_set_*` validators unchanged. See §7. |

## 7. Return-path nullable round-trip

The forward path (plugin schema → KnobSchema → frontend) is described above. The return path matters because lowered fields lose `anyOf` structure: a Pydantic `Optional[str]` field becomes `kind="text", nullable=true` on the wire. When the operator clears the field, the frontend submits JSON `null` in `edited_values.options[<field>]`. This must round-trip cleanly through `SetSourceArgumentsModel.model_validate` (`tools.py:2567`).

**Property to test:** `Optional[T]` field cleared via the renderer's clear-button affordance round-trips as `None` through the backend validator without rejection or coercion.

**Concrete test cases (in scope for `tests/unit/web/composer/test_knob_schema.py`):**

1. `Optional[str] = None`, no operator edit → submits absent → validator accepts.
2. `Optional[str] = None`, operator types value, then clears → submits `null` → validator accepts as `None`.
3. `Optional[int] = None`, operator types `0`, then clears → submits `null` (not `""`, not `"0"`) → validator accepts as `None`.
4. Required `str`, operator types value, clears → frontend disables Continue → no submit.
5. `Annotated[int, Field(json_schema_extra={"composer_tier": "advanced"})]` → wire emits `tier: "advanced"`.
6. Plain `int` with no annotation → wire omits `tier`.
7. `int = 5` (explicit default) → wire emits `default: 5`.
8. `int` (no default, required) → wire omits `default`.
9. Discriminator switches from variant A to variant B mid-edit: only variant-B fields appear in `edited_values.options`; variant-A user-typed values are discarded; `_execute_set_*` accepts the submission and validates against variant B's schema.
10. Operator submits an option value for a field whose `visible_when` does not match current discriminator: backend emits `guided_hidden_field_rejected` through the existing composer audit recorder, then returns HTTP 400 `code="hidden_field_submitted"` with per-field predicate detail. No silent drop.

`_execute_set_*` and the per-plugin configuration models are unchanged by this work.

## 8. Trust-tier implications

| Datum | Tier | Notes |
|-------|------|-------|
| `KnobSchema` (lowered) | 1 — full trust | We wrote it from a plugin model we control |
| `prefilled` values from `SourceInspectionFacts` | 3 — zero trust | Already labelled in current code; lowering does not change tier |
| `edited_values.options` returning from frontend | 3 → 2 transition | Coerced by `_execute_set_*` as today; new clear-button affordance preserves `None` semantics |
| `step_1_inspection_facts` on `GuidedSession` | 3 — zero trust | Stored as observed, not as truth |

## 9. Migration sequencing

Per Phase 9 convention 14 ("wire-contract changes ship in one commit"), the migration is not "incremental wire shapes" — it is one atomic wire-contract commit, preceded by a backend-only build-up and followed by a separate dispatch refactor.

| Step | Scope | Wire-shape change? |
|------|-------|---------------------|
| 1 | Land `knob_schema.py` (lowering + adapter) with full test corpus; no caller changes | No |
| 2 | Add `step_1_inspection_facts` to `GuidedSession` (additive field; persistence, no readers yet) | No (state schema only) |
| 3 | **Atomic:** all three emitters (`build_step_1_schema_form_turn`, `build_step_2_schema_form_turn`, `build_step_3_schema_form_turn`) switch to `KnobSchema`; frontend `SchemaFormTurn.tsx` rewrite to consume new shape; inspection-fact prefill goes live | **Yes — atomic** |
| 4 | Recipe-fold: add `mode` + `recipe_context` to `SchemaFormPayload`; `RecipeOfferTurn.tsx` deleted; recipe rendering composed inside `SchemaFormTurn` | **Yes — atomic** (separate commit from Step 3) |
| 5a | **Audit + fill before renderer cutover:** sweep every plugin's metadata-bearing configuration model, including discriminated variants returned by `discriminated_variants()`, fill missing `title` and `description` annotations, land as one commit (or one per-plugin commit chain) before Step 3 goes live | No |
| 5b | **Enforce:** land `scripts/cicd/enforce_options_metadata.py` as a CI-failing check. Wire to the CI matrix. Allowlist starts empty | No |
| 6 | `composer_tier` annotation pass — deferred until second plugin demonstrates partitioning need | No |

The Step 3 emitter is bundled with Step 1/2 in step 3 of this sequence because all three emitters use the same `schema_block` field — partial migration would leave the renderer with two incompatible payload shapes (python-engineering Critical 1).

## 10. Test strategy

`knob_schema.py` is the hot module. Test corpus:

- **Golden snapshots** for every metadata-bearing configuration model in the current plugin catalog, including provider variants returned by `discriminated_variants()`. Run on Pydantic upgrades to detect schema-generation changes before they reach the wire.
- **Live catalog smoke test:** instantiate the real plugin manager with `get_shared_plugin_manager()`, construct `CatalogServiceImpl`, and assert every registered plugin produces `knob_schema` without raising. This guards the live `$ref`, nullable array/object, object-map, array-of-model, complex-`anyOf`, and top-level-`oneOf` shapes found in the readiness review.
- **Hypothesis property test:** random `BaseModel` subclasses with declared scalar and nullable field types → lowering returns a valid `KnobSchema` (all required fields present, `kind` in the Literal set, `nullable` consistent with `anyOf` presence) or raises `KnobSchemaLoweringError` only for malformed schemas/invariant violations. Never silent garbage.
- **Discriminated-union round-trip:** lower `LLMTransform`'s discriminated union → verify the discriminator emerges as a single `kind="enum"` knob with both variant values; variant fields carry correct `visible_when` predicates; predicate-resolution against form state submits only the active-variant fields.
- **Synthetic discriminated-union corpus** (not just the LLMTransform real case): a two-variant union with shared fields; a three-variant union; a variant with no extra fields beyond the discriminator; a variant whose discriminator literal is a non-string. Each round-trips through `from_discriminated_model` with stable `fields` order and correct `visible_when` predicates.
- **`visible_when` negative cases:** predicate with `field` set to a non-existent KnobField name; predicate referencing a KnobField that is itself `visible_when`-gated (nesting attempt); predicate with extra keys beyond `{field, equals}`; predicate referencing a forward (later-declared) KnobField. Each raises `KnobSchemaLoweringError` at catalog load.
- **Hidden-field rejection round-trip:** operator submits an option for a field whose predicate doesn't match current discriminator value → backend emits `guided_hidden_field_rejected` through the existing composer audit recorder, then returns 400 with `code="hidden_field_submitted"` and the failing predicate in detail.
- **Totality test** over `typing.get_args(SlotType)` for `KnobSchema.from_slot_specs`.
- **Negative corpus:** malformed visibility predicates, malformed discriminator metadata, unknown `SlotType`, and unbounded `$ref` shapes should produce `KnobSchemaLoweringError`. Valid rich schemas must lower to `json-object`, `json-array`, or `json-value`.
- **Field-order preservation:** verify `fields` order matches `model_json_schema()["properties"]` insertion order against a model where alphabetical and declaration order differ.
- **Return-path round-trip:** the eight test cases in §7.

Frontend test surface gets simpler: the JSON textarea fallback tests are replaced by parametrised tests over `kind`. The `inferFieldType` helper is deleted.

## 11. CI lint enforcement

New script: `scripts/cicd/enforce_options_metadata.py`.

- Walks every plugin's metadata-bearing configuration model, including provider-specific models returned by `discriminated_variants()`; do not inspect only a discriminated plugin's base `config_model`.
- Asserts each field has non-empty `title` (or carries `Annotated[T, Field(title=...)]`).
- Asserts each field has non-empty `description`.
- When `json_schema_extra` is used on a field, asserts the canonical `Annotated[T, Field(json_schema_extra={...})]` form (python-engineering review verified both forms produce identical schema output, but mixed usage creates a refactoring hazard — direct `Field(...)` without `json_schema_extra` remains accepted for simple title/description-only fields).
- Allowlist via `config/cicd/enforce_options_metadata/` for legitimate exceptions.

This is in scope for the implementation plan, not for a future ticket. The systems-thinking review's #1 second-order effect — "Pydantic models become the UI contract" — is only safe if there is an enforcement floor for the metadata that the renderer now depends on.

## 12. Adapter retirement criterion

`KnobSchema.from_slot_specs` is the transitional path for non-plugin-backed recipes. Per CLAUDE.md no-legacy-code policy, transitional artefacts need an explicit retirement gate.

**Criterion:** The adapter is the permanent bridge for non-plugin-backed recipes unless and until their count reaches zero. Today, all three recipes in `recipes.py` (`classify-rows-llm-jsonl`, `split-by-numeric-threshold`, `fork-coalesce-truncate-jsonl`) are non-plugin-backed. Any retirement work is a separate design that re-authors each recipe against a plugin contract or migrates the recipe-construction surface entirely. **This design does not commit to that work; it commits only to the deletion gate.**

**Honest acknowledgement (per systems-thinking second-pass review):** The criterion is *passive* — nothing in this design schedules the recipe re-authoring work, and no Filigree issue currently owns it. If that work is never scheduled, the gate never fires and `from_slot_specs` lives indefinitely. That outcome is acceptable: the operator directive ("standardise on one knob") is satisfied at the **wire** (a single `KnobSchema` shape crosses the boundary in both cases). `from_slot_specs` is implementation detail in the lowering module — it is not a parallel typing system the way the original `SlotSpec` consumption was. If recipe re-authoring is never scheduled, the design's one-knob promise is still kept.

When a future change adds a new non-plugin-backed recipe, the implementer is responsible for either: (a) authoring it as a plugin contract instead, or (b) re-justifying the adapter's existence in a follow-up design.

## 13. Out of scope

- **Finding 6** (chat → form-patch tool): deferred. The KnobSchema is a strong foundation for this work (llm-diagnostician Finding 4) but the tool's design is its own concern. `kind` enum will need a glossary in the tool description when it lands (llm-diagnostician Finding 3).
- **Finding 8** (Step 3 transform-edit dispatcher persistence): **verified clean** during this design phase. `routes.py:3037-3112` correctly replaces `step_3_proposal.steps[edit_index]` in-place; no remediation needed.
- Backend tool/validator logic (`_execute_set_*`, runtime `_coerce_slot`): works.
- Audit-trail asymmetry for chat (`chat_solver` not emitting `ComposerLLMCall`): Phase B work, tracked separately.
- Multi-output sink wiring: MVP single-output constraint persists.
- Chain-solver plugin-schema consumption: the chain solver does not currently receive plugin option schemas at all (llm-diagnostician Finding 1). Any future work to provide schemas to the chain solver is a separate design; KnobSchema would be the right shape to ship.
- Bespoke UI editors for every complex plugin shape. The one-knob contract requires a stable wire shape, not a hand-built editor for every nested object or map. `json-object`, `json-array`, and `json-value` are acceptable first-pass controls for rich-but-valid plugin contracts.

## 14. Verification gate before implementation

Implementation may start only after the readiness repairs above are reflected in the plan:

- All-plugin startup smoke uses `get_shared_plugin_manager()` and `CatalogServiceImpl`.
- Implementation examples use live symbols: `config_model`, `GuidedSession.initial()`, and `get_recipe(match.recipe_name).description`.
- `SourceInspectionFacts` persistence includes a real deserializer for the live fields.
- Recipe folding preserves both accept and `build_manually` paths.
- FastAPI error tests assert the actual `{"detail": {"code": ...}}` envelope.
- The implementation plan carries the 2026-05-14 CHANGES_REQUESTED dispositions: strict `SourceInspectionFacts` persistence, protocol validation updates, property/golden tests, hidden-field audit emission, tier/freeze gates, and non-tautological Step 1 prefill.

## 15. Review summary

### First-pass reviews

| Reviewer | Verdict | Material findings folded |
|----------|---------|-------------------------|
| axiom-solution-architect | CHANGES-REQUESTED | H1 (recipe-fold scope), H2 (sequencing), M1 (tier deferral), M2 (inspection facts persistence), M3 (return-path) |
| axiom-python-engineering | 2 Critical + 3 Warnings | C1 (atomic commit per convention 14), C2 (BaseModel-class signature), exhaustiveness, canonical annotation form, parametrised totality |
| yzmir-systems-thinking | Validated | CI lint rule for metadata enforcement; adapter retirement criterion |
| yzmir-llm-specialist | 2 Medium | `NotRequired[default]` ambiguity, tier absent-on-default |

### Second-pass reviews (against revised spec)

| Reviewer | Verdict | Material findings folded |
|----------|---------|-------------------------|
| axiom-solution-architect | APPROVE-WITH-CHANGES | H-NEW-1 (metadata baseline pass: §9 split into 5a/5b), M-NEW-2 (catalog-load error operability: §6 row), M-NEW-3 (cache/hot-reload sentence: §3); H-NEW-2 (feature flag) rejected in favour of git-revert rollback after deconfliction with systems-thinking |
| yzmir-systems-thinking | Validated | `visible_when` mechanical enforcement (§6 row), §12 adapter retirement passivity (acknowledged honestly) |
| axiom-python-engineering | 2 Critical + 2 Warnings | C-1 (`discriminated_variants()` protocol method in §5), C-2 (`CatalogServiceImpl.__init__` pre-materialisation in §3), discriminator-switch semantics (§4.1 + §7 case 9), tagged-union `SchemaFormPayload` (§4) |
| yzmir-llm-specialist | Validated | `visible_when` evaluation semantics (§4.1), hidden-field rejection (§4.1 + §6 + §7 case 10), strike "field B is set" example (§4.1) |

All eight reviews (two passes × four lenses) have been folded. The structural decisions ratified across both passes. The remaining items (L-3 single-sourced glossary for future LLM tool descriptions, L-4 predicate validation as pre-requisite for future `propose_form_patch`, L-1 second-discriminated-union as redesign forcing function) are notes rather than spec changes.

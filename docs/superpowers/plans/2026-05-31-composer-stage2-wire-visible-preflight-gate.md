# Stage 2 — Wire-Visible Pre-Flight Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "a required wire-visible field (e.g. `web_scrape.http.abuse_contact`) is unset or placeholder" a *deterministic* readiness condition sourced from a *single declarative source* on the plugin, so the composer surfaces the operator-input gap early and reliably (instead of the cheap model inventing `example.com` and dead-ending), collapse the 7 scattered encodings of that rule, and make the hello-world tutorial seed both wire-visible fields so it never surfaces them.

**Architecture:** A new contractual classmethod `wire_visible_required_fields()` on the plugin base is the single source of truth (web_scrape overrides it). A detection function reads it and flags nodes whose wire-visible values are unset/empty/placeholder; `materialize_state_for_execution` returns a new typed blocker `WireVisibleInputRequired` (parallel to the existing `InterpretationReviewPending`), handled by the two existing callers. `implicit_decisions.py` is deduped to read the declarative source. The tutorial deterministically injects `elspeth@dta.gov.au` + a fixed scraping_reason during normalization.

**Tech Stack:** Python 3.13, pytest, pluggy plugins (`plugins/infrastructure/base.py`), the web readiness layer (`web/interpretation_state.py`, `web/execution/`), composer tutorial (`web/composer/tutorial_service.py`).

**Spec:** `docs/superpowers/specs/2026-05-31-composer-reset-debrief-and-rootcause-fixes-design.md` (Stage 2 + Post-recon corrections: parallel blocker not new `InterpretationKind`; declarative classmethod not `composer_hints`; 7 surfaces; nested paths; tutorial seeds both fields, tutorial-only).

---

## Verified facts (citations)

- `contracts/wire_visible_identity.py`: `is_placeholder_value(value)` (empty string is **not** a placeholder), `is_wire_visible_placeholder(value)` (alias), `reject_placeholder_value(value, *, field_name)`.
- web_scrape HTTP config (`plugins/transforms/web_scrape.py:73-110`): `abuse_contact` and `scraping_reason` are `Field(...)` (required); `_reject_empty` validator (`:100-110`) rejects empty + placeholder. Headers emitted `X-Abuse-Contact` / `X-Scraping-Reason` (`:825-826`). Assistance classmethod `get_agent_assistance` at `web_scrape.py:520`; base version at `plugins/infrastructure/base.py:798` (with `get_post_call_hints` at `:825`). `BaseTransform(ABC)` at `base.py:67`.
- `web/composer/state.py:760` `_validate_web_scrape_abuse_contact_not_reserved` — the RFC-reserved-domain validator that rejected `example.com`.
- `web/composer/implicit_decisions.py`: `_category_for_node_option` (`:197-202` → `"identity"`), `_provenance_for_path` (`:205-214` → `"explicit_source_required"`, **suffix** match), `_note_for_node_option` (`:235-238`, `node.plugin`+exact path). Public entrypoint `build_implicit_decisions_report` (`:56`) → `merge_implicit_decisions_meta` (`:80`), consumed only at `web/sessions/routes/_helpers.py:1697`. `DecisionCategory` includes `"identity"`, `DecisionProvenance` includes `"explicit_source_required"` (closed `Literal`s, `:18-31`). Nested option paths (`http.abuse_contact`) produced by `_flatten_options` (`:163-172`).
- Readiness seam: `web/interpretation_state.py` — `InterpretationReviewPending` (dataclass), `materialize_state_for_execution(state)` returns `InterpretationReviewPending` (pending) **or** the materialized state. Callers: `web/execution/service.py:505-506` and `web/execution/validation.py:1091-1093`, both `isinstance(..., InterpretationReviewPending)`.
- Regression guard: `tests/unit/web/sessions/test_routes.py:5856` (`test_state_data_persists_structured_implicit_decisions_report`) asserts `provenance == "explicit_source_required"` etc. — must stay green.
- Readiness-blocker test style: `tests/unit/web/test_interpretation_state.py:269`.
- Tutorial: `web/composer/tutorial_service.py` drives `CANONICAL_SEED_PROMPT` (`:48,111`), normalizes via `_normalise_tutorial_prompt_template` (`:242`), caches canonical result (`:121-153`).

---

## Task 1: Declarative `wire_visible_required_fields()` on the plugin base + web_scrape override

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/base.py` (add classmethod adjacent to `get_agent_assistance`, ~`:798`)
- Modify: `src/elspeth/plugins/transforms/web_scrape.py` (override on the transform class)
- Test: `tests/unit/plugins/test_wire_visible_required_fields.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/plugins/test_wire_visible_required_fields.py
from elspeth.plugins.transforms.web_scrape import WebScrapeTransform


def test_web_scrape_declares_wire_visible_required_fields():
    fields = WebScrapeTransform.wire_visible_required_fields()
    assert set(fields) == {"http.abuse_contact", "http.scraping_reason"}


def test_base_default_is_empty():
    from elspeth.plugins.infrastructure.base import BaseTransform
    # A transform that does not override returns no wire-visible-required fields.
    class _Bare(BaseTransform):  # minimal concrete subclass for the default
        pass
    assert _Bare.wire_visible_required_fields() == ()
```

(If `BaseTransform` has abstract methods preventing trivial subclassing, assert on the base classmethod directly: `assert BaseTransform.wire_visible_required_fields() == ()`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/plugins/test_wire_visible_required_fields.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'wire_visible_required_fields'`.

- [ ] **Step 3: Add the base classmethod (adjacent to `get_agent_assistance`, base.py:798)**

```python
    @classmethod
    def wire_visible_required_fields(cls) -> tuple[str, ...]:
        """Nested option paths whose values ship on the wire as identity and
        therefore MUST be operator/deployment-supplied (never model-invented).

        Single declarative source of truth for the wire-visible pre-flight
        gate (Stage 2, spec 2026-05-31). Paths are dot-joined nested option
        keys (e.g. ``"http.abuse_contact"``) to match the composer's
        flattened option paths. Default: no wire-visible-required fields.
        """
        return ()
```

- [ ] **Step 4: Override in web_scrape (`WebScrapeTransform`)**

```python
    @classmethod
    def wire_visible_required_fields(cls) -> tuple[str, ...]:
        # Both ship as HTTP headers (X-Abuse-Contact / X-Scraping-Reason) and
        # must come from the operator/deployment, never the model. See the
        # _reject_empty validator and _validate_web_scrape_abuse_contact_not_reserved.
        return ("http.abuse_contact", "http.scraping_reason")
```

(Place it next to the existing `get_agent_assistance` override in `web_scrape.py`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/plugins/test_wire_visible_required_fields.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/plugins/infrastructure/base.py src/elspeth/plugins/transforms/web_scrape.py tests/unit/plugins/test_wire_visible_required_fields.py
git commit -m "feat(plugins): declarative wire_visible_required_fields() single source"
```

---

## Task 2: Detection function + `WireVisibleInputRequired` blocker type

**Files:**
- Modify: `src/elspeth/web/interpretation_state.py` (add the blocker dataclass + detection fn)
- Test: `tests/unit/web/test_wire_visible_gate.py`

The detection scans state nodes; for each node whose plugin declares wire-visible-required fields, it flags any whose value (looked up at the nested option path) is missing, empty, or a placeholder (`is_wire_visible_placeholder`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/web/test_wire_visible_gate.py
import pytest

from elspeth.web.interpretation_state import (
    WireVisibleInputRequired,
    wire_visible_inputs_required,
)


def _state_with_web_scrape(abuse_contact):
    # Build a CompositionState with a single web_scrape node carrying the given
    # http.abuse_contact (None to omit). Mirror the helpers in test_interpretation_state.py.
    from elspeth.web.composer.state import CompositionState, NodeSpec
    http = {"scraping_reason": "monitoring"}
    if abuse_contact is not None:
        http["abuse_contact"] = abuse_contact
    node = NodeSpec(id="scrape", node_type="transform", plugin="web_scrape",
                    input="src_out", on_success="out", options={"http": http})
    return CompositionState.empty().with_nodes((node,))  # use the real builder API


def test_unset_wire_visible_field_is_detected(catalog):
    state = _state_with_web_scrape(abuse_contact=None)
    required = wire_visible_inputs_required(state, catalog)
    assert any(r.node_id == "scrape" and r.field_path == "http.abuse_contact" for r in required)


def test_placeholder_value_is_detected(catalog):
    state = _state_with_web_scrape(abuse_contact="changeme")
    required = wire_visible_inputs_required(state, catalog)
    assert any(r.field_path == "http.abuse_contact" for r in required)


def test_real_value_is_not_flagged(catalog):
    state = _state_with_web_scrape(abuse_contact="ops@agency.gov.au")
    required = wire_visible_inputs_required(state, catalog)
    assert all(r.field_path != "http.abuse_contact" for r in required)
```

Add a `catalog` fixture mirroring existing composer tests (a real `CatalogService` over the plugin registry) — copy from the nearest existing fixture in `tests/unit/web/` that constructs one. The exact `CompositionState`/`NodeSpec` builder calls must match `web/composer/state.py`; use the same construction the existing `test_interpretation_state.py` helpers use.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/test_wire_visible_gate.py -v`
Expected: FAIL — `ImportError: cannot import name 'WireVisibleInputRequired'`.

- [ ] **Step 3: Implement the blocker type and detection fn in `web/interpretation_state.py`**

```python
@dataclass(frozen=True, slots=True)
class WireVisibleRequiredFieldRef:
    """A wire-visible-required field whose value is unset/placeholder."""

    node_id: str
    plugin: str
    field_path: str  # nested option path, e.g. "http.abuse_contact"


@dataclass(frozen=True, slots=True)
class WireVisibleInputRequired:
    """Readiness blocker: operator must supply wire-visible identity values.

    Parallel to ``InterpretationReviewPending`` (Stage 2, spec 2026-05-31).
    Distinct from interpretation review: this is an unset *operator input*,
    not an LLM assumption awaiting review — so it is its own blocker type,
    NOT a new InterpretationKind.
    """

    fields: tuple[WireVisibleRequiredFieldRef, ...]

    def __post_init__(self) -> None:
        freeze_fields(self, "fields")


def _option_at_path(options: Mapping[str, Any], path: str) -> Any:
    cur: Any = options
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def wire_visible_inputs_required(
    state: CompositionState,
    catalog: CatalogService,
) -> tuple[WireVisibleRequiredFieldRef, ...]:
    """Flag wire-visible-required fields that are unset/empty/placeholder.

    Reads the single declarative source ``wire_visible_required_fields()``
    via the catalog-resolved plugin class for each node.
    """
    flagged: list[WireVisibleRequiredFieldRef] = []
    for node in state.nodes:
        plugin_cls = catalog.get_plugin_class(node.node_type, node.plugin)
        for path in plugin_cls.wire_visible_required_fields():
            value = _option_at_path(node.options, path)
            if value is None or (isinstance(value, str) and (not value.strip() or is_wire_visible_placeholder(value))):
                flagged.append(WireVisibleRequiredFieldRef(node_id=node.id, plugin=node.plugin, field_path=path))
    return tuple(flagged)
```

Confirm the catalog accessor name for a plugin *class* (the recon used `catalog.get_schema`; find the method that returns the plugin class so `wire_visible_required_fields()` can be called — likely `catalog.get_plugin_class(kind, name)` or via the registered class on the schema info). If only an instance/schema is available, add a thin `catalog.get_plugin_class` accessor. Import `is_wire_visible_placeholder`, `freeze_fields`, `Mapping`, `Any`, `CatalogService`, `CompositionState` at the top of the module.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/web/test_wire_visible_gate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/interpretation_state.py tests/unit/web/test_wire_visible_gate.py
git commit -m "feat(web): wire-visible input detection + WireVisibleInputRequired blocker"
```

---

## Task 3: Surface the blocker through `materialize_state_for_execution` + both callers

**Files:**
- Modify: `src/elspeth/web/interpretation_state.py` (`materialize_state_for_execution`)
- Modify: `src/elspeth/web/execution/service.py:505-506`
- Modify: `src/elspeth/web/execution/validation.py:1091-1093`
- Test: `tests/unit/web/test_wire_visible_gate.py`, `tests/unit/web/test_interpretation_state.py`

Decision: wire-visible inputs are checked **before** interpretation sites (an unset operator identity is a harder blocker than a pending LLM assumption), so `materialize_state_for_execution` returns `WireVisibleInputRequired` first when present.

- [ ] **Step 1: Write the failing test**

```python
def test_materialize_returns_wire_visible_blocker(catalog):
    from elspeth.web.interpretation_state import (
        materialize_state_for_execution, WireVisibleInputRequired,
    )
    state = _state_with_web_scrape(abuse_contact=None)
    result = materialize_state_for_execution(state)  # signature may take catalog; see Step 3
    assert isinstance(result, WireVisibleInputRequired)
    assert result.fields[0].field_path == "http.abuse_contact"
```

- [ ] **Step 2: Run it (fails — blocker not returned)**

Run: `.venv/bin/python -m pytest tests/unit/web/test_wire_visible_gate.py::test_materialize_returns_wire_visible_blocker -v`
Expected: FAIL (returns materialized state, not the blocker).

- [ ] **Step 3: Return the blocker from `materialize_state_for_execution`**

`materialize_state_for_execution` currently takes `(state)`. The detection needs the catalog. Thread the catalog in: add a `catalog: CatalogService` parameter (update both call sites in Step 4). At the top of the function body, before the interpretation-site check:

```python
    wire_visible = wire_visible_inputs_required(state, catalog)
    if wire_visible:
        return WireVisibleInputRequired(fields=wire_visible)
```

If threading `catalog` into `materialize_state_for_execution` is too invasive for its other callers, instead add a sibling `readiness_blockers(state, catalog)` that the execution callers invoke *before* `materialize_state_for_execution`, returning `WireVisibleInputRequired | None`. Prefer the parameter-threading approach unless a third caller exists without a catalog in scope — confirm caller count first (`grep -rn "materialize_state_for_execution(" src/elspeth`).

- [ ] **Step 4: Handle the new blocker at both callers (mirror the `InterpretationReviewPending` branch)**

In `web/execution/service.py` (after `:505`):

```python
        materialized_state = materialize_state_for_execution(composition_state, catalog)
        if isinstance(materialized_state, WireVisibleInputRequired):
            # Operator must supply wire-visible identity values before run.
            raise PipelineNotReadyError(_wire_visible_detail(materialized_state))
        if isinstance(materialized_state, InterpretationReviewPending):
            ...  # unchanged
```

Mirror the exact error/return shape the existing `InterpretationReviewPending` branch uses at each site (read `:506` and `validation.py:1093` and reproduce their pattern — raise the same exception type or return the same response model, with a wire-visible-specific message). Import `WireVisibleInputRequired`. Add `_wire_visible_detail(blocker)` building a message naming each `node_id`/`field_path`.

- [ ] **Step 5: Run tests (unit + the two execution caller tests)**

Run:
```bash
.venv/bin/python -m pytest tests/unit/web/test_wire_visible_gate.py tests/unit/web/test_interpretation_state.py -v
.venv/bin/python -m pytest tests/unit/web/execution/ -q
```
Expected: PASS (no regression in execution tests; the new blocker test passes).

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/interpretation_state.py src/elspeth/web/execution/service.py src/elspeth/web/execution/validation.py tests/unit/web/test_wire_visible_gate.py
git commit -m "feat(web): surface WireVisibleInputRequired through readiness materialization"
```

---

## Task 4: Dedupe `implicit_decisions.py` onto the declarative source

**Files:**
- Modify: `src/elspeth/web/composer/implicit_decisions.py` (`_category_for_node_option`, `_provenance_for_path`, `_note_for_node_option`)
- Test: `tests/unit/web/sessions/test_routes.py` (existing regression `:5856` must stay green) + new unit test

Replace the hardcoded `{"http.abuse_contact", "http.scraping_reason"}` sets with a lookup against the plugin's `wire_visible_required_fields()`. **Behaviour must be identical** for web_scrape (regression guard).

- [ ] **Step 1: Write the failing test (generality — a second plugin's wire-visible field classifies as identity)**

```python
def test_implicit_decisions_reads_declarative_wire_visible(monkeypatch, catalog):
    from elspeth.web.composer.implicit_decisions import _category_for_node_option, _provenance_for_path
    from elspeth.web.composer.state import NodeSpec
    node = NodeSpec(id="s", node_type="transform", plugin="web_scrape", input="i", on_success="o",
                    options={"http": {"abuse_contact": "x@y.gov.au", "scraping_reason": "r"}})
    # Still classified as identity / explicit_source_required via the declarative source.
    assert _category_for_node_option(node, "http.abuse_contact", catalog=catalog) == "identity"
    assert _provenance_for_path("node.s.options.http.abuse_contact", "x@y.gov.au", node=node, catalog=catalog) == "explicit_source_required"
```

(Signatures gain `catalog`/`node` kwargs in Step 3; match them to what the report builder can supply.)

- [ ] **Step 2: Run it (fails — functions don't take catalog / still hardcoded)**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_implicit_decisions_declarative.py -v`
Expected: FAIL.

- [ ] **Step 3: Refactor the three functions to read the declarative source**

Add a helper and thread `catalog` from `build_implicit_decisions_report` down:

```python
def _wire_visible_paths(node: NodeSpec, catalog: CatalogService) -> frozenset[str]:
    plugin_cls = catalog.get_plugin_class(node.node_type, node.plugin)
    return frozenset(plugin_cls.wire_visible_required_fields())
```

`_category_for_node_option`: replace `node.plugin == "web_scrape" and field_path in {...}` with `field_path in _wire_visible_paths(node, catalog)` → `"identity"`.
`_note_for_node_option`: same condition → the existing note string.
`_provenance_for_path`: this matches by *suffix* and has no node in scope. Keep the suffix behaviour but derive the suffix set from the declarative source across all plugins, or (cleaner) thread `node` here too so it uses `_wire_visible_paths`. Match whichever the report builder can supply; preserve the `endswith` semantics if a path-only signature must remain.

Thread `catalog` from `build_implicit_decisions_report(state)` — it must gain a `catalog` parameter; update its single consumer `merge_implicit_decisions_meta` (`:80`) and the call at `web/sessions/routes/_helpers.py:1697` to pass the catalog (available on `request.app.state`).

- [ ] **Step 4: Run the regression guard + new test**

Run:
```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_routes.py::test_state_data_persists_structured_implicit_decisions_report -v
.venv/bin/python -m pytest tests/unit/web/composer/test_implicit_decisions_declarative.py -v
```
Expected: BOTH PASS — `provenance == "explicit_source_required"` etc. unchanged for web_scrape.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/implicit_decisions.py src/elspeth/web/sessions/routes/_helpers.py tests/unit/web/composer/test_implicit_decisions_declarative.py
git commit -m "refactor(composer): implicit_decisions reads declarative wire_visible source (dedupe)"
```

---

## Task 5: Tutorial seeds both wire-visible fields deterministically (tutorial-only)

**Files:**
- Modify: `src/elspeth/web/composer/tutorial_service.py` (the `_normalise_tutorial_*` normalization path)
- Test: `tests/unit/web/composer/test_tutorial_service.py`

Inject `http.abuse_contact = "elspeth@dta.gov.au"` and `http.scraping_reason = "DTA composer tutorial — government website colour analysis"` into any `web_scrape` node whose value is unset/placeholder, during tutorial normalization. These are operator/deployment-supplied values (doctrine-clean). Tutorial-only — no deployment-wide default.

- [ ] **Step 1: Write the failing test**

```python
def test_tutorial_seeds_wire_visible_fields():
    from elspeth.web.composer.tutorial_service import _seed_tutorial_wire_visible_identity
    options = {"http": {"abuse_contact": "compliance@example.com", "scraping_reason": ""}}
    seeded = _seed_tutorial_wire_visible_identity("web_scrape", options)
    assert seeded["http"]["abuse_contact"] == "elspeth@dta.gov.au"
    assert seeded["http"]["scraping_reason"] == "DTA composer tutorial — government website colour analysis"


def test_tutorial_seed_is_noop_for_non_web_scrape():
    from elspeth.web.composer.tutorial_service import _seed_tutorial_wire_visible_identity
    options = {"model": "x"}
    assert _seed_tutorial_wire_visible_identity("llm", options) == options
```

- [ ] **Step 2: Run it (fails — function missing)**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tutorial_service.py -k wire_visible -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement the deterministic seed + wire it into normalization**

```python
_TUTORIAL_ABUSE_CONTACT = "elspeth@dta.gov.au"
_TUTORIAL_SCRAPING_REASON = "DTA composer tutorial — government website colour analysis"


def _seed_tutorial_wire_visible_identity(plugin: str, options: dict[str, Any]) -> dict[str, Any]:
    """Deterministically supply web_scrape wire-visible identity in the tutorial.

    These are operator/deployment-supplied values (the DTA deployment identity),
    not model fabrications — doctrine-clean. Tutorial-only; no deployment-wide
    default. Overwrites unset/empty/placeholder values so the tutorial never
    surfaces the wire-visible gap; leaves a real operator-set value untouched.
    """
    if plugin != "web_scrape":
        return options
    http = dict(options.get("http") or {})
    for key, seed in (("abuse_contact", _TUTORIAL_ABUSE_CONTACT), ("scraping_reason", _TUTORIAL_SCRAPING_REASON)):
        cur = http.get(key)
        if not isinstance(cur, str) or not cur.strip() or is_wire_visible_placeholder(cur):
            http[key] = seed
    return {**options, "http": http}
```

Call `_seed_tutorial_wire_visible_identity(node.plugin, node.options)` for each node in the tutorial normalization path (alongside `_normalise_tutorial_prompt_template`, `:242`). Import `is_wire_visible_placeholder`.

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tutorial_service.py -k wire_visible -v`
Expected: PASS

- [ ] **Step 5: Re-seed the tutorial cache (caveat from spec)**

The tutorial caches the canonical-prompt result (`tutorial_service.py:121-153`); a cached pipeline built before this change carries `example.com`. Invalidate/re-seed the cache so cached tutorials carry the seeded values. Document the operator action (the tutorial cache dir is `data/tutorial_cache`); deleting it forces a fresh canonical seed on next run. Add a test asserting a freshly-normalized canonical tutorial pipeline has `abuse_contact == "elspeth@dta.gov.au"`.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/tutorial_service.py tests/unit/web/composer/test_tutorial_service.py
git commit -m "feat(composer): tutorial seeds web_scrape wire-visible identity (elspeth@dta.gov.au)"
```

---

## Task 6: Gate reconciliation + full verification

**Files:** plugin hash, tier-model allowlist, suites.

- [ ] **Step 1: Refresh the web_scrape plugin source_file_hash (PH3 gate)**

Task 1 edited `plugins/transforms/web_scrape.py` — its `source_file_hash` is now stale (the gate is CI-only; commits go green locally with a stale hash). Refresh and co-land:
Run:
```bash
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth --format json > /tmp/tiercheck.json
.venv/bin/python -m scripts.cicd.plugin_hash   # refresh from the check JSON per project_plugin_hash_gate_ci_only
```
(Use the project's documented `scripts/cicd/plugin_hash` flow; co-land the refreshed hash in the same commit as the edit.)

- [ ] **Step 2: Tier-model allowlist / fingerprints**

Editing `base.py` / `web_scrape.py` / `interpretation_state.py` / `implicit_decisions.py` may rotate AST fingerprints. Run the tier-model check and reconcile per the project procedure:
Run: `env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth`
Expected: green (reconcile any rotations co-landed).

- [ ] **Step 3: Type + lint**

Run:
```bash
.venv/bin/python -m mypy src/elspeth/web/interpretation_state.py src/elspeth/web/composer/implicit_decisions.py src/elspeth/web/composer/tutorial_service.py src/elspeth/plugins/transforms/web_scrape.py src/elspeth/plugins/infrastructure/base.py
.venv/bin/python -m ruff check src/elspeth/web/ src/elspeth/plugins/
```
Expected: clean

- [ ] **Step 4: Targeted + adjacent suites**

Run:
```bash
.venv/bin/python -m pytest tests/unit/web/ tests/unit/plugins/test_wire_visible_required_fields.py -q
.venv/bin/python -m pytest tests/unit/web/sessions/test_routes.py -k implicit_decisions -q
```
Expected: PASS

- [ ] **Step 5: Commit any gate fixups**

```bash
git add -A && git commit -m "chore(composer): stage-2 gate reconciliation (plugin hash, tier-model)"
```

---

## Self-review checklist (completed by plan author)

- **Spec coverage:** declarative single source (Task 1); deterministic detection + parallel blocker not a new `InterpretationKind` (Tasks 2-3); dedupe 7 surfaces preserving nested paths + regression (Task 4); tutorial seeds both fields tutorial-only + cache re-seed (Task 5); gate reconciliation incl. plugin-hash (Task 6). ✓
- **Placeholder scan:** code shown for every code step. Two grounded "confirm the exact accessor/caller" notes (catalog plugin-class accessor in Task 2; `materialize_state_for_execution` caller count in Task 3) are verification instructions against cited code, not invented APIs. ✓
- **Type consistency:** `wire_visible_required_fields()`, `WireVisibleInputRequired`, `WireVisibleRequiredFieldRef`, `wire_visible_inputs_required`, `_wire_visible_paths`, `_seed_tutorial_wire_visible_identity` used identically across tasks; nested dot-path convention (`http.abuse_contact`) consistent with `_flatten_options`. ✓

## Risks

- **R2 (dedupe drifts audit classification):** Task 4 keeps `test_routes.py:5856` green as the behaviour-preservation guard.
- **Caller-count risk for `materialize_state_for_execution`:** Task 3 Step 3 includes the fallback (sibling `readiness_blockers` fn) if a catalog-less third caller exists.
- **Plugin-hash / tier-model fallout:** Task 6 co-lands the refreshes (CI-only gates, green-locally-stale trap).
- **Tutorial cache staleness:** Task 5 Step 5 re-seeds the cache.

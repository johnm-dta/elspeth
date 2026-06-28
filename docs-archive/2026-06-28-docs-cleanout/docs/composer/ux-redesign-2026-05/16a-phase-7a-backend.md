# Phase 7A — Backend: Plugin metadata schema extension + catalog API surface

> **⚠️ HISTORICAL — `data_trust_tier` scope rescinded 2026-05-18.**
>
> The original Phase 7A scope included a per-plugin `data_trust_tier`
> field (`int | None` with values `1`, `2`, `3`) and authored
> declarations on all 15 boundary plugins. **That scope was rescinded
> in commit `c76ecc0f2`** (operator review 2026-05-18) because the
> field failed the "each tag must represent a meaningful per-plugin
> decision" test — `data_trust_tier == 3` was structurally constant
> across every Source and every external-call Transform, and
> `data_trust_tier == 2` was structurally constant across every pure
> Transform. Authors were copy-pasting a value the kind already
> determined.
>
> The boundary predicate in
> `src/elspeth/web/audit_readiness/service.py` was rewritten to derive
> boundary status from `(kind, determinism)`, and the field + enum
> type + all 15 declarations + the `_INTERNAL_PLUGIN_CLASSES` parity
> test were deleted. A successor parity test
> (`test_boundary_predicate_parity.py`) exercises the new predicate
> against the same expected-boundary lists.
>
> Sections below referencing `data_trust_tier` are preserved as
> historical record of the Phase 7A planning state. New work should
> not re-introduce the field — see commit `c76ecc0f2` rationale and
> `feedback_catalog_is_reference_not_toolkit` memory.

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the backend half of Phase 7 — extend the plugin base classes
(`BaseSource` / `BaseTransform` / `BaseSink`) with new optional class
attributes for reference content (when-to-use prose, when-not-to-use prose,
example use snippet, capability tags, audit-characteristic flags, data
trust tier), extend the catalog API to surface these fields, derive the
audit-characteristic flags that can be derived from existing plugin
attributes (determinism), fill in `csv_source.py`
as the canonical example (Task 5), author `data_trust_tier = 3` on all 15
EXTERNAL_BOUNDARY plugins (all sources + the named transform and sink
allowlists from `trust.py`, Task 5 + Task 6), and delete
`src/elspeth/web/audit_readiness/trust.py` in the same commit as the final
plugin authoring — discharging the No-Legacy commitment at `trust.py:31-35`
(Task 6). Frontend wiring is in the companion plan,
[16b-phase-7b-frontend.md](16b-phase-7b-frontend.md).

**Architecture:** Schema-then-derivation-then-API. The new fields are
class-attribute level (matching the existing `determinism` /
`plugin_version` / `source_file_hash` precedent on `BaseSource`). The
catalog service derives the audit characteristics by composing the
plugin's declared `audit_characteristics` set with characteristics that
can be derived from `determinism`. The
extended `PluginSummary` model adds optional fields that default to
empty/None so existing plugins (and existing frontend consumers) keep
working unchanged.

**Tech Stack:** Python 3.13, Pydantic v2 (catalog response models), pytest.

**Sibling plan:** [16b-phase-7b-frontend.md](16b-phase-7b-frontend.md) —
frontend card-layout reshape, filter chips, search extension, synthetic
"Inline data from chat" entry, keyboard-shortcut regroup.

**Roadmap reference:**
[00-implementation-roadmap.md](00-implementation-roadmap.md). Design doc:
[08-catalog-reshape.md](08-catalog-reshape.md).

## Implementation worktree (batched with 16b + 16c)

Phase 7 ships as a single batched PR covering 16a + 16b + 16c. All work
goes in one worktree; this plan is the first to run (backend before
frontend). If the worktree doesn't exist yet, create it now:

```bash
git -C /home/john/elspeth worktree add .worktrees/phase-7-catalog -b feat/phase-7-catalog
cd /home/john/elspeth/.worktrees/phase-7-catalog
python3.13 -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

If it already exists (16b or 16c kicked off first), `cd
/home/john/elspeth/.worktrees/phase-7-catalog && source .venv/bin/activate`
and proceed.

**Order:** 16a → 16b → 16c. 16a and 16b can run in parallel after 16a Task 4 lands
(the API surface is then stable); 16c must wait for 16b.

**Discipline (operator-known gotchas):** activate `.venv` before any
`uv pip install` (installing without `--python` clobbers main's venv);
keep Python at 3.13 to match main (otherwise `enforce_tier_model.py`
reports ~300 spurious violations); when delegating to subagents, prefix
prompts with absolute paths and CWD discipline (subagents silently
misread the worktree path otherwise); prefer `mcp__filigree__*` over the
`filigree` CLI from inside the worktree.

**Final PR shape:** single PR from `feat/phase-7-catalog` → `RC5.2` once
all three plans land. See [16-phase-7-catalog-reshape.md](16-phase-7-catalog-reshape.md#implementation-worktree-batched)
for the full batch protocol.

---

## Scope boundaries

**In scope:**

- New class-attribute fields on `BaseSource`, `BaseTransform`, `BaseSink`
  in `src/elspeth/plugins/infrastructure/base.py`:
  - `usage_when_to_use: ClassVar[str | None] = None`
  - `usage_when_not_to_use: ClassVar[str | None] = None`
  - `example_use: ClassVar[str | None] = None`
  - `capability_tags: ClassVar[tuple[str, ...]] = ()`
  - `audit_characteristics: ClassVar[frozenset[str]] = frozenset()`
  - `data_trust_tier: ClassVar[int | None] = None`
- Extension of `PluginSummary` in `src/elspeth/web/catalog/schemas.py` to
  expose the new fields plus the *derived* audit characteristics (composed
  from declared `audit_characteristics` ∪ derived-from-determinism).
- Extension of `CatalogServiceImpl._to_summary` in
  `src/elspeth/web/catalog/service.py` to populate the new fields, with a
  helper `_derive_audit_characteristics(plugin_cls)` that does the
  derivation.
- One canonical filled-in plugin: `CSVSource` in
  `src/elspeth/plugins/sources/csv_source.py` gets full prose, tags, audit
  characteristics, and `data_trust_tier=3`. This is the **pattern example**;
  the full prose authoring for non-BOUNDARY plugins is an incremental task
  outside this phase's scope.
- `data_trust_tier = 3` on all 15 EXTERNAL_BOUNDARY plugins: all 6 sources
  (`csv`, `json`, `text`, `azure_blob`, `dataverse`, `null`) plus the 5
  named transform allowlist entries (`llm`, `web_scrape`, `rag_retrieval`,
  `azure_content_safety`, `azure_prompt_shield`) plus the 4 named sink
  allowlist entries (`azure_blob`, `chroma_sink`, `database`, `dataverse`),
  verified against the actual allowlists in
  `src/elspeth/web/audit_readiness/trust.py`. Non-BOUNDARY plugins keep the
  base-class default `data_trust_tier = None`.
- Deletion of `src/elspeth/web/audit_readiness/trust.py` in the same commit
  as the final plugin authoring (Task 6), per the binding No-Legacy
  commitment at `trust.py:31-35`. Replacement of `classify_plugin()` in
  `service.py` with direct `plugin_cls.data_trust_tier == 3` attribute
  lookup; `is_registered_plugin` inlined into `service.py`; `PluginKind`
  re-imported from `catalog/schemas.py`. Deletion of
  `tests/unit/web/audit_readiness/test_trust.py`.
- Test coverage for the new fields, the derivation logic, and a
  parametrized completeness guard asserting all BOUNDARY-class plugins
  have `data_trust_tier == 3`.

**Out of scope (handled in 16b or later phases):**

- Frontend catalog drawer reshape, filter chips, search extension,
  shortcuts regroup, synthetic "Inline data from chat" entry — all 16b.
- Per-plugin prose authoring for every plugin in the codebase (the
  `usage_when_to_use`, `usage_when_not_to_use`, `example_use`,
  `capability_tags`, and `audit_characteristics` fields on non-BOUNDARY
  plugins). That is an incremental documentation task. This plan ships the
  schema, the canonical CSV example (full prose), `data_trust_tier = 3` on
  all BOUNDARY plugins, and the trust.py deletion. Non-BOUNDARY plugins emit
  `None` / empty defaults; per design doc 08-§Risks: "Empty entries fall
  back to a generic 'see the technical description' message rather than
  blocking display."
- LLM-bootstrap prose generation. Open question D3 (per
  [00-implementation-roadmap.md §A](00-implementation-roadmap.md)) admits
  this as a possible future path; this plan does not include it.
- A backend "Inline data from chat" plugin. Reconnaissance confirmed no
  such plugin exists; per design doc 08-§"The 'Inline data from chat'
  entry," it is framed as "an option, not a plugin." 16b adds it as a
  synthetic frontend-only catalog entry.
- Schema-field-name search support (lazy-fetch conflict; flagged in 16b).
- Any UI changes whatsoever.

## Trust tier check (per CLAUDE.md)

The plugin metadata fields are **Tier 1**: they are system code, hand-
authored, type-checked, and committed alongside the plugin implementation.
There is no Tier 3 boundary on these — a CSV plugin's `usage_when_to_use`
string is part of the codebase, not external input.

The catalog API exposes these as part of `PluginSummary`, which is already
Tier 1 (`_StrictResponse` in `schemas.py` uses `strict=True,
extra="forbid"`). The Tier 1 guard already applies: a malformed plugin
attribute (wrong type, etc.) will crash at `PluginSummary` construction
time. No additional guards needed.

The *derived* audit characteristics merge a declared frozenset with
one inferred value. The inference is purely a function of the
class-level `determinism` attribute. If a plugin declares
`audit_characteristics={"signed"}` and its `determinism` is `IO_READ`,
the catalog service returns the sorted tuple `("io_read", "signed")`.
There is no Tier 3 input to this composition.

## File structure

**New files:** none. This phase extends existing files only.

**Modified:**

- `src/elspeth/plugins/infrastructure/base.py` — Add the six new
  class-attribute fields to all three of `BaseSource`, `BaseTransform`,
  `BaseSink`. Place them in the existing "Audit metadata" block adjacent
  to `determinism` / `plugin_version` / `source_file_hash`.
- `src/elspeth/web/catalog/schemas.py` — Extend `PluginSummary` with the
  six new optional fields. Keep `_StrictResponse` as the base; the new
  fields default to `None` / empty so existing emitters keep working.
- `src/elspeth/web/catalog/service.py` — Extend `_to_summary` to populate
  the new fields. Add `_derive_audit_characteristics(plugin_cls)` helper
  that composes declared + inferred characteristics.
- `src/elspeth/plugins/sources/csv_source.py` — Fill in the canonical
  example: `usage_when_to_use`, `usage_when_not_to_use`, `example_use`,
  `capability_tags`, `audit_characteristics`, `data_trust_tier=3`.
- `src/elspeth/plugins/sources/json_source.py`,
  `src/elspeth/plugins/sources/text_source.py`,
  `src/elspeth/plugins/sources/azure_blob_source.py`,
  `src/elspeth/plugins/sources/dataverse.py`,
  `src/elspeth/plugins/sources/null_source.py`,
  `src/elspeth/plugins/transforms/llm/transform.py`,
  `src/elspeth/plugins/transforms/web_scrape.py`,
  `src/elspeth/plugins/transforms/rag/transform.py`,
  `src/elspeth/plugins/transforms/azure/content_safety.py`,
  `src/elspeth/plugins/transforms/azure/prompt_shield.py`,
  `src/elspeth/plugins/sinks/azure_blob_sink.py`,
  `src/elspeth/plugins/sinks/chroma_sink.py`,
  `src/elspeth/plugins/sinks/database_sink.py`,
  `src/elspeth/plugins/sinks/dataverse.py` — Add `data_trust_tier:
  ClassVar[int | None] = 3` to each EXTERNAL_BOUNDARY plugin class.
- `src/elspeth/web/audit_readiness/service.py` — Replace
  `classify_plugin()` call with direct `plugin_cls.data_trust_tier == 3`
  lookup; inline `is_registered_plugin`; re-import `PluginKind` from
  `catalog/schemas.py`.

**Deleted:**

- `src/elspeth/web/audit_readiness/trust.py` — No-Legacy deletion; all
  callers updated in the same commit.
- `tests/unit/web/audit_readiness/test_trust.py` — Tests for deleted
  symbols; deleted with the module.

**Test additions:**

- `tests/unit/plugins/infrastructure/test_base_metadata.py` (new) — Verify
  the new class attributes exist on all three bases and default to
  `None` / empty.
- `tests/unit/web/catalog/test_schemas_extended.py` (new) — Verify
  `PluginSummary` accepts and round-trips the new fields.
- `tests/unit/web/catalog/test_service_audit_derivation.py` (new) —
  Verify `_derive_audit_characteristics` composition rules.
- `tests/unit/web/catalog/test_service.py` (modify if it exists, else
  create `test_service_extended_summary.py`) — Verify `_to_summary` emits
  the canonical CSV example's filled fields, and the `None` defaults for
  an unfilled plugin.
- `tests/unit/web/catalog/test_routes.py` (modify) — Verify the
  HTTP response shape includes the new fields. (Reconnaissance confirms
  this is the existing catalog-routes test module; there is no
  `tests/integration/web/catalog/test_catalog_routes.py`. The existing
  module is route-level, uses the in-process `TestClient`, and is
  conventionally treated as the wire-shape pin for `/api/catalog/*`.)
- `tests/unit/web/audit_readiness/test_boundary_attribute_parity.py`
  (new) — Parametrized completeness guard: every plugin previously
  classified BOUNDARY by `trust.py` must declare `data_trust_tier == 3`.
  Gates the trust.py deletion commit.
- `tests/unit/web/audit_readiness/test_trust.py` (delete) — Deleted with
  `trust.py`; tests exercised `classify_plugin` / `EXTERNAL_BOUNDARY_*`
  constants that no longer exist.
- `tests/unit/web/audit_readiness/test_service.py` (modify comments
  only) — Update inline citations from `classify_plugin("source", "csv")
  is BOUNDARY` to `CSVSource.data_trust_tier == 3`; behaviour under test
  is unchanged.

## Verification approach

Each task is TDD-shaped: failing test, minimal implementation, passing
test, commit. After all tasks land, the catalog API serves the extended
shape and the canonical CSV example is the one plugin with real reference
content. 16b will exercise the new shape end-to-end against the frontend.

## A note on derivation rules

The catalog API surfaces `audit_characteristics` as a **derived** set
combining:

1. The plugin's declared `audit_characteristics` frozenset (authoritative
   for traits the framework can't infer: `signed`, `network`, `credentials`,
   `provenance` for plugins that do extra provenance work beyond
   the standard pipeline).
2. **Inferred from `determinism`:**
   - `Determinism.IO_READ` → adds `"io_read"` to the set.
   - `Determinism.IO_WRITE` → adds `"io_write"`.
   - `Determinism.EXTERNAL_CALL` → adds `"external_call"` (the enum's
     `.value` verbatim; the frontend `auditCharacteristics.ts` translates
     this id to the user-facing label "Network call").
   - `Determinism.DETERMINISTIC` → adds `"deterministic"`.
   - `Determinism.SEEDED` → adds `"seeded"`.
   - `Determinism.NON_DETERMINISTIC` → adds `"non_deterministic"`.
3. **Source quarantine behaviour is declared, not inferred.**
   `_on_validation_failure` is an instance attribute set in each source's
   `__init__` from runtime config (verified at `csv_source.py:82` /
   `text_source.py:76` / `dataverse.py:209` — the class-level form is a
   bare type annotation with no value). Reading
   `plugin_cls._on_validation_failure` at catalog-build time would raise
   `AttributeError`. Authors must therefore declare `"quarantine"` in
   `audit_characteristics` themselves when their source supports
   non-discard quarantine routing. The CSV canonical example (Task 5)
   declares it explicitly.

This list is small, deliberate, and grows by declaration. The plan does
**not** try to infer `"provenance"` from "plugin uses Landscape" —
all plugins emit provenance through the standard pipeline, so the flag
would be uniformly true and uselessly noisy. Authors set it only when
their plugin does *additional* provenance work worth calling out.

Capability tags (e.g., `"text"`, `"file"`, `"http"`, `"csv"`) are
**purely authored**, not inferred. They drive the filter chips and search
in 16b.

---

## Task 1: Add new class attributes to the three plugin bases

**Files:**

- Modify: `src/elspeth/plugins/infrastructure/base.py` — add new fields
  to `BaseSource` (around line 1032 — the existing "Audit metadata"
  block), `BaseTransform` (around line 62), `BaseSink` (around line 681).
- Create: `tests/unit/plugins/infrastructure/test_base_metadata.py`.

- [ ] **Step 1: Identify the existing "Audit metadata" blocks**

Run:

```bash
grep -n "# Audit metadata" /home/john/elspeth/src/elspeth/plugins/infrastructure/base.py
```

Expected: three hits, one per base class. If fewer than three hits, the
block may be labeled differently on `BaseTransform` or `BaseSink`; in
that case, locate the `determinism:` declaration in each class — it sits
in the same block — and place the new fields immediately after.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/plugins/infrastructure/test_base_metadata.py`:

```python
"""Tests for the Phase-7A reference-content fields on plugin bases.

The new fields live on BaseSource / BaseTransform / BaseSink as class
attributes (matching the existing determinism / plugin_version /
source_file_hash precedent). They default to None or empty so existing
plugins keep working unchanged; authors fill them in when documenting a
plugin for the catalog's reference surface.
"""

from __future__ import annotations

from elspeth.plugins.infrastructure.base import BaseSink, BaseSource, BaseTransform


def test_base_source_has_reference_fields() -> None:
    assert BaseSource.usage_when_to_use is None
    assert BaseSource.usage_when_not_to_use is None
    assert BaseSource.example_use is None
    assert BaseSource.capability_tags == ()
    assert BaseSource.audit_characteristics == frozenset()
    assert BaseSource.data_trust_tier is None


def test_base_transform_has_reference_fields() -> None:
    assert BaseTransform.usage_when_to_use is None
    assert BaseTransform.usage_when_not_to_use is None
    assert BaseTransform.example_use is None
    assert BaseTransform.capability_tags == ()
    assert BaseTransform.audit_characteristics == frozenset()
    assert BaseTransform.data_trust_tier is None


def test_base_sink_has_reference_fields() -> None:
    assert BaseSink.usage_when_to_use is None
    assert BaseSink.usage_when_not_to_use is None
    assert BaseSink.example_use is None
    assert BaseSink.capability_tags == ()
    assert BaseSink.audit_characteristics == frozenset()
    assert BaseSink.data_trust_tier is None


def test_capability_tags_is_a_tuple_not_a_list() -> None:
    """Tuples are hashable and frozen — they should not be mutable list
    defaults that could surprise the next author."""
    assert isinstance(BaseSource.capability_tags, tuple)
    assert isinstance(BaseTransform.capability_tags, tuple)
    assert isinstance(BaseSink.capability_tags, tuple)


def test_audit_characteristics_is_a_frozenset() -> None:
    """Frozensets prevent accidental mutation of the class default and
    compose cleanly under set-union in the derivation logic."""
    assert isinstance(BaseSource.audit_characteristics, frozenset)
    assert isinstance(BaseTransform.audit_characteristics, frozenset)
    assert isinstance(BaseSink.audit_characteristics, frozenset)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/plugins/infrastructure/test_base_metadata.py -v
```

Expected: FAIL with `AttributeError: type object 'BaseSource' has no
attribute 'usage_when_to_use'` (and the equivalent on `BaseTransform`,
`BaseSink`).

- [ ] **Step 4: Add the fields to each base class**

In `src/elspeth/plugins/infrastructure/base.py`, immediately after the
existing `source_file_hash: str | None = None` declaration on each of
`BaseSource`, `BaseTransform`, and `BaseSink`, add:

```python
    # ── Reference content (Phase 7A) ────────────────────────────────────
    # These fields populate the catalog's reference cards. They are
    # documentation, not configuration — authors fill them in to explain
    # to a human reader (compliance, research, ops) what this plugin
    # does, when it's the right choice, when it isn't, and what audit
    # characteristics it has. Empty / None values render as a generic
    # "see the technical description" fallback in the catalog UI rather
    # than blocking display. See docs/composer/ux-redesign-2026-05/
    # 08-catalog-reshape.md for the per-field semantics and the
    # canonical csv_source.py example.

    usage_when_to_use: ClassVar[str | None] = None
    """Persona-facing prose. One short paragraph answering "when should I
    pick this plugin?" — written for compliance / research / ops readers,
    not for plugin developers. Avoid restating the technical
    description; that's what the docstring is for."""

    usage_when_not_to_use: ClassVar[str | None] = None
    """Persona-facing prose. One short paragraph answering "when should I
    *not* pick this plugin?" — gracefully redirecting users with the
    wrong shape of problem to the right plugin. The Marcus persona (per
    project_composer_personas) reads this to discover the plugin isn't
    a fit for his Zapier-shaped expectations."""

    example_use: ClassVar[str | None] = None
    """One-or-two-line YAML snippet showing realistic use. Format
    matches the pipeline YAML so a developer (Dev persona) can copy and
    paste into a composer session as a starting point. Indent under
    `source:` / `transform:` / `sink:` as appropriate for the plugin
    kind. Renders inside a <pre> block in the UI; preserve whitespace."""

    capability_tags: ClassVar[tuple[str, ...]] = ()
    """Short lowercase tags that drive catalog filter chips and fuzzy
    search. Examples: ("csv", "file", "batch") for csv_source;
    ("http", "network", "scraping") for a web-scrape transform. Tags
    are non-exhaustive; pick the two or three most useful for a user
    who is searching the catalog."""

    audit_characteristics: ClassVar[frozenset[str]] = frozenset()
    """Declared audit characteristics that the framework cannot derive
    from other attributes. The catalog service composes this set with
    the characteristic derived from `determinism` at summary-build time.
    Declare flags like 'signed', 'credentials', 'quarantine', 'provenance'
    (only for plugins that do extra provenance work beyond the standard
    pipeline). All tokens must be members of VALID_AUDIT_CHARACTERISTICS
    in catalog/service.py."""

    data_trust_tier: ClassVar[int | None] = None
    """Which tier of data this plugin handles at its boundary. 1 = our
    own data (audit, checkpoints); 2 = pipeline data (post-source); 3 =
    external data (source input, external API responses). Sources and
    external-call transforms = 3; pure row transforms = 2; sinks = 2
    (they emit, they don't ingest). See CLAUDE.md "Data Manifesto" for
    the tier definitions. Leave None to render as 'tier unspecified.'"""
```

The `ClassVar` import already exists on this file (it's used by the
existing `config_model: ClassVar[...]` declarations). If for some reason
it doesn't, add `from typing import ClassVar` to the existing typing
import block.

Place the new block in identical form on all three bases. The reason for
duplication rather than a shared mixin: the three bases already duplicate
`determinism` / `plugin_version` / `source_file_hash` for the same
reason (mixins fight with ABC + ClassVar + slots conventions used
elsewhere in the codebase). Match the existing precedent.

- [ ] **Step 5: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/plugins/infrastructure/test_base_metadata.py -v
```

Expected: PASS — all five tests green.

- [ ] **Step 6: Run the full plugin-infrastructure suite to catch regressions**

```bash
.venv/bin/python -m pytest tests/unit/plugins/infrastructure/ -v
```

Expected: PASS — no existing tests break. The new class attributes are
defaulted so existing plugins are unaffected.

- [ ] **Step 7: Commit**

```bash
git add tests/unit/plugins/infrastructure/test_base_metadata.py \
  src/elspeth/plugins/infrastructure/base.py
git commit -m "$(cat <<'EOF'
feat(plugins): add reference-content fields to plugin bases

Phase 7A.1 of composer UX redesign. Adds six new class-attribute fields
to BaseSource / BaseTransform / BaseSink: usage_when_to_use,
usage_when_not_to_use, example_use, capability_tags,
audit_characteristics, data_trust_tier. All default to None / empty so
existing plugins are unaffected; authors fill them in incrementally as
they document each plugin for the catalog's reference surface.

The catalog API will surface these in Task 3 of this phase; the
canonical csv_source example lands in Task 5. See
docs/composer/ux-redesign-2026-05/16a-phase-7a-backend.md and
docs/composer/ux-redesign-2026-05/08-catalog-reshape.md.
EOF
)"
```

- [ ] **Step 8: Mirror the same six fields onto all four protocol classes in `plugin_protocols.py`**

`contracts/plugin_protocols.py` is L0 — it must not import from anywhere above L0.
All types required for the new fields (`str | None`, `tuple[str, ...]`,
`frozenset[str]`, `int | None`) are stdlib, so no new imports are needed.

In `src/elspeth/contracts/plugin_protocols.py`, add the following block to
`SourceProtocol`, `TransformProtocol`, `BatchTransformProtocol`, and
`SinkProtocol`. Place it immediately after the existing `source_file_hash:
str | None` declaration in each class (matching the placement of the same
block on the base classes in Step 4 above):

```python
    # ── Reference content (Phase 7A) ────────────────────────────────────
    # Mirrors the fields added to BaseSource / BaseTransform / BaseSink.
    # Protocol declarations here let mypy verify that PluginClass-typed
    # variables in catalog/service.py can access these fields without
    # type: ignore suppressions. Types are stdlib-only (L0 constraint).
    usage_when_to_use: str | None
    usage_when_not_to_use: str | None
    example_use: str | None
    capability_tags: tuple[str, ...]
    audit_characteristics: frozenset[str]
    data_trust_tier: int | None
```

Note: protocol attribute declarations do not carry default values — that is
correct and expected. Defaults live on the base classes. The protocol
declares the shape; the base classes enforce the default.

Note: `BatchTransformProtocol` is the fourth protocol class and must
receive the same block. The brief mentioned "three protocols" but
`plugin_protocols.py` defines four independent declarations. Leaving
`BatchTransformProtocol` out would leave batch-transform members of the
`PluginClass` union without protocol coverage, perpetuating the exact gap
these steps are added to close.

- [ ] **Step 9: Write the failing test for protocol field presence**

Create `tests/unit/contracts/test_plugin_protocol_fields.py`:

```python
"""Tests that the Phase-7A reference-content fields are declared on all
four plugin protocol classes in contracts/plugin_protocols.py.

Uses protocol.__annotations__ rather than typing.get_type_hints() or
hasattr(). plugin_protocols.py defers context types (PluginSchema,
SourceContext, TransformContext, SinkContext, etc.) into TYPE_CHECKING
blocks; typing.get_type_hints() attempts to resolve every annotation as
a runtime expression and raises NameError on those forward references —
the test would error rather than assert, and would never pass in green
state. hasattr() is unconditionally banned (CLAUDE.md). SourceProtocol
and SinkProtocol are not @runtime_checkable, ruling out isinstance().
protocol.__annotations__ is runtime-safe: it returns the directly-
declared annotation dict on the class without resolving forward
references, which is all the test needs to assert field presence.
"""

from __future__ import annotations

from elspeth.contracts.plugin_protocols import (
    BatchTransformProtocol,
    SinkProtocol,
    SourceProtocol,
    TransformProtocol,
)

_PHASE_7A_FIELDS = {
    "usage_when_to_use",
    "usage_when_not_to_use",
    "example_use",
    "capability_tags",
    "audit_characteristics",
    "data_trust_tier",
}


def _assert_protocol_has_fields(protocol: type, protocol_name: str) -> None:
    hints = protocol.__annotations__
    missing = _PHASE_7A_FIELDS - hints.keys()
    assert not missing, (
        f"{protocol_name} is missing Phase-7A fields: {sorted(missing)}. "
        f"Add them to src/elspeth/contracts/plugin_protocols.py."
    )


def test_source_protocol_has_phase_7a_fields() -> None:
    _assert_protocol_has_fields(SourceProtocol, "SourceProtocol")


def test_transform_protocol_has_phase_7a_fields() -> None:
    _assert_protocol_has_fields(TransformProtocol, "TransformProtocol")


def test_batch_transform_protocol_has_phase_7a_fields() -> None:
    _assert_protocol_has_fields(BatchTransformProtocol, "BatchTransformProtocol")


def test_sink_protocol_has_phase_7a_fields() -> None:
    _assert_protocol_has_fields(SinkProtocol, "SinkProtocol")
```

**Why `protocol.__annotations__`**: `plugin_protocols.py` defers context
types (`PluginSchema`, `SourceContext`, `TransformContext`, `SinkContext`,
etc.) into `if TYPE_CHECKING:` blocks. `typing.get_type_hints()` attempts
to resolve every annotation as a runtime expression; it raises `NameError`
on those forward references, so the test would error rather than assert and
would never pass in green state. `hasattr()` is banned by CLAUDE.md.
`SourceProtocol` and `SinkProtocol` are not `@runtime_checkable`, ruling
out `isinstance()`. `protocol.__annotations__` is runtime-safe: it returns
the directly-declared annotation strings on the class without attempting
forward-reference resolution, which is exactly what the test needs.

- [ ] **Step 10: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/contracts/test_plugin_protocol_fields.py -v
```

Expected: FAIL — `AssertionError: SourceProtocol is missing Phase-7A
fields: ['audit_characteristics', 'capability_tags', ...]` (and likewise
for the other three protocols).

- [ ] **Step 11: Add the fields to all four protocol classes, then remove the `# type: ignore[attr-defined]` suppressions**

Apply the Step 8 block to all four protocols in
`src/elspeth/contracts/plugin_protocols.py`. Then open
`src/elspeth/web/catalog/service.py` and remove all seven
`# type: ignore[attr-defined]` comments:

- Two in `_derive_audit_characteristics`: on the `plugin_cls.audit_characteristics`
  and `plugin_cls.determinism` lines.
- Five in `_to_summary`: on the `plugin_cls.usage_when_to_use`,
  `plugin_cls.usage_when_not_to_use`, `plugin_cls.example_use`,
  `plugin_cls.capability_tags`, and `plugin_cls.data_trust_tier` lines.

**Hard requirement — `PluginClass` verification.** Before removing the
suppressions, verify the definition of `PluginClass` (the union type
parameter on `_to_summary`) in `service.py`. If `PluginClass` is defined
as a `Union` or `TypeAlias` that mypy cannot narrow to the four protocol
classes (for example, because it uses `type[Any]` or `type[object]`),
extending the protocols is necessary but not sufficient — mypy will still
report `attr-defined` errors. In that case, tighten `PluginClass`'s
definition so mypy can resolve the fields through the protocol declarations.
**Do not re-add `# type: ignore` comments as a substitute.** The suppressions
were temporary scaffolding; the protocols now carry the type information and
the suppressions have no place in the final implementation.

- [ ] **Step 12: Run tests to verify protocol fields pass, then run mypy to confirm suppressions are clean**

```bash
.venv/bin/python -m pytest tests/unit/contracts/test_plugin_protocol_fields.py -v
```

Expected: PASS — all four protocol tests green.

```bash
.venv/bin/python -m mypy \
  src/elspeth/contracts/plugin_protocols.py \
  src/elspeth/web/catalog/service.py
```

Expected: PASS — no `attr-defined` errors. If mypy still reports
`attr-defined` errors on `plugin_cls.*` accesses after the protocol
extension, the root cause is `PluginClass`'s definition (see Step 11 hard
requirement above); fix `PluginClass` rather than restoring suppressions.

```bash
git add src/elspeth/contracts/plugin_protocols.py \
  src/elspeth/web/catalog/service.py \
  tests/unit/contracts/test_plugin_protocol_fields.py
git commit -m "$(cat <<'EOF'
feat(contracts): extend plugin protocols with Phase-7A reference fields

Mirrors the six ClassVar fields from Phase 7A.1 onto all four protocol
classes (SourceProtocol, TransformProtocol, BatchTransformProtocol,
SinkProtocol) in contracts/plugin_protocols.py. Removes all seven
# type: ignore[attr-defined] suppressions from catalog/service.py that
were added as temporary scaffolding pending this protocol extension.

Protocol declarations carry no defaults (defaults live on the base
classes); the protocol declares the shape so mypy can verify PluginClass
members without suppressions.
EOF
)"
```

## Task 2: Extend `PluginSummary` schema with the new fields

**Files:**

- Modify: `src/elspeth/web/catalog/schemas.py` — add new optional fields
  to `PluginSummary`.
- Create: `tests/unit/web/catalog/test_schemas_extended.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/web/catalog/test_schemas_extended.py`:

```python
"""Tests for the Phase-7A extension of PluginSummary."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from elspeth.web.catalog.schemas import PluginSummary


def test_plugin_summary_accepts_all_new_fields() -> None:
    """A summary populated with every new field round-trips cleanly."""
    summary = PluginSummary(
        name="csv",
        description="Read rows from a CSV file.",
        plugin_type="source",
        config_fields=[],
        usage_when_to_use="When you have a reasonably large dataset already in a file.",
        usage_when_not_to_use="Small inline data — use 'Inline data from chat' instead.",
        example_use="source:\n  plugin: csv\n  options:\n    path: data/input.csv",
        capability_tags=("csv", "file", "batch"),
        audit_characteristics=("coerce", "io_read", "quarantine"),
        data_trust_tier=3,
    )
    assert summary.capability_tags == ("csv", "file", "batch")
    assert "io_read" in summary.audit_characteristics
    assert summary.data_trust_tier == 3


def test_plugin_summary_defaults_for_unfilled_plugin() -> None:
    """A summary with no reference content uses the documented defaults."""
    summary = PluginSummary(
        name="azure_blob",
        description="Read blobs from Azure storage.",
        plugin_type="source",
        config_fields=[],
    )
    assert summary.usage_when_to_use is None
    assert summary.usage_when_not_to_use is None
    assert summary.example_use is None
    assert summary.capability_tags == ()
    assert summary.audit_characteristics == ()
    assert summary.data_trust_tier is None


def test_plugin_summary_rejects_invalid_trust_tier() -> None:
    """Tier-1 strictness: data_trust_tier must be 1, 2, 3, or None."""
    with pytest.raises(ValidationError):
        PluginSummary(
            name="csv",
            description="...",
            plugin_type="source",
            config_fields=[],
            data_trust_tier=7,
        )


def test_plugin_summary_rejects_trust_tier_zero() -> None:
    """0 is below the ge=1 floor and must be rejected."""
    with pytest.raises(ValidationError):
        PluginSummary(
            name="csv",
            description="...",
            plugin_type="source",
            config_fields=[],
            data_trust_tier=0,
        )


def test_plugin_summary_rejects_negative_trust_tier() -> None:
    """-1 is below the ge=1 floor and must be rejected."""
    with pytest.raises(ValidationError):
        PluginSummary(
            name="csv",
            description="...",
            plugin_type="source",
            config_fields=[],
            data_trust_tier=-1,
        )


def test_plugin_summary_accepts_trust_tier_one() -> None:
    """1 is the lower bound (ge=1) and must be accepted."""
    summary = PluginSummary(
        name="csv",
        description="...",
        plugin_type="source",
        config_fields=[],
        data_trust_tier=1,
    )
    assert summary.data_trust_tier == 1


def test_plugin_summary_accepts_trust_tier_three() -> None:
    """3 is the upper bound (le=3) and must be accepted."""
    summary = PluginSummary(
        name="csv",
        description="...",
        plugin_type="source",
        config_fields=[],
        data_trust_tier=3,
    )
    assert summary.data_trust_tier == 3


def test_plugin_summary_rejects_extra_fields() -> None:
    """_StrictResponse uses extra='forbid'; unknown fields crash."""
    with pytest.raises(ValidationError):
        PluginSummary(
            name="csv",
            description="...",
            plugin_type="source",
            config_fields=[],
            mystery_field="surprise",  # type: ignore[call-arg]
        )


def test_audit_characteristics_serializes_as_list_for_json() -> None:
    """Pydantic emits tuple as a list in JSON. The derivation helper sorts
    before constructing the tuple, so the wire order is deterministic and
    matches `sorted(...)`; the frontend's `string[]` typing reads this
    directly."""
    summary = PluginSummary(
        name="csv",
        description="...",
        plugin_type="source",
        config_fields=[],
        audit_characteristics=("io_read", "quarantine"),
    )
    payload = summary.model_dump(mode="json")
    assert isinstance(payload["audit_characteristics"], list)
    assert payload["audit_characteristics"] == ["io_read", "quarantine"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/catalog/test_schemas_extended.py -v
```

Expected: most tests FAIL — `PluginSummary` doesn't accept the new fields;
pydantic raises `ValidationError` on unknown fields because of
`extra="forbid"`. Note: the S3 boundary rejection tests
(`test_plugin_summary_rejects_trust_tier_zero`,
`test_plugin_summary_rejects_negative_trust_tier`) pass in the red state
for the wrong reason (rejected as unknown fields); they provide real value
post-implementation by verifying the `ge=1, le=3` range constraints.

- [ ] **Step 3: Extend `PluginSummary` in `schemas.py`**

In `src/elspeth/web/catalog/schemas.py`, replace the existing
`PluginSummary` class with:

```python
class PluginSummary(_StrictResponse):
    """Lightweight plugin info for catalog browsing.

    Phase 7A adds reference-content fields (when-to-use prose, capability
    tags, audit-characteristic flags, data trust tier) so the catalog
    drawer can render persona-facing reference cards instead of a bare
    name+description. All new fields are optional and default to
    None / empty for plugins that haven't been authored yet; the
    frontend renders a fallback message rather than blocking display.

    `audit_characteristics` is the catalog service's *derived* set:
    declared characteristics from the plugin class composed with the
    characteristic inferred from `determinism`, then sorted into a
    deterministic tuple for stable wire-format ordering. Quarantine
    behaviour is author-declared (the `_on_validation_failure` signal
    is a per-instance attribute and cannot be inferred from the class
    object). See `CatalogServiceImpl._derive_audit_characteristics`.
    """

    name: str
    description: str
    plugin_type: PluginKind
    config_fields: list[ConfigFieldSummary]

    # Phase 7A reference-content fields
    usage_when_to_use: str | None = None
    usage_when_not_to_use: str | None = None
    example_use: str | None = None
    capability_tags: tuple[str, ...] = ()
    audit_characteristics: tuple[str, ...] = ()
    data_trust_tier: int | None = Field(default=None, ge=1, le=3)
```

Wire-format choice for `audit_characteristics`: the *plugin class*
declares its declared flags as a `frozenset[str]` (set semantics; dedup
under `|` composition). The *response model* exposes the composed set
as a **sorted tuple**. Two reasons:

1. `_StrictResponse` uses `strict=True` (Tier 1 emission discipline).
   Under strict mode, Pydantic v2 rejects list input where the field is
   typed `frozenset[str]` (verified). A `tuple[str, ...]` type round-
   trips cleanly through JSON (list ↔ tuple) and matches the frontend's
   `string[]` consumer type exactly.
2. Sets serialize with non-deterministic order. Sorting in
   `_derive_audit_characteristics` produces stable wire output, so the
   integration test and the frontend can both rely on a canonical order
   without an extra sort step at the consumer.

You will need to add `from pydantic import Field` to the imports at the
top of the file (currently the file imports `BaseModel, ConfigDict`).

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/web/catalog/test_schemas_extended.py -v
```

Expected: PASS — all nine tests green (five original + four `data_trust_tier` boundary tests).

- [ ] **Step 5: Run the full catalog-tests suite to catch regressions**

```bash
.venv/bin/python -m pytest tests/unit/web/catalog/ -v
```

Expected: PASS — existing tests do not break because the new fields are
optional with sensible defaults.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/catalog/schemas.py \
  tests/unit/web/catalog/test_schemas_extended.py
git commit -m "feat(web): extend PluginSummary with reference-content fields (Phase 7A.2)"
```

## Task 3: Audit-characteristic derivation helper

**Files:**

- Modify: `src/elspeth/web/catalog/service.py` — add
  `_derive_audit_characteristics(plugin_cls)` classmethod or module-level
  helper, plus per-`Determinism`-value mapping.
- Create: `tests/unit/web/catalog/test_service_audit_derivation.py`.

Prerequisite: Task 1 must be complete before running `test_all_plugin_audit_characteristics_are_valid`; the test accesses `cls.audit_characteristics` directly and will `AttributeError` on any plugin lacking the base-class default if Task 1 hasn't run.

This task isolates the derivation logic so it has its own test suite. The
next task wires it into `_to_summary`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/web/catalog/test_service_audit_derivation.py`:

```python
"""Tests for CatalogServiceImpl._derive_audit_characteristics.

The catalog service composes a plugin's declared audit_characteristics
frozenset with the characteristic inferred from determinism, sorted into
a deterministic tuple for stable wire ordering. Inference rules:

  Determinism.IO_READ            -> "io_read"
  Determinism.IO_WRITE           -> "io_write"
  Determinism.EXTERNAL_CALL      -> "external_call"
  Determinism.DETERMINISTIC      -> "deterministic"
  Determinism.SEEDED             -> "seeded"
  Determinism.NON_DETERMINISTIC  -> "non_deterministic"

Quarantine is author-declared, not inferred — `_on_validation_failure`
is a per-instance attribute set in each source's __init__ and does not
exist on the class object.
"""

from __future__ import annotations

from elspeth.contracts.enums import Determinism
from elspeth.web.catalog.service import (
    VALID_AUDIT_CHARACTERISTICS,
    _derive_audit_characteristics,
)


class _FakeSource:
    name = "fake_source"
    determinism = Determinism.IO_READ
    # quarantine is author-declared (not inferred from _on_validation_failure,
    # which is a per-instance attribute set in __init__).
    audit_characteristics = frozenset({"provenance", "quarantine"})


class _FakeSourceWithoutQuarantine:
    name = "fake_no_quarantine"
    determinism = Determinism.IO_READ
    audit_characteristics = frozenset()


class _FakeTransformWithNetwork:
    name = "fake_xfm"
    determinism = Determinism.EXTERNAL_CALL
    audit_characteristics = frozenset({"credentials"})


class _FakeTransformDeterministic:
    name = "fake_deterministic"
    determinism = Determinism.DETERMINISTIC
    audit_characteristics = frozenset()


class _FakeSink:
    name = "fake_sink"
    determinism = Determinism.IO_WRITE
    audit_characteristics = frozenset({"signed"})


class _FakeSeeded:
    name = "fake_seeded"
    determinism = Determinism.SEEDED
    audit_characteristics = frozenset()


class _FakeNonDeterministic:
    name = "fake_non_deterministic"
    determinism = Determinism.NON_DETERMINISTIC
    audit_characteristics = frozenset()


def test_source_declared_quarantine_passes_through() -> None:
    """`quarantine` is author-declared on the class. Composition preserves it."""
    derived = _derive_audit_characteristics(_FakeSource, plugin_kind="source")
    assert "quarantine" in derived  # declared by author
    assert "io_read" in derived  # inferred from determinism
    assert "provenance" in derived  # declared, preserved


def test_source_without_declared_quarantine_omits_it() -> None:
    """Source authors who don't declare quarantine don't get it in the
    derived set. Quarantine is author-declared in `audit_characteristics`;
    `_on_validation_failure` is per-instance and does not exist on the
    class object."""
    derived = _derive_audit_characteristics(
        _FakeSourceWithoutQuarantine, plugin_kind="source"
    )
    assert "quarantine" not in derived
    assert "io_read" in derived


def test_external_call_implies_external_call_flag() -> None:
    derived = _derive_audit_characteristics(
        _FakeTransformWithNetwork, plugin_kind="transform"
    )
    assert "external_call" in derived
    assert "credentials" in derived  # declared, preserved


def test_deterministic_transform() -> None:
    derived = _derive_audit_characteristics(
        _FakeTransformDeterministic, plugin_kind="transform"
    )
    assert "deterministic" in derived


def test_sink_io_write_inference() -> None:
    derived = _derive_audit_characteristics(_FakeSink, plugin_kind="sink")
    assert "io_write" in derived
    assert "signed" in derived  # declared, preserved


def test_transform_has_no_quarantine_inference() -> None:
    """quarantine is author-declared, not derived by the framework.
    Transforms don't quarantine at the boundary (they crash on type errors
    per the tier model); this test guards that no quarantine flag is
    injected for transforms."""
    derived = _derive_audit_characteristics(
        _FakeTransformWithNetwork, plugin_kind="transform"
    )
    assert "quarantine" not in derived


def test_quarantine_inference_is_not_attempted_from_instance_attribute() -> None:
    """Regression guard: the derivation MUST NOT read
    `plugin_cls._on_validation_failure`. That attribute is set in
    __init__ and does not exist on the class object — reading it would
    raise AttributeError at catalog-build time. The plan deliberately
    moved quarantine to author declaration."""
    # No `_on_validation_failure` attribute on this fixture; if the
    # implementation attempted to read it from the class, this would
    # AttributeError instead of returning a clean frozenset.
    class _NoInstanceAttr:
        name = "no_instance_attr"
        determinism = Determinism.IO_READ
        audit_characteristics = frozenset()

    derived = _derive_audit_characteristics(_NoInstanceAttr, plugin_kind="source")
    assert "io_read" in derived
    assert "quarantine" not in derived


def test_plugin_without_audit_characteristics_attr_does_not_crash() -> None:
    """Tier-1 plugin-attribute access on a plugin missing the declared
    audit_characteristics field should still work, because BaseSource /
    BaseTransform / BaseSink provide the default of frozenset()."""

    class _NoDeclaredChars:
        name = "no_declared"
        determinism = Determinism.DETERMINISTIC
        audit_characteristics = frozenset()  # the default from the base

    derived = _derive_audit_characteristics(_NoDeclaredChars, plugin_kind="transform")
    assert derived == ("deterministic",)


def test_returns_sorted_tuple() -> None:
    """Derivation returns a sorted tuple[str, ...] for stable wire-format
    ordering; the response model exposes this directly to the frontend."""
    derived = _derive_audit_characteristics(_FakeSource, plugin_kind="source")
    assert isinstance(derived, tuple)
    assert list(derived) == sorted(derived)


def test_seeded_implies_seeded_flag() -> None:
    """Determinism.SEEDED maps to the 'seeded' audit flag."""
    derived = _derive_audit_characteristics(_FakeSeeded, plugin_kind="transform")
    assert "seeded" in derived


def test_non_deterministic_implies_non_deterministic_flag() -> None:
    """Determinism.NON_DETERMINISTIC maps to the 'non_deterministic' audit flag."""
    derived = _derive_audit_characteristics(_FakeNonDeterministic, plugin_kind="transform")
    assert "non_deterministic" in derived


def test_determinism_to_audit_flag_covers_all_enum_values() -> None:
    """_DETERMINISM_TO_AUDIT_FLAG must be exhaustive over Determinism.

    If a new Determinism value is added to contracts/enums.py without a
    corresponding entry in _DETERMINISM_TO_AUDIT_FLAG, the subscript
    access in _derive_audit_characteristics raises KeyError at runtime.
    This test surfaces the gap at test time rather than at production
    catalog-build time.
    """
    from elspeth.web.catalog.service import _DETERMINISM_TO_AUDIT_FLAG

    assert set(_DETERMINISM_TO_AUDIT_FLAG.keys()) == set(Determinism)


def test_all_plugin_audit_characteristics_are_valid() -> None:
    """Every string in every plugin's audit_characteristics must belong
    to the closed audit-characteristic vocabulary maintained alongside
    08-catalog-reshape.md and codified as VALID_AUDIT_CHARACTERISTICS
    in catalog/service.py.

    This catches typos (e.g. 'io-read' instead of 'io_read') at CI time
    rather than letting them silently disappear from the rendered catalog
    card with no error raised. The test iterates all registered plugins
    (sources + transforms + sinks) so any new plugin with a misspelled
    characteristic fails CI immediately without any additional wiring.
    """
    from elspeth.plugins.infrastructure.manager import PluginManager

    manager = PluginManager()
    manager.register_builtin_plugins()

    all_plugin_classes = (
        list(manager.get_sources())
        + list(manager.get_transforms())
        + list(manager.get_sinks())
    )

    violations: list[str] = []
    for cls in all_plugin_classes:
        for token in cls.audit_characteristics:
            if token not in VALID_AUDIT_CHARACTERISTICS:
                violations.append(f"{cls.name}: {token!r} not in VALID_AUDIT_CHARACTERISTICS")

    assert not violations, "\n".join(violations)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/catalog/test_service_audit_derivation.py -v
```

Expected: FAIL — `ImportError: cannot import name 'VALID_AUDIT_CHARACTERISTICS'` (or `_derive_audit_characteristics` — whichever name is attempted first by the import collector).

- [ ] **Step 3: Implement the derivation helper**

In `src/elspeth/web/catalog/service.py`, add the following module-level
helper (place it near the top, after the existing module constants):

```python
from elspeth.contracts.enums import Determinism

# Map Determinism enum values to the audit-characteristic flag they
# imply. The catalog surfaces these as visual cues on the plugin card so
# a compliance-focused user (Linda persona) can see at a glance which
# audit traits apply without reading the technical description.
#
# Subscript access ([determinism]) is deliberate: if a future seventh
# Determinism value is added to contracts/enums.py without updating this
# table, the KeyError surfaces immediately at catalog-build time rather
# than silently returning None and dropping the inferred flag.
_DETERMINISM_TO_AUDIT_FLAG: dict[Determinism, str] = {
    Determinism.IO_READ: "io_read",
    Determinism.IO_WRITE: "io_write",
    Determinism.EXTERNAL_CALL: "external_call",
    Determinism.DETERMINISTIC: "deterministic",
    Determinism.SEEDED: "seeded",
    Determinism.NON_DETERMINISTIC: "non_deterministic",
}

# Closed vocabulary of valid audit-characteristic strings. Every token
# a plugin author places in `audit_characteristics: ClassVar[frozenset[str]]`
# must appear in this set. Typos (e.g. "io-read", "quarentine") fail CI
# via test_all_plugin_audit_characteristics_are_valid rather than
# silently disappearing from the rendered card. Extend this set together
# with 08-catalog-reshape.md when new visual cues are added to the UI.
#
# Members:
#   Determinism-derived (inferred by _derive_audit_characteristics):
#     io_read, io_write, external_call, deterministic, seeded, non_deterministic
#   Author-declared (from 08-catalog-reshape.md §"Audit-characteristic icons"):
#     provenance, retention, quarantine, coerce, signed, network, credentials
VALID_AUDIT_CHARACTERISTICS: frozenset[str] = frozenset({
    # Determinism-derived
    "io_read",
    "io_write",
    "external_call",
    "deterministic",
    "seeded",
    "non_deterministic",
    # Author-declared (08-catalog-reshape.md vocabulary)
    "provenance",
    "retention",
    "quarantine",
    "coerce",
    "signed",
    "network",
    "credentials",
})


def _derive_audit_characteristics(
    plugin_cls: type, *, plugin_kind: PluginKind
) -> tuple[str, ...]:
    """Compose declared + inferred audit characteristics for a plugin.

    The declared set comes from the plugin class's `audit_characteristics`
    attribute (defaulting to `frozenset()` on the base). The inferred
    set is derived purely from `determinism` (a class-level attribute on
    every plugin base): each Determinism value maps to a corresponding
    audit flag describing the plugin's reproducibility / side-effect
    surface.

    Quarantine behaviour is **author-declared, not inferred.** The
    source-quarantine signal lives in `_on_validation_failure`, which is
    set per-instance in `__init__` from runtime config — it does not
    exist on the class object. Reading `plugin_cls._on_validation_failure`
    here would AttributeError at catalog-build time. Sources whose
    runtime configuration supports non-discard quarantine routing
    declare `"quarantine"` in their `audit_characteristics` frozenset
    (the CSV canonical example does this).

    `plugin_kind` is retained in the signature to keep the boundary
    explicit and to allow future per-kind inferences without a signature
    change. It is unused in the current implementation.

    Direct attribute access (`plugin_cls.audit_characteristics`,
    `plugin_cls.determinism`) is correct here: the bases and protocols
    declare these with sensible defaults, so every plugin reachable via
    the catalog has them. A plugin without these attributes would be a
    malformed system plugin (Tier 1 bug); crash via AttributeError is
    the correct response, not defensive fallback.
    """
    del plugin_kind  # reserved for future per-kind inferences
    declared: frozenset[str] = plugin_cls.audit_characteristics
    determinism = plugin_cls.determinism

    # Subscript raises KeyError if a future Determinism value is added
    # to contracts/enums.py without updating _DETERMINISM_TO_AUDIT_FLAG.
    # That crash is correct: silent None return would drop the inferred
    # flag with no test failure and no audit-trail signal.
    inferred: frozenset[str] = frozenset({_DETERMINISM_TO_AUDIT_FLAG[determinism]})

    # Sort for stable wire-format ordering; the response model exposes
    # this as a tuple[str, ...] consumed by the frontend as string[].
    return tuple(sorted(declared | inferred))
```

**Why no quarantine inference.** Reconnaissance against the source tree
confirmed `_on_validation_failure` is set in each source's `__init__`
from runtime config (`csv_source.py:82` is a bare type annotation;
`text_source.py:76` and `dataverse.py:209` are instance assignments).
The attribute does not exist on the class object — reading
`plugin_cls._on_validation_failure` would `AttributeError` at
catalog-build time. A `getattr(plugin_cls, "_on_validation_failure", None)`
fallback would technically "work" by always returning `None`, which
would silently drop the inference rule for every real source — a
defective derivation masquerading as success, which is exactly the kind
of "silent pass-through that destroys the audit trail" CLAUDE.md
forbids. The plan therefore moves quarantine to author declaration; the
canonical CSV example (Task 5) declares it explicitly.

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/web/catalog/test_service_audit_derivation.py -v
```

Expected: PASS — all 13 tests green.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/catalog/service.py \
  tests/unit/web/catalog/test_service_audit_derivation.py
git commit -m "feat(web): derive audit characteristics from plugin metadata (Phase 7A.3)

- Enum-exhaustive _DETERMINISM_TO_AUDIT_FLAG lookup: subscript [determinism]
  crashes on unknown future values rather than silently returning None.
- VALID_AUDIT_CHARACTERISTICS allowlist (13 members) enforced by
  test_all_plugin_audit_characteristics_are_valid; typos in plugin
  declarations now fail CI rather than disappearing silently.
- Full Determinism coverage: SEEDED and NON_DETERMINISTIC test fixtures added."
```

## Task 4: Wire derivation into `_to_summary` and emit all new fields

**Files:**

- Modify: `src/elspeth/web/catalog/service.py` — `_to_summary` now
  populates the new `PluginSummary` fields.
- Create: `tests/unit/web/catalog/test_service_extended_summary.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/web/catalog/test_service_extended_summary.py`:

```python
"""Tests for CatalogServiceImpl._to_summary covering Phase-7A fields.

This file uses a hand-rolled PluginManager fake so the test doesn't
depend on the real plugin registry (which evolves). The fakes mimic
just enough of the protocol surface for _to_summary to work.
"""

from __future__ import annotations

from typing import Any, ClassVar

from elspeth.contracts.enums import Determinism
from elspeth.web.catalog.service import CatalogServiceImpl


class _BareTransform:
    """A transform with no reference-content fields filled in.

    Mimics the post-Phase-7A.1 baseline: defaults exist but no author
    has populated them yet. The catalog summary should round-trip the
    defaults as-is.
    """

    name = "bare"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = None
    usage_when_to_use: ClassVar[str | None] = None
    usage_when_not_to_use: ClassVar[str | None] = None
    example_use: ClassVar[str | None] = None
    capability_tags: ClassVar[tuple[str, ...]] = ()
    audit_characteristics: ClassVar[frozenset[str]] = frozenset()
    data_trust_tier: ClassVar[int | None] = None
    config_model = None
    is_batch_aware = False

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def get_config_model(cls) -> Any:
        return None


class _FilledSource:
    """A source with every reference-content field filled in.

    Mimics csv_source.py post-Phase-7A.5 — the canonical example.
    """

    name = "filled"
    determinism = Determinism.IO_READ
    plugin_version = "1.0.0"
    source_file_hash: str | None = None
    usage_when_to_use: ClassVar[str | None] = "When you have a CSV file."
    usage_when_not_to_use: ClassVar[str | None] = "When the data is inline; use chat instead."
    example_use: ClassVar[str | None] = "source:\n  plugin: filled"
    capability_tags: ClassVar[tuple[str, ...]] = ("file", "csv")
    # Declare both "coerce" (Tier-3 boundary trait) and "quarantine"
    # (runtime quarantine routing) explicitly; the catalog service does
    # not infer either, because `_on_validation_failure` is per-instance.
    audit_characteristics: ClassVar[frozenset[str]] = frozenset({"coerce", "quarantine"})
    data_trust_tier: ClassVar[int | None] = 3
    config_model = None

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def get_config_model(cls) -> Any:
        return None


class _FakePluginManager:
    def __init__(self, sources, transforms, sinks):
        self._sources = sources
        self._transforms = transforms
        self._sinks = sinks

    def get_sources(self):
        return self._sources

    def get_transforms(self):
        return self._transforms

    def get_sinks(self):
        return self._sinks


def test_bare_plugin_summary_uses_defaults() -> None:
    pm = _FakePluginManager(sources=[], transforms=[_BareTransform], sinks=[])
    svc = CatalogServiceImpl(pm)  # type: ignore[arg-type]
    summaries = svc.list_transforms()
    assert len(summaries) == 1
    s = summaries[0]
    assert s.usage_when_to_use is None
    assert s.usage_when_not_to_use is None
    assert s.example_use is None
    assert s.capability_tags == ()
    # Derived: DETERMINISTIC -> {"deterministic"}; no declared chars.
    # The response model exposes the composed set as a sorted tuple.
    assert s.audit_characteristics == ("deterministic",)
    assert s.data_trust_tier is None


def test_filled_source_summary_propagates_all_fields() -> None:
    pm = _FakePluginManager(sources=[_FilledSource], transforms=[], sinks=[])
    svc = CatalogServiceImpl(pm)  # type: ignore[arg-type]
    summaries = svc.list_sources()
    assert len(summaries) == 1
    s = summaries[0]
    assert s.usage_when_to_use == "When you have a CSV file."
    assert s.usage_when_not_to_use == "When the data is inline; use chat instead."
    assert s.example_use == "source:\n  plugin: filled"
    assert s.capability_tags == ("file", "csv")
    # Composed: declared {"coerce", "quarantine"} + inferred {"io_read"}.
    # quarantine is author-declared, not inferred from instance state.
    assert "coerce" in s.audit_characteristics
    assert "io_read" in s.audit_characteristics
    assert "quarantine" in s.audit_characteristics
    assert s.data_trust_tier == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/catalog/test_service_extended_summary.py -v
```

Expected: FAIL — `_to_summary` doesn't yet populate the new fields, so
the assertions on `usage_when_to_use`, etc., return `None` even for
the filled fake, and `audit_characteristics` is the empty frozenset
(rather than the composed set).

**Note on the test fakes:** `CatalogServiceImpl.__init__` eagerly calls
`_populate_schema_cache → _build_schema_info → _knob_schema →
validate_knob_schema`. The fakes above return `{}` for
`get_config_schema()` and `None` for `get_config_model()`, which should
make `_knob_schema` return `{"fields": []}` and `validate_knob_schema`
pass cleanly. If the validator rejects the empty schema (e.g., requires
a `plugin_kind` / `plugin_name` shape it can't infer from the fake),
swap to a minimal real plugin class as the fixture — `NullSource` from
`elspeth.plugins.sources.null_source` is the smallest existing source;
for the transform/sink cases, pick the simplest registered transform /
sink. The test is exercising `_to_summary`, not the schema validator,
so any plugin class that passes `__init__` is acceptable.

- [ ] **Step 3: Extend `_to_summary` to populate the new fields**

In `src/elspeth/web/catalog/service.py`, replace the existing
`_to_summary` method with:

```python
    def _to_summary(self, plugin_cls: PluginClass, plugin_type: PluginKind) -> PluginSummary:
        """Convert a plugin class to a PluginSummary.

        Phase 7A: also emits reference-content fields. Audit
        characteristics are the *derived* set: declared chars from
        `audit_characteristics` composed with the flag derived from
        `determinism`. The frontend reads audit_characteristics as a
        flat list of flag strings.
        """
        name: str = plugin_cls.name
        description = get_plugin_description(plugin_cls)
        schema = self._catalog_schema(plugin_cls, plugin_type)
        config_fields = self._extract_config_fields(schema)

        # Direct attribute access: the bases and protocols declare every
        # Phase-7A field with a default. A plugin missing them would be
        # a malformed system plugin (Tier 1 bug); crash is correct.
        usage_when_to_use = plugin_cls.usage_when_to_use
        usage_when_not_to_use = plugin_cls.usage_when_not_to_use
        example_use = plugin_cls.example_use
        capability_tags = plugin_cls.capability_tags
        data_trust_tier = plugin_cls.data_trust_tier

        audit_characteristics = _derive_audit_characteristics(
            plugin_cls, plugin_kind=plugin_type
        )

        return PluginSummary(
            name=name,
            description=description,
            plugin_type=plugin_type,
            config_fields=config_fields,
            usage_when_to_use=usage_when_to_use,
            usage_when_not_to_use=usage_when_not_to_use,
            example_use=example_use,
            capability_tags=capability_tags,
            audit_characteristics=audit_characteristics,
            data_trust_tier=data_trust_tier,
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/web/catalog/test_service_extended_summary.py -v
```

Expected: PASS — both tests green.

- [ ] **Step 5: Run the full catalog suite for regressions**

```bash
.venv/bin/python -m pytest tests/unit/web/catalog/ -v
```

Expected: PASS — existing catalog tests do not break. The new fields
appear on every summary; unfilled plugins get `None` / empty defaults
and a derived audit-characteristics set from their `determinism`.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/catalog/service.py \
  tests/unit/web/catalog/test_service_extended_summary.py
git commit -m "feat(web): emit reference-content fields in PluginSummary (Phase 7A.4)"
```

## Task 5: Canonical example — fill in `csv_source.py`

**Files:**

- Modify: `src/elspeth/plugins/sources/csv_source.py` — add the Phase-7A
  reference fields with hand-authored content.

This is the **canonical prose example** for the reference-content fields.
The remaining 14 EXTERNAL_BOUNDARY plugins receive `data_trust_tier = 3`
(without full prose authoring) in Task 6. Per-plugin prose authoring for
non-BOUNDARY plugins is an incremental task outside this phase's scope.

- [ ] **Step 1: Read the existing `csv_source.py` end-to-end**

```bash
.venv/bin/cat /home/john/elspeth/src/elspeth/plugins/sources/csv_source.py
```

Note the class structure: `class CSVSource(BaseSource):` with class-
attribute declarations near the top of the class body
(`name = "csv"`, `plugin_version = ...`, `source_file_hash = ...`,
`config_model = CSVSourceConfig`). The new fields slot in adjacent.

- [ ] **Step 2: Write the failing test**

Add a new test file: `tests/unit/plugins/sources/test_csv_source_metadata.py`:

```python
"""Tests asserting csv_source.py is the canonical Phase-7A example."""

from __future__ import annotations

from elspeth.plugins.sources.csv_source import CSVSource


def test_csv_source_has_when_to_use() -> None:
    assert CSVSource.usage_when_to_use is not None
    # Sanity-check that the prose is actually content, not a stub.
    assert len(CSVSource.usage_when_to_use) > 40


def test_csv_source_has_when_not_to_use() -> None:
    assert CSVSource.usage_when_not_to_use is not None
    assert len(CSVSource.usage_when_not_to_use) > 40


def test_csv_source_has_example_use() -> None:
    assert CSVSource.example_use is not None
    assert "csv" in CSVSource.example_use


def test_csv_source_has_capability_tags() -> None:
    tags = CSVSource.capability_tags
    assert "csv" in tags
    assert "file" in tags


def test_csv_source_data_trust_tier_is_three() -> None:
    """Sources surface Tier 3 (external) data at their boundary."""
    assert CSVSource.data_trust_tier == 3


def test_csv_source_declared_audit_characteristics_includes_coerce() -> None:
    """CSV source coerces external string data to typed columns per
    Tier-3 boundary rules; that's a notable audit trait worth declaring."""
    assert "coerce" in CSVSource.audit_characteristics
```

- [ ] **Step 3: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/plugins/sources/test_csv_source_metadata.py -v
```

Expected: FAIL — `usage_when_to_use is None` (the field exists but
hasn't been authored on `CSVSource`).

- [ ] **Step 4: Fill in the canonical example**

In `src/elspeth/plugins/sources/csv_source.py`, inside the `CSVSource`
class body, immediately after the existing class-attribute block
(`name = "csv"`, `plugin_version`, `source_file_hash`, `config_model`),
add:

```python
    # ── Reference content (Phase 7A canonical example) ──────────────────
    # This block is the canonical pattern for future plugin authors.
    # Copy this shape; replace the prose with your plugin's specifics.
    # The catalog drawer renders these fields as a persona-facing
    # reference card. Empty / None entries fall back to "see the technical
    # description" rather than blocking display — but the goal for every
    # plugin is to have these filled in eventually so the catalog is
    # useful as orientation material (per docs/composer/ux-redesign-2026-05/
    # 08-catalog-reshape.md).

    usage_when_to_use: ClassVar[str | None] = (
        "A reasonably large dataset (more than ~20 rows) that already "
        "exists as a CSV file. The source validates and coerces types "
        "at the boundary and quarantines malformed rows to a sink so the "
        "rest of the pipeline keeps running on the clean rows."
    )

    usage_when_not_to_use: ClassVar[str | None] = (
        "Small inline data — type it into chat instead (the composer "
        "creates a one-row source from your message). Streaming data — "
        "CSV is batch-only; no row is emitted until the full file is "
        "read. Data that arrives over HTTP — fetch it first, then point "
        "the CSV source at the downloaded file."
    )

    example_use: ClassVar[str | None] = (
        "source:\n"
        "  plugin: csv\n"
        "  options:\n"
        "    path: data/input.csv\n"
        "    on_validation_failure: quarantine"
    )

    capability_tags: ClassVar[tuple[str, ...]] = ("csv", "file", "batch", "tabular")

    audit_characteristics: ClassVar[frozenset[str]] = frozenset(
        {"coerce", "quarantine"}
    )
    # "io_read" is *inferred* by the catalog service from
    # determinism=IO_READ. "coerce" and "quarantine" are declared here:
    #   - "coerce" describes the CSV source's Tier-3 boundary behaviour
    #     (string cells -> typed columns) and cannot be inferred from
    #     determinism alone.
    #   - "quarantine" describes the runtime behaviour configured via
    #     `on_validation_failure`. The catalog service cannot infer this
    #     from the class because `_on_validation_failure` is a
    #     per-instance attribute set in `__init__`, not a class
    #     attribute. Authors of sources that support non-discard
    #     quarantine routing must declare `"quarantine"` themselves.

    data_trust_tier: ClassVar[int | None] = 3
    # Sources surface Tier 3 (external) data at their boundary. See
    # CLAUDE.md "Data Manifesto" for the tier definitions.
```

You will need `from typing import ClassVar` already imported (the file
imports `from typing import Any`; add `ClassVar` to that import line).

- [ ] **Step 5: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/plugins/sources/test_csv_source_metadata.py -v
```

Expected: PASS — all six tests green.

- [ ] **Step 6: Run the broader plugin tests for regression**

```bash
.venv/bin/python -m pytest tests/unit/plugins/sources/ -v
```

Expected: PASS — the new metadata fields are additive class attributes;
the source's behaviour is unchanged.

- [ ] **Step 7: Verify the end-to-end via the catalog API**

```bash
.venv/bin/python -m pytest tests/unit/web/catalog/test_routes.py -v -k csv
```

Expected: PASS — and the response payload for the CSV plugin should
include the new fields. Existing CSV-asserting tests live in
`TestListSources::test_csv_source_present` etc.; extend their coverage in
the next step.

- [ ] **Step 8: Add a route test pinning the wire shape**

In `tests/unit/web/catalog/test_routes.py`, add (alongside the existing
`TestListSources` class — note the fixture name is `client`, not
`client_with_user`):

```python
def test_csv_source_summary_includes_reference_content(self, client: TestClient) -> None:
    """Wire-shape pin: catalog API returns canonical CSV reference content."""
    resp = client.get("/api/catalog/sources")
    assert resp.status_code == 200
    sources = resp.json()
    csv = next(s for s in sources if s["name"] == "csv")
    assert csv["usage_when_to_use"] is not None
    assert "tabular" in csv["capability_tags"]
    assert "io_read" in csv["audit_characteristics"]  # inferred from determinism
    assert "coerce" in csv["audit_characteristics"]  # author-declared
    assert "quarantine" in csv["audit_characteristics"]  # author-declared
    assert csv["data_trust_tier"] == 3
```

The existing `client` fixture in `test_routes.py` already mounts the
`/api/catalog` router via the in-process `SyncASGITestClient`; no
additional setup is needed.

- [ ] **Step 9: Run the new route test**

```bash
.venv/bin/python -m pytest tests/unit/web/catalog/test_routes.py -v
```

Expected: PASS — the wire shape now exposes the reference content for
CSV; other plugins still have `None` / empty values, which is correct.

- [ ] **Step 10: Commit**

```bash
git add tests/unit/plugins/sources/test_csv_source_metadata.py \
  src/elspeth/plugins/sources/csv_source.py \
  tests/unit/web/catalog/test_routes.py
git commit -m "$(cat <<'EOF'
docs(plugins): fill canonical Phase-7A reference content on csv_source

Phase 7A.5 of composer UX redesign. csv_source.py becomes the canonical
example for the new reference-content fields: usage_when_to_use,
usage_when_not_to_use, example_use, capability_tags, declared
audit_characteristics, and data_trust_tier=3. Non-BOUNDARY plugins keep
their prose defaults (None / empty); per-plugin prose authoring is an
incremental task outside this phase's scope, with empty entries falling
back to a generic 'see the technical description' message rather than
blocking display. All EXTERNAL_BOUNDARY plugins receive data_trust_tier=3
in the subsequent Task 6 commit, which also deletes trust.py.

Tags an end-to-end integration test pinning the wire shape so the
frontend (Phase 7B) can rely on the new fields being present and
correctly composed (declared 'coerce' + inferred 'io_read' +
declared 'quarantine'). See
docs/composer/ux-redesign-2026-05/16a-phase-7a-backend.md.
EOF
)"
```

## Task 6: Author `data_trust_tier = 3` on all EXTERNAL_BOUNDARY plugins; delete `trust.py`

**Goal:** Discharge the No-Legacy deletion commitment at
`src/elspeth/web/audit_readiness/trust.py:31-35`. Task 5 establishes
the canonical pattern on `CSVSource`; this task propagates
`data_trust_tier = 3` to the remaining 14 EXTERNAL_BOUNDARY plugins,
verifies completeness, then deletes `trust.py` and replaces its sole
caller in `service.py` with a direct attribute lookup — all in one
commit.

**Why this is in scope:** Phase 7A introduces `data_trust_tier:
ClassVar` on the plugin bases. The `trust.py` module docstring carries
a binding commitment: "When Phase 7 adds `data_trust_tier: ClassVar` to
plugin base classes, delete this module entirely and replace
`classify_plugin()` callers with direct attribute lookup. CLAUDE.md
No-Legacy requires same-commit replacement." Phase 7A is that event.
Leaving `trust.py` alive after Phase 7A ships a parallel source of
truth: `PluginSummary.data_trust_tier` (from the plugin class attribute)
and `classify_plugin()` (from the module allowlists) would diverge on
any new plugin add, silently breaking the panel.

**EXTERNAL_BOUNDARY plugin list (verified against `trust.py` at
`src/elspeth/web/audit_readiness/trust.py:61-80`):**

Sources — all sources are BOUNDARY unconditionally (trust.py line
117-118 has no allowlist for sources; `if kind == "source": return
PluginTrust.BOUNDARY`). All 6 registered sources get `data_trust_tier
= 3`:

| Plugin name | Class | File |
|---|---|---|
| `csv` | `CSVSource` | `plugins/sources/csv_source.py` ← already done in Task 5 |
| `json` | `JSONSource` | `plugins/sources/json_source.py` |
| `text` | `TextSource` | `plugins/sources/text_source.py` |
| `azure_blob` | `AzureBlobSource` | `plugins/sources/azure_blob_source.py` |
| `dataverse` | `DataverseSource` | `plugins/sources/dataverse.py` |
| `null` | `NullSource` | `plugins/sources/null_source.py` |

Transforms — only the named allowlist entries get `data_trust_tier = 3`;
internal transforms keep `data_trust_tier = None` (the base default):

| Plugin name | Class | File |
|---|---|---|
| `llm` | `LLMTransform` | `plugins/transforms/llm/transform.py` |
| `web_scrape` | `WebScrapeTransform` | `plugins/transforms/web_scrape.py` |
| `rag_retrieval` | `RAGRetrievalTransform` | `plugins/transforms/rag/transform.py` |
| `azure_content_safety` | `AzureContentSafety` | `plugins/transforms/azure/content_safety.py` |
| `azure_prompt_shield` | `AzurePromptShield` | `plugins/transforms/azure/prompt_shield.py` |

Sinks — only the named allowlist entries get `data_trust_tier = 3`;
internal sinks keep `data_trust_tier = None`:

| Plugin name | Class | File |
|---|---|---|
| `azure_blob` | `AzureBlobSink` | `plugins/sinks/azure_blob_sink.py` |
| `chroma_sink` | `ChromaSink` | `plugins/sinks/chroma_sink.py` |
| `database` | `DatabaseSink` | `plugins/sinks/database_sink.py` |
| `dataverse` | `DataverseSink` | `plugins/sinks/dataverse.py` |

**Files:**

- Modify (14 plugin source files): one `data_trust_tier: ClassVar[int |
  None] = 3` declaration each (plus `ClassVar` added to the typing
  import if not already present) — see table above. `CSVSource` is
  already done by Task 5.
- Modify: `src/elspeth/web/audit_readiness/service.py` — replace the
  `classify_plugin()` call with a direct `plugin_cls.data_trust_tier ==
  3` check; inline `is_registered_plugin` logic; update imports.
- Delete: `src/elspeth/web/audit_readiness/trust.py`
- Delete: `tests/unit/web/audit_readiness/test_trust.py` (its tests
  exercised `classify_plugin` / `EXTERNAL_BOUNDARY_TRANSFORMS` /
  `EXTERNAL_BOUNDARY_SINKS`; those symbols no longer exist after
  deletion).
- Modify: `tests/unit/web/audit_readiness/test_service.py` — the four
  tests referencing `classify_plugin` or "boundary" behaviour remain
  valid; update their inline comments from "classify_plugin('source',
  'csv') is BOUNDARY" to "CSVSource.data_trust_tier == 3" so comments
  reflect the new mechanism. Behaviour under test is unchanged.
- New: `tests/unit/web/audit_readiness/test_boundary_attribute_parity.py`
  — completeness guard (see Step 2 below).

**Step 1: Read the failing evidence**

Before writing any code, confirm the commitment is live:

```bash
grep -n 'deletion commitment' \
  src/elspeth/web/audit_readiness/trust.py
```

Expected: line 31 prints the "Phase 7 deletion commitment" docstring.
This is the gate condition: Task 5's ClassVar now exists → deletion is
required.

**Step 2: Write the failing completeness test**

Create `tests/unit/web/audit_readiness/test_boundary_attribute_parity.py`:

```python
"""Completeness guard: every plugin previously classified BOUNDARY by
trust.py must declare data_trust_tier == 3 now that trust.py is gone.

This test is written BEFORE deleting trust.py. It fails red until all
14 remaining EXTERNAL_BOUNDARY plugins have data_trust_tier = 3.
Once green, trust.py and test_trust.py can be deleted in the same commit.

Do NOT use getattr() to read data_trust_tier — the field is a ClassVar
declared on BaseSource / BaseTransform / BaseSink; direct attribute
access is correct and will AttributeError loudly on missing fields,
which is the desired behaviour.
"""

from __future__ import annotations

import pytest

# Sources — all sources are BOUNDARY unconditionally per trust.py:117-118.
from elspeth.plugins.sources.azure_blob_source import AzureBlobSource
from elspeth.plugins.sources.csv_source import CSVSource
from elspeth.plugins.sources.dataverse import DataverseSource
from elspeth.plugins.sources.json_source import JSONSource
from elspeth.plugins.sources.null_source import NullSource
from elspeth.plugins.sources.text_source import TextSource

# Transforms — named EXTERNAL_BOUNDARY_TRANSFORMS allowlist from trust.py:61-69.
from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety
from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield
from elspeth.plugins.transforms.llm.transform import LLMTransform
from elspeth.plugins.transforms.rag.transform import RAGRetrievalTransform
from elspeth.plugins.transforms.web_scrape import WebScrapeTransform

# Sinks — named EXTERNAL_BOUNDARY_SINKS allowlist from trust.py:73-80.
from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink
from elspeth.plugins.sinks.chroma_sink import ChromaSink
from elspeth.plugins.sinks.database_sink import DatabaseSink
from elspeth.plugins.sinks.dataverse import DataverseSink

_BOUNDARY_PLUGIN_CLASSES = [
    # Sources
    CSVSource,
    JSONSource,
    TextSource,
    AzureBlobSource,
    DataverseSource,
    NullSource,
    # Transforms
    LLMTransform,
    WebScrapeTransform,
    RAGRetrievalTransform,
    AzureContentSafety,
    AzurePromptShield,
    # Sinks
    AzureBlobSink,
    ChromaSink,
    DatabaseSink,
    DataverseSink,
]


@pytest.mark.parametrize("plugin_cls", _BOUNDARY_PLUGIN_CLASSES, ids=lambda c: c.name)
def test_boundary_plugin_has_data_trust_tier_three(plugin_cls) -> None:
    """Every previously-BOUNDARY plugin must declare data_trust_tier == 3.

    This test gates the trust.py deletion in Task 6. It must be fully
    green before the deletion commit proceeds.
    """
    assert plugin_cls.data_trust_tier == 3, (
        f"{plugin_cls.__name__} (name={plugin_cls.name!r}) has "
        f"data_trust_tier={plugin_cls.data_trust_tier!r}; expected 3. "
        f"Author 'data_trust_tier: ClassVar[int | None] = 3' in the class body."
    )
```

**Step 3: Run test to verify it fails red**

```bash
.venv/bin/python -m pytest \
  tests/unit/web/audit_readiness/test_boundary_attribute_parity.py -v
```

Expected: 14 FAIL — all plugins except `CSVSource` (already done in
Task 5) fail because `data_trust_tier` is `None`. This confirms the
test is exercising the right thing.

**Step 4: Add `data_trust_tier = 3` to the 14 remaining plugins**

For each plugin file in the table above (excluding `csv_source.py`),
locate the class body's existing class-attribute block (the `name = ...`
declaration, `plugin_version`, and similar fields are near the top of
each class). Add immediately after the last existing class attribute:

```python
    data_trust_tier: ClassVar[int | None] = 3
    # Crosses a Tier-3 external boundary. See CLAUDE.md "Data Manifesto"
    # for tier definitions. Declaration required for trust.py deletion
    # per Phase 7A No-Legacy commitment (trust.py:31-35).
```

If `ClassVar` is not already imported in the file, add it:

- For files that already `from typing import ...`, add `ClassVar` to
  that import line.
- For files that have no `typing` import (rare), add
  `from typing import ClassVar` near the top of the import block.

Work through the 14 files. Each is a small targeted addition; do not
modify any other code in these files.

**Step 5: Run test to verify it passes green**

```bash
.venv/bin/python -m pytest \
  tests/unit/web/audit_readiness/test_boundary_attribute_parity.py -v
```

Expected: 15 PASS (including `CSVSource` from Task 5). Zero failures.
This is the gate condition for the deletion in the next step.

**Step 6: Replace `classify_plugin()` in `service.py` with direct attribute lookup**

The logic in `_build_plugin_trust_row` (`service.py:160-216`) currently
calls `classify_plugin(kind, name)` to determine if a plugin is
BOUNDARY. After this change it reads `plugin_cls.data_trust_tier == 3`
from the live catalog class directly.

The replacement requires:

1. **Resolve the plugin class** for each composition node (source /
   transform / sink) using the existing `PluginManager` or via the
   CatalogService already wired into `ReadinessService`. The simplest
   approach that avoids introducing a new dependency: inline a minimal
   `_get_plugin_class(kind, name)` helper that loads from `PluginManager`
   (mirroring the existing `_registered_plugin_names` pattern in
   trust.py, but without caching — or with a module-level
   `@lru_cache`).

2. **Inline `is_registered_plugin`** into `service.py`. The function is
   a 6-line helper (trust.py:97-106); it has no other callers. Copy the
   implementation verbatim, including the inner `_registered_plugin_names`
   `lru_cache`, renaming to module-private `_is_registered_plugin`. This
   keeps the L3 architecture intact (no new imports from outside
   `audit_readiness/`).

3. **Update the `_record` closure** (service.py:167-174): replace

   ```python
   trust = classify_plugin(kind, name)
   if trust is PluginTrust.BOUNDARY:
       boundary.append((kind, component_id, name))
   ```

   with a direct attribute check:

   ```python
   plugin_cls = _get_plugin_class_for_kind(kind, name)
   if plugin_cls.data_trust_tier == 3:
       boundary.append((kind, component_id, name))
   ```

   where `_get_plugin_class_for_kind` is a small helper:

   ```python
   def _get_plugin_class_for_kind(kind: PluginKind, name: str) -> type[BaseSource] | type[BaseTransform] | type[BaseSink]:
       """Return the registered plugin class for (kind, name).

       Raises StopIteration when the name is not in the catalog — caller
       must guard with _is_registered_plugin() first (as _record() does).
       Layer: L3. Called only after is_registered_plugin() confirms the
       name is present.
       """
       from elspeth.plugins.infrastructure.manager import PluginManager
       manager = PluginManager()
       manager.register_builtin_plugins()
       if kind == "source":
           return next(cls for cls in manager.get_sources() if cls.name == name)
       if kind == "transform":
           return next(cls for cls in manager.get_transforms() if cls.name == name)
       if kind == "sink":
           return next(cls for cls in manager.get_sinks() if cls.name == name)
       raise ValueError(f"unknown plugin kind: {kind!r}")
   ```

   Note: `PluginManager()` is called per-check here. If this is a
   performance concern at call volume, promote `_get_plugin_class_for_kind`
   to use the same `@lru_cache` pattern as the existing
   `_registered_plugin_names` stub in trust.py. Correctness first;
   optimise only if profiling shows a hot path.

4. **Update the import block** in `service.py`. Replace:

   ```python
   from elspeth.web.audit_readiness.trust import (
       PluginKind,
       PluginTrust,
       classify_plugin,
       is_registered_plugin,
   )
   ```

   with:

   ```python
   from elspeth.web.catalog.schemas import PluginKind
   ```

   `PluginTrust` is no longer needed (the comparison is now `== 3`, not
   `is PluginTrust.BOUNDARY`). `classify_plugin` and
   `is_registered_plugin` are deleted from trust.py; the inlined
   replacements live in service.py itself. `PluginKind` is re-imported
   from `catalog/schemas.py` where the canonical definition lives (it is
   already defined there at `schemas.py:16`, making `trust.py` the
   duplicate).

   Also add the necessary base-class imports for the return type
   annotation of `_get_plugin_class_for_kind`:

   ```python
   from elspeth.plugins.infrastructure.base import BaseSink, BaseSource, BaseTransform
   ```

   These may already be imported in service.py (check before adding).

**Step 7: Run the audit-readiness service tests to verify behaviour is preserved**

```bash
.venv/bin/python -m pytest \
  tests/unit/web/audit_readiness/test_service.py -v
```

Expected: PASS — the four plugin-trust-row tests pass unchanged. The
behaviour is identical: sources always produce a BOUNDARY result (their
`data_trust_tier` is 3); named transforms/sinks produce BOUNDARY; others
produce INTERNAL. The test comments may still reference
"classify_plugin" as a citation; update them to reference
"data_trust_tier == 3" so they don't describe a deleted function.

**Step 8: Delete `trust.py` and `test_trust.py`**

```bash
git rm src/elspeth/web/audit_readiness/trust.py
git rm tests/unit/web/audit_readiness/test_trust.py
```

After deletion, run:

```bash
grep -rn 'from elspeth.web.audit_readiness.trust' src/ tests/
```

Expected: 0 hits. If any remain, fix them before committing.

**Step 9: Run the full audit-readiness test suite**

```bash
.venv/bin/python -m pytest tests/unit/web/audit_readiness/ -v
```

Expected: PASS — `test_trust.py` is gone (deleted in Step 8);
`test_service.py` and `test_boundary_attribute_parity.py` both pass.

**Step 10: Run mypy on the modified files**

```bash
.venv/bin/python -m mypy \
  src/elspeth/web/audit_readiness/service.py \
  src/elspeth/plugins/sources/json_source.py \
  src/elspeth/plugins/sources/text_source.py \
  src/elspeth/plugins/sources/azure_blob_source.py \
  src/elspeth/plugins/sources/dataverse.py \
  src/elspeth/plugins/sources/null_source.py \
  src/elspeth/plugins/transforms/llm/transform.py \
  src/elspeth/plugins/transforms/web_scrape.py \
  src/elspeth/plugins/transforms/rag/transform.py \
  src/elspeth/plugins/transforms/azure/content_safety.py \
  src/elspeth/plugins/transforms/azure/prompt_shield.py \
  src/elspeth/plugins/sinks/azure_blob_sink.py \
  src/elspeth/plugins/sinks/chroma_sink.py \
  src/elspeth/plugins/sinks/database_sink.py \
  src/elspeth/plugins/sinks/dataverse.py
```

Expected: PASS — no new type errors. The `ClassVar[int | None] = 3`
declarations type-check cleanly against the base-class defaults.

**Step 11: Commit**

This commit is a single atomic unit: all 14 plugin additions + service.py
replacement + trust.py deletion + test_trust.py deletion + new
test_boundary_attribute_parity.py. Do not split across multiple commits.

```bash
git add \
  src/elspeth/plugins/sources/json_source.py \
  src/elspeth/plugins/sources/text_source.py \
  src/elspeth/plugins/sources/azure_blob_source.py \
  src/elspeth/plugins/sources/dataverse.py \
  src/elspeth/plugins/sources/null_source.py \
  src/elspeth/plugins/transforms/llm/transform.py \
  src/elspeth/plugins/transforms/web_scrape.py \
  src/elspeth/plugins/transforms/rag/transform.py \
  src/elspeth/plugins/transforms/azure/content_safety.py \
  src/elspeth/plugins/transforms/azure/prompt_shield.py \
  src/elspeth/plugins/sinks/azure_blob_sink.py \
  src/elspeth/plugins/sinks/chroma_sink.py \
  src/elspeth/plugins/sinks/database_sink.py \
  src/elspeth/plugins/sinks/dataverse.py \
  src/elspeth/web/audit_readiness/service.py \
  tests/unit/web/audit_readiness/test_boundary_attribute_parity.py \
  tests/unit/web/audit_readiness/test_service.py
git commit -m "$(cat <<'EOF'
refactor(audit-readiness): discharge trust.py No-Legacy commitment (Phase 7A)

Phase 7A.6 of composer UX redesign. trust.py:31-35 carries a binding
commitment to delete the module when data_trust_tier: ClassVar lands on
plugin base classes. That class attribute now exists (Task 1); this
commit discharges the commitment.

- Author data_trust_tier = 3 on all 14 remaining EXTERNAL_BOUNDARY
  plugins (5 sources + 5 transforms + 4 sinks; CSVSource was Task 5).
  Sources: json, text, azure_blob (source), dataverse (source), null.
  Transforms: llm, web_scrape, rag_retrieval, azure_content_safety,
  azure_prompt_shield. Sinks: azure_blob, chroma_sink, database,
  dataverse.
- Replace classify_plugin() in audit_readiness/service.py with direct
  plugin_cls.data_trust_tier == 3 attribute lookup. PluginKind is now
  imported from catalog/schemas.py (canonical definition). PluginTrust
  and classify_plugin are gone with trust.py.
- Delete src/elspeth/web/audit_readiness/trust.py — the parallel source
  of truth is eliminated.
- Delete tests/unit/web/audit_readiness/test_trust.py — tests for
  deleted symbols.
- Add tests/unit/web/audit_readiness/test_boundary_attribute_parity.py —
  parametrized completeness guard asserting every previously-BOUNDARY
  plugin has data_trust_tier == 3.

CLAUDE.md No-Legacy: no shims, no compatibility wrappers, no deferred
deletion. All call sites updated in this commit.

See docs/composer/ux-redesign-2026-05/16a-phase-7a-backend.md Task 6.
EOF
)"
```

---

## Task 7: Final regression sweep

- [ ] **Step 1: Run the full unit suite**

```bash
.venv/bin/python -m pytest tests/unit/ -x
```

Expected: PASS. If any unrelated tests fail, diagnose before declaring
this phase done; do not skip-or-suppress per CLAUDE.md "Fix errors you
encounter."

- [ ] **Step 2: Run the full web integration suite**

```bash
.venv/bin/python -m pytest tests/integration/web/ -x
```

Expected: PASS.

- [ ] **Step 3: Run mypy on the touched files**

```bash
.venv/bin/python -m mypy \
  src/elspeth/contracts/plugin_protocols.py \
  src/elspeth/plugins/infrastructure/base.py \
  src/elspeth/web/catalog/schemas.py \
  src/elspeth/web/catalog/service.py \
  src/elspeth/plugins/sources/csv_source.py \
  src/elspeth/web/audit_readiness/service.py
```

Expected: PASS — no new type errors. All seven `# type: ignore[attr-defined]`
suppressions were removed from `service.py` in Task 1 Steps 10–11, when
the same six fields were added to all four protocol classes in
`contracts/plugin_protocols.py`. Adding `plugin_protocols.py` to this
sweep means CI will catch any future drift between base-class fields and
protocol fields immediately, without needing a separate enforcement gate.

- [ ] **Step 4: Run ruff on the touched files**

```bash
.venv/bin/python -m ruff check \
  src/elspeth/plugins/infrastructure/base.py \
  src/elspeth/web/catalog/schemas.py \
  src/elspeth/web/catalog/service.py \
  src/elspeth/plugins/sources/csv_source.py \
  src/elspeth/web/audit_readiness/service.py \
  tests/unit/plugins/infrastructure/test_base_metadata.py \
  tests/unit/web/catalog/test_schemas_extended.py \
  tests/unit/web/catalog/test_service_audit_derivation.py \
  tests/unit/web/catalog/test_service_extended_summary.py \
  tests/unit/plugins/sources/test_csv_source_metadata.py \
  tests/unit/web/audit_readiness/test_boundary_attribute_parity.py \
  tests/unit/web/audit_readiness/test_service.py
```

Expected: PASS.

- [ ] **Step 5: Run the tier-model enforcement script**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check \
  --root src/elspeth --allowlist config/cicd/enforce_tier_model
```

Expected: PASS — no new defensive patterns introduced. The
`getattr` / `try-except` usage was deliberately avoided; the helper
uses direct attribute access throughout.

---

## What Phase 7A leaves the backend in

After all seven tasks land:

- `BaseSource`, `BaseTransform`, `BaseSink` carry six new optional
  class-attribute fields for reference content.
- The catalog API surface (`/api/catalog/{sources,transforms,sinks}`)
  emits the new fields on every `PluginSummary`. Unfilled plugins emit
  `None` / empty defaults; the canonical CSV example emits full
  authored content plus the derived audit-characteristic set.
- The audit-characteristic derivation rules are codified in
  `_derive_audit_characteristics` and have their own focused test
  suite.
- All 15 EXTERNAL_BOUNDARY plugins (all 6 sources + 5 named transforms
  + 4 named sinks, verified against the `trust.py` allowlists) declare
  `data_trust_tier = 3`. The No-Legacy deletion commitment in
  `trust.py:31-35` is discharged.
- `src/elspeth/web/audit_readiness/trust.py` is deleted; the parallel
  source of truth between `PluginSummary.data_trust_tier` and
  `classify_plugin()` is eliminated. `_build_plugin_trust_row` in
  `service.py` reads `plugin_cls.data_trust_tier == 3` directly.
- No frontend code has changed; the new fields are visible only to
  callers of the catalog API (curl, the upcoming Phase 7B reshape).

Phase 7B picks up from here: frontend card layout, filter chips, search
extension, "Inline data from chat" synthetic entry, shortcuts regroup.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Plugin authors don't write any prose, leaving every plugin's card empty | Per design doc 08-§Risks, empty entries fall back to "see the technical description" — non-blocking. The canonical CSV example sets voice and tone for incremental authoring. |
| The derivation rules surprise authors ("why does my source show `io_read`?") | The `_DETERMINISM_TO_AUDIT_FLAG` table is documented in this plan and in the service.py docstring. Authors reading the catalog source can see the derivation rules. |
| `data_trust_tier=3` for sources reads as "this is a bad / untrusted plugin" | The tooltip on the badge (16b) frames it as "what tier of data flows through this boundary," matching CLAUDE.md "Data Manifesto" language. The badge is descriptive, not pejorative. |
| Extending `PluginSummary` breaks existing JSON consumers | `_StrictResponse` uses `extra="forbid"` on the *parser* side, not the emitter. Pydantic emitting new fields doesn't break consumers that ignore unknown fields (the default JS behaviour); only consumers using strict-parse on the response would fail, and those don't exist outside the codebase. |
| Frozenset serialization order is non-deterministic | The frontend reads `audit_characteristics` as an unordered set of strings; rendering sorts them for stable display in 16b. The integration test asserts membership, not order. |
| Future plugin author copies the CSV example verbatim and forgets to update the prose | Code review catches this. The example_use field's YAML preview in the catalog drawer will look obviously CSV-shaped for a non-CSV plugin, surfacing the copy-paste error to the reviewer. |
| `is_registered_plugin` is inlined into `service.py` at Task 6; future callers wanting this helper must either duplicate it or extract it to a shared module | If a second caller appears, extract to `audit_readiness/utils.py` at that point. For now one caller is one definition. |
| `null_source` is assigned `data_trust_tier = 3` for mechanical equivalence with trust.py (which classifies all sources as BOUNDARY unconditionally) | Reviewer should confirm this is semantically correct. A `NullSource` emits no rows and makes no external calls; if the project decides BOUNDARY is wrong for it, change `data_trust_tier = None` in the same Task 6 commit. No downstream functional impact today; the panel correctly emits "no BOUNDARY plugins" when no source is set. |

## Memory references

- `feedback_catalog_is_reference_not_toolkit` — the design call this implements.
- `project_composer_personas` — informs the persona-facing voice of the
  prose fields (the canonical CSV example writes for Linda /
  Sarah / Marcus / Dev simultaneously).
- `feedback_no_calendar_shipping_commitments` — no calendar commitments
  in this plan.

## Review history

- 2026-05-15 reality-review: Fixed `Determinism.EXTERNAL_CALL` derived value from `"network"` to `"external_call"` (enum `.value` verbatim; frontend translates to "Network call" label); applied corrected `tests/integration/web/catalog/test_catalog_routes.py` path (was missing the `catalog/` subdirectory).
- 2026-05-16 reality-review (rev-2): Removed the `quarantine` inference rule from `_derive_audit_characteristics` — `_on_validation_failure` is a per-instance attribute set in each source's `__init__` (verified at `csv_source.py:82`, `text_source.py:76`, `dataverse.py:209`), not a class attribute. Reading `plugin_cls._on_validation_failure` at catalog-build time would `AttributeError`; a `getattr` fallback would silently mask the rule on every real source. Quarantine is now author-declared in `audit_characteristics`; the canonical CSV example declares `frozenset({"coerce", "quarantine"})`. Corrected the integration-test path: `tests/integration/web/catalog/test_catalog_routes.py` does not exist; the canonical wire-shape test module is `tests/unit/web/catalog/test_routes.py` and its fixture is named `client`, not `client_with_user`.
- 2026-05-17 OD1-Option-A expansion: Expanded scope to discharge the No-Legacy commitment in `trust.py:31-35`. Task 5 now authors `data_trust_tier=3` on all 15 EXTERNAL_BOUNDARY plugins (6 sources + 5 transforms + 4 sinks), not just `csv_source`. New Task 6 deletes `trust.py` and replaces `classify_plugin()` callers in `service.py` with direct `plugin_cls.data_trust_tier == 3` attribute lookup in the same commit; `is_registered_plugin` is inlined into `service.py` and `PluginKind` is re-imported from `catalog/schemas.py`. `tests/unit/web/audit_readiness/test_trust.py` is deleted in the same commit. Regression sweep renumbered from Task 6 to Task 7. Reviewer note: `null_source` gets `data_trust_tier=3` for mechanical equivalence with trust.py (which classifies all sources as BOUNDARY unconditionally); if this is semantically incorrect, it should be changed in the same commit with an explicit comment.
- 2026-05-17 Round-2 hardening (M2 + S2 + S3 + Round-1-fixup): (M2-A) Replaced `_DETERMINISM_TO_AUDIT_FLAG.get(determinism)` with a subscript `[determinism]` in the proposed `_derive_audit_characteristics` implementation; collapsed the now-dead `if flag is not None` guard; added exhaustivity test `test_determinism_to_audit_flag_covers_all_enum_values` asserting `set(map.keys()) == set(Determinism)`. (M2-B) Added `VALID_AUDIT_CHARACTERISTICS: frozenset[str]` constant (13 members drawn from the 08-catalog-reshape.md vocabulary plus the 6 Determinism-derived strings) and a membership test `test_all_plugin_audit_characteristics_are_valid` asserting every declared string across all registered plugins is in the allowlist; patterned after `test_every_external_call_plugin_is_on_allowlist_or_explicitly_excepted` in `test_trust.py`. (S2) Added `test_seeded_implies_seeded_flag` and `test_non_deterministic_implies_non_deterministic_flag` to close coverage gaps on `Determinism.SEEDED` and `Determinism.NON_DETERMINISTIC`. (S3) Added four boundary tests for `data_trust_tier` in `test_schemas_extended.py`: `=0` rejected, `=-1` rejected, `=1` accepted, `=3` accepted; Task 2 Step 4 expected count updated from "all five" to "all nine"; Task 3 Step 4 expected count updated from "all 8" to "all 13". Corrected test fixture `"emits_provenance"` → `"provenance"` to match 08-catalog-reshape.md authoritative vocabulary. (Round-1-fixup) Task 6 `_get_plugin_class_for_kind` docstring corrected: "Raises `KeyError`" → "Raises `StopIteration`" to match the `next(generator)` implementation.
- 2026-05-17 S1 + Round-2 carry-overs + dangling-reference cleanup: (S1) Extended all four protocol classes in `src/elspeth/contracts/plugin_protocols.py` (`SourceProtocol`, `TransformProtocol`, `BatchTransformProtocol`, `SinkProtocol`) with the same six Phase-7A ClassVar fields added to the base classes in Task 1 (`usage_when_to_use`, `usage_when_not_to_use`, `example_use`, `capability_tags`, `audit_characteristics`, `data_trust_tier`). Protocol extension mirrors the existing `determinism` / `plugin_version` / `source_file_hash` precedent. Added Task 1 Steps 8–12 covering: protocol field addition, a `typing.get_type_hints()`-based failing test asserting protocol field presence (chose `get_type_hints` over `isinstance` because `SourceProtocol` and `SinkProtocol` are not `@runtime_checkable`; `hasattr` is banned), removal of all seven `# type: ignore[attr-defined]` suppressions from `service.py` (five in `_to_summary`, two in `_derive_audit_characteristics`), and a hard requirement to verify `PluginClass`'s union definition in `service.py` — if protocol extension alone does not close the mypy gap, the implementer must fix `PluginClass`'s typing in the same commit rather than re-adding suppressions. Brief stated five suppressions; actual count is seven (confirmed by grep). Added `src/elspeth/contracts/plugin_protocols.py` to the Task 7 Step 3 mypy file list; updated Task 7 Step 3 prose from conditional "remove if not required" to present-tense "removed in Task 1 Steps 10–11." (MINOR-2) Added prerequisite note after Task 3 Files block: Task 1 must be complete before running `test_all_plugin_audit_characteristics_are_valid`. (NIT-1) Tightened Task 2 Step 2 red-test description to acknowledge that the two S3 boundary rejection tests pass in the red state for the wrong reason. (Dangling-ref) Rewrote the `test_all_plugin_audit_characteristics_are_valid` docstring to stand on its own without naming `test_trust.py` (which Task 6 deletes).
- 2026-05-17 Round-3.5 BLOCKER fix: Step 9 test replaced `typing.get_type_hints(protocol)` with `protocol.__annotations__` and removed the now-dead `import typing`. `typing.get_type_hints()` raises `NameError` at runtime against all four protocol classes because `plugin_protocols.py` defers context types (`PluginSchema`, `SourceContext`, `TransformContext`, etc.) into `if TYPE_CHECKING:` blocks; `get_type_hints()` attempts to resolve every annotation as a runtime expression and fails on the first unresolved forward reference. The red-state test would have errored with `NameError`, not the documented `AssertionError`, and the green-state test would never have passed. `protocol.__annotations__` is runtime-safe: it returns the directly-declared annotation strings on the class without attempting forward-reference resolution, which is all the test needs to assert field presence. Updated the Step 9 rationale paragraph to state this plainly.
- 2026-05-17 Round-4 textual cleanup (S4 + S5): (S4) Swept all stale "inferred-from-quarantine" / "on-validation-failure handling" prose. Nine sites corrected: (1) `**Goal:**` preamble — `"(determinism, validation-failure handling)"` → `"(determinism)"`; (2) `**Architecture:**` — dropped `"and the source quarantine setting"`; (3) Scope bullets union — dropped `"∪ inferred-from-quarantine"`; (4) `_derive_audit_characteristics` scope bullet — `"inference"` → `"derivation"`; (5) `test_source_without_declared_quarantine_omits_it` docstring — replaced "don't get it inferred / Inference from …" with factual author-declared framing; (6) `test_transform_has_no_quarantine_inference` docstring — replaced "source-only inference" with "author-declared, not derived"; (7) `_to_summary` docstring — replaced `"inferred from determinism and (for sources) on-validation-failure handling"` with `"derived from determinism"`; (8) Risks table row — swapped `quarantine` (now author-declared, so not a derivation surprise) for `io_read` as the example, and `"inference rules"` → `"derivation rules"`; (9) `audit_characteristics` base-class docstring — replaced `"inferred characteristics (from determinism, validation-failure handling)"` with `"characteristic derived from determinism"`, and added `'quarantine'` to the declare-flags examples since it is now author-declared. Rev-history entries left intact. (S5) Replaced `TestSources` → `TestListSources` in Task 5 Steps 7 and 8 (2 occurrences); verified against actual `tests/unit/web/catalog/test_routes.py` class names (`TestListSources`, `TestListTransforms`, `TestListSinks`) — no other mismatches found in the plan.

- 2026-05-18 Worktree batch protocol added: Added the `## Implementation worktree (batched with [siblings])` section near the top of this plan documenting the shared `.worktrees/phase-7-catalog` worktree, the execution order in the batch, the operator-known gotchas (venv leak / Python 3.13 / subagent CWD / filigree CLI), and the single-PR shipping shape. See `16-phase-7-catalog-reshape.md` for the canonical batch protocol.
- **2026-05-18 implementation complete**: All 7 tasks landed on `feat/phase-7-catalog`. 13 commits ahead of RC5.2: `7accf6bb1` (Phase 7A.0 mypy paydown prelude), `1aefdab31` (Phase 7A.0.1 prelude amendment for env drift), `2975efbe2` (Task 1.1 base-class fields), `1221c1c28` (Task 1.2 protocols + ClassVar drop), `d23386e34` (Task 2 PluginSummary extension), `b49a164de` (Task 3 derivation helper), `36be249a8` (Task 4 _to_summary wire-up), `3f963c7ac` (Task 5 canonical CSV reference content), `76df27a21` (Task 6 trust.py discharge), `d8968ccb1` (Task 6 fix: drift guards + cache + negative coverage), `1d6cbdef0` (Task 7 fix: TransformProtocol fake), `f008ba505` (post-review type hardening: DataTrustTier Literal + AuditCharacteristic StrEnum + 3 comment fixes). Per-plan PR-toolkit review verdict: zero BLOCKERs, zero unaddressed MAJORs after the post-review hardening commit. Test coverage analysis, silent-failure hunt, and general code review all approved on first pass. Type-design review found 4 MAJORs (all addressed in `f008ba505` — DataTrustTier Literal alias, AuditCharacteristic StrEnum, type aliases for wire-format asymmetry, sink-docstring contradiction fixed). Comment-accuracy review found 2 MAJORs (both addressed in `f008ba505` — test_explain.py docstring inaccuracy, test_boundary_attribute_parity.py import-section comment). Final unit-test count: 16053 passing. `mypy src/elspeth` clean across 420 source files. One pre-existing failure flagged (NOT 7A's fault): `test_happy_trivial_prompt_under_production_byte_envelope` (202237 vs 200000 byte envelope) reproduces identically on the merge-base — surfaced for operator awareness, not in scope for this plan.

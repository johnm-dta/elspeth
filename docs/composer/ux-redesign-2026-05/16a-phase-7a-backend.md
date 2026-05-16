# Phase 7A — Backend: Plugin metadata schema extension + catalog API surface

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the backend half of Phase 7 — extend the plugin base classes
(`BaseSource` / `BaseTransform` / `BaseSink`) with new optional class
attributes for reference content (when-to-use prose, when-not-to-use prose,
example use snippet, capability tags, audit-characteristic flags, data
trust tier), extend the catalog API to surface these fields, derive the
audit-characteristic flags that can be inferred from existing plugin
attributes (determinism, validation-failure handling), and fill in **one**
canonical example (`csv_source.py`) so future authors have a pattern to
copy. Frontend wiring is in the companion plan,
[16b-phase-7b-frontend.md](16b-phase-7b-frontend.md).

**Architecture:** Schema-then-derivation-then-API. The new fields are
class-attribute level (matching the existing `determinism` /
`plugin_version` / `source_file_hash` precedent on `BaseSource`). The
catalog service derives the audit characteristics by composing the
plugin's declared `audit_characteristics` set with characteristics that
can be inferred from `determinism` and the source quarantine setting. The
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
  from declared `audit_characteristics` ∪ inferred-from-determinism ∪
  inferred-from-quarantine).
- Extension of `CatalogServiceImpl._to_summary` in
  `src/elspeth/web/catalog/service.py` to populate the new fields, with a
  helper `_derive_audit_characteristics(plugin_cls)` that does the
  inference.
- One canonical filled-in plugin: `CSVSource` in
  `src/elspeth/plugins/sources/csv_source.py` gets full prose, tags, audit
  characteristics, and `data_trust_tier=3`. This is the **only** plugin
  authored in this phase; the rest get `None` defaults.
- Test coverage for the new fields and the derivation logic.

**Out of scope (handled in 16b or later phases):**

- Frontend catalog drawer reshape, filter chips, search extension,
  shortcuts regroup, synthetic "Inline data from chat" entry — all 16b.
- Per-plugin prose authoring for every plugin in the codebase. That is an
  incremental documentation task; this plan ships the schema + one
  canonical example and intentionally stops there. Per design doc 08-§Risks:
  "Empty entries fall back to a generic 'see the technical description'
  message rather than blocking display."
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
   `emits_provenance` for plugins that do extra provenance work beyond
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
**not** try to infer `"emits_provenance"` from "plugin uses Landscape" —
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
    """Declared audit characteristics that the framework cannot infer
    from other attributes. The catalog service composes this set with
    inferred characteristics (from determinism, validation-failure
    handling) at summary-build time. Declare flags like 'signed',
    'credentials', 'emits_provenance' (only for plugins that do extra
    provenance work beyond the standard pipeline)."""

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

Expected: FAIL — `PluginSummary` doesn't accept the new fields; pydantic
raises `ValidationError` on unknown fields because of `extra="forbid"`.

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

Expected: PASS — all five tests green.

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
from elspeth.web.catalog.service import _derive_audit_characteristics


class _FakeSource:
    name = "fake_source"
    determinism = Determinism.IO_READ
    # quarantine is author-declared (not inferred from _on_validation_failure,
    # which is a per-instance attribute set in __init__).
    audit_characteristics = frozenset({"emits_provenance", "quarantine"})


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


def test_source_declared_quarantine_passes_through() -> None:
    """`quarantine` is author-declared on the class. Composition preserves it."""
    derived = _derive_audit_characteristics(_FakeSource, plugin_kind="source")
    assert "quarantine" in derived  # declared by author
    assert "io_read" in derived  # inferred from determinism
    assert "emits_provenance" in derived  # declared, preserved


def test_source_without_declared_quarantine_omits_it() -> None:
    """Source authors who don't declare quarantine don't get it inferred.
    Inference from `_on_validation_failure` is impossible at the class
    level (it's a per-instance attribute set in __init__)."""
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
    """quarantine is a source-only inference (transforms don't quarantine
    at the boundary; they crash on type errors per the tier model)."""
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/catalog/test_service_audit_derivation.py -v
```

Expected: FAIL — `ImportError: cannot import name '_derive_audit_characteristics'`.

- [ ] **Step 3: Implement the derivation helper**

In `src/elspeth/web/catalog/service.py`, add the following module-level
helper (place it near the top, after the existing module constants):

```python
from elspeth.contracts.enums import Determinism

# Map Determinism enum values to the audit-characteristic flag they
# imply. The catalog surfaces these as visual cues on the plugin card so
# a compliance-focused user (Linda persona) can see at a glance which
# audit traits apply without reading the technical description.
_DETERMINISM_TO_AUDIT_FLAG: dict[Determinism, str] = {
    Determinism.IO_READ: "io_read",
    Determinism.IO_WRITE: "io_write",
    Determinism.EXTERNAL_CALL: "external_call",
    Determinism.DETERMINISTIC: "deterministic",
    Determinism.SEEDED: "seeded",
    Determinism.NON_DETERMINISTIC: "non_deterministic",
}


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
    `plugin_cls.determinism`) is correct here: the bases declare these
    with sensible defaults, so every plugin reachable via the catalog
    has them. A plugin without these attributes would be a malformed
    system plugin (Tier 1 bug); crash via AttributeError is the correct
    response, not defensive fallback.
    """
    del plugin_kind  # reserved for future per-kind inferences
    declared: frozenset[str] = plugin_cls.audit_characteristics  # type: ignore[attr-defined]
    determinism = plugin_cls.determinism  # type: ignore[attr-defined]

    inferred: set[str] = set()
    flag = _DETERMINISM_TO_AUDIT_FLAG.get(determinism)
    if flag is not None:
        inferred.add(flag)

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

Expected: PASS — all 8 tests green.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/catalog/service.py \
  tests/unit/web/catalog/test_service_audit_derivation.py
git commit -m "feat(web): derive audit characteristics from plugin metadata (Phase 7A.3)"
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
        characteristics are the *derived* set: declared chars composed
        with characteristics inferred from determinism and (for sources)
        on-validation-failure handling. The frontend reads
        audit_characteristics as a flat list of flag strings.
        """
        name: str = plugin_cls.name
        description = get_plugin_description(plugin_cls)
        schema = self._catalog_schema(plugin_cls, plugin_type)
        config_fields = self._extract_config_fields(schema)

        # Direct attribute access: the bases declare every Phase-7A
        # field with a default. A plugin missing them would be a
        # malformed system plugin (Tier 1 bug); crash is correct.
        usage_when_to_use = plugin_cls.usage_when_to_use  # type: ignore[attr-defined]
        usage_when_not_to_use = plugin_cls.usage_when_not_to_use  # type: ignore[attr-defined]
        example_use = plugin_cls.example_use  # type: ignore[attr-defined]
        capability_tags = plugin_cls.capability_tags  # type: ignore[attr-defined]
        data_trust_tier = plugin_cls.data_trust_tier  # type: ignore[attr-defined]

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

This is the **one** plugin authored in this phase. Future plugins follow
the same pattern; documenting each plugin is an incremental task outside
this phase's scope.

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
`TestSources::test_csv_source_present` etc.; extend their coverage in
the next step.

- [ ] **Step 8: Add a route test pinning the wire shape**

In `tests/unit/web/catalog/test_routes.py`, add (alongside the existing
`TestSources` class — note the fixture name is `client`, not
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
audit_characteristics, and data_trust_tier=3. Other plugins keep their
defaults (None / empty); per-plugin prose authoring is an incremental
task outside this phase's scope, with empty entries falling back to a
generic 'see the technical description' message rather than blocking
display.

Tags an end-to-end integration test pinning the wire shape so the
frontend (Phase 7B) can rely on the new fields being present and
correctly composed (declared 'coerce' + inferred 'io_read' +
inferred 'quarantine'). See
docs/composer/ux-redesign-2026-05/16a-phase-7a-backend.md.
EOF
)"
```

## Task 6: Final regression sweep

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
  src/elspeth/plugins/infrastructure/base.py \
  src/elspeth/web/catalog/schemas.py \
  src/elspeth/web/catalog/service.py \
  src/elspeth/plugins/sources/csv_source.py
```

Expected: PASS — no new type errors. The ClassVar declarations should
type-check cleanly; the `# type: ignore[attr-defined]` comments in
`service.py` shouldn't be needed if mypy follows the ClassVar
declarations through the protocol-vs-base distinction. Remove them if
they aren't required.

- [ ] **Step 4: Run ruff on the touched files**

```bash
.venv/bin/python -m ruff check \
  src/elspeth/plugins/infrastructure/base.py \
  src/elspeth/web/catalog/schemas.py \
  src/elspeth/web/catalog/service.py \
  src/elspeth/plugins/sources/csv_source.py \
  tests/unit/plugins/infrastructure/test_base_metadata.py \
  tests/unit/web/catalog/test_schemas_extended.py \
  tests/unit/web/catalog/test_service_audit_derivation.py \
  tests/unit/web/catalog/test_service_extended_summary.py \
  tests/unit/plugins/sources/test_csv_source_metadata.py
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

After all six tasks land:

- `BaseSource`, `BaseTransform`, `BaseSink` carry six new optional
  class-attribute fields for reference content.
- The catalog API surface (`/api/catalog/{sources,transforms,sinks}`)
  emits the new fields on every `PluginSummary`. Unfilled plugins emit
  `None` / empty defaults; the canonical CSV example emits full
  authored content plus the derived audit-characteristic set.
- The audit-characteristic derivation rules are codified in
  `_derive_audit_characteristics` and have their own focused test
  suite.
- No frontend code has changed; the new fields are visible only to
  callers of the catalog API (curl, the upcoming Phase 7B reshape).

Phase 7B picks up from here: frontend card layout, filter chips, search
extension, "Inline data from chat" synthetic entry, shortcuts regroup.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Plugin authors don't write any prose, leaving every plugin's card empty | Per design doc 08-§Risks, empty entries fall back to "see the technical description" — non-blocking. The canonical CSV example sets voice and tone for incremental authoring. |
| The derivation rules surprise authors ("why does my source show `quarantine`?") | The `_DETERMINISM_TO_AUDIT_FLAG` table is documented in this plan and in the service.py docstring. Authors reading the catalog source can see the inference rules. |
| `data_trust_tier=3` for sources reads as "this is a bad / untrusted plugin" | The tooltip on the badge (16b) frames it as "what tier of data flows through this boundary," matching CLAUDE.md "Data Manifesto" language. The badge is descriptive, not pejorative. |
| Extending `PluginSummary` breaks existing JSON consumers | `_StrictResponse` uses `extra="forbid"` on the *parser* side, not the emitter. Pydantic emitting new fields doesn't break consumers that ignore unknown fields (the default JS behaviour); only consumers using strict-parse on the response would fail, and those don't exist outside the codebase. |
| Frozenset serialization order is non-deterministic | The frontend reads `audit_characteristics` as an unordered set of strings; rendering sorts them for stable display in 16b. The integration test asserts membership, not order. |
| Future plugin author copies the CSV example verbatim and forgets to update the prose | Code review catches this. The example_use field's YAML preview in the catalog drawer will look obviously CSV-shaped for a non-CSV plugin, surfacing the copy-paste error to the reviewer. |

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

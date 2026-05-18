# 08 — Catalog Reshape (Reference, Not Toolkit)

## Decision

The Catalog button stays in the UI. Its drawer is reshaped from an
**interactive toolkit** (IDE-style "browse → pick → wire") to a
**searchable system-capability reference** (encyclopedia-style "browse →
read → learn").

The shift is in the affordances and framing, not in the underlying data
(which is the same plugin catalog already fetched from
`/api/plugins/{sources,transforms,sinks}`).

## Why the change

The interactive-toolkit framing implies a workflow nobody actually does.
No documented persona browses the plugin catalog and then picks plugins to
wire into their pipeline:

- **Linda** doesn't browse plugins. She describes what she needs in
  domain language.
- **Sarah** doesn't browse plugins. She describes her research goal.
- **Marcus** doesn't browse plugins. He describes the Zap he wants.
- **Dev** doesn't browse plugins; she names them directly.

Plugin selection happens through **the LLM driven by user intent**
(guided or freeform). The catalog's actual job is *orientation*: "what
can this system do? what shapes of plugin exist? what should I learn
about?"

That's a reference-surface job, not a toolkit-surface job.

## What stays vs what changes

| Element | Before (toolkit) | After (reference) |
|---|---|---|
| Button in chrome | Top-right of inspector header | Side rail, below "Export YAML" |
| Drawer position | Right-edge slide-over | Same |
| Three tabs (Sources / Transforms / Sinks) | Yes | Yes — but consider adding a "Recipes" or "Examples" tab |
| Fuzzy search | Yes | **Strengthened** — fuzzy across name + description + capability tags + schema field names |
| Per-plugin card | Yes | Yes, content rewritten |
| Schema expand-on-click | Yes | Yes |
| **"Use this" / "Select" buttons** | Implied by toolkit framing | **Removed** — there are no add-to-pipeline actions |
| **Drag handles** | None today | None — explicitly not added |
| **"In use" highlighting** | None today | None — explicitly not added (would imply a workflow link back to composition) |
| **Filters** | Three-tab segmentation | Add: filter by capability tag, by audit characteristics |
| **Audit characteristic icons** | Not present | **Added** — "emits provenance," "supports redaction," "deterministic," etc. — emitted only when the author made a per-plugin choice (kind-default determinism does NOT emit a flag) |
| **Persona-framed descriptions** | Single technical description | Multiple framings: "When you'd use this," "When you wouldn't," "What it does technically" |
| **Examples** | Schema only | Add: one-or-two-line example snippets showing realistic use |

## The "Inline data from chat" entry

Per [06-chat-as-data-entry.md](06-chat-as-data-entry.md), the catalog
should list **"Inline data from chat"** as the first source option, framed
as the lowest-friction starting point. This makes the catalog's reference
nature serve the user's actual choice space — the first row of the source
list shouldn't be a CSV plugin, it should be "you don't need a plugin for
small inputs; just type."

## Plugin card content design

Each plugin card has a consistent structure:

```text
  ┌──────────────────────────────────────────────────┐
  │  CSV Source                                      │
  │  ─────────                                       │
  │  Read rows from a CSV file. Validates and        │
  │  coerces types at the boundary; quarantines      │
  │  malformed rows.                                 │
  │                                                  │
  │  Audit: ✓ coerces types  ✓ quarantines bad rows │
  │                                                  │
  │  When you'd use this:                            │
  │    A reasonably large dataset (more than ~20     │
  │    rows) that already exists as a file.          │
  │                                                  │
  │  When you wouldn't:                              │
  │    Small inline data — use "Inline data from     │
  │    chat" instead. Streaming data — CSV is batch. │
  │                                                  │
  │  Example use:                                    │
  │    source:                                       │
  │      plugin: csv                                 │
  │      options: { path: data/input.csv }           │
  │                                                  │
  │  [ Schema → ]                                    │
  └──────────────────────────────────────────────────┘
```

The "When you'd use this" / "When you wouldn't" framings serve all four
personas:

| Persona | Reads | For |
|---|---|---|
| Linda | "When you'd use this" in compliance language | Choosing the right shape for her domain |
| Sarah | "When you'd use this" in research language | Confirming the system can serve her need |
| Marcus | "When you wouldn't" | Graceful rejection of his Zapier-shaped expectation |
| Dev | "Example use" code snippet, "Schema →" | Currency check on plugin names and options |

### Audit-characteristic icons

Quick visual cues on each card. Examples:

| Icon | Meaning |
|---|---|
| ✓ provenance | Records source data hash and per-row lineage |
| ✓ retention | Respects configured retention policy |
| ✓ quarantine | Quarantines malformed rows instead of crashing |
| ✓ coerce | Coerces types at the boundary (Tier 3 behaviour) |
| ✓ signed | Output is HMAC-signed |
| ⚠ external_call | Reaches an external system (network call) |
| ⚠ credentials | Requires user secrets |

Hover or tap reveals the explanation in plain language.

## Filters

Each tab (Sources / Transforms / Sinks) gets a filter strip at the top:

```text
  Filter: [ all ] [ ✓ provenance ] [ no external_call ]
```

The filter strip lets users narrow the catalog to "what works for my
sensitive-data pipeline" or "what doesn't make a network call" in one
click. The filters are non-exhaustive; the most useful are exposed as
chips, others reachable from "..."

## Search

Fuzzy search should hit:

- Plugin name
- Plugin description
- "When you'd use this" / "When you wouldn't" prose
- Capability tags
- Schema field names

So a user typing "url" finds:

- The web_scrape transform (description mentions URL)
- The "Inline data from chat" source (with "URL" in its examples)
- HTTP-related plugins

The current fuzzy implementation hits name + (lazy-fetched) description;
extending to capability tags and "when you'd use" prose is the main
addition.

## What the catalog does NOT do

- It does not have "Add to pipeline" or "Select" buttons on plugin cards.
  The user does not pick plugins from here; they describe their intent in
  chat.
- It does not show "plugins currently in use" highlighting. The catalog
  is reference; the graph view is the surface for "what's in my pipeline."
- It does not have configuration forms inside it. Plugin options are
  configured in turn widgets (guided) or chat (freeform).
- It does not have "plugin author resources" or "create a new plugin"
  affordances. Plugins are system-owned code (per CLAUDE.md); the
  composer is not a plugin development surface.
- It does not have "recently used" or "favorites" sections. The catalog
  is canonical reference; personal usage history doesn't belong here.

## Keyboard shortcut placement

The current Catalog shortcut is `Ctrl+Shift+P`, which sits alongside
action shortcuts (`Ctrl+Shift+V` Validate, `Ctrl+E` Execute). Since the
Catalog is reshaping into reference, its shortcut should reshape too:

- Move toward documentation conventions: `?` or `F1` are conventional for
  help (already used; `?` opens keyboard shortcuts help).
- A reasonable alternative: keep `Ctrl+Shift+P` but reframe the
  shortcuts help to group it under "Reference" rather than "Actions."

The shortcut placement is a small detail; flag in [11-open-questions.md](11-open-questions.md).

## Implementation notes

| Component | Touch-point |
|---|---|
| Backend | Plugin metadata needs "when you'd use this" / "when you wouldn't" / "example use" fields. These are documentation, not config — they can live in the plugin's docstring or in a sidecar markdown file per plugin. |
| Backend | Audit-characteristic flags can be inferred from existing plugin contracts (provenance, retention, etc.). New endpoint or extended existing one to return them as flags. The determinism-derived flag is **suppressed** when the plugin inherits the kind default (every Source's IO_READ, every Sink's IO_WRITE, every Transform's DETERMINISTIC) — surfacing it on every card teaches the user nothing per-plugin and fails the "every tag must represent a meaningful per-plugin decision" test. |
| Frontend | New card layout matching the spec above. |
| Frontend | Filter chips and search-across-prose. |
| Documentation | Each plugin author needs to write "when you'd use this" / "when you wouldn't" prose. This is a per-plugin documentation task; can be done incrementally. |

## Risks

| Risk | Mitigation |
|---|---|
| Plugin authors write low-quality "when you'd use this" prose | Provide a template; review during PR. Empty entries fall back to a generic "see the technical description" message rather than blocking display. |
| Catalog becomes a documentation tool that drifts from the engine | Plugin documentation lives next to the plugin code; the catalog reads it. Drift then = the plugin's docs are out of date, which is a normal staleness problem. |
| The "Inline data from chat" entry is confusing — it's not really a "plugin" | Frame it explicitly as an option, not a plugin. Different visual style; first in the list with explanatory framing. |

## Memory references

- `feedback_catalog_is_reference_not_toolkit`
- `project_composer_personas` (each persona's catalog use case)
- `project_composer_dynamic_source_from_chat` (the inline-data entry)

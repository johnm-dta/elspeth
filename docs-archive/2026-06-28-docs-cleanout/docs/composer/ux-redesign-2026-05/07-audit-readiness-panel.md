# 07 — Audit-Readiness Panel

## The problem

ELSPETH's entire product positioning hinges on the audit trail. The README
opens with *"high-assurance pipeline substrate for consequential workflows."*
The data-trust model, the declaration-trust framework, the runtime VAL
manifests, the terminal-outcome model, the JSONL change journal — all of
it exists to make pipeline outputs defensible.

**The composer surfaces none of this during composition.**

A user can compose a beautiful-looking pipeline in the current UI and have
no idea whether:

- The source plugin records absence-as-`None` vs fabricates defaults
- The LLM transform's prompt template is recorded with each call
- The sink's output is hash-chained
- The retention policy aligns with the user's actual policy needs
- The trust tiers of the plugins they've selected match the data
  sensitivity

The validation indicator dot (`●/✓/⚠/✗` in the current inspector) covers
*structural* validation — does the pipeline wire up cleanly. It does not
cover *audit readiness* — does this pipeline produce the evidence the
user's domain requires.

## The surface

A **persistent panel** in the composer's right rail. Visible at all times
during composition. Updates continuously as the pipeline changes.

The panel summarises audit readiness across four-to-six axes, with each
axis collapsible to a single status indicator (✓ / ⚠ / ✗ / —) and
expandable to its detail.

## Content design

The panel speaks Linda's vocabulary, not the engine's. Where the engine
says "VAL manifest," the panel says "validation evidence recorded." Where
the engine says "declaration trust," the panel says "plugin claims
verified." The vocabulary mapping is intentional and load-bearing.

### Top-level rows (always visible)

```text
  ┌─ AUDIT READINESS ────────────────────────┐
  │                                          │
  │  Validation        ✓  All checks pass    │
  │  Plugin trust      ✓  All Tier 1/2       │
  │  Provenance        ⚠  See details        │
  │  Retention         —  Not configured     │
  │                                          │
  │  [Explain →]   [Refresh]                 │
  └──────────────────────────────────────────┘
```

### What each row means

| Row | What it checks | ✓ when | ⚠ when | ✗ when |
|---|---|---|---|---|
| **Validation** | Schema, route targets, edge compatibility (existing validate_pipeline preflight) | All checks pass | Warnings present (e.g. identity passthroughs) | Errors that would prevent execution |
| **Plugin trust** | Trust tier of each plugin against the data the pipeline handles | All plugins meet expected tier | Mixed tiers with documented justification | Tier 3 plugin handling Tier 1 data without explicit coercion boundary |
| **Provenance** | Will every output row have a complete lineage chain? | All sources, transforms, sinks record hashes | Identity passthroughs detected (per ADR-019 family) | A plugin in the path doesn't record provenance |
| **Retention** | Does the configured retention align with what the user said they need? | Configured and consistent | Configured but defaults | Not configured for a pipeline that handles sensitive data |
| **LLM interpretations** | (Conditional — shown only if pipeline has LLM transforms) Are all subjective-term interpretations user-accepted? | All accepted | Pending acceptance | Run blocked until accepted |
| **Secrets** | (Conditional — shown only if pipeline references secrets) Are all secret references resolvable and audit-fingerprinted? | All resolved | Some unresolved | Required secret missing |

### The "Explain" view

Clicking "Explain →" opens a detail view that walks the user through what
will happen when this pipeline runs, in narrative form:

```text
  When you run this pipeline, ELSPETH will record:

  • Source data — 5 URLs from your typed input.
    SHA-256 hash recorded for each URL.

  • Web fetch — HTTP status, response time, response hash for each URL.

  • LLM ratings — for each URL: the full prompt (with your accepted
    definition of "cool"), the full response, the model and version,
    the timestamp. Recorded in the audit database.

  • Output file — written to your session storage. SHA-256-hashed;
    chain-of-custody recorded with timestamp.

  • Run metadata — when, who (you), with which plugin versions in use.

  Retention: 90 days by default. You can change this in the source
  configuration or your session settings.

  This evidence is sufficient to answer questions about any output
  row of this pipeline, including which LLM judged it and why.
```

The explanation is generated from the actual plugin configurations in the
current composition, not from canned text. If the user changes a plugin,
the explanation updates.

### Vocabulary mapping (engine → panel)

| Engine term | Panel term | Why |
|---|---|---|
| Validation preflight | Validation | Linda thinks "review." Closest neutral term. |
| Plugin trust tier (1/2/3) | Plugin trust | "Trust" is plain language; the tier numbers can appear in the Explain view |
| Provenance / lineage chain | Provenance | Linda's domain term works |
| `identity_node_advisory` | "Identity passthrough — provenance gap" | Plain English; the technical name in the Explain detail |
| Declaration trust | "Plugin claims verified" | Most users don't know what "declaration" means |
| VAL manifest | "Validation evidence recorded" | "Manifest" is engine vocabulary |
| Terminal outcome / disposition | "Final state of each row" | More natural |
| Retention policy | Retention | Domain term works |
| Run accounting | (Not surfaced) | Internal concept |
| Composite primary key | (Not surfaced) | Internal concept |

## Behaviour

### When does the panel update?

Continuously. Specifically, on:

- Any change to the composition state (plugin added / removed / configured)
- Any change to the user's accepted LLM interpretations
- Any change to secrets references
- Any change to retention configuration

The audit-readiness check is the same backend logic the Validate route runs;
this surface just makes it visible always rather than on-demand.

### What if the panel says ⚠ or ✗?

Each row's status indicator is clickable. Clicking ⚠ or ✗ opens:

1. A description of what's wrong, in panel-vocabulary
2. A suggested fix (e.g., "Replace the identity passthrough with a
   transform that records provenance")
3. A jump-to-where button that selects the affected node in the graph
   and scrolls the relevant turn into view

This builds on the existing `identity_node_advisory`-style repair-hint
infrastructure introduced in RC-5.1.

### Continuity from the tutorial

The audit-readiness panel is **the same surface** the hello-world tutorial's
turn 5 introduces (see [04-first-run-tutorial.md](04-first-run-tutorial.md)).
Users meet the audit story at the end of the tutorial; the panel persists
that view into all subsequent sessions.

The continuity matters for retention. Without the tutorial introduction,
the panel is a column of green checks that nobody reads. With the tutorial
introduction, the panel is "the place that proves my pipeline is
defensible — like the tutorial showed me."

## Persona-specific framing

| Persona | What they read | What they ignore |
|---|---|---|
| **Linda** | Every row, especially Provenance and Retention. Clicks Explain frequently. | Nothing — for her, this panel is the central reason to use the tool. |
| **Sarah** | Validation, Provenance. Reads Explain at the end of a successful run. | Plugin trust (trusts institution) |
| **Marcus** | Glances at the top-level dots before clicking Run. | Explain detail — he'll come back if Run produces something unexpected. |
| **Dev** | Glances at validation, opens YAML, reads the actual config. | Most of the panel — the YAML is more efficient for her. |

## What the panel does NOT do

- It does not block execution. A user can run a pipeline with ⚠ status if
  they choose. The panel surfaces; the run command decides.
- It does not log to the audit trail itself. The audit trail records what
  the pipeline actually does at runtime; the panel is a composition-time
  preview.
- It does not replace the Validate button — it subsumes it. The "Validation"
  row carries the validation status, and the existing validate_pipeline
  call powers it.
- It does not require new engine work. All the underlying signals
  (validation, trust tiers, plugin records, retention config) already
  exist. The panel is presentation.

## Implementation notes

| Component | Touch-point |
|---|---|
| Frontend | New persistent panel component, reads composition state, calls existing validate endpoint plus a new "audit-readiness summary" endpoint |
| Backend | New endpoint that aggregates: validate_pipeline result + plugin trust-tier check + provenance-chain check (per identity_node_advisory logic) + retention config + LLM interpretations status + secrets status |
| Backend | The aggregation endpoint should not duplicate logic — compose existing checks |
| Audit recorder | No changes required; this is read-only presentation |

## Risks

| Risk | Mitigation |
|---|---|
| Panel feels like noise once it's "all green" | Reduce visual weight when all-green; keep one-line "Audit ready ✓" rather than four rows. Expand to detail on hover or click. |
| Confusing language ("Plugin trust ⚠") for Marcus | Tooltips in plain language. The Explain view is canonical. |
| Panel disagrees with what actually happens at runtime | Both must call the same underlying validation routes. If they disagree, that's a bug — file it; don't paper over it. |
| Performance — recomputing on every composition change | Debounce updates. The existing validate route is cheap enough; the additional checks are similar order. |

## Memory references

- The audit-readiness panel is recommendation §critical in the original
  review pass; not memory-pinned as a separate entry because it follows
  from the personas and the README positioning.
- See `project_composer_personas` for Linda-vocabulary requirements.

# Composer Guided/Freeform Capability Parity

**Status:** Proposed for written-spec review
**Date:** 2026-07-13
**Priority:** High
**Scope:** Composer authoring architecture, freeform and guided skill packs, guided session migration, validation feedback, and cross-surface evaluation

## Summary

Freeform and guided composer modes must be able to author the same pipelines.
Guided mode may stage the conversation, constrain when decisions are made, and
add review checkpoints. It must not define a smaller pipeline language.

The current guided implementation violates that rule. It asks the model for a
list of transform steps and deterministically materializes them as one linear
chain. The proposal cannot represent multiple sources, gates, arbitrary routes,
forks, queues, coalesces, aggregations, explicit edges, or independently named
outputs. Later stages can display some of those structures when they already
exist, but guided mode cannot author them.

This design removes the guided-only pipeline model. Freeform and guided will use
one canonical full-pipeline proposal schema, one discovery and planning
capability, one validation path, and one commit executor. The modes will differ
only in interaction and review.

## 1. Context and root cause

### 1.1 Product history

The limitation resulted from a capability-graduation failure:

1. The original guided/tutorial scaffold deliberately optimized for a novice
   happy path: few clicks, low token use, a small prompt, and a linear example.
2. Its conversational interaction became the superior authoring experience.
3. The guided-mode reframe promoted that experience into the primary guided
   surface while treating the existing chat-to-build path as fixed.
4. The tutorial-era proposal and materialization contracts were never upgraded
   to the complete composer topology.
5. Later eval remediations made the inherited restriction explicit instead of
   removing it.

The important distinction is that promoting the tutorial interaction was
correct; promoting its capability boundary was not. Tutorial is a workflow
profile, not a pipeline-language profile.

### 1.2 Current architectural restriction

The restriction exists at four layers:

- `ProposeChainPayload` and `ChainProposal` contain only
  `{plugin, options, rationale}` transform steps.
- The chain solver's terminal tool schema accepts only `steps` and `why`.
- `handle_step_3_chain_accept()` converts every step into
  `node_type="transform"`, synthesizes `chain_in`/`chain_N`/`main` labels, and
  submits `edges=[]`.
- Guided skill text describes a single linear spine and explicitly says guided
  cannot express a gate.

The final wire stage carries and renders richer topology fields. Tests that
feed it hand-built fork/coalesce states prove display support, not authoring
support. This stage-boundary mismatch allowed apparent DAG coverage without a
guided path that creates the DAG.

### 1.3 Historical evidence

- Commit `39ebe6729` introduced the original guided design on 2026-05-11,
  co-authored by Claude Opus 4.7. It defined a small novice protocol, independent
  skills, and freeform exit for graph manipulation.
- Commits `72eba9169`, `88c754dd10`, and `c87633be7` introduced the transform-only
  proposal and linear materializer on 2026-05-11/12.
- Commit `b91106dfe` added the literal "Single linear spine" rule on 2026-06-24
  as part of the staged tutorial wire work.
- Commit `4cbfcb6aae` promoted the staged tutorial path and green passive
  tutorial E2E on 2026-06-28 while deleting recipe-match capability and tests.
  The artifact shows capability removal tied to the tutorial goal; repository
  history cannot establish subjective intent beyond that evidence.
- Commit `44ca8f664` responded to the 2026-07-10 filtering eval by teaching the
  model that guided could not express gates and by rejecting the false filter
  claim. This correctly stopped a silent semantic failure but converted the
  missing capability into a product disclaimer.

The original specifications also contradicted themselves: they called guided a
construction surface for existing engine capabilities and required shape
preservation for fork-and-merge, while prescribing a transform-only sequential
implementation. Tests enforced the implementation, not the product claim.

## 2. Decision and invariant

### 2.1 Capability invariant

Define `P` as the set of canonical composer payloads that pass boundary and
semantic checks **and produce a globally valid, runnable `CompositionState`**
under the deployment's shared web policy. A mutation tool may currently return
a candidate state plus validation telemetry even when that state is invalid;
such incomplete candidates are not members of `P`.

For every `p` in `P`:

```text
freeform_can_author(p) = true
guided_full_can_author(p) = true
guided_staged_can_author(p) = true
```

The following rules are load-bearing:

1. Freeform and guided have identical pipeline expressiveness.
2. A global security or deployment policy may remove a capability from `P`, but
   it must remove it from every composer mode equally.
3. A guided-to-freeform handoff does not satisfy guided capability parity.
4. A recipe match does not satisfy capability parity unless the non-recipe
   planner can author the same pipeline.
5. Tutorial may constrain its teaching scenario and visible sequence. It must
   use the same planner and proposal schema as ordinary guided mode.
6. Adding a canonical composer capability must make it available to every mode
   without adding a second topology representation.

### 2.2 Interaction is the only mode distinction

Freeform accepts an end-to-end request and may refine the result incrementally.
Guided decomposes the same request into source, output, transformation/topology,
and wire-review conversations. Both produce the same canonical pipeline draft.

Guided stages may lock reviewed facts. They may not prevent the shared planner
from representing required wiring. Store incomplete early-stage work as guided
interaction facts and operator intent, not a partial pipeline IR. If a later
requirement invalidates an earlier fact, guided performs a typed rewind to that
stage, updates the fact, and replans. It does not declare the topology
unsupported.

## 3. Architecture

### 3.1 Target flow

```text
                         operator intent
                               |
                 +-------------+-------------+
                 |                           |
          freeform controller         guided controller
          end-to-end dialogue         staged dialogue/review
                 |                           |
                 +-------------+-------------+
                               |
                   shared composer planner
             shared capability + plugin discovery
                               |
              canonical set_pipeline arguments
                               |
                  shared validation/explanation
                               |
             shared audited set_pipeline dispatch
                               |
                    immutable CompositionState
```

The controllers package context and present decisions. They do not define
different authoring languages.

### 3.2 Canonical proposal contract

The canonical `set_pipeline` declaration is the authoring-schema authority. A
new `PipelineProposal` is only an approval/audit envelope around those exact
arguments:

```python
@dataclass(frozen=True, slots=True)
class PipelineProposal:
    pipeline: Mapping[str, Any]  # exact canonical set_pipeline arguments
    why: str
    base_composition_version: int
    base_composition_content_hash: str
    draft_hash: str
    reviewed_anchor_hash: str
    provenance: ProposalProvenance

    def __post_init__(self) -> None:
        freeze_fields(self, "pipeline", "provenance")
```

`PipelineProposal` must not introduce another node, edge, source, or output
model. Its `pipeline` JSON schema is referenced or generated from the registered
`set_pipeline` tool declaration. `SetPipelineArgumentsModel` remains the typed
boundary validator. The design must not create a third schema between those two
existing authorities. A structural compatibility test must fail when the
LLM-facing declaration and typed validation model disagree on properties,
requiredness, nested strictness, or supported shapes.

The server computes `draft_hash` after deep-freezing the custody-safe canonical
arguments and verifies it again on restore and acceptance. A supplied or stored
hash mismatch is a Tier-1 integrity failure. `frozen=True` without recursive
freezing is insufficient because nested dicts/lists could otherwise change
between hashing and commit.

`ProposalProvenance` records the planner configuration, actual per-call skill
hash, proposal schema version, model/provider identifiers already permitted by
the existing audit contract, optional `supersedes_payload_hash`, and optional
version-7 migration lineage. It contains no topology fields and therefore does
not become another pipeline representation.

The payload represents all currently web-authorable structure:

- the legacy single `source` form or one or more named `sources`;
- transforms, gates, aggregations, queues, and coalesces;
- input, success, error, route, fork, branch, policy, merge, and trigger fields;
- explicit edges and connection labels;
- the canonical number of distinctly named outputs and their write-failure
  policies;
- plugin options, interpretation requirements, and authoring metadata allowed
  by the canonical boundary.

Row expansion or deaggregation behavior remains plugin behavior expressed
through these canonical node types and their output settings; it is not a new
node type.

Persist the raw accepted canonical arguments in the proposal envelope after
secret handling and inline-content custody have converted forbidden raw values
to safe references. Persist the normalized `CompositionState` produced by the
executor separately. Never round-trip state through a new guided topology
representation.

### 3.3 Shared planner

Extract a shared composer-planning loop used by freeform new-pipeline builds and
guided topology builds. It receives:

- complete operator intent;
- current `CompositionState`;
- reviewed/locked facts from guided stages, when present;
- the canonical pipeline schema;
- the same read-only catalog, model, plugin-schema, plugin-assistance, source,
  sink, and topology discovery surfaces;
- validation or revision feedback from the previous proposal.

The planner returns a `PipelineProposal`. The freeform controller may immediately
stage the proposal for its normal approval/mutation workflow. The guided
controller emits a `propose_pipeline` turn and waits for review. Neither planner
uses a transform-only intermediate representation.

All topology-producing configurations call one public `plan_pipeline()`
entrypoint. Freeform, guided-full, guided-staged, and tutorial supply prompt and
interaction context to that function; they do not own separate planning loops.
Architecture tests forbid active imports/calls of `solve_chain` outside the
version-7 compatibility adapter once migration completes.

Existing incremental freeform mutation tools remain available for edits. They
do not become a second new-pipeline planning implementation.

### 3.4 Guided stage responsibilities

Version-8 guided checkpoint state must represent plural interaction facts. It
replaces singular `step_1_result`, source intent/chosen-plugin staging, and
single-item sink staging with:

- a stable-id mapping of reviewed `SourceResolved` facts;
- a stable-id sequence/mapping of reviewed `SinkOutputResolved` facts;
- pending source/output intents keyed by stable item id;
- an optional active `{kind, id}` edit target.

These are reviewed dialogue facts, not a partial topology IR. The canonical
pipeline exists only when `plan_pipeline()` produces a complete proposal.

#### Source stage

Replace singular `SourceResolved` authoring assumptions with a named source-set
resolution. The stage can propose and review one or more sources, including the
options and source-level failure policy allowed by the canonical schema.

The source stage does not permanently assign final connection labels. The full
planner owns topology wiring after it knows the complete request.

#### Output stage

Resolve a set of distinctly named outputs. Remove the `sink_name="main"`
collision and hard-coded `on_write_failure="discard"`. Preserve per-output
required fields, options, and policies.

#### Transformation and topology stage

Replace `propose_chain` with `propose_pipeline`. The stage may author every node
type and routing construct in the canonical contract. It receives the complete
source and output sets plus the original operator intent, not only a reduced
source/sink contract.

#### Wire-review stage

Continue using the existing full-DAG rendering capability. Review the canonical
proposal and validated `CompositionState`; do not reconstruct or enforce a
single spine. The review checks:

- every produced connection has valid consumers;
- fan-out and fan-in are explicit and policy-compliant;
- field guarantees satisfy downstream requirements on every path;
- fork/coalesce participation and merge policies are coherent;
- failure routes terminate or rejoin intentionally;
- the graph matches the operator's requested topology.

### 3.5 Guided commit

On acceptance, guided performs these checks:

1. `base_composition_content_hash` still matches the current pipeline content.
2. Reviewed locked facts have not changed, using `reviewed_anchor_hash`.
3. The proposal validates against the canonical schema.
4. A shared non-persisting candidate build runs global web policy, plugin
   contracts, and full graph validation and returns `validation.is_valid=true`.
5. The shared audited `set_pipeline` dispatch accepts the exact proposal and
   produces the same validated candidate content hash.

The handler passes `proposal.pipeline` unchanged to the same audited
`set_pipeline` dispatch as freeform. That dispatch invokes
`_execute_set_pipeline()` as its semantic executor and records the canonical
redacted `ComposerToolInvocation`. Guided must not call the executor directly
unless it explicitly produces and persists that identical audit record. It must
not synthesize node identifiers, labels, types, edges, routes, or output names.

`base_composition_version` remains audit and diagnostic context. It is not an
acceptance gate because guided history persistence may create a new composition
version without changing pipeline content.

Extract the candidate constructor/validator from the canonical executor so
every planner configuration uses identical construction semantics. Candidate
validation may audit the rejected attempt, but it does not publish a new
`CompositionState`. This is necessary because the existing mutation tool can
return `success=True` together with an invalid candidate for incremental
authoring. Planner acceptance is stricter: invalid candidates feed the repair
loop and never become the current version.

A no-translation regression captures the arguments observed at the canonical
executor boundary and asserts deep equality with the accepted proposal. Secret
resolution and redaction that occur inside the canonical executor are outside
that comparison.

If checks 1 or 2 fail, guided regenerates against current state instead of
committing stale work. If checks 3–5 fail, the shared planner receives the
structured error and uses the configured product repair budget before surfacing
a named error to the operator. A capability disclaimer is not an error-recovery
path.

## 4. Skill-pack design

### 4.1 Shared capability core

Create one shared capability core assembled into every composer planning
prompt. It defines:

- the canonical connection model;
- all structural node types and their roles;
- source and output cardinality;
- routing, fan-out, fan-in, fork/coalesce, queue, and failure-path semantics;
- field-contract propagation and exact downstream names;
- recipe status as optional acceleration only;
- the requirement to preserve requested topology rather than simplify it.

This material may be generated partly from the canonical declaration and
partly from reviewed structural guidance. The prompt loaders, not copied
Markdown, compose it into each surface.

### 4.2 Freeform big-bang skill

The freeform skill contains:

- the shared capability core;
- freeform interaction and convergence policy;
- batching and incremental-repair guidance;
- dynamic plugin assistance;
- concise canonical examples, including fork/coalesce and multi-source queue
  fan-in.

The current large skill should not receive another duplicated catalog of plugin
options. Plugin-specific shapes remain dynamically discovered.

### 4.3 Guided full skill

`load_guided_skill()` composes:

- the shared capability core;
- guided interaction policy;
- every stage overlay;
- the same dynamic plugin and structural assistance available to freeform.

The full guided solver must not carry a smaller discovery set or a smaller
terminal proposal schema than the shared planner.

### 4.4 Guided staged skills

`load_step_chat_skill(step)` composes:

- the shared capability core or its byte-stable capability index;
- guided interaction policy;
- only the current stage overlay;
- dynamically retrieved detail relevant to the current intent.

The staged prompt is smaller because it omits irrelevant conversation rules,
not because it removes authoring capabilities. In the topology stage it has
the complete canonical proposal schema and discovery surface.

If prompt caching requires a small stable head, keep the shared capability
index in that head and place full schemas/assistance in cacheable tool results.
Prompt-size targets are performance targets, never acceptance gates that permit
capability removal.

### 4.5 Tutorial profile

Tutorial uses the staged guided planner with fixed sample data and teaching
copy. It may preselect or explain the next relevant decision. It must not:

- substitute a tutorial-only planner or proposal model;
- remove capabilities from the planner schema;
- rewrite ordinary guided skill rules;
- count a tutorial recipe or canned topology as proof of guided parity.

### 4.6 Forbidden guidance

Tests reject capability-disclaimer text such as:

- "guided chains cannot include gates";
- "add this after the guided build";
- "switch to freeform for custom branching";
- "this mode supports only a single linear spine".

Guidance may say a requested capability is globally unavailable only when
canonical discovery proves it is absent from `P` for the deployment.

## 5. Discovery and plugin contracts

### 5.1 Canonical structural discovery

Expose the canonical pipeline-authoring schema and reviewed structural
assistance to the shared planner. Do not hand-copy the `set_pipeline` schema
into `chain_solver.py` or a guided skill.

The planner must be able to discover:

- registered source, transform, and sink plugins;
- plugin option schemas and assistance;
- installed LLM models;
- canonical node types and structural fields;
- the current pipeline state and validation result.

Guided may keep mutation tools server-side. Read-only planning capability and
the proposal schema must remain equal to freeform.

### 5.2 Structured LLM output schema

The LLM transform already supports structured multi-query output, but catalog
discovery exposes `queries` as untyped nested dictionaries. Replace those
dictionaries with typed configuration models so generated schema describes:

- query name;
- `input_fields`;
- prompt/template;
- `response_format`;
- `output_fields` with suffix and type;
- collision and pass-through behavior where applicable.

Add one concise, canonical multi-output example to shared LLM plugin assistance.
Both freeform and guided must receive the same assistance.

### 5.3 Validation-probe correctness

Before evaluating planner behavior, fix the composer construction probe for
secret-bearing plugin configuration. The probe currently passes a persisted
`{secret_ref: ...}` object into an LLM constructor that expects the resolved
string form, producing an opaque validation error and hiding valid output-field
guarantees.

Apply `redact_secret_refs_for_validation()` after thawing and stripping
authoring metadata, before `create_transform()`. Add a regression proving that:

1. an LLM config with a secret reference passes the resolver-free probe;
2. structured query fields become guaranteed fields;
3. a downstream field mapper can require those fields;
4. the complete fork/coalesce proposal validates without probe warnings.

This is a P0 prerequisite for meaningful live parity evaluation.

## 6. Persistence, audit, and migration

### 6.1 Versioning

Bump `GUIDED_SESSION_SCHEMA_VERSION` from 7 to 8. Version 8 replaces persisted
`step_3_proposal: ChainProposal | None` with
`pipeline_proposal: PipelineProposal | None`.

Do not bump `SESSION_SCHEMA_EPOCH` solely for this JSON-envelope change. That
epoch would require destructive SQLite recreation and would not protect
PostgreSQL. Instead, add an explicit version dispatcher at the composition-state
restore boundary:

- version 8 receives strict native decoding;
- version 7 receives strict legacy decoding followed by state-aware migration;
- version 6 and earlier remain rejected because their nested shape is not
  compatible with the current checkpoint;
- unknown future versions fail closed.

Keep Tier-1 strictness inside each supported decoder. Do not default missing
fields and do not rewrite persisted rows while reading them.

Version-7 decoding is permanent, not a temporary migration window. Historical
composition versions preserve their original `composer_meta`; revert and fork
can make those bytes current again years later. A historical version-7 row
stays version 7. Its next guided mutation may write a version-8 checkpoint with
migration provenance.

### 6.2 Version-7 migration

Use a state-aware `restore_guided_session(raw, composition_state)` adapter rather
than weakening `GuidedSession.from_dict()`. Migration depends on session
position:

- **Step 1, Step 2, or Step 3 without a proposal:** preserve profile, history,
  chat history, terminal state, and counters. Map the version-7 source result to
  the version-8 reviewed-source mapping under stable id `source`; map every
  `SinkResolved.outputs` entry to the reviewed-output collection using its
  existing name/stable identity; map singular pending intent/chosen-plugin
  fields to the corresponding active item. Set the new proposal to `None`.
- **Step 3 with a pending `ChainProposal`:** require the legacy preconditions of
  exactly one source and at least one output. Translate once using the historical
  source -> `chain_in`, `guided_xform_N`, `chain_N`, final -> `main` rules while
  preserving current composition metadata. Mark provenance
  `legacy_linear_v7`. If the snapshot violates legacy preconditions, return a
  named recovery conflict rather than fabricating a proposal.
- **Pending legacy step edit:** convert `step_3_edit_index` to stable component
  id `guided_xform_N`; new version-8 edits use stable `{kind, id}` targets.
- **Step 4 or completed:** treat the paired `CompositionState` as the
  authoritative graph and leave the executable proposal absent. Runtime-normalized
  state may contain resolved path/custody metadata that cannot safely round-trip
  through the authoring boundary. The UI may render a distinct non-executable
  state snapshot; it must not fabricate `set_pipeline` arguments from state.
- **Exited sessions:** preserve the terminal fact. No proposal is required
  unless the stored graph is being shown.

Preserve the original version-7 serialized proposal in immutable audit/history
records. Migration changes resumable checkpoint state; it does not rewrite
historical events. Record `from_schema_version=7`, the legacy proposal hash, and
the derivation mode as migration lineage.

### 6.3 Turn and frontend compatibility

Keep `TurnType.PROPOSE_CHAIN` and its renderer permanently for historical
evidence. Add `PROPOSE_PIPELINE` as the version-8 active turn; do not rename the
old enum value in place.

For a pending version-7 proposal, GET appends and audits a new
`propose_pipeline` turn that explicitly supersedes the old payload hash. It does
not rewrite the old `TurnRecord`, request/response hashes, or content-addressed
payload. Preserve a legacy POST acceptance path for a stale browser that submits
its already-rendered `propose_chain` without first performing GET. That path
uses the deterministic version-7 adapter and then writes version-8 state.

The restore boundary returns a version-tagged wrapper containing the strict
version-8 session view plus a typed `LegacyV7Context` while a legacy proposal or
edit remains pending. Route dispatch handles a stale legacy response before
discarding that context, preserving the old `why`, rationale, steps, and edit
index. The new proposal provenance carries `supersedes_payload_hash`, and a
`guided_turn_superseded` audit event links the immutable old and new payload
hashes with reason `schema_v7_to_v8`.

The proposal review UI renders a graph diff and node/output summaries from the
canonical proposal. Reuse the existing full-DAG graph and wire-stage components
instead of creating another linear step widget. The frontend closed union keeps
both turn types. `propose_pipeline` responses echo `draft_hash` and use stable
component edit targets rather than array indices.

Deploy server support for both protocols before deploying the new frontend.
New fields are optional at the global request boundary and required when
handling `propose_pipeline`, so old and new clients can coexist during rollout.

### 6.4 Audit requirements

The accepted proposal commits through the same audited canonical `set_pipeline`
dispatch as freeform, not an unaudited direct state constructor. Audit records
include:

- proposal schema version;
- `draft_hash` of the exact canonical arguments;
- base composition version;
- `base_composition_content_hash`, covering sources, nodes, edges, outputs, and
  metadata while excluding version and guided metadata;
- reviewed anchor hash and stable edit target;
- planner surface (`freeform`, `guided_full`, `guided_staged`, or
  `tutorial_profile`);
- repair count and the existing allowlisted/redacted terminal validation shape;
- migration provenance when applicable.

Proposal persistence has two explicit representations:

1. The private resumable checkpoint stores the exact, deep-frozen,
   custody-safe canonical arguments required for later acceptance. It may
   contain approved paths, prompts, options, and safe secret references, but no
   credential literals, resolved secret values, or raw inline content.
2. Guided audit/payload-store events persist an allowlisted/redacted projection
   suitable for audit display. They store `audit_payload_hash` over that
   projection and `draft_hash` as the binding to the exact checkpoint; they do
   not treat the redacted projection as executable.

Acceptance runs under the session compose lock and compares the echoed
`draft_hash` plus base content hash. A mismatch returns 409 and replans/rebases;
it never commits a stale full-state replacement.

All hashes use the existing `canonical_json()` and `stable_hash()` functions
over versioned, domain-separated objects:

- `draft_hash` hashes
  `{"schema": "composer.pipeline-proposal.v1", "pipeline": <exact custody-safe args>}`;
- `base_composition_content_hash` retains the existing `_composition_content_hash`
  preimage: sources, nodes, edges, outputs, and metadata only;
- `reviewed_anchor_hash` hashes
  `{"schema": "guided.reviewed-anchors.v1", "facts": <deep-frozen reviewed facts>}`;
- `audit_payload_hash` hashes the separately redacted audit projection;
- per-call skill evidence hashes the actual rendered messages and advertised
  tool schemas for that invocation, not the aggregate staged-file set.

Safe secret-reference marker objects participate in hashes exactly as stored.
Credential literals, resolved secret values, raw inline content, timestamps,
and mutable object identity never enter these preimages.

Do not store credential literals, resolved secret values, or raw inline-blob
content in proposal, checkpoint, or validation audit payloads. Safe
`{secret_ref: NAME}` markers remain persistable and receive the existing
redaction treatment. Raw validation messages are forbidden; audit stores only
the current allowlisted validation fields.

When intent includes inline source content, the controller performs audited,
idempotent session-blob materialization before the proposal becomes reviewable,
replaces legacy `source.inline_blob` with `source.blob_id`, and computes
`draft_hash` afterward. It must not write `source.options.blob_ref`, which the
canonical boundary rejects. Named `sources.<name>.blob_id`/`inline_blob` remain
globally unavailable while canonical `set_pipeline` v1 rejects them; both modes
share that limitation until the canonical contract changes. Retries reuse the
materialized blob. Rejection leaves it under the existing session-blob
retention/cleanup policy rather than performing an unaudited destructive
cleanup. Canonical `set_pipeline` redaction does not
automatically protect the separate guided payload-store copy, so tests must
exercise that projection directly for accepted, rejected, failed, and migrated
proposals.

The event set includes the redacted canonical `set_pipeline` invocation,
`guided_turn_answered`, `guided_step_advanced`, committed composition version,
and the hashes and migration lineage above.

### 6.5 Rollout and rollback

Use expand/contract deployment:

1. Deploy permanent dual version-7/version-8 decoding, both turn types, the
   legacy acceptance adapter, new audit fields, and a disabled version-8 writer.
2. Deploy the frontend that understands both protocols.
3. Enable version-8 writes and `propose_pipeline` for new and resumed work.
4. Later disable legacy authoring while retaining version-7 decode and history
   rendering permanently.

The restored-session wrapper retains `source_schema_version`, and persistence
requires an explicit `target_schema_version`. While version-8 writing is
disabled, unrelated mutations of a restored version-7 session serialize back
to version 7. Enabling `propose_pipeline` selects version 8 deliberately. This
prevents phase 1 from writing version-8 checkpoints merely because an in-memory
migration occurred.

Rollback means disabling the new planner while retaining the version-8 reader
and tolerant API. Downgrading to the current version-7-only binary after any
version-8 write is unsafe. A rollback release must include the version-8 decoder
and accept the new protocol fields. Database deletion is not a rollback.

## 7. Validation and error handling

All surfaces use the same validation and explanation pipeline. Error responses
must distinguish:

- canonical schema violation;
- missing or unavailable plugin/model;
- plugin option contract failure;
- field-guarantee failure;
- invalid graph topology;
- stale proposal/version conflict;
- global web-policy rejection;
- internal construction-probe failure.

Repair feedback includes the exact safe schema fragment and affected component.
It must not collapse internal probe failures into advice that the model change a
valid plugin configuration.

The general product keeps its configured bounded repair maximum; this design
does not silently change that operational policy. The colour eval defines a
stricter `eval_max_automatic_repairs=1`. One automatic validation repair may
pass. Any operator correction after the initial intent fails the primary
derivation score, and no retry path may switch modes or simplify topology.

## 8. Verification strategy

### 8.1 Deterministic parity corpus

Verification treats interaction mode, planner prompt assembly, and workflow
profile as independent axes:

| Configuration | Interaction | Planner prompt | Profile | Invocation proof |
| --- | --- | --- | --- | --- |
| `freeform_big_bang` | Freeform | Freeform full | Ordinary | Normal composer request |
| `guided_full` | Guided planner harness | Guided full | Ordinary | Direct shared-planner eval seam |
| `guided_staged` | Guided stage protocol | Shared core + current stage | Ordinary | `/guided/start` through wire-ready |
| `tutorial_profile` | Guided stage protocol | Same staged planner | Tutorial | Schema/hash identity plus fixed tutorial journey |

The first three are arbitrary authoring configurations and run the full topology
corpus. Tutorial is not counted as a fourth arbitrary authoring journey because
its public experience intentionally fixes a lesson scenario. Mechanical
schema/planner identity proves that its profile cannot narrow capability, and
its existing scenario supplies the profile-specific behavioral regression.

Build canonical fixtures for these topology classes:

| Class | Required proof |
| --- | --- |
| Linear transform | Ordered transformations and field propagation |
| Conditional gate | True/false routing without transform emulation |
| Multi-output | Distinct output names and policies |
| Fork/coalesce | Independent branches and require-all merge |
| Multi-source queue | Named sources feeding explicit queue fan-in |
| Aggregation | Batch/aggregate node and downstream field contract |
| Row expansion | Aggregation -> `batch_replicate` -> downstream consumer, preserving upstream guaranteed fields and cardinality contract |
| Error routing | Explicit failure path with valid termination/rejoin |
| Structured LLM | Typed multi-output query fields consumed downstream |

For every fixture, exercise:

1. `freeform_big_bang`;
2. `guided_full`;
3. `guided_staged`.

Each path must commit through the shared audited `set_pipeline` dispatch.
Cross-surface semantic
equivalence uses graph isomorphism: it may canonicalize generated node ids,
connection names, stable map/list ordering, output temp paths, audit timestamps,
composition version, rationales, and session/profile metadata. It must preserve
node and plugin kinds, normalized options, directed edge roles, route labels,
topology, conditions, policies, merge modes, field contracts, output business
schemas, and failure policies. Assert equivalent `CompositionState`, validation
result, runtime graph, and generated public YAML under that normalization.

The deterministic matrix therefore contains at least 27 positive cases: three
arbitrary authoring configurations by nine topology fixtures. It uses no
network/provider calls, contains no skips or expected failures, and completes
within 60 seconds on CI.

Schema identity plus examples do not mathematically prove the universal
capability invariant. Add property-based generation of valid canonical pipeline
arguments/DAGs and pass them through all three planner configurations. Assert
the advertised schema accepts the same generated values and the shared commit
path yields an equivalent runtime graph. The nine fixtures remain readable
structural regression examples.

Use a deterministic fake-completion seam that emits each fixture/generated
payload through the real planner response schema, tool-call parser, proposal
construction, candidate validation, and audited commit boundary. Do not inject
an already-constructed `PipelineProposal` or `CompositionState`, which would
bypass the behavior under test.

### 8.2 Skill-surface contract tests

Tests must assert:

- the shared capability-core hash participates in freeform, guided-full, and
  guided-staged prompt hashes;
- all planning surfaces use the canonical proposal schema;
- guided prompts contain no capability disclaimers;
- staged prompt scoping changes interaction guidance only;
- the topology stage exposes the complete discovery set;
- adding a canonical node/structural field causes a failing parity-coverage
  test until shared assistance covers it.

Add mutation controls that deliberately remove `fork_to`, queue support,
coalesce `merge`, or a second source from one guided schema and prove the parity
gate fails.

Remove tests that treat `chain_in`/`main` as universal guided wiring or prompt
line count as a capability acceptance criterion. Retain linear topology as one
fixture, not the definition of guided mode.

### 8.3 Repair and persistence contracts

Add two deterministic repair cases:

1. The first canonical proposal fails with a real structured validation error;
   the planner repairs once and commits. Assert one repair, atomic rejection of
   the first mutation, and no mode handoff.
2. The repaired proposal also fails. The surface fails closed after its bounded
   budget without simplifying topology, importing hidden YAML, applying a
   recipe-only substitute, or switching to freeform.

Version-7 migration/resume cases must normalize to the same graph signature as
newly authored canonical proposals.

Security regressions inspect private checkpoints, redacted audit projections,
failed/rejected proposals, and migrated version-7 records and prove that none
contains credential literals, resolved secrets, or raw inline-blob content.

The persistence regression corpus includes strict version-8 round-trip,
unknown-version rejection, version-7 restore at every wizard step, pending
proposal and pending edit, Step 4, completed and exited sessions, process
restart, stale-client legacy acceptance, revert to version-7 history followed
by continued editing, fork from both versions, hash-conflict rejection, exact
historical payload-hash preservation, and behavior rollback with persisted
version-8 rows.

### 8.4 Harness surface dimension

Extend the composer eval harness with an explicit authoring surface:

```text
surface = freeform | guided_full | guided_staged | tutorial_profile
```

`guided_staged` and `tutorial_profile` scenarios must call `/guided/start` and
drive the real stage protocol. Posting only to the general message endpoint
does not exercise those configurations. `guided_full` uses the explicitly
defined direct `plan_pipeline()` eval seam.

Score normalized topology and field contracts, not exact model prose, tool-call
sequence, or generated connection names. A freeform handoff, canned recipe, or
manual test-state injection cannot count as a guided pass.

### 8.5 Two-LLM colour split/merge acceptance

Materialize this exact committed eval fixture and record its SHA-256:

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

Use the outcome-oriented request from the companion colour design/run sheet.
The request must say, in plain English:

- use two independent LLMs in parallel;
- both receive `color_name` and `hex`;
- blue/red amount fields are integers in `0..100`;
- blue/red confidence fields are numbers in `0..1`;
- reasons are concise non-empty sentences;
- wait for both assessments and emit exactly the eight named fields;
- remove raw responses, token/model metadata, and branch bookkeeping;
- write successes as one JSON array of ten objects and failures to a separate
  JSON output;
- use deterministic sampling where supported and an economical configured
  model.

It may name the required plugins, but it must not name composer tools,
prescribe call order, provide tool argument JSON, or import hand-authored YAML.

The accepted topology is:

```text
colour source (10 rows)
        |
  explicit row fork
      /     \
 blue LLM  red LLM
    |  \     /  |
    | failures  |
    +-----> failure JSON
      \     /
 require-all union coalesce
        |
 exact-field cleanup
        |
 success JSON array
```

Acceptance requires:

- two independent LLM nodes, not one node with two queries;
- both branches receive every source row;
- a real require-all union coalesce produces one hybrid row;
- one success JSON sink and one failure JSON sink;
- every relevant LLM, coalesce, and cleanup error path reaches the failure sink;
- exactly ten successful output rows and zero failure rows;
- exactly these eight semantic fields:
  `color_name`, `hex`, `blue_amount`, `blue_confidence`, `blue_reason`,
  `red_amount`, `red_confidence`, `red_reason`;
- amount fields are integers, confidence fields are numbers, and reason fields
  are strings;
- amount fields are true integers in `0..100`; confidence fields are finite
  non-boolean numbers in `0..1`; reasons are non-empty strings;
- raw LLM responses, usage, model metadata, and branch helper fields are absent;
- valid graph and downstream field contracts;
- no capability handoff or manual graph correction;
- no timeout and at most one validation repair.

The ten input identities must appear exactly once in the successful artifact.
As a semantic smoke test, Pure Blue and Navy must have more blue than red, while
Pure Red and Orange must have more red than blue. Do not require exact numeric
equality across modes.

Run accounting must show ten blue assessments, ten red assessments, ten
completed coalesces, ten successful hybrid rows, zero failed rows, zero pending
tokens, and closed audit accounting. The public terminal run-status accounting
is authoritative for `source.rows_processed`, `tokens.succeeded`,
`tokens.failed`, `tokens.pending`, and `integrity.closure`. Landscape audit
records grouped by logical row/token lineage and LLM/coalesce node id are
authoritative for twenty logical assessments and ten coalesces; provider retry
attempts are reported separately and do not count as additional logical
assessments.

For the primary derivation score, allow no ordinary-language correction after
the initial request. Permit one automatic validation repair. An
operator-corrected run may remain diagnostic evidence but does not pass intent
derivation.

Live limits per surface are:

- freeform request to valid runnable draft: 180 seconds;
- guided full-planner call: 180 seconds;
- guided-staged start to wire-ready: 360 seconds;
- execution to terminal state: 300 seconds;
- Playwright retries: zero.

Capture the session and run ids, surface/profile, exact per-call skill hash,
composition version, repair count, proposal/tool outcomes, final state/YAML
hashes, validation result, output artifact SHA-256, and screenshots.

Run three distinct proofs:

1. `freeform_big_bang`: live composer API/browser journey from the plain-English
   request through execution and artifact verification.
2. `guided_full`: live direct `plan_pipeline()` eval seam using the full guided
   prompt, followed by the shared candidate/commit/run path.
3. `guided_staged`: Playwright journey through `/guided/start`, every stage,
   wire review, execution, and artifact download.

All three compare the same normalized topology and business-output contract.
Only the guided-staged proof is required to be browser E2E; the other two must
use live provider-backed planning and the same server commit/runtime paths.

### 8.6 Tutorial regression

The tutorial's existing example must remain green, but its test is not the
parity oracle. Add a mechanical assertion that tutorial profile uses the same
planner, canonical proposal-schema hash, effective discovery/tool schemas, and
shared-capability-core hash as guided-staged. The tutorial may provide fixed
data and teaching copy only.

## 9. Documentation changes

Update the user manual to remove the claim that guided is only for linear
pipelines or that freeform is required for custom branching, multiple sources,
or aggregation. Replace it with:

- guided and freeform have identical authoring capability;
- guided stages and reviews decisions;
- freeform accepts end-to-end and incremental requests;
- switching modes changes interaction, not the pipeline language.

Document tutorial as a guided workflow profile rather than a separate or
reduced composer implementation.

## 10. Alternatives considered

### 10.1 Expand `ChainProposal` into a guided DAG model

Add identifiers, node types, routes, edges, queues, merges, sources, and outputs
to the existing guided proposal.

**Rejected:** this creates a second copy of the canonical authoring schema. It
would require permanent parity adapters and would drift when new composer
capabilities arrive.

### 10.2 Hand complex guided requests to freeform

Keep the linear guided planner and treat freeform as its escape hatch.

**Rejected:** this preserves the defect, makes the default surface misleading,
and cannot satisfy guided capability parity.

### 10.3 Add more guided recipes

Create recipes for fork/coalesce, multi-source, aggregation, and common LLM
patterns.

**Rejected as the primary solution:** recipes improve speed and reliability for
known shapes but cannot cover the open-ended canonical language. They also
weaken the intended test that the model can derive a valid pipeline from plain
English. Recipes remain optional accelerators after parity exists.

## 11. Consequences

### Positive

- Guided becomes the superior interaction surface without tutorial-era limits.
- New canonical capabilities reach every composer mode by construction.
- One validation and commit path reduces semantic drift.
- Cross-surface eval failures identify planning quality rather than missing
  topology fields.
- The colour demonstration becomes a reusable foundation for later aggregators
  and statistical analysis.

### Negative

- The change touches persisted guided sessions, protocol types, frontend review,
  prompt composition, and eval infrastructure.
- Full canonical proposals are larger than transform-step lists and require
  deliberate prompt caching and bounded schema delivery.
- Guided stage rewind semantics must handle source/output changes cleanly.
- Version-7 session migration requires permanent dual-version decoding and
  careful rollout.

### Neutral

- Guided can still hide complexity from novice users; it simply cannot hide
  capabilities from its planner.
- Freeform incremental editing remains useful even though new-pipeline planning
  is shared.
- Recipes remain useful but lose their role as capability substitutes.

## 12. Acceptance criteria

The design is implemented only when all of the following are true:

1. No guided-only pipeline topology model remains in the active authoring path.
2. Freeform, guided-full, and guided-staged use the canonical full-pipeline
   proposal schema and shared audited `set_pipeline` commit seam.
3. Guided supports multiple sources, multiple outputs, every canonical node
   type, arbitrary valid routing, queues, fork/coalesce, aggregation, and
   failure paths.
4. Full and staged skill packs include the shared capability core and identical
   plugin/structural assistance.
5. Tutorial profile uses the guided-staged planner without reducing its schema.
6. Version-7 sessions resume or migrate without audit-history rewriting.
7. The secret-reference construction-probe regression is fixed.
8. Deterministic fixtures and property-generated valid DAGs pass on all three
   arbitrary authoring configurations; tutorial profile passes planner/schema
   identity and its fixed behavioral regression.
9. The live ten-row two-LLM colour scenario passes on freeform, guided-full,
   and guided-staged; guided-staged supplies the Playwright execution proof.
10. User documentation no longer recommends freeform for a topology that guided
    should be able to author.
11. Invalid planner candidates never become the current `CompositionState`, and
    accepted proposal arguments reach the audited commit seam unchanged.
12. Checkpoint/audit/hash tests prove permanent version-7 readability,
    version-8 rollback readability, proposal immutability, and absence of
    credential literals, resolved secrets, or raw inline content.

## 13. Implementation decomposition

This architecture spans multiple persistence, planner, protocol, frontend, and
evaluation boundaries. Implementation planning must produce an ordered plan set
rather than one monolithic change:

1. Fix the secret-reference construction probe and lock the existing canonical
   authoring schema with structural compatibility tests.
2. Extract the shared planner and canonical proposal envelope without changing
   guided persistence.
3. Add permanent version-7/version-8 restore, protocol, concurrency, audit, and
   frontend compatibility.
4. Switch guided-full, guided-staged, and tutorial-profile planning to the
   canonical proposal and commit path.
5. Replace duplicated prompt rules with the shared capability core and typed
   plugin assistance.
6. Land the three-configuration deterministic/property parity corpus plus the
   tutorial profile identity guard before enabling version-8 authoring by
   default.
7. Enable the new path, update user documentation, and run the live colour
   acceptance on all required surfaces.

Each slice must leave the repository valid and rollback-readable. No individual
slice may close the capability-parity work before the full acceptance criteria
pass.

## 14. References

- `src/elspeth/web/composer/guided/protocol.py`
- `src/elspeth/web/composer/guided/state_machine.py`
- `src/elspeth/web/composer/guided/chain_solver.py`
- `src/elspeth/web/composer/guided/steps.py`
- `src/elspeth/web/composer/guided/prompts.py`
- `src/elspeth/web/composer/guided/skills/base.md`
- `src/elspeth/web/composer/guided/skills/step_3_transforms.md`
- `src/elspeth/web/composer/guided/skills/step_4_wire.md`
- `src/elspeth/web/composer/tools/sessions.py`
- `src/elspeth/web/composer/state.py`
- `src/elspeth/plugins/transforms/llm/`
- `tests/integration/web/composer/guided/`
- `tests/unit/web/composer/guided/`
- `evals/composer-harness/`
- `docs/guides/user-manual.md`
- `docs/superpowers/specs/2026-07-13-two-llm-colour-hybrid-pipeline-design.md`
- `docs/superpowers/plans/2026-07-13-two-llm-colour-hybrid-demo-run-sheet.md`
- `docs-archive/2026-06-28-docs-cleanout/docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md`
- `docs-archive/2026-06-28-docs-cleanout/docs/superpowers/specs/2026-06-25-llm-primary-guided-creation-design.md`
- `docs-archive/2026-07-08-docs-cleanout/docs/superpowers/specs/2026-06-30-guided-mode-reframe-design.md`

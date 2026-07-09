# ELSPETH Composer Guide

**Document date:** 8 July 2026
**Release covered:** 0.7.0
**Audience:** Evaluators, programme teams, operators, and technical reviewers
**Register:** Public-facing / lightly technical
**Status:** Current capability guide

## What Composer Is

Composer is the web authoring surface for ELSPETH pipelines. It helps a user
describe a workflow, choose data sources and destinations, add transformation
steps, validate the result, and either run the pipeline or hand it to someone
else for review.

The important point is not that Composer uses chat. The important point is that
Composer turns a conversation into an auditable pipeline artifact. The generated
pipeline is still validation-gated, exportable as YAML, and backed by the same
audit and lineage model as hand-authored ELSPETH pipelines.

Use Composer when you want a guided way to build a pipeline without starting
from a blank YAML file.

## What You Can Do

| Need | Composer capability |
|---|---|
| Start from plain-language intent | Describe the workflow in chat or use guided mode to let the model build each stage from an operator instruction. |
| Build with structure | Add or revise a source, sink, transforms, final wiring, routing, batch steps, and plugin options through controlled UI turns. |
| Keep the operator in control | Accept, reject, or edit proposed changes before they become part of the composition. |
| Check readiness | Use the audit-readiness and live verification panels to see validation, plugin trust, provenance, retention, LLM interpretation, and secret status. |
| Review the shape | Inspect the graph view and rendered YAML before running or sharing. |
| Handle credentials safely | Reference secrets by name instead of placing secret values in pipeline configuration. |
| Preserve work in progress | Resume after an interrupted authoring session with transcript, redacted tool rows, and state diffs. |
| Finish in the right way | Save for review, run the pipeline, or export YAML depending on the user's workflow. |

## The Authoring Experience

Composer supports three authoring paths.

| Path | Best for | How it feels |
|---|---|---|
| First-run tutorial | New users learning the vocabulary | A short guided walkthrough that builds a simple pipeline and explains the audit story. |
| Guided mode | Users who want structure | A conversational builder that uses an LLM to construct source, sink, transforms, and final wiring one stage at a time. |
| Freeform mode | Experienced users with a clear pipeline in mind | A chat-first surface where the user describes the pipeline and reviews proposed changes. |

Guided mode is the default for new sessions unless the user changes their
Composer preference. Users can switch modes during a session without changing
their account default.

In 0.7.0, guided mode is LLM-primary. The operator stays in control, but the
model proposes the concrete stage change through `/guided/chat`; ELSPETH applies
that proposal to the in-progress pipeline only after validation, presents the
plain-language gloss and graph impact, and gates final wiring on advisor
sign-off.

## How Composer Keeps Work Auditable

Composer treats authoring as part of the evidence chain.

- The system records LLM calls made during composition, including provider,
  model, status, latency, and token counts.
- The system records tool invocations used to change the composition state.
- Tool arguments are redacted before they are shown back to the user where
  sensitive fields may be involved.
- Generated YAML is validated before execution.
- The audit-readiness panel shows whether the composition has enough evidence
  to run or share.
- If an LLM-assisted step depends on a subjective interpretation, Composer can
  surface that interpretation for review instead of silently deciding it.
- Guided sessions surface pending interpretation cards before persistence can
  advance, and the wire stage can request an advisor rather than auto-complete.

This does not make the language model an authority. The language model proposes
changes. ELSPETH records, validates, and gates the resulting pipeline.

## Readiness Panel

The audit-readiness panel is the operator's compact answer to "is this pipeline
safe to move forward?"

| Row | What it tells you |
|---|---|
| Validation | Whether the current YAML shape passes runtime-oriented validation. |
| Plugin trust | Whether the chosen plugins fit the expected trust and capability model. |
| Provenance | Whether source and composition evidence can be traced. |
| Retention | Whether output and audit retention expectations are visible. |
| LLM interpretations | Whether subjective LLM interpretation points have been reviewed or opted out. |
| Secrets | Whether credentials are referenced safely and resolvably. |

Warnings are not hidden. They are there so a user can fix the pipeline before
running it, or share it with an explicit caveat.

## Completion Options

Composer gives three ways to finish a composition.

| Action | What happens |
|---|---|
| Save for review | Composer marks the current composition as ready for another person to inspect, creates a signed share link, and shows the reviewer the same readiness and YAML evidence. |
| Run pipeline | ELSPETH starts a background run, streams progress, and records the run in the audit trail. |
| Export YAML | Composer renders the pipeline as YAML so an operator or engineer can run, review, or store it outside the web UI. |

These actions are deliberately separate. A compliance reviewer may want to save
for review without running anything. A researcher may want to run immediately.
An engineer may want YAML and the CLI.

## Recovery

If a Composer session is interrupted, the recovery panel can show:

- the assistant transcript;
- redacted tool-call rows;
- the before-and-after composition diff;
- the visible failure reason where one is available.

The goal is not simply to "try again." The goal is to show what happened and let
the user decide what to keep.

## What Composer Is Not

Composer is not a replacement for assurance review, operational ownership, or
agency approval.

It does not certify a pipeline as fit for a regulated workload. It helps a user
build a pipeline, see its validation state, preserve evidence of authoring, and
hand the result to the right next step.

## Where To Go Next

- Read [`guarantees.md`](guarantees.md) for the audit and lineage guarantees
  that apply once a pipeline runs.
- Read [`platform-architecture.md`](platform-architecture.md) for the broader
  system shape.
- Read [`assessment-mapping.md`](assessment-mapping.md) for the public-sector
  evaluation map and caveats.

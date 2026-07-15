# State Engine Assurance Hub

This directory is the permanent home for analysis, verification, remediation,
and decision preparation for Elspeth state engines. The current programme covers
the **durable token scheduler** from source-plugin ingress to sink-plugin
durability. The web session engine is explicitly out of scope and will receive a
separate assessment.

## Current verdict

> **Not yet confirmed complete.** The implementation is mapped, Wave 1 has
> executed evidence, and Wave 2 is edit-ready. Confirmed defects and proof gaps
> still prevent a complete or production-assured verdict.

| Property | Current value |
| --- | --- |
| Current engine | Durable token scheduler |
| Current assessment | [2026-07-15 14:59 AEST](assessments/2026-07-15-1459/00-coordination.md) |
| Repository baseline | `release/0.7.1` at `0dcd61acaa44082d93ec205683700e798748ee6d` |
| Executed campaign state | Wave 1 complete; Wave 2 reconnaissance complete and test edits planned |
| Wave 1 result | 3 Confirmed legs; 15 Gap legs across the assessed Wave 1 scope |
| Current hard blockers | Invalid scheduler subtypes, implicit unfenced writes, and unproved crash/concurrency seams |
| Normative ADR status | Deferred until the verification campaign resolves every state-engine leg |

The verdict applies only to the named baseline and evidence recorded in the
current assessment. It is not a release certification.

## Evergreen documents

- [Completeness criteria](completeness-criteria.md) defines the stable boundary,
  proof standard, hard gates, and completion rule.
- [Assessment framework](assessment-framework.md) defines how to collect,
  classify, execute, and preserve evidence.
- [Legacy canonical scheduler map](../token-scheduler-state-engine.md) is the
  committed transition-level baseline from which this information area was
  created. Future assessments live here; the legacy path remains stable while
  existing documentation still links to it.

## Current assessment

- [Coordination and scope](assessments/2026-07-15-1459/00-coordination.md)
- [Discovery findings](assessments/2026-07-15-1459/01-discovery-findings.md)
- [State-engine map](assessments/2026-07-15-1459/02-state-engine-map.md)
- [Executed evidence](assessments/2026-07-15-1459/03-executed-evidence.md)
- [Gap analysis](assessments/2026-07-15-1459/04-gap-analysis.md)
- [Remediation plan](assessments/2026-07-15-1459/05-remediation-plan.md)
- [Evidence inventory](assessments/2026-07-15-1459/evidence/README.md)
- [Assessment provenance](assessments/2026-07-15-1459/provenance/README.md)

## Document roles

| Layer | Mutable? | Purpose |
| --- | --- | --- |
| This README | Yes | Point to the current verdict and latest assessment. |
| Completeness criteria | Rarely | Define the contract for declaring a state engine complete. |
| Assessment framework | Rarely | Define repeatable evidence and scoring practice. |
| Dated assessments | No | Preserve what was observed, executed, and concluded at one baseline. |
| Superseding ADR | Versioned decision | State durable guarantees only after the campaign confirms them. |

## Update policy

1. Never rewrite a completed dated assessment to make it match newer code.
2. Correct a material error with an explicit addendum in that snapshot, then
   carry the correction into the next assessment.
3. Create a new timestamped assessment when code, schema, runtime choreography,
   proof criteria, or a material verdict changes.
4. Update this README only after the new assessment passes its validation gate.
5. Keep observed implementation, executed evidence, remediation intent, and
   accepted architecture decisions in separate documents.
6. Promote only positively confirmed, deduplicated gaps into Filigree.

## Naming

- `TS-*`: durable scheduler status transitions.
- `AUX-*`: state-preserving or adjacent compare-and-swap mutations.
- `PB-*`: plugin and orchestrator boundary paths.
- `RM-*`: orchestration read-model predicates or decisions.
- `F-*`: forbidden, refused, or intentionally absent paths.

These identifiers remain stable across assessments. If implementation changes a
leg's meaning, the new assessment records the change instead of silently reusing
the old definition.

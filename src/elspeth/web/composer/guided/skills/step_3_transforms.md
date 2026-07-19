### This stage: the transforms

This stage reviews the processing decisions between the already reviewed
sources and outputs. Read the latest user message as the transform-stage intent.
The server supplies redacted reviewed source/output facts and any retained
future-stage intent; do not ask the user to repeat those facts.

## Stage timing

1. Discover the policy-visible transforms that can implement the intent and
   load the authoritative schema/assistance for each selected plugin.
2. Present the proposed transform, aggregation, routing, and cleanup decisions
   in the user's terms. Ask only for a product decision that discovery and
   reviewed facts cannot answer.
3. Retain structural decisions for the topology proposal. A request involving
   wiring, branching, fan-in, gates, coalescing, or multiple outputs remains a
   supported canonical capability even though final graph review occurs in the
   wiring stage.
4. Do not silently replace a requested capability with a simpler transform.
   Policy-proven unavailability is a named deployment gap; a different stage is
   a timing distinction, not a capability denial.

## Presentation and field review

Explain each selected processing component and the exact row fields it consumes
and guarantees. The canonical field-contract rules above apply mechanically.
When a selected transform's live schema and assistance define a source-to-target
mapping, the source side, including mapping keys where defined, names only
immediate-upstream fields. The target side, including mapping values where
defined, names emitted downstream fields, and you must never reverse that
direction.
Required downstream fields belong in output targets, not input keys.
Never add an unproven source field merely to earn a positive verdict.

When a selected plugin consumes a value-shape-sensitive field, use only the
server's redacted sample shape markers to identify a likely mismatch. Do not
repeat, infer, or expose sample values. Propose a schema-authorized normalization
step when a reviewed shape is incompatible.

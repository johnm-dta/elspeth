// data-trust.typ — ELSPETH Data Trust Primer (RC-5.2)
// Audience: technical users, reviewers, integrators, and assurance staff.
//
// This document compresses docs/guides/data-trust-and-error-handling.md
// into a release-PDF primer. It intentionally avoids internal
// "mandatory reading" and coding-instruction framing; the goal is to
// explain how the product thinks about trust, errors, quarantine, and
// audit integrity.

#import "tokens.typ": *
#import "theme.typ": *
#import "data.typ" as data

#show: document-frame.with(
  title: "ELSPETH Data Trust",
  subtitle: "RC-5.2 -- " + data.doc-date,
  draft: false,
  h1-pagebreak: false,
)

#cover-page(
  title: "Data Trust Primer",
  subtitle: "How ELSPETH decides whether to quarantine, continue, or crash.",
  doc-date: data.doc-date,
  version: "RC-5.2",
  author: "John Morrissey, CTO Branch",
  affiliation: "Digital Transformation Agency",
  audience: "Technical users, reviewers, integrators, assurance staff",
  classification: default-classification,
  status: "Primer -- current as of " + data.doc-date,
  distribution: "Internal -- Digital Transformation Agency only",
)

#outline(
  title: text(font: font-body, size: size-h1, weight: "bold",
    fill: c-navy, "Contents"),
  indent: auto,
  depth: 2,
)
#pagebreak()

= Why trust tiers matter

ELSPETH handles different kinds of data differently. The right error
response depends on whose data failed: external input, pipeline data
that already crossed validation, or ELSPETH's own audit record.

#callout(kind: "note", title: "The short version")[
  External data is validated and may be quarantined. Pipeline data is
  trusted for type but can still fail value-level operations. Audit
  data is ELSPETH's legal record; anomalies there crash immediately
  rather than being smoothed over.
]

#pdf.artifact(align(center, diagram-trust-tiers()))

#text(size: size-small, fill: c-ink-soft)[
  *Figure -- trust flow.* External source data starts at zero trust,
  crosses validation into pipeline data, and is then recorded into
  the Landscape audit trail. The trust level determines the failure
  response.
]

= The three tiers

== Tier 3: External data

External data is treated as zero trust. CSV files, JSON responses,
database rows from outside ELSPETH, HTTP payloads, LLM outputs, and
message-queue payloads may be malformed, incomplete, surprising, or
hostile.

At this tier, validation and coercion are appropriate. A source may
turn `"42"` into `42` or `"true"` into `true` if its contract allows
that normalisation. Data that cannot be validated is quarantined, and
the quarantine reason becomes part of the audit trail.

== Tier 2: Pipeline data

Pipeline data has crossed a validation boundary. ELSPETH trusts its
types: if a transform expects an integer, it should receive an integer.
Wrong types at this point indicate an upstream system bug, not a user
data problem to silently repair.

Tier 2 still requires care when performing operations on values. A
valid integer can be zero in a divisor, a valid string can be an
invalid date, and a valid list can be empty. Those operation-level
failures are handled as row outcomes rather than hidden as defaults.

== Tier 1: Audit data

Audit data is ELSPETH's own record: the Landscape database, run
configuration stored for a run, node states, call records, payload
hashes, and terminal outcomes. ELSPETH wrote this data and relies on
it to explain what happened.

At this tier, anomalies are not repaired by coercion or defaulting.
If the audit trail contains an impossible value, the system should
fail loudly because silent repair would undermine the evidence record.

= Failure responses

#data-table(
  columns: 4,
  header: ("Tier", "Typical data", "Correct response", "Why"),
  align-rules: (left, left, left, left),
  ("Tier 3", "Source rows, provider responses, external files", "Validate, coerce if allowed, quarantine malformed input", "The data belongs to the outside world"),
  ("Tier 2", "Validated pipeline rows", "Trust declared types; handle value-level operation failures", "The contract was already checked, but values can still be unusable"),
  ("Tier 1", "Landscape and audit records", "Crash on anomaly", "The audit trail is the evidence record"),
)

#callout(kind: "advisory", title: "Crash can be the safer outcome")[
  In an audit-bearing system, silently producing a plausible but wrong
  answer is worse than stopping. A crash tells the operator the record
  cannot be trusted; a quiet default may create false assurance.
]

= Boundaries inside transforms

Trust tiers are about data flow, not just plugin type. A transform may
receive Tier 2 row data and then call an external service. The row
remains Tier 2, but the response from the external service is Tier 3
until it is parsed and validated.

The practical rule is to keep the distance between external response
and validation as short as possible. Parse the response, check its
shape, validate required fields, and only then add the validated
result back to the pipeline row.

== Common boundary patterns

#data-table(
  columns: 3,
  header: ("Boundary", "Validate immediately", "After validation"),
  align-rules: (left, left, left),
  ("LLM response", "JSON parse, object shape, required fields", "Use explicit fields as pipeline data"),
  ("HTTP API", "Status, content type, response schema", "Record selected fields and continue"),
  ("External database", "Row shape, missing fields, type coercion rules", "Treat validated result as pipeline input"),
  ("File read in transform", "File format and expected fields", "Continue with validated content"),
  ("Message queue", "Payload format and schema", "Quarantine malformed messages"),
)

= Plugin ownership

ELSPETH plugins are system-owned code. Sources, transforms,
aggregations, and sinks are developed and tested as part of ELSPETH;
users configure them, but do not provide arbitrary plugin code in the
current architecture.

That distinction matters. A plugin defect is ELSPETH's bug and should
stop the run rather than quietly inventing an answer. A user-data
defect is part of the domain being processed and should be recorded,
quarantined, or surfaced according to policy.

#data-table(
  columns: 3,
  header: ("Scenario", "Response", "Register"),
  align-rules: (left, left, left),
  ("Plugin method throws unexpectedly", "Crash", "System bug"),
  ("Plugin returns wrong contract type", "Crash", "System bug"),
  ("User row has a malformed field", "Quarantine row", "External data problem"),
  ("External provider returns invalid JSON", "Return recorded error result", "External boundary failure"),
  ("Audit database contains impossible state", "Crash", "Audit integrity failure"),
)

= Coercion rules

Coercion belongs at external boundaries. Source plugins may normalise
external data when their contract allows it. Transforms and sinks do
not coerce pipeline row types; a type mismatch after source validation
is an upstream defect to fix.

#data-table(
  columns: 3,
  header: ("Surface", "Coercion", "Reason"),
  align-rules: (left, left, left),
  ("Source input", "Allowed", "Normalises zero-trust external data at ingestion"),
  ("Transform row input", "Not allowed", "Receives validated pipeline data"),
  ("Transform external response", "Allowed at response boundary", "The response is Tier 3 until validated"),
  ("Sink input", "Not allowed", "Receives validated pipeline data"),
  ("Audit record", "Never", "The record must be exact or fail"),
)

= Reading an error

When ELSPETH reports an error, the first question is "whose data
failed?" That question usually determines whether the system should
quarantine a row, record a provider failure, or stop the run.

#data-table(
  columns: 2,
  header: ("Question", "Interpretation"),
  align-rules: (left, left),
  ("Did external input fail validation?", "Quarantine or reject the affected input with a recorded reason"),
  ("Did a valid row fail an operation?", "Record a row-level error outcome"),
  ("Did a provider fail?", "Record the external call failure and apply configured retry policy"),
  ("Did system-owned code violate a contract?", "Crash and fix the bug"),
  ("Did the audit record look corrupt?", "Stop; do not invent a corrected record"),
)

= Assurance value

The trust model is not just an implementation style. It is how ELSPETH
keeps audit evidence honest. External data can be messy without
invalidating the whole run. Validated pipeline data can still produce
explainable row-level failures. The audit trail remains exact enough
that a reviewer can trust it when it says what happened.

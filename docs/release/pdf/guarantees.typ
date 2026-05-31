// guarantees.typ — ELSPETH Assurance Guarantees (RC-5.2)
// Audience: users, integrators, auditors, and assurance staff.
//
// This document compresses docs/release/guarantees.md into a
// release-PDF formatted assurance appendix. It is intentionally
// shorter than the markdown: the markdown keeps versioned clause
// wording, while this PDF gives reviewers a clear release-reading
// copy of what ELSPETH promises and where the boundaries sit.

#import "tokens.typ": *
#import "theme.typ": *
#import "data.typ" as data

#show: document-frame.with(
  title: "ELSPETH Guarantees",
  subtitle: "RC-5.2 -- " + data.doc-date,
  draft: false,
  h1-pagebreak: false,
)

#cover-page(
  title: "Assurance Guarantees",
  subtitle: "What ELSPETH promises about lineage, execution, data, recovery, identity, and authored pipelines.",
  doc-date: data.doc-date,
  version: "RC-5.2",
  author: "John Morrissey, CTO Branch",
  affiliation: "Digital Transformation Agency",
  audience: "Users, integrators, auditors, assurance staff",
  classification: default-classification,
  status: "Assurance appendix -- current as of " + data.doc-date,
  distribution: "Internal -- Digital Transformation Agency only",
)

#outline(
  title: text(font: font-body, size: size-h1, weight: "bold",
    fill: c-navy, "Contents"),
  indent: auto,
  depth: 2,
)
#pagebreak()

= The core promise

Every output produced by ELSPETH can be traced back to its source
with a complete audit trail. For a produced output, a reviewer can
ask "why?" and recover the source row, transforms applied, external
calls made, routing decisions taken, and the final disposition.

#callout(kind: "note", title: "Assurance posture")[
  These are not aspirational features. The guarantees below are
  release-blocking commitments: if a current release cannot uphold
  one of them, that is a bug to fix before public release.
]

== What the audit trail must answer

#data-table(
  columns: 3,
  header: ("Question", "Recorded evidence", "Why it matters"),
  align-rules: (left, left, left),
  ("Where did this output come from?",
   "Original source row and stable source hash",
   "The result is attributable to received data, not just to an output file"),
  ("What happened to it?",
   "Node-by-node transform chain with input and output hashes",
   "The processing path is inspectable after the run"),
  ("Did it leave the system?",
   "Terminal outcome: completed, routed, forked, consumed, coalesced, quarantined, failed, or expanded",
   "No row disappears without a recorded outcome"),
  ("Did external systems influence it?",
   "External call request, response, timing, provider, and status metadata",
   "LLM and HTTP decisions remain reviewable"),
  ("Why did it route there?",
   "Gate decision and destination",
   "Routing is visible policy, not hidden control flow"),
)

= Audit and execution

ELSPETH records complete lineage for rows, tokens, routing decisions,
external calls, sink outputs, and terminal outcomes. Hashes are
stable and versioned, payload retention preserves metadata and
integrity evidence, and `explain()` reports when retained payloads
are no longer available.

Execution follows the declared DAG. Transforms run in dependency
order, forked tokens remain isolated, and gates route deterministically
to configured destinations. Retries are explicit: each attempt is a
separate audit record with a clear final outcome.

#callout(kind: "advisory", title: "Timeout boundary")[
  Timeout aggregation is currently checked when a new row arrives or
  when the source completes. ELSPETH does not claim true idle-timeout
  behaviour in RC-5.2.
]

== Runtime guarantees

#data-table(
  columns: 3,
  header: ("Area", "Guarantee", "Boundary"),
  align-rules: (left, left, left),
  ("DAG order", "Dependencies are respected before downstream transforms run", "The configured graph defines the order"),
  ("Forking", "Branches receive isolated token state", "Sibling branches do not share mutable row state"),
  ("Routing", "Gate destinations are validated and decision reasons recorded", "Invalid destinations fail configuration validation"),
  ("Retries", "Attempts, backoff, permanent failure, and final status are recorded", "Retry policy must be explicit"),
  ("Retention", "Hashes and metadata survive payload deletion", "Expired payload bytes may be unavailable by policy"),
)

= Data, configuration, and recovery

Source data is validated at the boundary. Malformed rows are
quarantined with their reason; valid rows continue. Schema contracts
declare what plugins require and produce, and DAG construction rejects
incompatible contracts before execution.

Configurations are validated before execution. Invalid plugin
references, invalid route targets, schema mismatches, and missing
required fields fail fast. Environment variables expand at load time,
missing required values fail clearly, and execution is explicit.

Interrupted runs can resume from checkpoints. Already-processed rows
are not reprocessed, aggregation state is restored, and idempotency
keys allow compatible sinks to be safely retried.

#callout(kind: "success", title: "Safe default")[
  ELSPETH favours explicit execution and recorded failure over silent
  data modification. A dry run is safe; a real run is a deliberate
  action.
]

= External systems

External calls are part of the audit surface. ELSPETH records what
was sent, what came back, how long it took, and whether the call
succeeded or failed.

ELSPETH cannot guarantee the quality, consistency, uptime, or rate
limit behaviour of third-party systems. It guarantees that the
interaction is recorded and that configured retry/rate-limit policy
is respected.

== External-call evidence

#data-table(
  columns: 2,
  header: ("Recorded field", "Meaning"),
  align-rules: (left, left),
  ("Request and response", "Payloads and hashes used for review"),
  ("Provider and status", "Which service responded and how"),
  ("Latency", "Timing evidence for performance and incident review"),
  ("Error type", "Failure mode when the external call did not produce a usable response"),
)

= Identity, sessions, and secrets

RC-5.2 adds assurance guarantees for authenticated web and Composer
surfaces. ELSPETH supports local username/password, generic OpenID
Connect, and Microsoft Entra ID providers. Authenticated actions
record the principal that performed them.

Sessions are isolated by authenticated principal. In-progress Composer
work is persisted before the response is acknowledged, and session
mutations record both the principal and the operation.

Secrets are referenced by name in configuration and resolved at run
time. Audit records store the reference, source, timestamp, latency,
and HMAC fingerprint, not the secret value itself.

#callout(kind: "risk", title: "Authentication is not authorisation")[
  ELSPETH authenticates principals and records evidence. The deploying
  organisation remains responsible for deciding which principals may
  perform which actions and which network controls protect the service.
]

= Composer authoring

The Composer does not weaken the engine. A generated pipeline is a
configuration produced through an audited authoring surface; execution
still passes through the same validation, runtime, audit trail, and
guarantees as a hand-written YAML pipeline.

Composer transcripts are preserved so a reviewer can inspect how a
pipeline was produced. Tool-call results are recorded, and sensitive
payloads are redacted according to the redaction manifest; the fact
of redaction is itself recorded.

A failed Composer operation does not silently modify committed
pipeline state. Failed validation, failed save, and failed recovery
paths preserve the prior committed state and surface the failure to
the operator.

= What is not guaranteed

#data-table(
  columns: 2,
  header: ("Not guaranteed", "Reason"),
  align-rules: (left, left),
  ("High-throughput streaming or sub-millisecond latency", "Correctness and auditability are prioritised over throughput"),
  ("Third-party model quality", "ELSPETH records external responses; it does not control the provider"),
  ("Organisational access policy", "Identity and evidence are recorded, but deployment policy remains organisational"),
  ("Network isolation", "TLS, reverse proxy, firewall, and hosting controls belong to the deployment environment"),
  ("True idle aggregation timeout", "RC-5.2 timeout checks depend on new row arrival or source completion"),
)

= The test

The practical test is simple: given any output ELSPETH produced, the
audit trail must prove complete lineage to the source row, processing
chain, external calls, and terminal outcome. If that cannot be shown,
the guarantee is broken and the release must treat it as a blocking
defect.

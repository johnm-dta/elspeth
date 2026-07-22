# Assessment Review Record

Review is a technical challenge record, not an approval receipt.

Review outcome: complete

## First-pass design reviews

| Lens | Material findings | Disposition |
| --- | --- | --- |
| Documentation topology | Establish one canonical hub; make the old map a temporary pointer; keep snapshots small; separate full and delta modes; warn that the old remediation plan is not live authority | Accepted and incorporated into the hub, history, framework, and historical errata |
| Proof model | Define stable legs and ten dimensions; keep sink-effect seams as cases beneath PB-06/PB-07; derive leg/global verdicts; retain explicit hard gates and limitations | Accepted, then expanded from 54 to 63 legs for run coordination and to 68 when fresh review found five load-bearing resume/barrier read models |
| Reproducibility | Record full Git/tree/worktree identity, environment and lock hashes, command vectors, exact result/collection facts, tracker/index snapshots, establishes/does-not-establish, and historical rerun semantics | Accepted, except documentation-specific unit tests: direct parsing, link checks, literal reruns, and independent readers are the validation mechanism |

## Integrated-package reviews

Three independent, read-only readers inspected the assembled package. They made
no repository edits and created no document-package unit tests.

| Lens | Material finding | Disposition and changed files |
| --- | --- | --- |
| Architecture | The 54-leg catalog omitted distinct leader-seat and worker-registry transitions | Accepted. Added RC-01 through RC-07, RM-07/RM-08, the coordination state model, and the coordination repository to `architecture.md`, `catalog.json`, the criteria, matrix, manifest, and evidence model. |
| Architecture | Leg verdicts were asserted rather than mechanically derived | Accepted. Added hard-gate `affected_leg_ids`; the direct validator now expands cells, applies open gates, derives every leg verdict, derives family/total counts, and derives the overall result. |
| Architecture | Token subtypes and edges did not distinguish pending-sink ownership or barrier hold type, and omitted TS-13 from `LEASED -> TERMINAL` | Accepted. Corrected the architecture diagram, subtype table, edge labels, and production close description in `architecture.md`. |
| Architecture | The transaction-boundary table omitted sink reservation, inspection, plan, publication/response, finalization/response, and callback seams | Accepted. Expanded the table and linked the exhaustive PB-07 cases in `architecture.md`. |
| Evidence | The proof-matrix family counts contradicted the manifest | Accepted. The current matrix derives 68 legs, 44 gaps, and 24 unknowns; the validator checks every family row and the total row against the manifest. |
| Evidence | Literal commands and retained evidence were insufficiently reproducible | Accepted. Replaced the unavailable `python -m pip` command with `uv pip freeze`; created artifact directories before writes; parameterized assessment paths; recorded relative cwd, timeout, safe environment, resources, exact coverage tuples, node indexes and hashes, result counts, limitations, and reproducibility class. |
| Evidence | Dirty overlays and structural/history requirements could not be reconstructed literally | Accepted. Moved overlay capture outside the worktree, retained patches plus an untracked archive/path list, narrowed Loomweave retention to load-bearing claims, and split strict v1 reruns from legacy best-effort reconstruction. |
| Future agent | Profiles, initializer, delta encoding, evidence promotion, N/A, tracker ownership, and historical reruns were underspecified | Accepted. Added closed execution profiles and per-family dimension contracts, a full initializer, fully materialized deltas with parent identity and changed tuples, catalog-owned N/A validation, coherent-theme tracker ownership, and two historical rerun paths. |
| Future agent | The package still had an integrated-review placeholder | Accepted by replacing it with this challenge record and requesting fresh re-review after the changes. |

## Validation after integration

- Before fresh re-review changes, the documented contract program derived 63
  legs, 39 gaps, 24 unknowns, and `not_complete`. The fresh-review iteration
  expanded the universe to 68 legs; final validation is recorded below.
- Fresh `--collect-only` execution matched all retained node indexes exactly:
  EV-001 46, EV-002 38, and EV-003 30.
- The package uses direct JSON/contract parsing, link checks, literal command
  execution, and independent readers. No unit tests were added for the
  documentation package.

## Fresh re-review

The first fresh re-review found additional material gaps; all were accepted.

| Lens | Fresh finding | Second-iteration disposition |
| --- | --- | --- |
| Architecture | Plugin discovery returned 31 transforms while the catalog listed 24 | Added the seven omitted AWS, Azure, LLM, and RAG transforms; direct validation now compares the catalog inventory to live discovery. |
| Architecture | Resume, redrive, completion, journal restore, and barrier intake depend on five uncataloged reads | Added RM-09 through RM-13 and expanded HG-07, architecture, matrix, manifest, and proof counts from 63 to 68 legs. |
| Architecture | RM-07/RM-08 misdescribed occupied expired seats and grace-adjusted worker expiry | Rewrote both contracts and exit gates around `seat_live` and `now - grace_seconds`. |
| Architecture | RC-04 can mutate membership after a losing seat CAS; RC-07 relies on caller preselection to exclude leaders | Exposed both source/contract mismatches in architecture and HG-10, assigned `elspeth-b8d0c9b40a` and `elspeth-33c1793a26`, and kept the legs/gates open. |
| Architecture | Sink preparation ownership is a durable RESERVED subtype and restart seam; TS-14 is an ownerless pending-sink repair exception | Added the subtype and three PB-06 cases; corrected pending-sink terminalization prose and the transaction table. |
| Contract | Delta metadata was not compared with its parent; an `intentionally_absent` verdict could not be derived; evidence promotion ignored failed executions | Removed the unreachable verdict, added exact parent digest/cell/evidence/gate comparison, and required successful evidence for pass/partial promotion. |
| Contract | Baseline/environment/family/HG-09/review completeness and retained raw outputs were under-validated | Added identity/profile/plugin/family/total/HG-09/review checks. Re-executed EV-001–003 and retained JUnit/stdout/stderr plus hashes. |
| Future agent | Initializer omitted the dated README; the manifest template was an invalid empty duplicate; strict rerun lost access to the package checkout | Added a README template, removed the redundant manifest template, and split package-bearing and detached execution roots. |
| Future agent | Python optimization disabled all validator assertions; pending review records still passed | Forced `PYTHONOPTIMIZE=0`, fail closed when `__debug__` is false, scan the dated package, and require `Review outcome: complete`. |

Final fresh-reader outcome: no material findings remain. The architecture
reader verified the 68-leg/source model and open RC-04/RC-07 gaps; the contract
reader verified baseline drift, JUnit/count, artifact, delta, derivation, and
review-path enforcement; the future-agent reader verified initialization,
validation, evidence capture, and strict historical rerun instructions.

The branch moved from `36146eac4` to `422415009` during final review through
two documentation-only commits. Git proved no non-document diff, Loomweave
reported no modified indexed files, and EV-001 through EV-003 were recaptured
at the final baseline. Final direct validation records 68 legs, 44 gaps, 24
unknowns, ten open hard gates, 114 exact collected/passing nodes, nine retained
raw artifacts with matching hashes, and valid links across 28 Markdown files.
No unit tests were added for this documentation package.

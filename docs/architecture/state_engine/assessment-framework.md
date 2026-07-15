# State Engine Assessment Framework

## Purpose

Use this framework for every state-engine assessment. It makes later snapshots
comparable and keeps discovery, execution, remediation, and architectural
decision-making distinct.

## Required inputs

Record these before collecting evidence:

- branch, full commit SHA, local timestamp, Python version, and database mode;
- complete worktree status, including unrelated user changes;
- structural-index commit and freshness/disjointness evidence;
- applicable state-engine ADRs, contracts, runbooks, and prior assessments;
- live Filigree issues and observations that overlap the assessed legs;
- production repositories, orchestrators, plugin boundaries, schemas, and read
  models in scope;
- exact test commands and environmental limitations.

If the worktree contains unrelated changes, preserve them and record why they do
not affect the evidence baseline.

## Assessment workflow

### 1. Freeze the baseline

Create `assessments/YYYY-MM-DD-HHMM/` and write `00-coordination.md`. State the
scope, exclusions, baseline, intended outputs, and whether the assessment may
modify code or tracker state.

### 2. Discover production authority

Trace each public runtime entry to the authoritative state mutation or read. Do
not infer runtime use from a facade method merely because tests call it. Record:

- entry point and caller chain;
- transaction boundary;
- state and subtype preconditions;
- row, event, auxiliary, and plugin-visible consequences;
- refusal and compatibility arms;
- subsequent cross-transaction seam.

Use the structural index when fresh. When it is stale, refresh it or prove that
the indexed source/test graph is disjoint from the committed drift.

### 3. Build the leg matrix

Use stable identifiers and one row per leg:

| Leg | Production entry | Success image | Refusal image | Rollback | Concurrency | Plugin path | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `TS-00` | Exact caller chain | Test/evidence | Test/evidence | Test/evidence | Test/evidence or N/A | Test/evidence or N/A | Mapped/Candidate/Confirmed/Gap |

Never mark a leg Confirmed solely because multiple partial tests collectively
sound persuasive. The matrix must show every applicable proof dimension.

### 4. Execute the narrowest representative evidence

Run production-path tests first, then direct repository tests for atomic detail,
then property or model tests for breadth. For each command, record:

- exact command and node IDs;
- observed pass/fail/skip counts and duration;
- what the result proves;
- what it explicitly does not prove.

Mocks may prove orchestration decisions or call ordering. They cannot prove
database atomicity, process races, external plugin durability, or replay.

### 5. Probe candidate defects safely

For a suspected defect:

1. capture the complete pre-operation durable image;
2. invoke the real public method with the smallest reproducer;
3. capture the complete post-operation image;
4. demonstrate the violated invariant;
5. rerun a valid positive control;
6. deduplicate against live Filigree before filing.

Do not weaken a fail-closed invariant merely to make the reproducer pass.

### 6. Classify and reconcile

Classify every finding as one of:

- implementation defect;
- missing or inadequate evidence;
- concurrency-bound proof;
- crash-seam uncertainty;
- policy/ADR decision;
- stale documentation or tracker description;
- intentionally absent behavior.

Link pre-existing tracker ownership. File only positively confirmed gaps that do
not already have a coherent owner.

### 7. Write the remediation plan

Order work by invariant risk and dependency:

1. fail-closed subtype and fencing defects;
2. small proof ratchets that reveal the true surface;
3. cross-transaction crash/restart seams;
4. real registered multi-process proof;
5. plugin lifecycle and forbidden-path closure;
6. comprehensive ADR and CI enforcement.

Each task names exact files, tests, assertions, commands, expected RED behavior,
exit gates, and tracker ownership.

### 8. Validate and freeze

Before publishing an assessment:

- resolve every relative Markdown link;
- scan for placeholders and ambiguous verdicts;
- run `git diff --check`;
- render or lint Mermaid when tooling is available;
- verify that test counts match captured outputs;
- confirm no Candidate or Unknown claim is written as a guarantee;
- record validation limitations.

After validation, treat the dated directory as immutable.

## Reusable evidence templates

### Verification run

| Field | Value |
| --- | --- |
| Command | Exact shell command |
| Baseline | Full commit SHA |
| Result | Pass/fail/skip/warning counts and duration |
| Establishes | Narrow guaranteed conclusion |
| Does not establish | Explicit remaining boundary |

### Gap record

| Field | Value |
| --- | --- |
| Candidate | Stable candidate or leg ID |
| Classification | Defect/evidence/concurrency/crash/policy/docs |
| Reproducer | Exact test or probe |
| Durable consequence | Complete observed before/after difference |
| Existing owner | Filigree ID or `none found` |
| Next proof | Exact test to add |
| Exit gate | Observable condition for closure |

### Crash-seam record

| Field | Value |
| --- | --- |
| Durable step A | Last committed state before interruption |
| Interruption point | Exact call boundary or injected exception |
| Expected restart authority | Component that must reconcile the state |
| Allowed replay | Explicitly idempotent work, if any |
| Forbidden result | Loss, duplicate durable identity, stale mutation, or repeated external effect |
| Test form | Same-database restart, preferably process death |

## Snapshot policy

- Use local time in `YYYY-MM-DD-HHMM` directory names and record the timezone.
- Copy evidence outputs or summarize them under `evidence/`; place worker briefs,
  baseline facts, and tool limitations under `provenance/`.
- Link source by stable path and symbol. Record line numbers only as baseline
  hints because later edits may move them.
- Never delete an old assessment when a verdict improves.
- Update the hub's current pointer only after the new snapshot passes validation.

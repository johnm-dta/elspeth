# DAG completeness reassessment baseline and method

**Assessment started:** 2026-07-17 17:39 AEST
**Source branch:** `release/0.7.1`
**Assessment branch:** `codex/dag-reassessment-20260717`
**Commit:** `6e8a6bf5f2f8542bf5b95b1669ce3d3df68d93e3`
**Assessment owner:** `codex-dag-assessment`
**Filigree owner:** `elspeth-1e817cb78d`
**Verdict:** **Not complete**

## Why this snapshot exists

The 2026-07-15 seed assessment identified two reproduced scheduler-subtype
defects and an incomplete disposition proof package. Those defects are closed,
and TS-07 through TS-10 now have a maintained truth table plus complete
state/event/effect rollback coverage. Closing a recovery or atomicity hard gate
is an explicit reassessment trigger in the evergreen framework.

This package is therefore a new assessment, not an edit or re-score of the seed
snapshot. It applies the normalized 15-dimension framework but does not
calculate an aggregate while mandatory dimensions remain `U`. The seed
assessment's legacy 11-dimension indicator is not a comparison baseline.

## Frozen workspace

The live `release/0.7.1` worktree initially contained unrelated changes owned
by other in-progress issues. All source inspection, test execution, and
document editing for this assessment occurred in the clean isolated worktree:

```text
/home/john/elspeth/.worktrees/dag-reassessment-20260717
```

The worktree was initially created from `dffa6e7635b4b1cf3ce4444cb8e5da509a712e8a`.
The release branch advanced during the assessment, including sink-effect and
guided-authoring changes, so the assessment commit was rebased onto the exact
final code baseline above. Every executed evidence group was rerun against that
final baseline; the counts and outcomes in this package are the rerun results.
Live Filigree state is identified by capture time because it can postdate the
frozen code commit.

## Loomweave index

A fresh, worktree-local, non-incremental analysis ran against the frozen tree:

```text
run: 4f698166-9a0c-423a-aea9-8c7bf1c5a256
entities: 55,274, including 129 subsystems
edges: 118,568
Filigree emission: disabled
```

The analyzer reported one bounded limitation:

- the reference-site cap was exceeded in `src/elspeth/web/aws_ecs_acceptance.py`.

No negative structural conclusion relies on that incomplete edge set. Claims
in the affected area use exact source and executed tests.

## Assessment method

The assessment followed `../../assessment-framework.md`:

1. freeze one commit and record environmental limitations;
2. inventory the complete product chain;
3. evaluate all 15 mandatory dimensions;
4. execute current positive, negative, fault, contention, browser, and scale
   evidence where it exists;
5. leave missing, skipped, or plan-only cells Partial, Fail, or Unknown;
6. reconcile live tracker ownership and create proper issues for unowned gaps;
7. calculate the normalized maturity indicator only when no dimension remains
   `U`, without overriding hard gates;
8. publish an independently reviewed dated package; and
9. update the permanent DAG hub without rewriting historical assessments.

Three independent read-only evidence passes covered structural/authoring,
runtime/recovery, and security. The root pass independently captured the clean
baseline, broad unit result, contract gate, scale slice, live tracker state, and
final synthesis.

## Baseline health

| Gate | Result | Interpretation |
|---|---|---|
| Unit suite excluding the signed fingerprint baseline | `25,022 passed, 33 skipped` | Broad current regression baseline is green for the executed scope. |
| Signed fingerprint baseline | `1 failed` | Chroma and sink-effect fingerprint drift; tracked by `elspeth-18fe6e759e`. |
| Contract-boundary gate | failed | Four misplaced type definitions, 29 `dict[str, Any]` violations, and one stale whitelist entry. |
| Worktree integrity | clean | No evidence command changed tracked files. |

The assessment does not describe the repository as globally green. The two
failing gates are retained as maintained-contract limitations.

## Evidence limitations

- Live cloud services and third-party credentialed plugin combinations were not
  exercised.
- The four seeded browser correctness files contain six describe-level skipped
  tests; collection is not browser acceptance.
- The performance suite contains functional scale slices but no declared,
  enforced release envelope.
- Missing crash, contention, parity, and round-trip cells were not inferred from
  nearby tests.
- Live Filigree status is a reconciliation source, not proof that a frozen-code
  behavior passes.

# State Engine Assessment — 2026-07-18 16:31 AEST

This full reassessment introduces the v1 proof universe and evaluates code at
`42241500931926c5fd914ab7d92b479d9da1f8c2` (`release/0.7.1`). The assessment
documents were authored as a non-behavioral worktree overlay; no source or test
file differed from that commit during evidence execution.

## Verdict

**Not complete.** The engine has strong narrow evidence for strict fencing,
pending-sink admission, barrier atomicity, and built-in sink response-loss
recovery. It lacks a complete ten-dimensional proof package for every one of
the 68 mandatory legs, and repository-wide hard gates remain open.

The six concrete implementation defects called hard blockers by the July 15
snapshot are fixed at this baseline. That does not make the older `3 / 18`
score comparable to this result: v1 evaluates 68 legs and refuses to promote a
leg while any mandatory dimension or case is unresolved.

## Package

- [assessment.json](assessment.json) — baseline, environment, evidence,
  tracker capture, all 68 leg classifications, hard gates, and derived result.
- [evidence.md](evidence.md) — exact commands, fresh results, and limitations.
- `nodes/` — exact collected pytest node IDs, hashed from `assessment.json`.
- `artifacts/` — retained JUnit, stdout, and stderr with manifest hashes.
- [review.md](review.md) — independent findings, dispositions, and re-review.
- [Current proof matrix](../../proof-matrix.md) — readable family and gap view.

## Reproduce

Follow the parent [assessment program](../../assessment-program.md). For this
baseline, create a clean detached worktree at the full commit, run
`uv sync --frozen --all-extras`, verify the recorded Python/lock identity, and
execute EV-001, EV-002, and EV-003 exactly. A later rerun must preserve this result and
record any environment or output divergence separately.

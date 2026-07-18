# Executed Evidence

Evidence was executed on 2026-07-18 AEST against code commit
`42241500931926c5fd914ab7d92b479d9da1f8c2`. The worktree contained only the
assessment-document overlay listed by `git status`; source and test files were
unchanged from the commit.

The assessment began at `36146eac4`; two concurrent documentation-only commits
moved the branch to the baseline above before final evidence capture. Git
reported no difference outside `docs/` between those commits, and EV-001
through EV-003 were recaptured at the final baseline.

## EV-001 — strict fencing and pending-sink admission

Argument vector:

```text
[".venv/bin/python", "-m", "pytest", "-p", "no:dotenv", "-q", "-n", "0", "--junitxml=docs/architecture/state_engine/assessments/2026-07-18-1631/artifacts/EV-001.junit.xml", "tests/unit/core/landscape/test_scheduler_fencing.py", "tests/unit/core/landscape/test_scheduler_pending_sink_claim.py"]
```

- Working directory: `.` relative to captured repository root `/home/john/elspeth`
- Started: `2026-07-18T17:53:28+10:00`
- Ended: `2026-07-18T17:53:34+10:00`
- Shell-measured duration: 6.536 seconds
- Exit: 0
- Result: 46 passed, 0 failed/errors/skipped/xfail, 1 configuration warning,
  in 4.99 seconds
- Collection: the same selectors with `--collect-only -q -n 0` collected
  exactly 46 nodes in 0.03 seconds.

Establishes narrow SQLite repository and executable source-contract evidence
for strict non-optional authority, stale-token refusal without payload
mutation, isolation of the named legacy adapter, complete pending-sink bundle
admission, malformed-bundle refusal, and atomic predicate recheck.

Does not establish real leader/follower orchestration, independent-process
contention, lease expiry and reclaim, every state/subtype bundle, production
plugin composition, or crash/restart convergence.

## EV-002 — barrier atomicity and built-in sink recovery

Argument vector:

```text
[".venv/bin/python", "-m", "pytest", "-p", "no:dotenv", "-q", "-n", "0", "--junitxml=docs/architecture/state_engine/assessments/2026-07-18-1631/artifacts/EV-002.junit.xml", "tests/integration/pipeline/test_builtin_sink_effect_recovery.py", "tests/unit/core/landscape/test_scheduler_repository_complete_barrier.py"]
```

- Working directory: `.` relative to captured repository root `/home/john/elspeth`
- Started: `2026-07-18T17:56:18+10:00`
- Ended: `2026-07-18T17:56:22+10:00`
- Shell-measured duration: 3.966 seconds
- Exit: 0
- Result: 38 passed, 0 failed/errors/skipped/xfail, 1 configuration warning,
  in 2.45 seconds
- Collection: the same selectors with `--collect-only -q -n 0` collected
  exactly 38 nodes in 0.05 seconds.

Establishes direct repository evidence for exhaustive barrier snapshots,
atomic consume/emission, exact refusal arms, and injected rollback. It also
executes built-in CSV/JSON sink response-loss reconciliation, diversion
ordering, per-effect member classification, and zero-publication virtual
effects through the pipeline.

Does not establish abrupt operating-system process death, external database or
network sinks, a long external call beyond the effect lease, generation-fenced
takeover, every PB-06/PB-07 seam, aggregation continuation after a committed
TS-15 consume, or complete plugin lifecycle behavior.

## EV-003 — TS-07 through TS-10 disposition images

Argument vector:

```text
[".venv/bin/python", "-m", "pytest", "-p", "no:dotenv", "-q", "-n", "0", "--junitxml=docs/architecture/state_engine/assessments/2026-07-18-1631/artifacts/EV-003.junit.xml", "tests/unit/core/landscape/test_scheduler_events.py::test_normal_dispositions_refuse_reclaimed_sink_redrive_without_mutation", "tests/unit/core/landscape/test_scheduler_events.py::test_transform_disposition_truth_table_commits_exact_row_event_and_branch_loss", "tests/unit/core/landscape/test_scheduler_events.py::test_transform_disposition_truth_table_refuses_stale_owner_without_mutation", "tests/unit/core/landscape/test_scheduler_events.py::test_transform_disposition_truth_table_refuses_departed_member_without_mutation", "tests/unit/core/landscape/test_scheduler_events.py::test_transform_disposition_truth_table_rolls_back_when_event_insert_fails", "tests/unit/core/landscape/test_scheduler_events.py::test_branch_loss_failure_rolls_back_disposition_row_and_event", "tests/unit/core/landscape/test_scheduler_events.py::test_mark_blocked_refuses_missing_release_key_without_mutation", "tests/unit/core/landscape/test_scheduler_events.py::test_mark_pending_sink_rejects_incomplete_bundle_without_mutation"]
```

- Working directory: `.` relative to captured repository root `/home/john/elspeth`
- Started: `2026-07-18T17:56:22+10:00`
- Ended: `2026-07-18T17:56:25+10:00`
- Shell-measured duration: 2.717 seconds
- Exit: 0
- Result: 30 passed, 0 failed/errors/skipped/xfail, 1 configuration warning,
  in 1.15 seconds
- Collection: the same selectors with `--collect-only -q -n 0` collected
  exactly 30 nodes in 0.03 seconds.

Establishes direct SQLite repository evidence that TS-07 through TS-10 commit
their exact row, scheduler-event, and applicable branch-loss images; refuse
sink-redrive subtypes, stale owners, departed members, missing release keys,
and incomplete sink bundles without mutation; and roll back row/event/loss
images when a component insert fails.

Does not establish production transform/gate/follower composition, independent-
connection or process contention, crash/restart convergence, every supported
database profile, plugin lifecycle, or downstream read-model truth tables.

## Environment

| Fact | Captured value |
| --- | --- |
| Python | 3.13.1; `/home/john/elspeth/.venv/bin/python`; build `main`, 2024-12-19 |
| pytest | 9.0.3 |
| uv | 0.10.2 |
| Git | 2.43.0 |
| SQLite | 3.47.1 |
| SQLAlchemy | 2.0.45 |
| Kernel | Linux 6.8.0-124-generic x86_64 |
| Locale | `en_US.UTF-8` |
| Timezone | Australia/Canberra, UTC+10 at capture |
| `PYTHONHASHSEED` | Unset |
| `.env` | Present, but pytest-dotenv explicitly disabled; values not loaded or captured |
| `pyproject.toml` SHA-256 | `d084c07c35783c8fce3313d77ac0662df66c8db191b302770438f880f6fcf9e4` |
| `uv.lock` SHA-256 | `fac5b295e29ff4ee796a1ca9317c32d9bc7ff4ca1e6863d16f82666d655d680f` |

Each run emitted one expected `PytestConfigWarning` because disabling
pytest-dotenv leaves the repository's `env_files` option unknown. This is
recorded rather than suppressed. Exact collected node IDs are retained in the
`nodes/` directory with hashes in `assessment.json`.

JUnit, stdout, and stderr are retained under `artifacts/` for every run; their
SHA-256 values are recorded in `assessment.json`. Empty stderr files are
retained deliberately so the absence of process-level stderr is auditable. No
credential or `.env` value was captured. A historical rerun compares exit,
collection, result counts, and JUnit semantics, not timestamps or elapsed time.

# Verification Runs

## Retained audit-grade command

```bash
.venv/bin/pytest -q \
  tests/unit/core/test_count_ready_in_set.py \
  tests/unit/core/test_multi_source_foundation.py \
  -k 'unresolved_work_excludes_durable_sink_handoffs or count_ready or count_failed'
```

Observed result:

```text
16 passed in 3.70s
```

This proves the selected RM-01/RM-05-adjacent helpers were green immediately
before the planned Wave 2 edit. It does not supply the missing RM-02–RM-06 truth
table.

## Prior Wave 1 retained results

| Scope | Observed result | Limitation |
| --- | --- | --- |
| Three specialist packages | 83 passing node invocations | Some nodes overlapped; this is not a unique-test count. |
| Root cross-package selection | 16 passed, 1 warning in 4.88s | Representative selection, not every repository test. |
| Direct subtype probes | Both candidates reproduced | Purpose-built probes, not yet permanent regressions at that point. |

## Supplementary Wave 2 result

A specialist ran a focused read-model, relinquishment, membership-fence,
processor-mode, and eviction-ordering bundle with this observed result:

```text
63 passed in 3.64s
```

The exact command string was not retained in the root transcript. Treat this as
supporting health evidence, not as audit-grade command provenance. The next
assessment must retain the exact command when the new tests execute.

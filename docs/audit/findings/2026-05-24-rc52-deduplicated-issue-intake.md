# RC5.2 deduplicated issue intake

Status: filed in Filigree on 2026-05-24.

Source: operator-provided review transcript and follow-up committed-window review
summary from 2026-05-24. This document deduplicates repeated batch summaries,
removes findings later closed by subsequent commits, and preserves the remaining
actionable work as Filigree-ready issue cards.

Suggested goal:

> Turn `docs/audit/findings/2026-05-24-rc52-deduplicated-issue-intake.md`
> into Filigree issues, preserving type, priority, parentage, and acceptance
> checks. Do not start implementation until the tracker intake is complete.

Filigree intake result:

- Judge parent: `elspeth-2ed3bb0f7d`
- RC5.2 non-judge parent: `elspeth-41c92e76e2`
- Intake label: `rc52-dedup-intake-2026-05-24`

| Item | Filigree issue | Intake action |
| ---- | -------------- | ------------- |
| 1 | `elspeth-f6b4f54fbd` | Created |
| 2 | `elspeth-3f7952064f` | Updated existing matching issue |
| 3 | `elspeth-e9679a7ea3` | Created |
| 4 | `elspeth-14f3db187a` | Created |
| 5 | `elspeth-536336ccd0` | Updated existing matching issue |
| 6 | `elspeth-15aedf9944` | Created |
| 7 | `elspeth-b0b061e86f` | Created |
| 8 | `elspeth-612406fb75` | Created |
| 9 | `elspeth-f5bad373ae` | Created |
| 10 | `elspeth-322cae638b` | Created |
| 11 | `elspeth-46ea4d15df` | Created |
| 12 | `elspeth-cd6a59805e` | Created |
| 13 | `elspeth-339cc8dc46` | Created |
| 14 | `elspeth-d2f8d897a7` | Created |
| 15 | `elspeth-96640cb37b` | Created |
| 16 | `elspeth-ae32d3537d` | Created |
| 17 | `elspeth-b0125a607d` | Created |

## Priority mapping

- Review CRITICAL / pre-merge blocker -> Filigree `P0`
- Review High / MAJOR -> Filigree `P1`
- Review Medium / test-gap -> Filigree `P2`
- Review Low / hygiene -> Filigree `P3`

## Recommended grouping

- Parent epic or phase: RC5.2 review remediation, or the existing judge CLI
  umbrella if the operator elects to keep that umbrella as the whole-review
  tracker.
- The seven judge CLI findings should stay grouped together because four are
  pre-merge blockers and several share the same audit-integrity contract shape.
- The RC5.2 committed-window findings can be filed as a separate child group
  unless an existing RC5.2 remediation parent already owns them.
- The tracker and ops cleanup items are tasks, not code bugs.

## Judge CLI / pre-merge findings

### 1. trust_boundary invariant liveness — rule accepts raising-shape-only tests

- Type: bug
- Priority: P0
- Bound to source: T10 / commit `c29e0e1fd`
- Review severity: CRITICAL
- Summary: `trust_boundary.tests` checks that malformed input raises, but does
  not prove the decorated symbol is invoked with the documented invariant.
  Module docstring and metadata advertise the stronger invariant-liveness
  contract. Review notes call this C5-2/C6-1, with six-agent CRITICAL
  convergence and an unresolved operator decision from section 5.3: extend the
  rule or downgrade the documented claim.
- Acceptance checks:
  - The rule verifies the documented invariant is exercised, or the docstring
    and metadata are downgraded to match the actual raising-shape contract.
  - A regression fixture fails under raising-shape-only coverage.
  - The rule metadata no longer overstates what the check proves.

### 2. Judge metadata tamper binding — verdict quartet is editable plain text

- Type: bug
- Priority: P0
- Bound to source: C8-3 / commit `31f69cef3`
- Review severity: CRITICAL
- Summary: `file_fingerprint` and `ast_path` are computed and verified, but
  `judge_verdict`, `judge_rationale`, `judge_model`, and `judge_recorded_at`
  remain forgeable plain text. An in-place edit that softens the rationale can
  pass the current gates. The review explicitly called for hash-chain or HMAC
  signing of judge metadata.
- Acceptance checks:
  - Judge verdict metadata is included in an authenticated binding, or the
    prototype explicitly rejects those fields as mutable non-audit evidence.
  - In-place mutation of verdict/rationale/model/timestamp fails validation.
  - Threat-model notes document key management or the chosen non-HMAC fallback.

### 3. Allowlist YAML scalar round-trip — single hyphen writes unparseable audit data

- Type: bug
- Priority: P0
- Bound to source: C2-6 / commit `f8dffa4d7`
- Review severity: CRITICAL
- Summary: The Hypothesis property
  `test_property_inline_scalar_round_trips_through_safe_load` falsifies on
  `value="-"`. The writer's bare-safe character set permits a single hyphen,
  emitting `x: -`, which YAML parses as a block-sequence indicator and rejects.
  A plausible command such as `elspeth-lints justify --owner -` can write audit
  YAML that the loader cannot read.
- Acceptance checks:
  - `value="-"` is quoted or otherwise serialized safely.
  - The property test passes for YAML indicator scalars.
  - Emitted audit YAML round-trips through the production loader.

### 4. Allowlist YAML append atomicity — concurrent justify calls can lose updates

- Type: bug
- Priority: P0
- Bound to source: C4-1 / commit `c464cfb27`
- Review severity: CRITICAL
- Summary: The atomic-write helper locks only the write half. Callers such as
  `_append_entry_to_yaml` and `apply_plan` still perform `read_text -> mutate ->
  atomic_write_text` with the lock acquired only during replace. Two concurrent
  justify calls can both read the same old file, both mutate, and then serialize
  one update over the other.
- Acceptance checks:
  - The full read -> mutate -> write sequence is protected as one critical
    section where mutation depends on current file content.
  - A concurrent append regression test proves no lost allowlist entries.
  - The helper contract clearly states whether it protects writes only or
    compound read/modify/write transactions.

### 5. Judge gate workflow concurrency — allowlist gate can race PR and push runs

- Type: bug
- Priority: P1
- Bound to source: T3+T4 / commit `8d966207e`
- Review severity: MAJOR
- Summary: `.github/workflows/enforce-allowlist-judge-gates.yaml` has no
  `concurrency:` block. `ci.yaml` already shows the intended pattern. Concurrent
  PR and push triggers can race and report stale or conflicting policy-gate
  state.
- Acceptance checks:
  - The judge-gates workflow mirrors the repository's chosen concurrency policy.
  - Structural workflow tests or CI policy checks cover the concurrency block.

### 6. Judge excerpt redaction verification — written redaction records are not loaded

- Type: bug
- Priority: P1
- Bound to source: C8-3 / commit `31f69cef3`
- Review severity: MAJOR
- Summary: The writer emits a `judge_excerpt_redactions` block, but the review
  found no corresponding loader parse/validation path. If correct, the redaction
  record is presence-only rather than load-verified audit data.
- Acceptance checks:
  - Loader parses and validates `judge_excerpt_redactions`.
  - Malformed redaction metadata fails closed.
  - Tests prove writer and loader stay symmetric.

### 7. Python AST parser fault coverage — file-read errors lack direct regression tests

- Type: task
- Priority: P2
- Bound to source: T2 / commit `2d686747b`
- Review severity: MINOR
- Summary: `PythonFileReadError` is wired through callers, but there are no
  direct unit tests for the caught exception types: `UnicodeDecodeError`,
  `PermissionError`, and `OSError`. Coverage currently exists only
  integration-by-caller.
- Acceptance checks:
  - Direct parser tests cover all three caught exception types.
  - Any comprehension-scope behavior called out by the T2 review is pinned.

## RC5.2 committed-window findings

### 8. Telemetry delivery accounting — exporter failures can be counted as emitted

- Type: bug
- Priority: P1
- Review severity: High
- Evidence from review:
  - `src/elspeth/telemetry/manager.py:230`
  - `src/elspeth/telemetry/manager.py:264`
  - `src/elspeth/telemetry/manager.py:532`
  - `src/elspeth/telemetry/exporters/otlp.py:218`
  - `src/elspeth/telemetry/exporters/azure_monitor.py:268`
- Summary: `TelemetryManager` treats any exporter result other than `False` as
  success, increments `events_emitted`, and flush failures only increment
  per-exporter failure counts. OTLP and Azure exporters also ignore SDK exporter
  return status. Dropped telemetry can therefore look delivered and bypass
  total-failure handling.
- Acceptance checks:
  - Exporter return statuses flow into manager-level success/failure accounting.
  - Failed flushes do not increment emitted-success counters.
  - Tests cover partial and total exporter failure.

### 9. Save-for-review dialog semantics — modal loses dialog role and aria-modal

- Type: bug
- Priority: P1
- Review severity: High
- Evidence from review:
  - `src/elspeth/web/frontend/src/components/composer/SaveForReviewDialog.tsx:58`
  - `src/elspeth/web/frontend/src/components/composer/SaveForReviewDialog.tsx:110`
- Summary: The component comment still requires WAI-ARIA dialog behavior, but
  the rendered root is now a plain `section` without `role="dialog"` or
  `aria-modal="true"`. Current tests only catch the test id, not modal
  semantics.
- Acceptance checks:
  - Rendered modal root restores dialog semantics.
  - Tests assert accessible dialog behavior, not only DOM presence.

### 10. Composer schema discovery guard — empty pipelines appear safe to mutate

- Type: bug
- Priority: P2
- Review severity: Medium
- Evidence from review:
  - `src/elspeth/web/composer/prompts.py:183`
  - `src/elspeth/web/composer/prompts.py:280`
  - `src/elspeth/web/composer/skills/pipeline_composer.md:551`
- Summary: `schemas_gap` is computed only from plugins already present in state,
  while the skill text says an empty gap means mutation is allowed. For a new or
  empty pipeline, the composer may infer it is safe to mutate before discovery.
- Acceptance checks:
  - Empty/new pipeline prompts still require schema discovery before mutation.
  - Tests cover first-authoring behavior.
  - Skill text and prompt computation agree.

### 11. gov-pages-rate-cool scoring — AMBER tool-call range is scored GREEN

- Type: bug
- Priority: P2
- Review severity: Medium
- Evidence from review:
  - `evals/composer-rgr/scenarios/gov-pages-rate-cool/scenario.json:24`
  - `evals/lib/composer_rgr_score.py:528`
  - `evals/lib/composer_rgr_score.py:643`
- Summary: The scenario says 9-12 tool calls should be AMBER, but the scorer
  only forces RED above 12 and the final GREEN/AMBER collapse can report GREEN.
- Acceptance checks:
  - 9-12 tool calls produce AMBER.
  - Scorer logic and scenario prose agree.
  - Regression test covers the boundary values.

### 12. Tool-call info target size — chat disclosure button is below 24px

- Type: bug
- Priority: P2
- Review severity: Medium
- Evidence from review:
  - `src/elspeth/web/frontend/src/components/chat/chat.css:974`
  - `src/elspeth/web/frontend/src/components/chat/ToolCallCard.tsx:21`
- Summary: The tool-call info trigger is rendered as a 16x16 interactive button,
  below the 24x24 WCAG 2.5.8 target-size threshold.
- Acceptance checks:
  - Interactive target is at least 24x24 CSS pixels.
  - Visual layout remains stable in compact chat cards.

### 13. Skill inventory drift gate — generated check is absent from CI

- Type: task
- Priority: P3
- Review severity: Low
- Summary: `scripts/cicd/generate_skill_inventory.py --check` is wired into
  pre-commit but not CI, even though the script describes `--check` as the CI
  gate.
- Acceptance checks:
  - CI runs the skill-inventory check.
  - The CI lane fails on generated inventory drift.

### 14. CSS barrel coverage — split tests bypass runtime stylesheet import

- Type: task
- Priority: P3
- Review severity: Low
- Summary: CSS split tests read constituent files directly rather than the
  runtime barrel import. Missing imports in `styles/index.css` can escape the
  focused tests.
- Acceptance checks:
  - Tests exercise the runtime `styles/index.css` barrel.
  - Import-order omissions fail in the focused CSS test surface.

## Tracker and ops cleanup tasks

### 15. Judge review backlog grooming — critical umbrella contains raw tail findings

- Type: task
- Priority: P1
- Summary: `elspeth-2ed3bb0f7d` was originally framed as "pre-merge CRITICAL
  only", but its child set appears to contain the broader MAJOR/MINOR tail.
  Review notes report 131 open children in `triage`, split 66 P1 and 65 P2, with
  no assignee and no WIP. The umbrella description also appears to enumerate 156
  items while 153 are tracked.
- Acceptance checks:
  - Operator chooses whether the umbrella remains pre-merge-only or becomes the
    full review tracker.
  - Raw triage children are split, opened, closed as duplicates, or reparented.
  - The 153 vs 156 discrepancy is explained or recorded as non-load-bearing.

### 16. Historical SSH excerpt audit — scrubber transition window may contain leaked bodies

- Type: task
- Priority: P2
- Summary: The review found no current worktree gap after T8c/T8d, but audit
  YAML written between commit `a726fae44` and the later T8c SSH-body fix may
  contain unredacted SSH bodies.
- Acceptance checks:
  - Determine whether any reaudit YAML was generated during the vulnerable
    window in dev or staging.
  - Inspect, rotate, redact, or remove affected artifacts if any exist.
  - Record the result in the relevant tracker issue or release notes.

### 17. Stale claim cleanup — unrelated codex claims need operator-driven release

- Type: task
- Priority: P3
- Summary: The pasted review corrected an earlier claim-state misread. No codex
  claim exists on the judge umbrella or its 131 open children. Six unrelated
  stale codex claims remain on other work areas and should only be released or
  reassigned if the operator asks for that cleanup.
- Acceptance checks:
  - Re-check live Filigree state before changing claims.
  - Release or reassign only the unrelated stale claims the operator confirms.

## Items intentionally not filed from this transcript

- T1 judge core: verified, no material concern.
- T5 Tier-1 metadata gates: verified; stale commit message about C8-3 is an
  audit-trail note, not a code defect.
- T5+C8-3 follow-ups: verified; administrative punt is explicitly scoped.
- C3-2/C3-3 reaudit sweep isolation: verified.
- T6b/T6c/T6d sidecar work: verified.
- Outbound-boundary current code: verified; historical SSH-window task retained
  separately.
- T8b scrubber coverage partial: superseded by T8c/T8d production-path and
  exhaustiveness gates.

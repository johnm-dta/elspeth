# Independent validation

**Reviewers:** `dag_corpus_stage_fix_review` spec reviewer and
`dag_integration_delta_quality_review` quality reviewer
**Review scope:** merge integration, live manifest delta, dated assessment,
hub navigation, and evidence-to-status boundaries
**Final status:** **Approved / ready** — no Critical or Important findings remain

## Review criteria

The reviewer was asked to verify that:

- the 2026-07-17 assessment remains unchanged;
- every promoted cell follows from current exact executable evidence;
- closed issue state is not substituted for scenario evidence;
- no generic repository fix promotes an end-to-end scenario by analogy;
- live owners name remaining work rather than closed defects;
- command provenance and limitations are reproducible; and
- the **Not complete** verdict and no-aggregate rule still follow from the
  current matrix.

## Findings and dispositions

| Severity | Finding | Disposition |
| --- | --- | --- |
| Important | Atomic parent disposition and PostgreSQL contention did not directly prove sequential replay through another batch member. | Added `cardinality-identity-11` for `TestExpandToken::test_batch_expansion_claim_is_scoped_to_batch_not_selected_parent`, attached it only to Contracts, Runtime, and Audit, and executed the exact locator. |
| Important | Recovery cited new runtime/audit evidence and described concurrent replay even though its attached evidence was only the existing per-child durable-payload recovery reference. | Recovery now cites only `cardinality-identity-07`; its reason states the durable payload evidence, the closed reproduced bypass, and the still-unexecuted child-enqueue process-death seam. |
| Important | The assessment package lacked an independent validation record. | Added this validation file and included it in the current assessment link-resolution contract. |
| Important | Several evidence-ledger commands used abbreviated hashes, ellipses, or prose summaries. | Replaced them with full hashes and exact commands; recorded the two dynamic selector scripts verbatim and separated clean-merge evidence from reconciliation-working-tree checks. |
| Important | The contract pinned evidence IDs and locators but not every record's kind, claim, and stages; affected-cell evidence tuples could drift independently. | Added a deterministic SHA-256 contract over all fields of all 52 ordered evidence records and explicit expected tuples for row-expansion Contracts, Runtime, Audit, Recovery, and Concurrency. |
| Minor | Contracts and Runtime described their remaining gap in terms of the Recovery process-death seam. | Contracts now names the missing complete parent/child identity contract, Runtime names missing exact production child work/outcomes, and process death remains confined to Recovery. |
| Minor | The ledger described `ruff format --check` as formatting files and over-scoped `git diff --check` to the committed merge. | Recorded "already formatted" and limited the whitespace claim to the uncommitted reconciliation delta. |
| Minor | A pre-stage `git diff --check` could not inspect newly added assessment files; the commit hook then found Markdown hard-break spaces. | The hook removed the spaces, all files were restaged, and the ledger now records the complete `git diff --cached --check` gate. |

## Status boundary confirmed in review

| Row-expansion cell | Final status | Owner | Why it is not higher |
| --- | --- | --- | --- |
| Contracts | Partial | `elspeth-ef29ef6ba4` | Repository-level parent/batch identity and replay refusal pass, but no production corpus contract case exists. |
| Runtime | Partial | `elspeth-ef29ef6ba4` | Selected child-set and refusal behavior passes below the full config-to-runtime scenario chain. |
| Audit | Partial | `elspeth-ef29ef6ba4` | Durable parent disposition and batch claim pass without complete production lineage, payload, and outcome reconciliation. |
| Recovery | Partial | `elspeth-7cdc4da434` | The reproduced bypass is closed, but the exact child-enqueue process-death and resume seam remains open. |
| Concurrency | Partial | `elspeth-ef29ef6ba4` | One PostgreSQL batch race passes; the complete production multi-process expansion matrix is absent. |

No output-contract, journal, URL-redaction, shutdown-diversion, or branch-loss
fix promotes another scenario cell. Those results repair local hard gates or
strengthen adjacent evidence without completing a mandatory scenario row.

The final re-review reported no Critical or Important findings. Its only
mechanical requests were to mark this record approved, refresh the final
focused-suite count, and use reconciliation-working-tree terminology; all
three are reflected in the published package.

The subsequent quality re-review approved the complete registry fingerprint,
the five exact row-expansion evidence bindings, ledger wording, unchanged
historical assessment, and narrow status promotions with no remaining
Critical, Important, or Minor findings.

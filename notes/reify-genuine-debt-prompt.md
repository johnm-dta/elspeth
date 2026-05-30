# Task prompt — Reify genuine tier_model debt (~136 entries)

You are continuing the cicd-judge tier_model backfill. The pre-screen + enhanced-prompt
re-judge are done; this task is the **code-reification** half of the remaining 272-entry
BLOCK queue: suppressions the cicd-judge rejected as *dishonest* (defensive code dressed
as a trust boundary), which must be fixed in code rather than suppressed.

## Read first (do not skip)
- Memory: `project_cicd_judge_tier_model_backfill_2026-05-30.md` (full context: how we got here).
- Worklist: `notes/reaudit-rejudge-block-queue-2026-05-30.md` — the **"Reify code (~136)"** section.
- Per-entry rationales: `notes/reaudit-rejudge-postprompt-2026-05-30.md` — the judge's `PRE_JUDGE_FRESH_BLOCK`
  table gives a specific reason for *each* entry. **This is your spec — the judge already diagnosed each fix.**
- Doctrine: `CLAUDE.md` (Three-Tier Trust Model; Defensive Forbidden / Offensive Encouraged; Plugin
  Ownership; telemetry primacy) and the enhanced judge prompt `elspeth-lints/src/elspeth_lints/core/judge.py`.

## The split is a heuristic — verify category before acting
The "reify" vs "@trust_boundary migrate" buckets were split by whether the judge rationale mentioned
`@trust_boundary`. For each entry, **read its rationale in the report and the actual code** before
acting. Reclassify freely:
- If the rationale says "use `@trust_boundary`" AND the rule is **R1 or R5** → it belongs in the
  *migration* task (`notes/trust-boundary-migration-prompt.md`), not here.
- If it's a genuine Tier-3 boundary on an R2/R6/R7/R8/R9 finding (NOT decorator-suppressible) and the
  guard is honest → it is neither reify nor decorator-migrate: re-judge it and, if ACCEPTED, it becomes a
  *signed allowlist entry* (operator signs), or collapse a cluster into a `per_file_rule`. Flag these.

## Special flavor — boilerplate-rationale entries (honest boundary, dishonest rationale)
The 2026-05-30 signing pass surfaced 16 entries (mostly `web/composer/llm_response_parsing.py` R2)
that `reaudit` ACCEPTed but `justify` BLOCKed. Cause: `justify` feeds `rationale_duplicate_count`
to the judge (reaudit does not), and ~29 of these share an identical copy-paste rationale
("Tier 3 LiteLLM/provider metadata boundary"). The judge correctly flags this as a rationale copied
rather than written for the site. These are likely **honest Tier-3 boundaries with a dishonest
(boilerplate) rationale** — the fix is NOT code reification but **rewriting each rationale to be
site-specific (addressing the specific rule's concern at that exact line), then re-judging** via
`justify`. If the rule is R2 (not decorator-suppressible) and the boundary is genuine, a site-specific
rationale should sign; if it won't, reify the code. Watch for this flavor across the R2 web clusters.

## Fix taxonomy (from the rationales)
1. **Tier-1 "let it crash"** (Landscape writes, checkpoint reads, audit-grade updates): remove the
   suppression; raise a typed `AuditIntegrityError` (or let the existing exception propagate). Our data,
   crash on anomaly. e.g. `data_flow_repository.record_transform_error`, `checkpoint/recovery.get_resume_point`,
   `retention/purge` (also a telemetry-primacy fix: route the failure to Landscape, not `logger.warning`).
2. **First-party contract shape-guard** (`isinstance`/`getattr` on our own typed return, Plugin Ownership):
   delete the guard; fix/tighten the producing contract so the shape is guaranteed; let a violation crash.
   e.g. `state_guard._extract_audit_evidence_context` (R4+R5 on `AuditEvidenceBase.to_audit_dict`).
3. **Silent-except on code we own** (R6): replace the swallow with explicit dispatch on the real condition,
   or let it raise. e.g. `core/dag/graph.warn_divert_coalesce_interactions`, `mcp/server._find_audit_databases`.
4. **`.get()`/`setdefault` on internal Tier-2 data** (R1/R8): direct subscript and let `KeyError` surface
   (fix the schema/contract), or branch explicitly. e.g. `schema_contract_factory`, `core/events.EventBus.emit`.

## Method (per entry — TDD, mandatory)
1. Read the entry's judge rationale (the report) + the code at the fingerprinted site.
2. Confirm it's genuine debt (not mis-bucketed). If mis-bucketed, move it and note why.
3. Write a failing test that pins the *offensive* behaviour (the crash/raise/explicit-result the fix
   should produce). Watch it fail.
4. Make the code change. Watch the test pass; run the touched module's existing tests.
5. **Delete the now-unnecessary allowlist entry** from `config/cicd/enforce_tier_model/<module>.yaml`.
6. Re-run the gate on the touched files (agent-keyless → shape-only mode):
   `env PYTHONPATH=elspeth-lints/src ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth`
   — the finding must no longer fire (because the code is fixed), and budget must stay green.
7. Run ruff + mypy on touched files.

## Sequencing & scope
- **Spine first** (`core/`, `contracts/`, `engine/` — ~10 entries): highest audit value, closest to the
  legal record, best proving ground. Then `plugins/` (23), then `web/` (95).
- Batch by subsystem; surface each batch for review (`superpowers:requesting-code-review`).
- Each reification *removes* a defensive pattern AND frees the tight permanent-allow_hits budget (309/320).
- Worktree: ask the operator first (default yes) per project convention.

## Constraints
- **Never** hold or use `ELSPETH_JUDGE_METADATA_HMAC_KEY` (operator-only custody). You don't need it here —
  reification removes allowlist entries; it doesn't sign them.
- No dishonest "fixes" (don't replace one suppression with another). The fix must make the code genuinely
  offensive/correct so the rule legitimately stops firing.
- Co-land allowlist removals with the code fix in the same commit; update the fingerprint baseline if the
  edit shifts AST positions of *other* entries in the file.

## Done when
All ~136 (minus reclassifications) are either fixed-in-code with their allowlist entries removed and the
gate green, or explicitly reclassified (to migration / signed-allowlist / per-file-rule) with rationale.

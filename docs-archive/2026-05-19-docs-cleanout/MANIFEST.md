# Docs Archive Manifest - 2026-05-19 Cleanout

This archive moves point-in-time, generated, completed, or superseded
documentation out of active `docs/` while preserving relative paths under this
directory. To map an old active path, prefix it with
`docs-archive/2026-05-19-docs-cleanout/`.

## Current Replacements

| Archived group | Reason | Active replacement |
| --- | --- | --- |
| `docs/release/rc2-checklist.md`, `rc3-checklist.md` | Obsolete release gate checklists | Git history and this archive |
| `docs/release/feature-inventory.md` | RC-3-era statements that read like current truth | `docs/release/elspeth-progress-rc1-to-rc5.md`, `docs/README.md`, `docs/reference/`, and `docs/release/rc-3-release-notes.md` for release history |
| `docs/release/guarantees.md` | RC-3-era guarantees snapshot; useful contract content was surfaced, stale release wrapper was deleted | `docs/contracts/assurance-contract.md` |
| `docs/release/rc4-executive-brief.md` | RC-4 planning snapshot superseded by RC-5 composer work and the RC-5.2 audience brief | `docs/release/executive-summary.md`, `docs/release/elspeth-progress-rc1-to-rc5.md`, Filigree RC5/RC5.2 work items, and `docs/composer/ux-redesign-2026-05/` |
| `docs/release/rc-2-checkpoint-fix.md` | Historical release note; useful migration note folded into the resume runbook | `docs/runbooks/resume-failed-run.md` |
| `docs/analysis/security-posture-brief.md`, `security-comparison-blog.md` | March RC-3.3 assurance snapshots with stale metrics | Fresh security or assurance work should start from live code and Filigree |
| `docs/arch-analysis-2026-04-29-1500/` | Frozen architecture-analysis workspace | `ARCHITECTURE.md`, `docs/architecture/adr/`, and this archive for provenance |
| `docs/arch-pack-2026-04-29-1500/` | Frozen architecture presentation pack | `ARCHITECTURE.md`; PDF tooling can still discover this archived source |
| `docs/superpowers/meta/web-ux-program.md`, `prompt-sync-plans-with-r4-spec-delta.md`, `prompt-verify-sub1-before-execution.md` | Prompts reference missing old web-UX specs/plans | Current composer docs and Filigree issue state |
| `docs/superpowers/plans/completed/composer-progress-persistence/` | Completed plan/review corpus | Filigree feature `elspeth-90b4542b63`; live code/tests for implementation truth |
| `docs/superpowers/handovers/2026-05-12-phase-*.md`, `2026-05-13-per-step-chat-phase-a-handover.md` | Completed handoff notes | Filigree and branch history |
| `docs/composer/ux-redesign-2026-05/*.review*.json` plus architecture/quality/reality/systems review sidecars | Generated reviewer sidecars; active plan prose stays in place | Paired plan Markdown files and Filigree phase issues |
| `docs/audits/pr-review-72h-audit-2026-04-21.md` | Point-in-time PR audit; critical findings were filed and closed in Filigree | Closed issues include `elspeth-460ab42c12`, `elspeth-b9b1091e3`, `elspeth-e5a6ce8eb9`, `elspeth-081d7aa5ee`, `elspeth-f97f3f10da` |
| `docs/audits/2026-05-14-composer-ux-review-actions.md` | Stale checklist form; represented by the RC5-UX remediation epic | `elspeth-de91358c30` and children |
| `docs/audits/test-infrastructure-audit-2026-03-01.md` | Old test-infrastructure snapshot; too stale to remain active truth | Fresh validation should be filed under Filigree before implementation |
| `docs/architecture/semi-autonomous/design-review-minutes.md`, `design-review-synthesis.md`, `independent-critique-gemini.md` | Review history around the retained design | `docs/architecture/semi-autonomous/design.md` |
| `CHANGELOG-RC1.md`, `CHANGELOG-RC2.md` | Reconstructed historical changelogs covering Pre-RC1 through RC2; ongoing `/CHANGELOG.md` is the authoritative line-by-line record from RC3 onward. Retained here so the RC1/RC2 narrative remains discoverable. | `/CHANGELOG.md` for RC3+; this archive for the RC1 build and RC2 hardening narrative |
| `scripts/generate_test_bugs.py` | One-off test-bug fixture generator; unreferenced anywhere in the active tree. Bug-tracking discipline now lives in Filigree. | Filigree (`filigree session-context`) and the per-suite test fixtures under `tests/` |

## Not Moved

| Path/group | Reason |
| --- | --- |
| `docs/release/README.md`, `docs/release/executive-summary.md`, `docs/release/elspeth-progress-rc1-to-rc5.md`, `docs/release/elspeth-velocity-rc1-to-rc5.md` | Retained because they were created for specific current audiences: public-sector evaluators, engineering leadership, and engineering reviewers. |
| `docs/audit/findings/*.md` | Retained because `_tickets-to-file.md` still marks CI allowlist subticket filing as incomplete under `elspeth-297b8f5c5d`. |
| `docs/audit/2026-05-19-cicd-allowlist-audit.md` | Current audit synthesis and still active. |
| `docs/assets/*.pdf` | No tracked files were present in this checkout. PDF build outputs now default to `tools/pdf/out/`. |
| `docs/bugs/generated/edited-48h/` | No tracked files were present in this checkout; `docs/bugs/generated/` remains ignored/generated. |

## Notes

- The main checkout had pre-existing dirty docs state when this worktree was
  created. This archive branch lives in an isolated worktree, merged current
  `RC5.2`, and does not modify that main-checkout dirty state.
- The archive intentionally preserves old internal links inside archived
  snapshots. Active docs should point either to current sources or to this
  manifest.

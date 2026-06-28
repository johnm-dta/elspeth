# Docs Archive Manifest - 2026-06-28 Cleanout

Release-preparation cleanout. The lens applied: *is this internal work product
the public should see?* Roadmaps and high-level plans stay; implemented
detailed plans/specs, evaluation evidence, point-in-time audits, agent prompts,
and per-period progress/velocity ("time accounting") reports move here.

This archive preserves relative paths under this directory. To map an old
active path, prefix it with `docs-archive/2026-06-28-docs-cleanout/` — e.g.
`docs/superpowers/specs/X.md` now lives at
`docs-archive/2026-06-28-docs-cleanout/docs/superpowers/specs/X.md`. Prose and
code-comment citations elsewhere in the tree (under `src/`, `tests/`, `evals/`,
`config/`) were left in place and resolve via this prefix rule; clickable
markdown links in retained docs were repointed here.

## Relocated Groups

| Archived group | Reason | Active replacement |
| --- | --- | --- |
| `docs/plans/` (incl. `reviews/`, `README.md`) | Implemented design and implementation plans; none in flight | Live code/tests for implementation truth; Filigree for open work |
| `docs/superpowers/plans/` and `docs/superpowers/specs/` | Implemented SDD per-phase plans and design specs; none in flight | Live code/tests; `docs/architecture/adr/` for the durable decisions |
| `docs/composer/evidence/` | Composer evaluation reports and investigation notes (point-in-time) | `docs/release/composer-guide.md`, `docs/reference/composer-tools.md` |
| `docs/composer/ux-redesign-2026-05/` | RC-5.2 composer UX planning and implementation phase docs | Shipped composer surface; `docs/architecture/adr/022`, `025`, `027` for retained decisions |
| `docs/audit/` (`*.md`, `findings/`, `source-risk/`, `test-suite/`) | Internal audit syntheses and review-agent finding dumps | Filigree (open items, e.g. `elspeth-297b8f5c5d` for CI allowlist) |
| `docs/design/` (`composer-ux-spec.md`, `tutorial-design-review-2026-06-28.md`) | Internal design specs and review notes | Shipped composer/tutorial surface and ADRs |
| `docs/prompts/` (risk-tree prompts, `multi-token/`) | Internal agent prompts / prompt-engineering work product | Not customer-facing; retained here for provenance |
| `docs/requirements/web-ux-composer-feedback-requirements.md` | Internal requirements capture, now implemented | Composer surface and Filigree |
| `docs/filigree/` (`SCHEMA.md`, `subsystems.yaml`) | Documents the internal issue-tracker tooling, not the product | `filigree --help` / MCP tool schemas |
| `docs/assets/prompt-guide.md` | Internal review/redesign of the composer prompt | `src/elspeth/web/composer/skills/pipeline_composer.md` (the live prompt) |
| `docs/bugs/BUGS.md` | Internal bug list | Filigree (`filigree session-context`) |
| `docs/architecture/audit-remediation.md` | Internal remediation tracking, not an architecture reference | Filigree; `docs/architecture/adr/` |
| `docs/release/public-release-docs-audit-rc5-2.md` | Internal audit of the release-doc set | The release docs it reviewed, now current |
| `docs/release/elspeth-progress-rc1-to-rc5.md` | Per-period "what shipped" cumulative-output report — internal work product, not evaluator-facing | `/CHANGELOG.md` for release history |
| `docs/release/elspeth-velocity-rc1-to-rc5.md` | Per-day commit volume / tempo — maintainer time accounting | `/CHANGELOG.md`; git history |
| `docs/release/pdf/progress.typ`, `velocity.typ` | Typst sources for the two reports above; their PDF build targets were removed from `docs/release/pdf/Makefile` and `build.sh` | Re-add the targets here if the reports are ever republished |

## Deleted, Not Archived

Generated or regenerable content, removed rather than relocated (all gitignored
except where noted):

| Path | Reason |
| --- | --- |
| `docs/bugs/generated/` | Generated bug-sweep reports (gitignored, regenerable) |
| `docs/audit/triage/` | Generated triage batches/findings (gitignored, regenerable) |
| `docs/release/pdf/fonts/public-sans/sources/` | Upstream Public Sans `.ufo` design sources (~13 MB, gitignored); the build only needs the bundled `fonts/ttf/` |
| `docs/release/pdf/fonts/public-sans/__MACOSX/` | macOS zip-extraction cruft |
| `docs/research/` | Empty directory |

## Not Moved (retained in active `docs/`)

| Path/group | Reason |
| --- | --- |
| `docs/guides/`, `docs/reference/`, `docs/runbooks/`, `docs/contracts/`, `docs/operator/` | Customer/operator-facing tutorials, reference, procedures, and contracts |
| `docs/architecture/` (ADRs + overviews, incl. `semi-autonomous/design.md`) | Durable architecture decision record and system orientation |
| `docs/release/` core set (`README.md`, `executive-summary.md`, `composer-guide.md`, `platform-architecture.md`, `assessment-mapping.md`, `guarantees.md`) + PDF build system | Evaluator-facing release/assurance surface |
| `docs/product/` (incl. `roadmap.md`, `vision.md`) | Live `/own-product` tool workspace; holds the high-level public roadmap. Out of scope for a plans/reporting cleanout |
| `docs/elspeth-lints/` | Contributor-facing docs for the CI tier-model linter |

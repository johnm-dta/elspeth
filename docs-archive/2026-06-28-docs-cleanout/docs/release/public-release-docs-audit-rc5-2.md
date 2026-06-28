# RC-5.2 Public Release Documentation Audit

**Audit date:** 2026-05-20
**Scope:** root Markdown files, `README.md`, `docs/README.md`, `docs/release/`, and high-visibility linked documentation surfaced from those indexes.
**Purpose:** identify anything that could make the public release look unfinished, internally focused, unclear, or risky for an agency-facing government project.

## Release blockers

- [ ] **Boss-owned: replace draft executive-summary markers before publication.**
  - Files: `docs/release/executive-summary.md`, `docs/release/README.md`, `docs/README.md`, `docs/release/pdf/executive-summary.typ`.
  - Current issue: the executive summary is explicitly marked `DRAFT`, `awaiting review`, and `awaiting operator sign-off`. That is appropriate before approval, but it must not be the state of the public release entry point.
  - Professional landing action: leave this with the boss/reviewer; no automated cleanup in this pass.

- [x] **Fix the root README stale release badge and RC-5.1 framing.**
  - File: `README.md`.
  - Completed: status badge now says RC-5.2 and the README includes an RC-5.2 update section.

- [x] **Fix the broken root README assurance link.**
  - File: `README.md`.
  - Completed: the README now links to `docs/release/guarantees.md`.

- [x] **Remove or archive the root `PR_DESCRIPTION.md` before public release.**
  - File: `PR_DESCRIPTION.md`.
  - Completed: removed from the root public release surface.

- [x] **Decide whether internal agent/process files are public release files.**
  - Files: `CLAUDE.md`, `AGENTS.md`, and public links to them from `README.md`, `PLUGIN.md`, `CONTRIBUTING.md`, and `docs/README.md`.
  - Completed: public orientation links now point to durable trust-boundary and architecture docs rather than `CLAUDE.md` / `AGENTS.md`.

## High-priority cleanup

- [x] **Remove internal tracker language from public indexes.**
  - Files: `docs/README.md`, `docs/release/README.md`, `docs/release/elspeth-progress-rc1-to-rc5.md`, `docs/release/pdf/progress.typ`.
  - Completed: public indexes no longer route readers to Filigree, active audit packets, or in-progress branch state.

- [x] **Fully refresh or de-emphasise `guarantees.md`.**
  - File: `docs/release/guarantees.md`.
  - Completed: relabelled as a layered assurance appendix and softened "contractual" framing in the release index.

- [x] **Stabilise progress and velocity docs to post-merge state.**
  - Files: `docs/release/elspeth-progress-rc1-to-rc5.md`, `docs/release/elspeth-velocity-rc1-to-rc5.md`.
  - Completed: live branch/HEAD/remotes language was replaced with release-snapshot wording and post-RC-5.2 follow-up framing.

- [ ] **Treat legal, ATO, and ownership statements as approval-required.**
  - File: `docs/release/executive-summary.md`.
  - Current issue: the summary includes sensitive claims about MIT licensing, personal copyright assignment, DTA adoption, interim ATO scope, a real-world pilot, and single-contributor continuity risk.
  - Professional landing action: keep the honesty, but require legal/comms/security owner sign-off before publication. Consider moving the copyright-assignment discussion out of the executive summary into a legal/licensing note.

- [x] **Cordon off generated bug and audit material.**
  - Files/directories: `docs/bugs/generated/`, `docs/audit/findings/`, `docs/audit/test-suite/`, `docs/README.md`.
  - Completed for public routing: `docs/README.md` no longer links readers into active audit findings or generated report packets.

- [x] **Remove generated PDF build artifacts unless they are intentional release assets.**
  - Files/directories: `docs/release/pdf/out/`, `docs/release/pdf/fonts/public-sans/__MACOSX`.
  - Completed for tracked release state: these paths are not tracked by git in this checkout; no public tracked artifact was removed.

- [x] **PDFify the assurance and trust-model documents selected for the release pack.**
  - Files: `docs/release/pdf/guarantees.typ`, `docs/release/pdf/data-trust.typ`, `docs/release/pdf/build.sh`, `docs/release/pdf/Makefile`, `docs/release/pdf/README.md`, `.gitignore`.
  - Completed: the PDF pipeline now builds seven PDFs, including `elspeth-guarantees.pdf` and `elspeth-data-trust.pdf`. The operator/deployment candidates are intentionally left for a future wiki rather than forced into the release PDF set.

## Clarity improvements

- [x] **Create a clean public documentation route.**
  - Suggested shape: `README.md` -> `docs/README.md` -> `docs/release/README.md` -> final executive summary / guarantees / progress / velocity.
  - Avoid routing public readers into active work trackers, draft findings, generated reports, or agent instructions unless intentionally marked as maintainer-only.

- [x] **Use one naming convention for the release.**
  - Current surface mixes `RC5.2`, `RC-5.2`, `0.5.2`, and "Current HEAD".
  - Suggested convention: use `RC-5.2 / 0.5.2` in prose and reserve branch names for provenance footnotes.

- [x] **Keep the progress and velocity documents, but frame them as management evidence.**
  - These documents are useful for showing pace and workflow to executives. Preserve that intent.
  - Strengthen the opening note so readers understand commit counts are delivery telemetry, not a substitute for assurance, maintainability, or product value.

- [ ] **Preserve negative evidence, but place it carefully.**
  - Example: the RC-5.1 `compliance@example.com` evaluation defect is honest and useful engineering evidence, but it is very specific and agency-visible.
  - Suggested treatment: keep it in progress/engineering provenance, ensure "evaluation only, no production traffic affected" remains prominent, and avoid duplicating it in broad user-facing docs.

- [ ] **Replace personal namespace examples if an official home exists.**
  - Files: `README.md`, `PLUGIN.md`, `CONTRIBUTING.md`, Docker and deployment docs.
  - Current issue: clone and container examples use `github.com/johnm-dta/elspeth` / `ghcr.io/johnm-dta/elspeth`.
  - Professional landing action: switch to the official agency repository/container namespace when available, or add a note that the namespace is temporary for the RC.

## Quick verification performed

- Root Markdown files found before cleanup: `AGENTS.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `PLUGIN.md`, `PR_DESCRIPTION.md`, `README.md`.
- Broken relative links checked across the root README and release docs: one missing target found before cleanup, `README.md` -> `docs/contracts/assurance-contract.md`; fixed in this pass.
- Draft/internal/publication-risk grep pass covered root Markdown, `docs/README.md`, `docs/release/*.md`, release PDF Typst sources, guides, reference docs, and runbooks.
- Release PDF pipeline rebuilt successfully with Typst 0.14.2; outputs generated for executive summary, progress, velocity, architecture, Composer, guarantees, and data trust.

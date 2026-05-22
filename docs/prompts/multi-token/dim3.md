/goal Please do a detailed static analysis of the changed files in the `.worktrees/multi-source-token-scheduler` worktree, focusing on migration dust in policy, documentation, CI, release, lint, redaction, secret-reference, security architecture, and operator guidance.

Begin by identifying the changed files in the `.worktrees/multi-source-token-scheduler` worktree relative to its base branch. Review only changed files and directly related context needed to understand those changes. If the base branch cannot be determined, treat that as a hard precondition failure and report it.

Bring in and apply these skill packs:
- security architecture
- quality engineering
- systems thinking
- solution architecture
- Python engineering, where docs or CI refer to code behaviour
- determinism and replay theory, where docs or policy describe auditability, replay, evidence, or reproducibility

This repo uses policy and docs as operational controls, so stale assumptions are operationally significant. The engine has moved from single token to concurrent tokens, and from single source to multi-source, but some supporting materials may still reflect the old model.

Review changed files and sections for:
- stale references to single-token or single-source operation
- outdated lint, redaction, release, secret-reference, CI, or secret-handling assumptions
- docs that describe old flows, old commands, old env vars, old failure modes, or old trust boundaries
- operator guidance that would cause incorrect behaviour in the upgraded engine
- security controls that do not account for multiple sources, multiple tokens, source isolation, or concurrent execution
- policy or docs that no longer match the code path they are meant to control
- incomplete migration notes, missing warnings, ambiguous runbook steps, or misleading examples
- documentation around replay, audit evidence, reproducibility, redaction, and release safety that no longer matches implementation reality
- comments, README sections, examples, test names, and CI labels that encode obsolete assumptions
- easy documentation, CI, release, or policy cleanups that reduce operational risk

CRITICAL RULE: Do not edit any files under any circumstances. Only stop early for hard precondition failures, such as the worktree being unavailable, changed files being impossible to identify, Filigree being unavailable, or the repository being unreadable. Do not stop merely because tests fail, docs are stale, or the branch is messy; create tickets for those issues instead. Because this audit will be run concurrently with other specialist audit passes, prefer high recall over strict deduplication. Create tickets for concrete findings even if they may overlap with another lens. Do not suppress a finding merely because another audit agent might also report it. The consolidation pass will de-duplicate and normalise severity later.

Create Filigree tickets for every stale assumption, documentation error, operator-control mismatch, CI/release/policy drift, security-control mismatch, redaction or secret-reference issue, misleading example, or easy operational cleanup you find.

Use ticket titles in this format:
[area] concise issue summary

Examples:
[ux] UX not updated to handle multiple sources
[engine] Source identity can be lost during token retry
[replay] Concurrent source ordering is not deterministic
[docs] Operator guide still describes single-source secret references
[tests] Multi-source partial failure path lacks assertions

Each ticket should include:
- file and relevant section
- issue summary
- why it matters
- likely impact
- suggested fix
- relevant skill-pack lens used

For each ticket, include:
- severity: merge blocker, should fix before merge, should fix soon after merge, nice-to-have, documentation-only, or needs investigation
- confidence: high, medium, or low
- evidence type: direct code evidence, inferred design risk, missing test coverage, stale documentation, or operational concern

Include a short “dedupe key” field describing the underlying issue in stable terms, such as `single-source-assumption-in-token-state`, `non-deterministic-source-ordering`, `operator-docs-stale-secret-reference-flow`, or `sqlite-transaction-boundary-replay-risk`.

Do not create vague tickets that only say to “review”, “clean up”, or “consider improving” an area. If the issue is broad, identify the specific file, section, stale assumption, missing invariant, or concrete behaviour that makes it actionable.

At the end, provide a report summarising the tickets created and your overall assessment of the engine upgrade quality.

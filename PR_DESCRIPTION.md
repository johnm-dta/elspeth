# RC5.2 — guided composer, durable progress, recovery UX, and release-doc refresh

## Summary

RC5.2 is the Web Composer release train from `RC5.2` into `main`. I reviewed the
full branch log against `main`: the merge base with `main` is `7bf364da`, and
the committed branch contains **771 commits** over `main` (**230 first-parent
commits**) across 2026-05-08 through 2026-05-19 after this release-doc refresh.

The branch is no longer a single frontend recovery slice. It folds together:

- guided composer mode;
- composer progress persistence and manifest-keyed redaction;
- per-step guided chat, chat-as-data-entry, and interpretation review;
- composer preferences, tutorial flow, completion gestures, shareable reviews,
  catalog reshape, and audit-readiness UI;
- engine/plugin correctness work including `AggregationBatchContext`,
  `report_assemble`, determinism declarations, and one-knob composer schemas;
- auth, execution, checkpoint/resume, frontend accessibility, and staging
  hardening;
- CI/CD consolidation, `elspeth-lints`, CodeQL, Playwright gating, telemetry
  trailer enforcement, release reports, and docs cleanup.

## First-Parent Integration Map

Key first-parent milestones reviewed from `git log --first-parent main..HEAD`:

- `ff46b8091` — guided composer wizard, Phases 1-10 and PR review fixes.
- `0e7e5cdfa` / `1ab1419d6` / `e560e61f2` — composer progress persistence
  Phase 1A/1B/1C/2 integration and changelog pass.
- `5379f41a8` — per-step guided chat integration.
- `cc429d5f6` through `badf17b28` — RC5.2 auth, execution, checkpoint/resume,
  plugin, frontend a11y, and staging import-root hardening.
- `159598db3` — Phase 3 compose-loop persistence.
- `2e003296f` through `b29117523` — Phase 4 recovery payload, store, panel,
  diff, transcript, and trigger wiring.
- `b2f454ce9`, `2135908f6`, `c936562aa`, `7a8187c0c` — composer UX proposal
  lifecycle, one-knob schema, preferences, and audit-readiness phases.
- `3dee19f8d` — Phase 5 chat data-entry and interpretation-events merge.
- `dd20888f0` — Phase 6 completion gestures and shareable-review flow.
- `e36841361` — Phase 7 catalog reshape and audit-characteristic surface.
- `586ce3103` / `37e26db24` — CI allowlist burn-down and CI master-plan work.
- `ca9bc05bd` — Phase 4 hello-world tutorial.
- `cde78aa5a` — 9 P0 composer-UX defects plus dependency promotion.
- `f905828f8` — solo-maintainer delivery governance ADR.
- `dc5ee3350` — docs archive cleanout merge.
- This release-doc refresh commit — refreshed changelog, lockfile alignment, and
  PR body.

## Changelog / Release Notes

`CHANGELOG.md` has been refreshed for the full RC5.2 train and now dates
`0.5.2` to 2026-05-19. It records:

- guided composer mode and `ComposerLLMCall` audit rows;
- progress persistence Phase 1A schema, Phase 1B atomic turn persistence,
  Phase 1C Postgres portability, Phase 2 redaction, Phase 3 compose-loop
  persistence, and Phase 4 recovery UX;
- chat-as-data-entry and interpretation review;
- composer preferences, first-run tutorial, completion gestures, shareable
  reviews, catalog reshape, and audit-readiness UI;
- engine/plugin additions including `report_assemble`, determinism declarations,
  and one-knob schema lowering;
- CI/CD consolidation, `elspeth-lints`, CodeQL, Playwright gating, telemetry
  trailer enforcement, and docs archive cleanup;
- operational notes for session DB recreation, coupled session/Landscape reset,
  frontend build deployment, and PR scope.

## Docs-Cleanout Refresh

The branch now includes the dated docs-cleanout package:

- `docs-archive/2026-05-19-docs-cleanout/MANIFEST.md` maps relocated snapshots.
- RC-1/RC-2 changelog fragments and stale release snapshots moved into the dated
  archive.
- Frozen architecture packs, generated review sidecars, completed handovers,
  old audits, and one-off prompt/test-bug artifacts moved or removed from active
  paths.
- `docs/README.md`, `docs/release/README.md`, progress/velocity reports,
  composer evidence docs, PDF tooling docs, and active cross-links were updated
  to point readers at current sources.

That package is intentionally a documentation hygiene layer on top of the RC5.2
train, not a separate product feature.

## Verification Evidence

Evidence already recorded in branch history and PR discussion includes:

- focused composer Python suite: 1774 passed, 9 warnings;
- frontend unit suite: 51 files passed, 518 tests passed;
- frontend typecheck: passed;
- frontend lint: passed with existing `react-hooks/exhaustive-deps` warnings;
- frontend build: passed with existing Vite chunking warnings;
- local Chromium recovery smoke: fanout guard, recovery panel coexistence,
  transcript fetch with `include_tool_rows=true&limit=500`, provider payload
  hidden;
- visible recovery panel axe audit: 0 critical violations, 2 non-critical
  violations;
- final focused gates from the earlier PR body: mypy on composer service,
  tier-model check, and focused compose-loop/session telemetry tests.

Additional evidence from this refresh:

- `filigree session-context` ran successfully in the RC5.2 checkout.
- `git log --first-parent main..HEAD` and `git log --no-merges main..HEAD` were
  reviewed to derive this PR body and the refreshed changelog.
- `gh pr view 39` confirmed the open PR target is `RC5.2` -> `main`.

## Operational Notes

- Session schema changes still require recreating `sessions.db` per
  `docs/runbooks/staging-session-db-recreation.md`.
- Phase 5 interpretation-events deployment requires the session DB and Landscape
  audit DB to be reset together so `resolved_prompt_template_hash` handoff
  remains coherent.
- Frontend-only deploys require rebuilding `src/elspeth/web/frontend/dist/`;
  backend Python, dependency, environment, systemd, or Caddy changes require
  restarting `elspeth-web.service`.
- No staging restart or live staging verification was performed as part of this
  documentation refresh.

## Current PR State

As of this refresh, PR #39 is open at
`https://github.com/johnm-dta/elspeth/pull/39`.

The final release-doc refresh push started a fresh check set. The latest
observed remote state after that push showed CodeQL `Analyze Python` in
progress; re-check GitHub before treating this PR as merge-ready.

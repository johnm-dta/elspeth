# Paste-ready prompt — re-run the composer LLM evaluation

Copy everything between the `---` lines into a fresh Claude Code session.
Assumes the same machine (or one with equivalent staging access).

---

I'd like you to evaluate how well the ELSPETH composer LLM (the chat-driven
pipeline builder on the staging deploy) is at generating workflows. Drive it
end-to-end as a regular user would — only the public HTTP API, no source
edits, no in-process MCP, no Python helper calls. The point is to capture
what an unprivileged user actually experiences.

## Environment

- Staging URL: https://elspeth.foundryside.dev
- Staging is a source-checkout systemd/Caddy deploy on this host
  (`systemctl status elspeth-web.service`). It is not the
  `scripts/deploy-vm.sh` flow — see memory note "staging deployment".
- Auth provider: `local`, registration closed.
- Test user: `dta_user` / `dta_pass`. The password is not sensitive — it
  guards ~$10 of OpenRouter credit which the user has authorized burning
  on this eval.
- Composer model is configured via `ELSPETH_WEB__COMPOSER_MODEL` in
  `deploy/elspeth-web.env`. Composer budget knobs:
  `COMPOSER_MAX_COMPOSITION_TURNS`, `COMPOSER_MAX_DISCOVERY_TURNS`,
  `COMPOSER_TIMEOUT_SECONDS`, `COMPOSER_RATE_LIMIT_PER_MINUTE`.

## What "act like a user" means

You may use the following surface and only this:

- `POST /api/auth/login` for the JWT, then `Authorization: Bearer <jwt>` on
  every call.
- `GET /openapi.json` and `GET /api/catalog/{sources,transforms,sinks}`
  (and per-plugin `/schema`) for ground-truth discovery — used **only** to
  score the LLM's output, never to bypass anything.
- `POST /api/sessions` to create chats; `POST /api/sessions/{sid}/messages`
  to talk to the composer LLM.
- `POST /api/sessions/{sid}/blobs/inline` for file uploads (the same
  endpoint the SPA uses for "drop a file"). Don't write files to
  `/tmp/...` and ask the LLM to wire that path — the composer will (and
  should) refuse.
- `GET /api/sessions/{sid}/state`, `…/state/yaml`, `…/state/versions`,
  `…/messages`, `…/composer-progress` for read-side inspection.
- `POST /api/sessions/{sid}/validate` for the runtime pre-flight.
- `POST /api/sessions/{sid}/execute` to actually run the pipeline; then
  `GET /api/runs/{rid}` and `…/diagnostics` to read the outcome.
- `cat data/outputs/*.jsonl` (or whatever path the run reports) to confirm
  real on-disk artefacts.

You may inspect the codebase, sqlite databases under `data/`, and journald
(`journalctl -u elspeth-web.service`) to **diagnose** failures after a
user-flow finishes — but the conclusions you draw must be about what the
user-flow itself revealed. Don't edit ELSPETH source. Don't use
`mcp__elspeth-composer__*` tools (those bypass the HTTP path).

## Methodology

Run a portfolio of scenarios chosen to exercise different LLM behaviours:

1. **Monolithic complete-pipeline ask** in one message (CSV → LLM classify
   → routed sinks). This tests how the composer behaves when the user
   submits a "build it all at once" prompt — including whether it survives
   the per-message wall-clock and turn budget.
2. **Same task, broken into incremental turns** (one component per
   message). Tests fidelity, plugin/option correctness, and tool-selection
   judgement (`set_pipeline` vs `patch_*`).
3. **Stateful aggregation** (e.g. count rows per group, emit a single
   summary JSON). Tests the LLM's grasp of aggregation triggers,
   batch-aware transforms, and `group_by` semantics.
4. **Gate routing with a follow-up correction and revert** (split rows by
   field, then change the predicate, then ask to revert). Tests whether
   the LLM patches surgically or rebuilds, and whether it knows about
   `/state/revert`.
5. **Vague prompt** ("I want to do something with my Excel file"). Tests
   whether the LLM asks clarifying questions or hallucinates a plugin /
   pretends it can read xlsx.

For each scenario:
- Record the assistant message, the resulting `composition_state` (and
  diff vs the previous version), the YAML, and the `/validate` result.
- Then run `/execute`. Capture run id, status, errors, and any output
  files. If something fails, **always** feed the literal runtime error
  back to the composer in the next user message and let the LLM attempt
  its own recovery — that measures the LLM's debugging behaviour, not
  yours.
- For each LLM choice (plugin name, option fields, schema shape), compare
  against the catalog ground-truth and call out (a) confabulations,
  (b) judgement calls that turned out right, (c) judgement calls that
  turned out wrong, (d) anything the LLM volunteered unprompted.

Long-running calls: `POST /messages` is synchronous and the server budget
is ~180s. Run them with `run_in_background=true` and a `Monitor`
watching the response file rather than chained sleeps. Use
`/composer-progress` for liveness while you wait.

## What to report

Write the report to `/home/john/elspeth/notes/composer-llm-eval-<date>.md`
(also keep a `/tmp/elspeth_eval/REPORT.md` working copy). Structure:

- Front matter: deployment, composer model in use, budget knobs, test
  user, date.
- "How this eval was run" — the constraint that you only used the public
  HTTP surface, with the auth/discovery/exec sequence enumerated. This
  is what makes the bug claims credible.
- Per-scenario findings: outcome, what the LLM did well, what it
  fabricated, what surfaced as a server bug vs an LLM gap.
- Cross-cutting findings — especially any case where composer
  `state.validate()` says OK and runtime `/validate` (or `/execute`) says
  not OK. That gap is the architectural finding worth surfacing.
- Final scoreboard: composer-happy / validate-happy / engine-executed /
  real-output for every scenario.
- File real bugs in `filigree` (the project's tracker — see CLAUDE.md for
  CLI/MCP reference). Use labels `composer` and `cluster:rc5-ux`. Include
  reproducers anyone can replay through the HTTP API.

## What to avoid

- Don't conclude "the LLM is bad" from a single failed run. Try the same
  ask twice if the failure mode is non-deterministic (it often is for
  GPT-5.5).
- Don't conclude "the LLM is great" from one successful run either.
  Especially watch for cases where the composer says "valid and complete"
  but `/validate` or `/execute` later rejects — the model trusts the
  composer's `is_valid` signal, so that whole class of failure is
  invisible from the chat alone.
- Don't burn excessive credit on layer-after-layer recovery loops on the
  same scenario. If you're three runtime errors deep into one pipeline
  and each fix exposes a new constraint, that **is** the finding. Stop,
  describe the gap, move on.
- Don't propose Python source edits or composer-internal fixes in the
  report unless you've actually located the responsible file/line. If
  you do, link to the exact path.

## Prior art

A previous eval lives at
`/home/john/elspeth/notes/composer-llm-eval-2026-04-28.md` with
transcripts in `/tmp/elspeth_eval/`. Read it first to see the format
expected, the bugs already filed (`elspeth-411435710b`,
`elspeth-178f765792`), and the workarounds that worked. If you reproduce
those same failures, note that — sustained reproducibility across runs is
itself a finding. If they no longer reproduce, even better — note the
delta and confirm the fix.

When you're done: leave the report durable on disk before declaring
completion. Then summarise out loud the headline finding (one sentence),
the per-scenario scoreboard, and the bugs filed/updated.

# Aborted rerun — 2026-05-06

This directory contains the **first turn only** of a `p1_t1_happy` rerun
attempted on 2026-05-06 against the post-restart staging service running
the `feat/eval-per-row-output-archival` code (closing
`elspeth-77d2641032`).

## What happened

- `harness.sh p1_t1_happy` created session `62e2a68f-1ce2-43a6-8a9a-fc5ca294dca7`
  and uploaded the q3 customer interactions CSV cleanly.
- `post_message.sh p1_t1_happy 1` posted Linda's verbatim opening prompt
  (`turn1.user.txt`).
- The composer hit the configured 300s wall-clock budget without
  converging — see `msg.t1.resp.json`:
  ```json
  {"detail": {"error_type": "convergence",
              "detail": "Composer did not converge within 0 turns
                         (budget exhausted: timeout)..."}}
  ```
- After 301.2s only the source was set in composer state
  (`state.after.t1.json`: `version=1`, `is_valid=false`,
  `nodes=[]`, `outputs=[]`).
- The rerun was aborted; the historical `p1_t1_happy/` directory
  (2026-05-03 v1 evidence) was restored as the canonical record.

## Why

Comparing this turn's metrics to the historical 3-turn full-eval
metrics:

| Metric | 2026-05-03 (3 turns total) | 2026-05-06 (turn 1 only) |
|---|---|---|
| Wall seconds | 159.79 | 301.2 |
| Composer convergence | Built complete pipeline | Stuck at v1 source-only |

The 2026-05-03 eval ran against `gpt-5` (the previously-deployed
composer model). The 2026-05-06 staging deploy runs `gpt-5-mini`
(`ELSPETH_WEB__COMPOSER_MODEL=openrouter/openai/gpt-5-mini` in
`deploy/elspeth-web.env`) — which on this prompt, this run, did not
converge within the 300s budget. This replicates `composer-obs-8f82c91147`
(3-of-3 happy-path repro of LLM convergence-timeout).

This is **not** a regression in the harness or the new `/api/runs/.../outputs`
endpoint — both are independently verified working. See
`evals/2026-05-06-endpoint-validation/` for the lightweight validation
that exercised the per-row endpoint with real audit-DB rows and
on-disk artifacts.

## Closing the audit-evidence gap fully

To produce a successful composer-driven hardmode rerun that lands
per-row outputs in this scenario directory:

1. Bump `ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS=300.0` to `600.0` in
   `deploy/elspeth-web.env`.
2. `sudo systemctl restart elspeth-web.service`.
3. Re-run `harness.sh p1_t1_happy`, drive the persona loop to DONE,
   then `finalize_scenario.sh p1_t1_happy` (the new finalize step
   pulls outputs from the working endpoint).

Or upgrade the composer back to `gpt-5` (which historically converges
in ~50s/turn).

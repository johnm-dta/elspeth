# Roadmap — ELSPETH                     Updated: 2026-07-07 (PDR-0001)

> Sequencing, WSJF / cost-of-delay, and dated forecasts are produced by
> /axiom-program-management. This file records bets as INTENT, not a delivery
> schedule. Do not compute WSJF here; hand the committed bet over for sequencing.

## Now  (committed, in-flight)

- **Web hardening to GA** *(primary bet, operator-confirmed 2026-06-14)* — close
  the Web-surface assurance gaps so the Composer path is safe for real users.
  Five clusters: auth/OIDC/JWKS · sessions/Alembic/session-db · blobs integrity &
  MIME · secrets store concurrency/portability · execution service terminal-state
  & path allowlist. · tracker: elspeth-250f698aaf, elspeth-ef52049338,
  elspeth-0fd9dfcb7e, elspeth-16ddaa7d02, elspeth-248536c9e6 · metric: north-star
  (run assurance completeness) + Web-GA input metric.

- **Ship the 0.7.0 line** *(in-flight delivery)* — LLM-primary composer work,
  CLI/TUI operator-surface refresh, and release hardening on the active
  release line. · branch: release/0.7.0 and feature worktrees · tracker:
  elspeth-82c3914f95 for the CLI/TUI refresh · metric: audit-integrity and
  operator explainability guardrails.

## Next (shaped, decreasing certainty)

- **Composer assurance parity** — composer correctness, LLM-evaluation
  remediation (validator parity / runtime dry-run / operator visibility), and
  deterministic advisor checkpoints. The "authoring without weakening assurance"
  bet made falsifiable. · tracker: elspeth-e1ab67e55a, elspeth-528bde62bb,
  elspeth-dac6602a2b · metric: composer authoring success (activation).

- **Fork/coalesce audit integrity** — schema reconciliation, field provenance,
  merge safety. · tracker: elspeth-e20903300c.

## Later (directional bets, no order, no dates)

- **Compiler facade** — seal YAML + composer input into one compiled, secret-safe
  pipeline artifact bound to Landscape provenance; run web and CLI execution from
  the verified artifact instead of reparsing YAML. The stated post-RC-5 direction
  in README. · tracker: none yet.
- **Plugin expansion (web research pipeline)** — OpenSearch, browser scrape,
  report sink, Chroma upgrade. · tracker: elspeth-868c55d712.

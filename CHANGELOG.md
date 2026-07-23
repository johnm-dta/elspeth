# Changelog

All notable changes to ELSPETH are documented here.

---

## 0.7.1 - 2026-07-23 (Recoverable effects and Composer proposal-validation coverage)

0.7.1 makes both pipeline publication and web authoring recoverable. It adds a
durable external-effect protocol for built-in sinks and audit exports, closes
the largest guided/freeform Composer capability gaps for complex DAGs, and adds
the supported AWS ECS deployment profile. The notes below intentionally cover
only release-level changes and critical correctness or security fixes.

**Breaking pre-1.0 schema cutover:** `SESSION_SCHEMA_EPOCH` advances from 26
to 35, guided checkpoints advance from schema 7 to 10, and Landscape
`SQLITE_SCHEMA_EPOCH` advances from 22 to 29. ELSPETH does not migrate either
predecessor database in place before 1.0. Archive or export required evidence,
stop the old service, recreate stale session and Landscape stores, then install
0.7.1. Do not roll older code back over the recreated databases; keep the
service drained and repair this release forward.

### Major changes

- **Durable, replay-safe external effects** — built-in file, object, database,
  Dataverse, and Chroma sinks now reserve and persist an immutable publication
  plan before external publication, fence the active worker, and reconcile
  uncertain outcomes before retrying. Audit exports use the same effect
  coordinator and sealed snapshots. The protocol prevents a crash or lost
  response from silently duplicating publication; an unprovable result remains
  explicitly blocked. Operators resume a recovered export with
  `elspeth export-resume <run-id> --execute`; see the
  [sink-effect recovery runbook](docs/runbooks/sink-effect-recovery.md).
- **Shared Composer proposal and validation contract** — guided-staged
  authoring covers seven of nine maintained parity fixtures through the
  shared proposal and validation contract used by freeform and guided-full
  authoring.
  The current code-proven limitations are cross-sink write-failure fallback and
  require-all coalesce; freeform and guided-full can author those shapes. A
  candidate remains separate from committed state until review and wire
  confirmation, and closed rejection codes feed bounded, auditable repair
  rather than silent replanning.
- **Durable Composer operations** — guided planning, start admission, failed
  operation evidence, fork replay, and proposal confirmation are persisted and
  fenced. Competing mutations receive a fast conflict, stale responses settle
  without overwriting newer state, and every planner attempt records its final
  disposition.
- **Supported AWS ECS runtime profile** — the release adds a lean Fargate web
  image, validate-only startup and deployment doctor, Aurora PostgreSQL, EFS,
  task-role S3 access, Bedrock provider and guardrails, Cognito authorization
  code with PKCE, plus CloudWatch and X-Ray telemetry. See the
  [AWS ECS deployment runbook](docs/runbooks/aws-ecs-deployment.md).
- **Principal-bound web plugin policy** — catalog, Composer, import,
  validation, execution, and delayed export now use one auditable plugin-policy
  snapshot. Operator profiles expose approved Azure or AWS model controls
  without writing private provider bindings into pipeline state.
- **First-class structural queue and text endpoints** — queues can fan multiple
  upstream routes into one downstream node without merging records, while text
  joins CSV and JSON as a supported source and atomic sink format.

### Critical fixes

- **Crash recovery no longer repeats completed sink or export work** — durable
  effect streams, ordered members, target-side ledgers, remote spooling, and
  exact reconciliation close the publication gap between external success and
  local acknowledgement. Failsink members retain the exact primary effect that
  produced them.
- **State-engine transitions are atomic at crash and contention seams** —
  source completion, child scheduling, aggregation continuation, coalesce
  completion, routing decisions, batch membership, and terminal outcomes now
  persist with their controlling state transition. Run-scoped ownership and
  lease fencing prevent cross-run or stale-worker mutation.
- **Composer cancellation and concurrency cannot wedge a session** — aborted
  freeform or guided turns settle before durable resynchronisation; startup
  waits for backend timing; fork quota failures replay as stable HTTP 413; and
  concurrent guided submissions settle as retryable conflicts instead of
  queued duplicate work.
- **DAG contracts propagate through routing and barrier nodes** — nested
  pass-through fields, pure-routing gates, queue fan-in, fork destinations, and
  coalesce union guarantees now reach downstream validation in graph order.
  The fixes prevent valid graphs from being rejected and invalid sink contracts
  from reaching execution.
- **Security-sensitive failures remain inside their trust boundary** — provider
  error text and blob redirect URLs are redacted from audit data, unknown LLM
  providers return closed rejection codes, remote OTLP endpoints require TLS,
  malformed ODBC brace syntax fails closed, and auth state compensates when its
  audit write cannot be persisted.

## 0.7.0 - 2026-07-09 (LLM-primary guided pipeline creation)

Guided pipeline creation becomes LLM-primary. The guided composer is
reworked so each stage of a pipeline — source, then sink, then transforms,
then the final wiring — is driven by a language model through the
`/guided/chat` endpoint, which applies the model's proposed change to the
in-progress pipeline in place rather than asking the operator to author
plugin options by hand. A revise mode reopens a committed stage and amends
it against its current state, and the flow terminates at a new wiring stage
whose completion is gated by an advisor sign-off. Current user-facing guidance
lives in `docs/release/composer-guide.md`; implemented design records were
removed from the active public docs tree, with git history remaining the public
provenance record and maintainers optionally preserving local ignored archives.

The web session database and Landscape audit database both reset on upgrade:
`SESSION_SCHEMA_EPOCH` advances to 26 and `SQLITE_SCHEMA_EPOCH` advances to
22. The audit schema now run-scopes `routing_events` with composite state/edge
foreign keys so routing decisions cannot cross audit-run boundaries. The
untagged 0.6.1 maintenance line is folded into this release rather than tagged
separately (the same precedent as 0.5.4 into 0.6.0).

### Added

#### LLM-primary staged guided pipeline creation

- **Each guided stage is built by a language model through `/guided/chat`**
  — the guided composer drives source, sink, transform, and wiring as
  ordered stages; at each stage the model interprets the operator's chat and
  applies the resulting change to the in-progress pipeline in place, instead
  of the operator hand-authoring plugin options. The source, sink, and
  transform drivers accept the current node so a committed stage can be
  revised, and a revise-mode addendum carries the prior stage state into the
  model's context (`revise_context`). The free-text sink driver auto-drops
  its wrapper, and a source driver routes a URL-row source to the web-scrape
  recipe.
- **`WorkflowProfile` value type** — a new closed-enum profile is threaded
  through guided-session bootstrap and persisted on `GuidedSession`,
  distinguishing a tutorial walk from an ordinary guided session. It is
  surfaced on every `GuidedSessionResponse` (via a `WorkflowProfileResponse`
  field), mirrored onto the TypeScript `GuidedSession`, seeded by a new
  idempotent `POST /guided/start` entry endpoint, and stripped on fork so a
  tutorial profile cannot leak into a forked session.
  `GUIDED_SESSION_SCHEMA_VERSION` advances 5→7 across the release.
- **`STEP_4_WIRE` wiring stage with a confirm-wiring contract** — a terminal
  wiring stage is added to the guided order; it rebuilds the pipeline edges
  from the model's connection labels, renders them with a contract overlay,
  and is gated by an explicit `CONFIRM_WIRING` payload with pinned
  required/nested keys. An invalid confirm re-emits the wire turn rather than
  completing, and a stale terminal stamp is cleared on a wire redirect.
- **Advisor sign-off authority at the wire stage** — a persisted-counter-
  bound wire sign-off runner gates the `STEP_4_WIRE` terminal on an advisor
  checkpoint (`run_signoff_checkpoint`), with a sign-off verdict classifier,
  differentiated sign-off audit names, fail-closed handling of blocked
  findings, and a whole-pipeline `REQUEST_ADVISOR` escape that re-emits —
  never auto-completes — the wire turn. The advisor pass counter is
  persisted on the guided session, and an optimistic-concurrency
  `step_index` 409 guards a guided respond against a lost update.
- **Pending-interpretation gate in the guided flow** — guided sessions
  project pending interpretation cards (invented source, pipeline decision,
  …) at the persist seam and block advancement until they are resolved,
  extending the interpretation-surfacing model into the guided flow with a
  backend run-tier backstop.
#### Guided mode reframed as a conversational builder

- **A docked chat paired with an always-visible verification panel** — the
  guided surface is reworked into a docked chat conversation paired with an
  always-visible verification panel: a plain-language gloss and a validation
  summary sit above the live pipeline graph so the operator can read what the
  model built as it is built, with a `ComposingIndicator` during the build.
- **Read-only interpretation decision cards with a View→Approve gate** — each
  interpretation decision renders as a read-only summary card that leads with
  the model's own rationale, carries an `Explain` button backed by grounded
  advisory context, and gates advancement behind a two-stage View→Approve
  acknowledgement over a card stack.

#### Always-on prompt-shield review

- **Prompt-shield review is now always-on** — the previously advisory
  (RC5.2-demoted) prompt-shield review is made always-on, with its State A
  decoupled and a three-state (A/B/C) shield-state helper plus an
  availability resolver; the B-vs-C shield refinement is wired into the
  confirm-wiring route, where the repair turn stages the fail-safe shield
  C-draft.

#### Passive first-run tutorial — synthetic web-scrape recut

- **Canonical first-run scenario retargeted to web-scrape extraction** — the
  hello-world tutorial now teaches source→transform→sink by having a model
  rate a set of synthetic government-style pages, over a deterministic
  synthetic-sample URL backed by an SSRF resolver; the staging rubric harness
  is retargeted to match, with assumption and shield-override teaching-moment
  copy. The tutorial runs as a staged guided walk through
  `TutorialGuidedShell` (a welcome bookend wrapping an embedded guided
  `ChatPanel`) with prepopulated, locked stage prompts, and never falls
  through to the freeform composer body.
- **Tutorial wired to public GitHub Pages** — the scrape pages are served
  from public GitHub Pages rather than synthetic in-app pages, and the
  tutorial uses the DTA abuse contact rather than a personal dev domain.
- **`allowed_hosts` SSRF slot on the web-scrape recipe** — an optional
  `allowed_hosts` slot is added to the web-scrape recipe as an SSRF
  enforcement boundary (not a slot-inaccessibility control), with the
  tutorial-sample GET surface injecting the allowed host at the accept seam
  and a blank `tutorial_sample_base_url` rejected at config time.
- **Five-input tutorial cache key** — the tutorial cache key now folds the
  staged guided skill hash and the recipe-catalog content hash
  (`guided_staged_skill_hash`, `recipe_catalog_content_hash`) alongside its
  existing inputs, so a change to either the staged skill or the recipe
  catalog invalidates a stale tutorial build.
- **Tutorial workspace promoted to the guided layout and relaid out** — the
  first-run tutorial adopts the guided layout as its canonical shell: a
  freeform-parity conversation column, an action-zone decision surface, and a
  320px artifact rail, with the guided composer docked at the bottom to match
  the live composer. The tutorial can be reset from preferences at any time,
  and its stage is now persisted so a mid-tutorial reload resumes at the saved
  stage — including post-run — rather than restarting at the welcome bookend
  (backed by the new `user_preferences` tutorial-resume columns; see
  Operational).

#### Azure Document Intelligence enrichment transform

- **`azure_document_intelligence` enrichment transform** — a new transform
  plugin enriches rows by sending a document to Azure AI Document
  Intelligence and folding the extracted layout/content back onto the row. It
  is registered under the `azure_document_intelligence` plugin name and
  declared as an external-call boundary in the audit-readiness expectations,
  with fail-closed page-count handling and host:port pinning on its endpoint.

#### Blob-backed document ingestion transforms

- **`blob_fetch` transform** — a new HTTP(S) fetch transform stores an
  operator-authorised remote document in the run payload store and emits a
  blob reference plus fetch metadata (`content_type`, `size_bytes`,
  `sha256`, HTTP status, final URL, and final resolved IP). The transform is
  an external-call audit boundary, uses the existing SSRF-safe HTTP client,
  requires wire-visible `abuse_contact` and `fetch_reason` headers, and keeps
  the accepted file surface behind an explicit content-type allowlist for
  text, CSV, JSON/JSONL, and XML payloads.
- **`blob_csv_expand` transform** — a new row-expansion transform reads a
  payload-store CSV blob and emits one output row per CSV data row while
  preserving the upstream row, so URL/document identifiers continue to
  disambiguate rows after expansion. It supports header normalization,
  explicit columns for headerless CSVs, field mappings, row-index emission,
  blob-size and row-count limits, and schema-contract propagation for the
  expanded rows.
- **Web/composer integration for blob workflows** — guided composer guidance
  now treats remote document ingestion as a manifest source followed by
  `blob_fetch` and a parser transform, not as a new source plugin. Web
  execution validation applies the same public-only SSRF policy to
  `blob_fetch` as `web_scrape`, audit-readiness now classifies parser
  transforms with `Determinism.IO_READ` as boundary surfaces, and
  `examples/blob_transforms/` demonstrates both offline CSV blob expansion
  and an opt-in hosted tutorial HTML fetch.

#### ELSPETH design system and marketing website

- **Typed React primitive library (`components/ui`)** — a typed primitive
  library (Button, Card, Input, Textarea, Tabs, AlertBanner, StatusBadge,
  TypeBadge, WordMark) lands under
  `src/elspeth/web/frontend/src/components/ui/`, font weights and wordmark
  tracking are tokenized, and `LoginPage` is migrated onto the primitives as
  an exemplar.
- **Standalone marketing website (`website/`)** — a static, design-token-
  based marketing site is added at `website/`, separate from the application
  frontend, and hardened for public release: the example pipeline YAML uses
  the real secret-reference syntax (`${VAR}` for the CLI, `{secret_ref: NAME}`
  for the Composer) so a published example runs as written; the data-source
  list, the two authoring modes, and the Web Composer on-ramp (`elspeth web`)
  are reconciled against the source tree; WCAG 2.1 AA contrast and Level-A
  semantics (skip link, `<main>` landmark, heading order, native data tables)
  are added; the icon set is vendored locally in place of a third-party CDN;
  and the colour theme follows the operating-system preference and persists
  across pages.

#### CLI and TUI inspection

- **Machine-readable plugin catalog inspection** — `elspeth plugins list
  --format json` and `elspeth plugins inspect <source|transform|sink> <name>
  --format json` expose catalog descriptions, config fields, JSON Schema, and
  Composer knob schema for automation and docs tooling.
- **Graph-backed `elspeth explain` TUI** — the lineage explorer now renders a
  selectable run/branch/node/token/status tree from recorded graph edges,
  including branch labels, repeated DAG joins, focused evidence views, refresh,
  and non-interactive `--no-tui` / `--json` fallbacks.

#### Web Composer auth and local user operations

- **Local Composer user-management CLI** — `elspeth composer users add` and
  `elspeth composer users remove` manage `data/auth.db` without requiring
  operators to run Python snippets against `LocalAuthProvider`.
- **Email-verified local registration mode** —
  `ELSPETH_WEB__REGISTRATION_MODE=email_verified` creates pending local users,
  writes one-use verification links to `data/email-verifications.jsonl`, and
  activates users through `/api/auth/verify-email` / `?verify_token=...`.
  Non-local deployments must set `ELSPETH_WEB__PUBLIC_BASE_URL`.

#### Release review tooling

- **Codex panel-review foundation** — `scripts/codex_panel_review.py` adds a
  single-file, multi-lens SME review runner with strict structured findings,
  cache-friendly prompt layering, dry-run/cache-hit reporting, and a
  fail-closed evidence gate for unanchored findings.

### Changed

- **Composer routes decomposed into an area package** — the monolithic
  `routes/composer.py` is decomposed into a `routes/composer/` area package,
  splitting the route module without changing its HTTP surface.
- **Release documentation cleanout completed** — the remaining implemented
  plans and design specs under `docs/plans/`, `docs/superpowers/plans/`, and
  `docs/superpowers/specs/` were removed from tracked active docs while
  keeping `docs/` focused on current user, operator, architecture, and release
  documentation. Maintainers may preserve the removed files in a local ignored
  archive; public provenance remains available through git history.
- **Explicit `ProcessorMode` and a public follower-drain surface** — the
  engine's leader/follower processor mode is made an explicit value and the
  follower drain is promoted to a public surface, replacing the previous
  implicit mode selection so the run coordinator's behaviour is legible at the
  seam rather than inferred.
- **Dead `aggregation_boundaries` checkpoint knob removed** — the unused
  aggregation-boundaries checkpoint configuration option is deleted from the
  config surface (pre-release tech-debt removal; there is no data to migrate).
- **Dynamic attribute probes replaced with explicit contracts** — the
  remaining `hasattr`/fallback attribute probing is removed across provider
  response parsing, export sink shape handling, plugin lifecycle attributes,
  journal platform checks, composer trust seams, and web route application
  state; each site now declares the contract it relies on and fails closed
  when that contract is not met. Provider response parsing keeps to data-only
  reads while still mapping Pydantic extra fields, so provider token counts,
  costs, and cache-usage siblings remain captured.

### Fixed

Assorted engine, web, and frontend correctness fixes land alongside the
headline work.

- **Run/session durability and liveness** — the run-event replay cursor is
  made durable; follower liveness is forced per claim and unregistered
  claimants are fenced; session quota writers are serialized; a dependency
  settings-hash snapshot is taken before a run; empty assistant audit rows
  are rejected; and coalesce-branch sink collisions are rejected.
- **Composer and guided robustness** — dynamic-source-from-chat resolves to a
  committable inline source; blob source names are validated; guided audit
  payload refs are persisted; the guided source-chat resolver is guarded;
  tutorial runs are rate-limited; stale guided default entries no longer
  break the frontend; and a completed guided wizard no longer re-fires on
  remount.
- **Provider and MCP boundary hygiene** — OpenRouter preflight response
  bodies are validated and a stale catalog is cleared on a failed refresh;
  MCP enum tool arguments are validated, malformed mutation validation is
  deferred, and runtime preflight summary operations are counted; malformed
  `Content-Length` is rejected; and non-standard HTTP status codes are
  audited.
- **Catalog, config, and allowlist** — aliased plugin knobs surface under
  their user-facing alias; env schema option overrides are normalized; and a
  dangling allowlist `allow_hits` entry degrades to skip-and-warn instead of
  crashing the run.
- **Engine fail-closed hardening** — a wave of integrity fixes make the run
  engine refuse to proceed on ambiguous state rather than silently guess:
  `resume()` re-verifies checkpoint currency and topology at entry; checkpoint
  writes, the heartbeat, and best-effort transforms fail closed on Tier-1
  integrity errors; coalesce journal restore fails closed on unknown or
  overlapping branches; unmapped traversal nodes fail closed via an explicit
  structural allowlist; scheduler disposition writes are membership-fenced;
  transform-error audit is recorded before terminal completion; diverted sink
  anchors are closed when the primary audit fails; `PENDING_SINK` replay
  preserves the persisted error hash; and condition-trigger fire times are
  sampled at observation rather than backdated.
- **Guided and tutorial front-end robustness** — the guided chat seam gains a
  discriminator, self-heal, and reload-resume; a dead resume session is
  recovered and the tutorial runs once rather than re-firing on remount;
  advisory replies reject tool-scaffold leaks and clamp the decision headline;
  assistant markdown renders in the flat transcript; and the wire-stage
  advisor dead-end is closed.
- **Chat and node-config layout tamed on wide displays** — chat bubbles are
  held to a centred reading column and the node-config inspector stacks its
  option rows, so neither sprawls edge-to-edge on a wide viewport.
- **YAML export/import round-trip fidelity** — a pipeline exported to YAML and
  re-imported preserves its `source_blob_ids` sidecar and reattaches the
  guided `blob_ref` without leaking a filesystem path; export is unblocked for
  secret-bearing pipelines; and the import path is hardened against pasted
  input.
- **Validation-error legibility** — a pydantic missing-source / missing-sink
  error is reframed into novice-facing copy at settings-load, and chat
  validation-failure injections are humanised through a store-safe frontend
  module (`validationHumaniser.ts`).
- **Secret-field heuristic false positives** — structural field names that
  merely end in `_key` (`data_key`, `alternate_key`) are exempted from the
  secret-field heuristic by exact name, so a JSON source's `data_key` or a
  Dataverse sink's `alternate_key` is no longer misclassified as a credential.
- **Sink, provider, and HTTP correctness** — CSV sink stringification
  `ValueError`s propagate instead of being swallowed; non-string Chroma
  required fields are diverted; generic Dataverse 4xx responses fail closed;
  lost-branch union schema fields are tolerated; absolute-path SQLite DSNs keep
  their slash count through sanitization; and `AuditedHTTPClient` no longer
  double-decompresses gzip bodies.
- **Silent-degradation paths converted to structured failures** — the
  FAILED-resume counter baseline handles only database `OperationalError`, so
  other audit corruption signals propagate instead of degrading to partial
  counters; race-created journal parent directories are verified owner-only
  through a symlink-refusing check rather than skipping enforcement when the
  directory already exists; catalog plugin-schema parsing is extracted into a
  typed parser so a malformed plugin schema can no longer under-detect
  required secret fields; guided respond reads required turn-response keys
  directly so schema drift crashes instead of silently evaluating to
  `False`; and guided-session payloads serialized before the
  `on_validation_failure` field existed are refused at session rehydrate
  rather than silently defaulted (pre-release session stores are recreated
  at deploy).

### Security

A large trust-boundary hardening body lands in this release. The fixes keep
tool arguments, provider diagnostics, and audit anchors from leaking caller
data or being forged, and tighten transport and endpoint authentication.

- **Declarative redaction of composer tool arguments and responses** — source
  option values, recipe slots, set-pipeline options, blob-discovery extra
  arguments, advisor unknown arguments, and unexpected mutation tool
  arguments are redacted before they reach an audit record or a model. CSV
  header-mismatch evidence is redacted; advisor and OpenRouter provider /
  diagnostic errors are redacted before LLM evaluation; exception redaction
  is preserved through the operation audit; MCP preflight crash errors and
  Azure managed-identity auth failures are sanitized; and embedded PEM key
  blocks are redacted.
- **Forged audit anchors blocked** — forged or operator-anchored LLM audit
  anchors are rejected, so a model cannot inject a counterfeit audit
  reference; the HTML fragment separator is escaped and file-output audit
  URIs are encoded.
- **Transport and endpoint authentication tightened** — websocket
  authentication moves from a JWT-in-query parameter to tickets; HTTPS is
  required for bearer and OpenRouter endpoints; the Prometheus metrics
  endpoint requires authentication; managed-identity RAG configs are blocked;
  blob mutations are gated behind approval; recovery execution is bound to
  reviewed state; web-authored OpenRouter base URLs are gated to the canonical
  endpoint; AWS IPv6 metadata endpoints are blocked alongside the existing
  SSRF-safe web controls; and the secret inventory is made metadata-only.
- **CI judge-tool fail-closed gates and signature verification
  (`elspeth_lints`)** — agent judge tools fail closed; judge-tool reads are
  gated through the scrubber and blocked on signing paths; rejudge and new
  judge-metadata signatures are verified in the audit gates; justify prompts
  are bound to the scan snapshot; stale v1 scope migrations are refused; and
  unverified fork-allowlist signatures are rejected.
- **Key-separated judge-signature handoff (`elspeth_lints`)** — agents now
  stage trust-tier review bundles through the key-free `elspeth-judge` MCP
  server, while operators mint or rotate signed judge metadata only through
  the key-bearing `elspeth-lints sign-bundle` / `rekey` CLI. CI is verify-only
  and the obsolete `scripts/cicd/sign_accept_backlog.py` path is removed.
- **Bounded budgets** — sequential LLM retry budgets and MCP Landscape query
  limits are bounded so a runaway model or query cannot exhaust the host.
- **Key Vault `vault_url` restricted to approved Azure endpoints** — the Azure
  Key Vault `vault_url` is constrained to the approved `*.vault.azure.net`
  endpoint class by a host check (which a bare suffix test cannot enforce),
  with an optional exact-URL pin via `ELSPETH_KEYVAULT_ALLOWED_VAULT_URLS`,
  closing an SSRF vector where a foreign vault URL could redirect the
  managed-identity challenge; edge whitespace is normalised before validation.
- **Blob sink path allowlist scoped to the owning session** — composer blob
  writes and reads are confined to the session that owns them, and the session
  id is threaded through shareable-review validation, so one session cannot
  address another session's blob paths.
- **Database credentials scrubbed before audit persistence** — `odbc_connect`
  connection-string passwords and database-node DSN passwords are scrubbed
  before they can reach an audit record or a fingerprint.
- **Output-echoed fields reject environment-variable placeholders** — the
  `report_assemble` transform rejects env-var placeholders in the fields it
  echoes to output, so an unresolved `${VAR}` cannot be surfaced as literal
  content.
- **Host environment exposure reduced** — commencement gates no longer receive
  the process environment by default, and in-memory pipeline YAML loading no
  longer expands host environment variables unless the caller explicitly opts in.
- **Further audit-export and gate hardening** — raw failing rows are redacted
  from audit exports by default; CSV audit exports neutralize spreadsheet
  formulas; JSONL filesystem exports are staged before publish; harness
  credentials are kept out of the eval `curl` argv and backfill sink paths are
  restricted to data roots; and the `elspeth_lints` gates gain reaudit-limit,
  allowlist-expiry, stale-judged-signature, and override-hash-IO fail-closed
  handling.

### Operational

- **The web session database and Landscape audit database both reset on
  upgrade.** `SESSION_SCHEMA_EPOCH` advances to 26 and
  `SQLITE_SCHEMA_EPOCH` advances to 22. `GUIDED_SESSION_SCHEMA_VERSION`
  reached 7 over the release (5→7), and those guided fields live in the
  `composition_states.composer_meta` JSON blob, so they add no SQL column. The
  final 25→26 session epoch bump adds first-run-tutorial resume columns
  (`tutorial_stage`, `tutorial_session_id`, `tutorial_run_id`,
  `tutorial_source_data_hash`) to `user_preferences`; the Landscape epoch bump
  run-scopes `routing_events` with composite state/edge foreign keys. Before
  first start on 0.7.0, stop `elspeth-web.service`, archive and remove
  `data/sessions.db` plus sidecars and the configured Landscape audit DB plus
  sidecars, then restart so both schemas are recreated. `data/auth.db` is
  separate, so local user accounts survive. Procedure:
  `docs/runbooks/staging-session-db-recreation.md`, including the hardened
  archive steps — `PRAGMA wal_checkpoint(TRUNCATE)` before copying each
  database, destroy or secure the archives at the end of the deploy window,
  and rotate `settings.secret_key` so an archived copy of encrypted
  `user_secrets` rows is inert under the new key.
- **Ship a frontend dist rebuilt from this release's source.** The web UI is
  served from `src/elspeth/web/frontend/dist/`, which is gitignored and built
  out of band (`cd src/elspeth/web/frontend && npm run build`); it is produced
  by neither CI nor the Docker image. The 0.7.0 frontend carries the new
  `components/ui` primitive library, the `LoginPage` migration onto it, the
  tokenized font weights, and the guided wiring-stage / tutorial-shell
  changes, so producing the 0.7.0 web deploy MUST rebuild the dist from
  `release/0.7.0` HEAD rather than reuse a prebuilt bundle; a stale bundle
  will not render the new login, email-verification, design-token, or guided
  wire-stage surfaces.
- **CLI/MCP audit database defaults consolidate under `data/`.** Ad-hoc CLI and
  MCP Landscape defaults now point at `data/audit.db` instead of the older
  `state/audit.db` path, matching the release's system-database layout and
  reducing accidental split-brain audit stores during local operator work.
- **Local and CI gate behaviour is explicit.** The operator-approved pre-commit
  cleanup removes the whole-tree wardline hook from the local hook set; release
  agents still run `wardline scan . --fail-on ERROR` explicitly before handoff
  when external-input boundaries are touched. CodeQL now runs the
  `security-extended` query suite without `security-and-quality`, matching the
  release's security-gate posture and avoiding quality-only noise from blocking
  the release.

---

## 0.6.0 - 2026-06-20 (cross-process multi-worker run coordination)

The single-worker-to-multi-worker transition. Multiple cooperating
processes on one host may now operate against a single run backed by one
WAL SQLite audit database: one leader (source ingest, barrier trigger
evaluation, checkpoints, finalization, sink I/O) and any number of
claim-only followers attached through the new `elspeth join` entry point.
The deployment shape, its alternatives, and the operator requirements are
recorded in ADR-030. Builds on the 0.5.4 maintenance fixes below.

The audit database schema epoch advances to 21; per the delete-the-DB
migration policy operators delete the prior database before first run on
this version.

### Added

- **`elspeth join <run_id>`** — attach a follower to a RUNNING run. A
  second worker joining is the supported feature; racing `resume()`
  remains refused. Admission is one atomic transaction gated on run
  status, pipeline-config-hash equality, and a live leader seat, after a
  filesystem preflight that fails with an operator-actionable error when
  the worker cannot write the database, its directory, or the WAL
  sidecars.
- **Run-level worker registry and heartbeat** — a dedicated heartbeat
  thread beats the worker row and the leader seat in one transaction;
  liveness drives takeover and reaping decisions.
- **Dead-leader takeover** — a run left RUNNING under a dead leader is now
  resumable via `elspeth resume` (previously a wedged, unrecoverable
  state); a leader frozen holding the write lock surfaces an
  operator-actionable error naming the process to terminate before
  resuming.
- **`examples/multi_worker` and `examples/multi_worker_showcase`** — two
  runnable examples for the multi-worker feature. `multi_worker` is
  self-verifying: it backgrounds a leader, attaches a follower via
  `elspeth join`, and asserts that two or more workers each completed at
  least one row using `scheduler_events` lease-owner attribution.
  `multi_worker_showcase` demonstrates a wider multi-source / concurrent
  spectacle without a correctness assertion. Both are registered in
  `examples/README.md` and `examples/AGENTS.md`.
- **`examples/concurrent_scheduler`** — a runnable proof that the scheduler
  keeps multiple token lifecycles open at once: two three-row CSV sources
  feed a count-6 `batch_stats` barrier that can only fire if all six tokens
  are alive simultaneously (one-at-a-time draining would deadlock). It
  ships with read-only SQL attribution queries confirming both sources fed
  the single batch.

### Changed

- **Barrier buffers are journal-first** — aggregation and coalesce
  acceptance is durable in the scheduler journal before it enters
  executor memory, so barrier state spans workers and survives takeover
  (ADR-029 amended).
- **Leader writes are epoch-fenced** — finalization, run-status changes,
  checkpoints, barrier completion, lease recovery, and source ingest each
  verify leadership inside their transaction; a superseded leader's writes
  are refused, not applied.
- **Cross-process write discipline** — writable Landscape transactions
  take the WAL write lock at `BEGIN IMMEDIATE`; dashboard and audit-story
  read paths open read-only (ADR-030 §D5).
- **Lease recovery is liveness-aware** — an expired item lease held by a
  worker whose registry heartbeat is still live is revived, not reaped, so
  a long in-flight model call is no longer mistaken for a dead worker.
- **Web composer enforces implicit-required contracts for named typed
  sources** — the predicate that decided whether a producer was a typed
  source matched only the bare string `"source"`, so it returned `False`
  for every *named* source (producer id `source:<name>`). The
  implicit-required parity check therefore never fired for the headline
  multi-source case: the pipeline validated green at compose time but
  runtime Phase-2 type validation rejected it. The predicate now uses
  `is_source_producer_id()`, closing the compose-green / runtime-red
  divergence (elspeth-3332619032).

See the rewritten `docs/runbooks/scheduler-lease-recovery.md` for N>1
recovery procedures and operator guidance (worker count, lease sizing,
shared group/clock requirements, per-worker forensic journals).

### Fixed

Plugin-boundary correctness — the plugins-subsystem remediation. Each of
these defects let a plugin coerce, fabricate, or self-contradict at the
trust boundary without an honest audit record; the fixes keep the boundary
truthful (no silent Tier-3 coercion, no fabricated statistics, no
fail-open scanning, no row that fails its own contract).

- **`AzureBlobSource` CSV parses strictly** — the source built its
  `csv.reader` without `strict=True`, so malformed quoting (data after a
  closing quote) was silently merged into adjacent fields and, when the
  field count still matched, passed through as a valid row with no
  quarantine and no audit record. It now parses with `strict=True` like
  `CSVSource`: the malformed row is recorded and quarantined, and
  processing stops because the parser state is no longer trustworthy
  (elspeth-ebe13515f4).
- **`batch_stats` skips and reports `None` instead of crashing the run** —
  a `None` value field raised `TypeError`, which the aggregation executor
  records FAILED and re-raises, aborting the whole run; an all-`None` group
  additionally hit a `mean = 0/0` `ZeroDivisionError`. `None` is now
  skipped and reported in `skipped_missing` / `skipped_missing_indices`
  (mirroring `skipped_non_finite`), and a group with no aggregatable values
  returns an audited `validation_failed` error naming each skipped row
  rather than a fabricated `count=0`/`sum=0`. Genuine wrong types still
  raise at the boundary (elspeth-e62478e5db).
- **`batch_outlier_annotator` no longer fabricates `robust_z_score=0.0`
  when MAD=0** — any batch where more than half the values are identical
  (common for score/count data) drove the median absolute deviation to
  zero, which the annotator reported as a robust z-score of `0.0`,
  silently disabling outlier detection exactly when it mattered. It now
  applies the Iglewicz–Hoaglin mean-absolute-deviation modified z-score
  when spread is still present, and emits an honest `None` (never `0.0`)
  for a wholly identical batch (elspeth-a46c6e361f).
- **Non-finite scanner uses `isinstance`, not exact-type checks** —
  `_find_non_finite_value_path` gated container recursion on exact type, so
  a `NaN`/`Infinity` nested inside a `Mapping` subclass (e.g. `OrderedDict`)
  or a `tuple` subclass (e.g. a namedtuple) bypassed the source-boundary
  non-finite gate entirely — a fail-open leak into audit hashing. The
  float, mapping, sequence, and ndarray checks now use `isinstance`; `bool`
  stays excluded and masked-array `NaN` is still treated as absent.
- **`value_transform` retypes the output contract on a typed-field
  overwrite** — overwriting an existing typed field with a
  different-typed result (e.g. an `int` price recomputed as a `float`) left
  the stale `FieldContract.python_type` in place, so the emitted row failed
  its own `out.contract.validate(...)` — a self-contradictory audit record.
  An overwrite whose runtime type differs from the declared type now
  rebuilds that field's contract to the value's type, preserving
  `original_name` / `required` / `source` / `nullable`; `object`/`any`
  fields are left untouched.
- **`AzureBlobSink` CSV no longer diverts valid rows in flexible/observed
  mode** — the per-row staging probe trial-encoded each row against only
  the first row's keys, while the real serializer computes field names
  cumulatively across all rows. A later row carrying a valid extra field
  (legitimate in flexible/observed mode) tripped `DictWriter`'s
  `extrasaction='raise'` and was routed to the failure sink before the
  serializer that would have accepted it ever ran — silent data loss. The
  probe now uses the same schema-aware field set as the serializer, so a
  valid late-appearing field is kept while a genuine fixed-mode
  column-lock violation is still diverted per-row.
- **`batch_experiment_compare` no longer fabricates inferential statistics
  for a singleton arm** — `stdev` is undefined at n=1 and was reported as
  an honest `None`, but the comparison then coerced that `None` to `0.0`
  in the standard-error term, so an arm with a single value still produced
  a real `standard_error`, `z_score`, and 95% confidence interval from a
  zero-variance assumption. When either arm has undefined variance the
  standard error, z-score, and interval are now all `None`.
- **`batch_effect_size` emits `None` pooled dispersion for two singletons**
  — when both groups had a single value the pooled-variance denominator
  (`n₁+n₂−2`) is zero and the pooled standard deviation is undefined, but
  it was reported as a real `0.0` (while the per-group stdevs and Cohen's
  *d* were correctly `None`). The pooled standard deviation is now `None`
  in that case, consistent with the other undefined dispersion statistics.
- **`json_source` and `dataverse` narrow their header catch to
  configuration faults** — both caught broad `ValueError` around header
  normalization, so a configuration mistake (a `field_mapping` collision,
  an unknown key) was silently quarantined as if it were a per-row data
  error. The catch is now scoped to `ExternalHeaderError`, so a config
  fault surfaces at the boundary instead of as a phantom quarantine.
- **`CSVSink` and `AzureBlobSink` trial-encode each row before staging** —
  a single unencodable character (e.g. an emoji written to a `cp1252`
  sink) bypassed the per-row divert path and raised `UnicodeEncodeError`
  at write time, aborting the whole batch with no per-row audit record.
  Each row is now trial-encoded individually during staging, so the codec
  fault is caught as a per-row diversion and the surrounding good rows are
  still written.
- **`database_sink` diverts `DataError` per row and serializes flexible
  extra columns** — a `DataError` (e.g. integer overflow on a typed
  column) is per-row attributable but was missing from the divert arms, so
  it crashed the batch instead of diverting the offending row. Separately,
  typed-field serialization now covers all fields, not only declared ones,
  so a flexible-mode row carrying an extra typed column is serialized
  correctly.
- **RAG `count()` inside `search()` is guarded like `query()`** — the bare
  `count()` call had no exception handling, so a transient backend error
  (`ChromaError`, `ConnectionError`, `TimeoutError`, `OSError`) escaped as
  a raw traceback that crashed the run with no quarantine and no audit
  record. It now shares the `query()` except arms.
- **Chroma readiness probe widens its catch and closes its client** — the
  probe missed plain `ValueError` (raised by `chromadb` when the server is
  unreachable) and `httpx` transport errors, which escaped as raw
  tracebacks and aborted the commencement gate; it also leaked the
  `HttpClient`. The except set is widened and the client is closed in a
  `finally` block.
- **Sequential multi-query transforms retry transient errors within a
  bounded budget** — a transient 429/5xx/network error on a sequential
  multi-query row propagated immediately and failed the row. Such errors
  are now retried locally within a `max_capacity_retry_seconds` budget
  (mirroring the Azure capacity-retry pattern), without re-running queries
  that already succeeded for the same row; on budget exhaustion the row
  diverts as `retry_timeout`, while non-retryable errors (context-length,
  content-policy) still fail immediately.
- **RAG query returns a `missing_field` error instead of a bare
  `KeyError`** — a missing query field raised an uncaught `KeyError` with
  no audit record; a presence check now returns a structured
  `missing_field` error, consistent with the rest of the RAG boundary.
- **`web_scrape` treats unenumerated 4xx responses as errors, not
  content** — codes outside the handled set (400, 402, 405, 410, …) were
  returned as successful responses and their error-page body was
  fingerprinted and audited as real content, corrupting change detection.
  They now raise a non-retryable client error; 408 Request Timeout raises
  a retryable one.
- **`web_scrape` guards content type and body size before extraction** —
  there was no `Content-Type` check and no size cap, so a binary response
  (image, PDF, `application/octet-stream`) was extracted and fingerprinted
  as text. A content-type guard now rejects non-text responses and a
  configurable cap rejects oversized bodies before extraction. (The size
  cap is a post-buffer guard; the streaming pre-buffer cap follow-up is
  noted below.)
- **`batch_classifier_metrics` accepts an all-negative batch as data** —
  when `positive_label` was configured but absent from the batch (a
  legitimate outcome in rare-positive monitoring) the plugin returned
  `validation_failed`. An all-negative batch now yields honest
  zero-positive metrics; only a batch with no labels at all remains an
  error.
- **LLM `chat_completion` records the call before re-raising on a
  malformed response** — the `response.usage` /
  `response.choices[0].message` reads happened outside the guarded region,
  so a malformed provider response left no Landscape record even though
  tokens had been consumed. The reads are now guarded; on failure the call
  is recorded as ERROR with the raw response captured before a
  non-retryable client error is raised.
- **Field-resolution rebuild is a union, not a replace** — in
  `json_source`, `azure_blob_source`, and `dataverse`, a later sparse row
  introducing a new key replaced the field-resolution set with only that
  row's fields, discarding previously seen keys and corrupting the
  field-resolution audit record. The rebuild now merges all observed keys
  across all rows.
- **`AzureBlobSource` rejects a headerless CSV with no columns or schema at
  configuration time** — the `has_header=False` / no-columns path invented
  numeric field names (`"0"`, `"1"`, …) that are not valid identifiers,
  broke `field_mapping`, and skipped field resolution entirely, leaving the
  audit record permanently absent. That configuration is now refused at
  startup with an operator-actionable error.
- **Statistical batch transforms emit `None` for every undefined statistic,
  never a fabricated `0.0`** — extending the singleton-arm fixes above to
  the rest of the family: `batch_distribution_profile` emits `None` stdev
  at n=1; `batch_paired_preference` emits `None` for an all-tie preference
  rate and for the standard error / interval when an arm has n≤1; and
  `batch_outlier_annotator` emits `None` (not `0.0`) for the standard
  z-score when the batch standard deviation is zero.
- **Six batch transforms guard against non-finite group keys** — a `NaN`
  float or `Decimal('NaN')` used as a group-by key silently fragments
  groups (because `NaN != NaN`). The non-finite group-key guard already in
  `batch_effect_size` is now ported to `batch_experiment_compare`,
  `batch_distribution_profile`, `batch_paired_preference`, `batch_stats`,
  `batch_top_k`, and `batch_drift_compare`, including the `Decimal` arm.
- **`batch_paired_preference` errors on a duplicate variant within a
  pair** — two rows sharing the same variant label silently used the first
  and dropped the second (silent data loss). The plugin now detects and
  errors on a duplicate variant before processing the pair.
- **`batch_replicate` records the quarantine row index, not the row body** —
  the invalid-copies quarantine entry embedded the full row content in its
  audit metadata, leaking Tier-2/3 data into the audit record. It now
  records the row index only.

Two engine-level correctness fixes also land in this release:

- **Aggregation resume no longer bricks a run after a FAILED-flush crash** —
  a crash between marking aggregation tokens terminally failed and
  releasing their BLOCKED scheduler rows left durable BLOCKED rows whose
  tokens were already terminal; on resume the restore path saw an empty
  live-BUFFERED set and raised `AuditIntegrityError` on every attempt,
  making the run permanently unresumable. The aggregation restore arm now
  journal-releases BLOCKED rows whose tokens carry a terminal outcome
  before restoring, mirroring the proven coalesce path (elspeth-55546a6fd6).
- **Multi-worker N>1 leader/follower coordination hardened** — the real
  `elspeth join` end-to-end exercise surfaced and closed several N>1
  defects: a follower whose leader finalized mid-drain now exits clean
  rather than reporting eviction; the post-finalize source-lifecycle write
  no longer downgrades a terminal `EXHAUSTED` state; `elspeth run` /
  `elspeth resume` surface a takeover as a clean eviction exit instead of a
  fatal traceback; the session-fork blob-copy invariant is no longer
  suppressed when the blob map is empty; the peer-lease wait is bounded to
  the liveness window and honours shutdown; follower teardown propagates
  Tier-1 audit-integrity errors instead of swallowing them; and the
  `PENDING_SINK` drain correctly recognizes peer-owned work and loops until
  no peer leases or scheduled work remain.

The shipped `examples/multi_worker` harness also had its pass/fail verdict
corrected: a follower exiting non-zero was logged but not folded into the
final decision, so the run could print PASS after a worker failed. The
verdict now requires both the attribution assertion (≥2 workers each
completed a row) and a clean exit from the leader and every follower.

A follow-up is tracked for `web_scrape`'s `max_body_bytes`, which is
enforced only after `AuditedHTTPClient` has already buffered and
audit-captured the full body; a true pre-buffer cap belongs in the shared
client as a streaming byte-limit (elspeth-a6f246d02a).

### Operational

- **The web session database also resets on upgrade (in addition to the
  audit DB above)** — the session database schema epoch advances to 19
  (`SESSION_SCHEMA_EPOCH`). It is not migrated in place: 0.6.0 boot fails
  closed on a pre-0.6.0 session DB with `SessionSchemaError: Session DB
  schema version 18 does not match SESSION_SCHEMA_EPOCH=19. Pre-release
  ELSPETH does not migrate session databases. Delete the session DB file
  and restart.` Before first start on 0.6.0, stop `elspeth-web.service`,
  back up and remove `data/sessions.db` (and its `-wal`/`-shm` sidecars),
  and restart; the bootstrap recreates the schema on first start.
  `data/auth.db` is a SEPARATE file — local user accounts survive the
  reset. Procedure: `docs/runbooks/staging-session-db-recreation.md`.
- **Ship a frontend dist rebuilt from this release's source** — the web UI
  is served from `src/elspeth/web/frontend/dist/`, which is gitignored and
  built out of band (`cd src/elspeth/web/frontend && npm run build`); it is
  produced by neither CI nor the Docker image. Producing the 0.6.0 web
  deploy MUST rebuild the dist from `release/0.6.0` HEAD rather than reuse a
  prebuilt bundle. A dist built before `ebbf90fcd` (the ADR-025
  plural-`sources` migration) reads the dropped singular `state.source` and
  crashes the chat panel with `TypeError: Cannot read properties of
  undefined (reading 'options')` on guided source-select and on reloading
  any composed session.

---

## 0.5.4 - 2026-06-20 (maintenance fixes folded into 0.6.0)

Maintenance line opened on top of RC-5.3, carrying correctness fixes that
post-date the 0.5.3 cut. These fixes ship as part of the 0.6.0 release
rather than as a standalone 0.5.4 tag.

### Fixed

- **`tier_1_decoration` lint no longer crashes on vendored trees** — the
  repo-wide TDE2 pass and its candidate scan now walk the tree through the
  shared `iter_python_files` excluded-dir filter instead of a raw
  `rglob("*.py")`, so an unparseable third-party file under `.venv` / cache
  directories can no longer hard-crash the gate.

---

## 0.5.3 - 2026-06-08 (RC-5.3 — Correctness, Audit Integrity, and Release Gating)

RC-5.3 is a correctness and hardening release on top of the RC-5.2 composer
train. It carries one new composer capability (operator-set sampling), tightens
the release pipeline so images cannot publish ahead of their required checks,
and lands a broad body of audit-integrity, trust-tier, output-contract, and
frontend recovery fixes. The theme is evidence integrity under failure: audit
rows are no longer written before the work they describe is real, output
contracts reject sparse or malformed rows at the sink boundary rather than
downstream, trust-tier error semantics raise typed faults instead of crashing
or fabricating, and the web composer no longer renders stale responses after
navigation. No schema migrations are introduced.

A second body of work landed on the `release/0.5.3` branch after the initial
RC-5.3 cut and is folded into this release. Two structural changes anchor it —
the **orchestrator god-class decomposition** (the run/resume engine split into
focused collaborators) and a new **tutorial-reliability e2e battery** — alongside
the **composer advisor-authority redesign** (the advisor is now a mandatory,
model-distinct reviewer with deterministic early/end checkpoints, replacing the
old reactive self-trigger) and two systematic bug-fix burn-downs across the CI/CD
enforcement scanners (`elspeth_lints`) and the engine/core/plugins trust-tier and
audit-integrity paths.

### Added

- **Operator-set sampling configuration (ADR-027)** — the web composer now
  sources its LLM sampling parameters from operator-set configuration rather
  than per-call defaults, threaded through the guided chain/chat solvers, the
  boot probe, and auto-title. The `ComposerLLMCall` audit contract records the
  resolved sampling so an auditor can reconstruct exactly how each model call
  was parameterised. See
  `docs/architecture/adr/027-composer-operator-set-sampling.md`.
- **Release required-checks gate** — `scripts/cicd/check_release_required_checks.py`
  gates container-image publication in `build-push.yaml` on the release's
  required checks having passed, so an image can no longer be pushed ahead of
  its CI evidence.
- **Large-scale checkpoint policy** — documented checkpoint behaviour for
  large-scale runs (`docs/reference/configuration.md`, the
  `examples/large_scale_test` settings) with a policy test pinning it.
- **OpenRouter catalog source check** — Landscape database-compatibility guards
  now require an OpenRouter catalog source check before relying on catalog
  provenance.
- **Composer advisor authority** — the web orchestrator now requires a
  model-distinct advisor and runs deterministic checkpoints rather than relying
  on the model to ask for review. A non-blocking **early advisory checkpoint**
  fires once per session, and an **end gate acts as final authority** with a
  fail-closed re-review loop on its own separate budget. Both reuse the existing
  audited LLM call path so the review itself is captured in the trail.
- **Prescriptive government URLs in the hello-world tutorial**, so the
  first-run experience exercises the canonical "rate these gov pages" flow
  against known-good sources.
- **Tutorial-reliability e2e battery** — a new frontend Playwright harness
  (`tests/e2e/`) drives the hello-world tutorial against a live staging backend
  across N runs, scoring four dimensions (a/b/c/d) with per-run record types, a
  batch aggregator, and a version-stamped trend log, so first-run reliability
  regressions are caught before release.

### Changed

- **Trust-tier branch-join correctness** — the `tier_model` rule and its
  `trust_boundary` suppression now resolve branch joins correctly; the
  `core` / `plugins` / `web` enforce-tier allowlists were reconciled to match.
- **Fail closed on conflicting original headers** — `display_headers` now fails
  closed when original headers conflict instead of silently picking one.
- **Stale allowlists fail the gate** — non-tier stale allowlist entries now fail
  CI rather than passing silently.
- **Repo hygiene** — `AGENTS.md` and `CLAUDE.md` are no longer tracked; the
  session hook regenerates them on disk, so tracking them only produced
  perpetual churn.
- **Advisor is now mandatory** — the reactive advisor self-trigger is retired
  (proactive-security trigger retained) and the advisor-disabled code paths are
  removed, so a composer session can no longer run un-reviewed. Composer-skill
  critical rules were hoisted into high-attention zones, and the skill now
  documents backend-run advisor checkpoints instead of the reactive framing.
- **ReorderBuffer clamps negative `buffer_wait_ms`** on clock skew rather than
  scheduling a negative wait.
- **Orchestrator decomposition** — the run/resume engine (`core.py`, ~2,950 LOC)
  was split into focused collaborators (`RunCeremony`, `SourceIterationDriver`,
  `CheckpointCoordinator`, `RunExecutionCore`, `ResumeCoordinator`), reducing the
  god-class to ~1,030 LOC with no behavioural change to the run path. The
  decomposition surfaced several pre-existing audit-path defects that the
  burn-downs below then fixed.
- **Tutorial path is no longer special-cased on the backend** — the
  tutorial-only state normalization that repaired the composed pipeline only on
  the tutorial path was removed, so a tutorial run now exercises the same backend
  code as any other run (composer correctness is fixed at the source rather than
  papered over with a tutorial-only shim).

### Fixed

- **Audit integrity and determinism** — artifact streaming now verifies bytes
  against the audit hash before serving, composer MCP delete preserves audit
  history, `resolved_prompt_template_hash` is preserved across read/export and
  validated before insert, SSRF-safe success and empty-corpus Chroma retrievals
  are each recorded once, and the replayer advances only on a concrete result.
  Per-item detail in git history.
- **Output contracts and sink/source boundaries** — sparse-field output
  contracts (Azure Blob, Dataverse, `JSONSource`, narrative-summary), complete
  custom-CSV-header enforcement, database/Chroma/JSON sink write-safety
  preflights, and Dataverse OData-unsafe-lookup rejection were corrected at the
  boundary.
- **Trust-tier and web error semantics** — web-session Tier-1 paths raise typed,
  upstream-interpretable errors, composer routes surface provider detail, and
  redaction masks the file-blob storage-path carrier (storage-path leak closed).
- **Web composer recovery and stale-response guards** — stale responses are
  guarded across session selection, navigation, blob loads, execution-start,
  YAML refetch, and run-outputs state; stores clear on logout; loop closure
  during progress scheduling is treated as shutdown.
- **Sessions, Landscape, and CI/CD gates** — session index/constraint validation
  checks column sets, Landscape read-only mode keeps live WAL audit DBs visible,
  and the adapter-budget / cicd-judge gates and RC5.3 allowlist drift were
  repaired.
- **Composer reliability and execution orchestrator** — a fail-closed pre-run
  validation gate, a preflight-repair gate, advisor-blocked orphan-placeholder
  surfacing, advisor end-gate reach into prompt-template pipelines, and
  boot-probe / authoring-guard robustness closed the composer-reliability fixes.
- **Trust-tier and audit-integrity burn-down (engine / core / plugins)** — typed
  `AuditIntegrityError` for malformed checkpoints, teardown that no longer masks
  the primary failure, shape-checked retrieval boundaries, telemetry-accounting
  integrity, replay-safe Chroma audit hashing, config-time validation hardening,
  and sparse-row `field_mapping` correctness. Per-item detail in git history.
- **CI/CD enforcement-scanner burn-down (`elspeth_lints`)** — the allowlist-
  expiry, attribution (`gve_attribution`), frozen-annotation, freeze-guard,
  component-type, contract-manifest, `tier_1_decoration` / `audit_evidence_nominal`
  / `tier_model`, and composer `catch_order` rules were all made fail-closed and
  broadened, and the CI lanes themselves now fail closed. Per-rule detail in git
  history.

### Security

- **Frontend dependency advisories patched** within existing semver ranges
  (lockfile-only; no major-version or source change) — `vite` ≤6.4.1 → 6.4.3
  (HIGH: dev-server path traversal and arbitrary file read), `dompurify`
  ≤3.3.3 → 3.4.8 (four XSS sanitizer bypasses), `mermaid` 11.14 → 11.15 (Gantt
  DoS and `classDef` CSS/HTML injection), and `uuid` <11.1.1 (buffer-bounds,
  transitive). `npm audit` reports zero remaining advisories; the frontend
  suite and build remain green.
- **SSRF Host header brackets literal IPv6 authorities** instead of emitting an
  ambiguous, splittable host.
- **Secret-echo closed** — the commencement gate no longer echoes an environment
  secret in its failure reason, and `record_validation_error` no longer leaks row
  content via `str(e)`.
- **DSN audit sanitization no longer double-encodes `odbc_connect`**, and
  rate-limit bucket-name sanitization is now injective (distinct buckets cannot
  collide).
- **Env-var-name regex anchored with `\A…\Z`** rather than `^…$`, closing a
  multiline-injection gap.

### Dependencies

- **Runtime image → Python 3.13** — the Docker base moves from
  `python:3.12-slim` to `python:3.13-slim` (digest-pinned), aligning the
  shipped runtime with the release interpreter that dev and CI exercise.
  `requires-python` remains `>=3.12`. (Dependabot proposed 3.14; declined in
  favour of the tested 3.13 baseline.)
- **GitHub Actions bumped to current majors** (SHA-pinned) — `actions/checkout`
  4→6, `actions/upload-artifact` 4→7, `docker/build-push-action` 5→7,
  `actions/setup-python` 5→6, `actions/setup-node` 4→6, and the `docker/*`
  setup/login/metadata actions.
- The breaking frontend majors dependabot grouped with the security fixes
  (React 18→19, Vite 6→8, TypeScript 5.7→6.0) are **deferred** to a dedicated
  migration — blocked upstream by `openapi-typescript` still requiring
  TypeScript 5.x.

> **Baseline drift — resolved before release.**
> `tests/unit/elspeth_lints/fixtures/fingerprint_baseline.json` drifted as
> RC-5.3 commits landed. The baseline was regenerated after the signed enforce
> gate verified green and the signed `tier_model` allowlists were reconciled;
> `test_baseline_capture_is_self_consistent` passes at the release HEAD.

---

## 0.5.2 - 2026-05-19 (RC-5.2 — Guided Composer, Durable Progress, and Recovery UX)

RC-5.2 is the large Web Composer release train that folds the guided-mode
wizard, composer progress persistence, manifest-keyed redaction, per-step chat,
chat-as-data-entry, interpretation review, shareable completion gestures,
frontend recovery UX, CI/CD gate consolidation, release-documentation refreshes,
and RC5.2 hardening back onto `main`. It moves the composer from a best-effort
interactive surface toward an audited, recoverable authoring system: model
calls, tool dispatch, redacted tool payloads, persisted transcript rows,
interpretation decisions, recovery diffs, and operator-visible failure causes
now share one evidence story.

### Added

#### Composer Guided Mode

- **Composer guided mode** — new structured-protocol wizard for first-time
  pipeline authors. Source → sink → transforms in three steps; closed
  six-turn taxonomy; deterministic recipe pre-match; LLM-read-only with
  respect to pipeline state. Ships alongside the unmodified freeform
  composer; mode transition uses progressive disclosure. Implementation design
  detail is preserved in git history rather than active docs.
- **ComposerLLMCall audit channel** — every `solve_chain` invocation in
  guided mode now records a `ComposerLLMCall` audit row (provider, model,
  status, latency, prompt/completion tokens). Pairs with the existing
  `ComposerToolInvocation` audit channel so an auditor can reconstruct
  both what the model was asked and what tools it then dispatched.

#### Composer Progress Persistence (durable, recoverable authoring)

- **Schema (Phase 1A)** — new audit columns on `chat_messages` (tool-call
  linkage, `sequence_no`, `writer_principal`) with biconditional CHECK
  constraints, a `composition_states.provenance` enum, and new `run_events` and
  (INERT, read-side) `audit_access_log` tables with per-session indices. Applied
  by recreation, not Alembic — see *Operational* below.
- **Single-transaction turn primitive** — `persist_compose_turn` /
  `persist_compose_turn_async` writes the assistant message, redacted tool rows,
  and composition state atomically, with advisory-lock serialization and
  contiguous per-session `sequence_no` reservation under concurrent writers. A
  testcontainer-backed Postgres lane exercises the cross-DB semantics SQLite
  cannot model. Implementation design detail is preserved in git history rather
  than active docs.
- **Redaction walker + MANIFEST** — every composer tool now has an explicit
  redaction policy (type-driven Pydantic argument models plus declarative
  entries) so LLM-supplied argument and response payloads are made Tier-1-safe;
  Pydantic validation errors route through the `ToolArgumentError` ARG_ERROR
  channel without leaking per-field detail, and a manifest-parity adequacy guard
  pins a byte-identical redaction snapshot. Per-entry detail in git history.
- **Atomic compose-loop persistence** — `_compose_loop` persists assistant
  messages, tool-call breadcrumbs, redacted payloads, and state snapshots through
  `persist_compose_turn` even when tools fail, cancellation lands mid-turn, or a
  plugin crash triggers recovery; a bounded per-turn tool-call cap emits
  `tool_call_cap_exceeded`, and audit-grade transcript reads opt into
  `include_tool_rows` through `audit_access_log`.
- **Frontend recovery panel** — the frontend detects recoverable composer
  failures and opens a dedicated recovery surface rendering the assistant
  transcript, redacted tool rows, and a before/after state diff, plus an ESLint
  `npm run lint` gate for the recovery / guided-mode UI.

#### RC5.2 Hotfix Integration

- **Auth, audit, and execution hardening** — local/Entra auth flows now audit
  token issuance, failure classes, login outcomes, and web-run attribution into
  Landscape (redacting JWKS failure detail, suppressing token-response caching),
  and web execution classifies validation errors, sanitizes broad execution
  errors, persists resolved run config, and rejects misplaced secret refs.
- **Engine/plugin and frontend fixes** — checkpoint resume/coalesce parsing,
  pending batch row identity, JSON sink parent creation, sink preflight timing,
  Web Scrape fail-closed boundaries, and LLM provider preflight were tightened,
  alongside guided/catalog/run accessibility and theming fixes (contrast,
  forced-colors, cross-tab theme sync, screen-reader-safe status). Per-item
  detail in git history.

#### Composer UX Redesign, Preferences, and Review Flow

- **Composer preferences** — a new preferences schema, service, and routes let
  users choose their composer starting mode, with backend validation and
  write-failure alerting.
- **Chat as data entry** — short chat inputs can project into audited inline blob
  sources with source provenance, hash evidence, MIME parsing, and ambiguity
  handling.
- **LLM interpretation review** — ambiguous prompt-template decisions can pause
  for operator review, persist append-only interpretation events, gate execution
  while unresolved, and surface guided/freeform review widgets in chat.
- **Completion gestures and shareable reviews** — composer sessions gained
  completion events, HMAC-signed shareable review links, YAML-export audit
  events, narrative result views, and shared-inspection views.
- **First-run tutorial and mode guidance** — a hello-world tutorial introduces
  the composer with persistent tutorial state, cache-skip telemetry, and explicit
  freeform/guided switching guidance.
- **Catalog reshape and audit-readiness UI** — plugin catalog cards, filters,
  audit characteristics, inline chat-source entry, SideRail audit status, and
  graph/YAML modals were rebuilt around the new composer information architecture.

#### Engine, Transform, and Plugin Additions

- **Batch-aware aggregation context** — transforms can receive
  `AggregationBatchContext`, enabling the new `report_assemble` transform with
  metadata-collision checks.
- **Composer knob schema lowering** — plugin option metadata now lowers into
  one-knob composer schemas, with discriminated plugin protocol support,
  visible-when scope guards, and recipe-slot adapters.
- **Determinism declaration enforcement** — plugin infrastructure enforces
  determinism declarations mechanically via `__init_subclass__`.

#### CI, Lints, and Release Documentation

- **`elspeth-lints` static analyzer (ADR-023)** — custom CI analyzers moved into
  the `elspeth-lints` package with rule fixtures, parity harnesses, SARIF output,
  and rule-author docs; CI now consolidates static analysis, gates RC branches,
  runs CodeQL and dependency/license checks, makes Playwright E2E a required
  signal, and enforces cohort-attribution trailers on telemetry-backfill commits.
- **Release docs cleanout** — RC-1/RC-2 changelog fragments, superseded release
  snapshots, frozen architecture packs, and completed handover corpora were
  removed from active docs, while current release/assurance/audit documents stay
  linked from `docs/README.md`; generated PDF output now defaults under
  `tools/pdf/out/`.

### Changed

- **`SessionServiceImpl` write contracts tightened** — `add_message()` now
  requires `writer_principal=`, `save_composition_state()` requires
  `provenance=`, and `__init__()` requires `telemetry=` and `log=` (constructed
  via `build_sessions_telemetry()`); each principal/provenance value is enforced
  by a CHECK constraint.
- **Exit-from-COMPLETED terminal returns 200** — guided sessions in
  `kind=completed` accept `control_signal=exit_to_freeform` via POST
  `/api/sessions/{id}/guided/respond` and transition to `kind=exited_to_freeform`
  (previously returned 409).
- **Guided-mode prompt loading** — the guided composer skill pack is split into
  base plus step-specific prompt files, allowing step-scoped context without
  flattening all guidance into one prompt; guided mode also gains a separate
  per-step advisory chat channel with persisted history and `ComposerChatTurn`
  audit rows.
- **Composer dependency packaging** — `chromadb`, `html2text`, and
  `beautifulsoup4` are now mandatory dependencies rather than optional `rag` /
  `web` extras, matching the composer and web-scrape surfaces that import them
  during normal operation.
- **Session schema durability** — session SQLite engines now set WAL,
  `busy_timeout`, synchronous PRAGMA, schema epoch guards, orphan PENDING-row
  recovery, and cross-DB hash spot-checks for interpretation runtime handoff.
- **Composer chain-solver and frontend validation refinements** — the guided
  chain solver constrains tool-response shape and surfaces malformed responses
  via the auto-drop channel; MCP finite-status fields use narrower literals with
  CI-checked cross-language drift; frontend validation became cache-aware; and
  the hash router moved off retired Spec/Runs tabs to graph/YAML/catalog modal
  actions. Per-item detail in git history.

### Removed

- **Composer replacement-shape machinery** —
  `_runtime_preflight_failure_message`, `_enforce_replacement_non_prefix_invariant`,
  `_ReplacementBranch`, and `_INTERCEPTED_ASSISTANT_HISTORY_PREFIX` are deleted
  (with the 7 tests pinning them); the compose loop's augmentation shape is now
  the sole codepath.
- **Retired composer IA surfaces** — the old Spec tab, Runs tab, SessionSidebar,
  and inspector panel/page were removed after their capabilities moved into the
  SideRail, modals, session switcher, run history, output panels, and validation
  banner.
- **Optional dependency extras and active-doc clutter** — the `web` and `rag`
  extras were retired once their packages became mandatory, and superseded prompt
  files, generated review sidecars, completed handovers, and archived checklists
  were removed from active docs.

### Operational

- **Sessions DB schema deployment requires recreation** — Phase 1A's
  new columns, tables, CHECKs, and partial unique indices are not
  applied via Alembic. Per `project_db_migration_policy`: stop the
  service, archive the old `sessions.db`, restart. The bootstrap
  creates the new schema on first start. Procedure documented in
  `docs/runbooks/staging-session-db-recreation.md`.
- **Phase 5 interpretation deployment requires two DBs to move together** —
  `interpretation_events_table`, `interpretation_review_disabled`, and the
  Landscape `calls_table.resolved_prompt_template_hash` handoff require the
  session DB and Landscape audit DB to be reset together on schema cutover.
  The staging recreation runbook documents the coupled reset.
- **Frontend-only deploys require static rebuild, not service restart** —
  `npm run build` refreshes `src/elspeth/web/frontend/dist/`; backend Python,
  dependency, environment, systemd, or Caddy changes still require
  `elspeth-web.service` restart.
- **Current PR check posture** — PR #39 now includes more than 770 committed
  branch commits over `main`, including the docs-cleanout merge. The PR body and
  release docs should be treated as the review map for the whole RC5.2 train,
  not for a single feature slice.

---

## [0.5.1] - 2026-05-11 (RC-5.1 — Composer Correctness, Validator Hardening, and Audit-Integrity Coverage)

RC-5.1 is a correctness and assurance follow-up to RC-5. The Web Composer's
authoring loop, validation surface, and run-evidence views all received targeted
hardening; a new `identity_node_advisory` validator detects identity passthroughs
that obscure observed-sink lineage; the `data_dir` config is now resolved to an
absolute path at validation time; pipeline recipes, source inspection, and a
forced-repair loop with proof diagnostics extend the composer authoring surface;
and the Landscape audit-integrity test surface gains direct coverage for four
ADR-019-family invariants that previously had zero unit tests. Frontend UX received
a Tier-1 panel-review pass covering accessibility, focus management, and operator
visibility.

### Added

#### RC5-UX Substrate, Composer, and Execution Evidence

- **Substrate framing** — the README now presents ELSPETH as a high-assurance
  pipeline substrate with two authoring surfaces (hand-edited YAML and the Web
  Composer) that target the same primitives, graph-validation contracts,
  executor, and Landscape audit trail rather than a bolt-on workflow builder.
- **Expanded Web Composer surface** — authenticated sessions, versioned
  composition state, blob management, secret references, chat-first authoring,
  graph/spec/YAML inspection, validation, execution, cancellation, and output
  artifact review are now part of the documented authoring surface.
- **Audited, runtime-shaped composer loop** — composer plugin discovery, state
  mutation, validation, YAML export, blob/secret tools, and advisor hints all
  happen through explicit tool contracts; previews, `/validate`, and `/execute`
  use runtime assembly and graph-validation contracts rather than a separate
  best-effort UI validator.
- **Run evidence and cancellation visibility** — web execution exposes
  Landscape-derived run accounting, diagnostics snapshots, discard summaries, and
  the full output artifact manifest/content surfaces, and in-progress runs carry
  a distinct cancellation-requested state while work drains toward `cancelled`.

#### Composer Reliability and Operator Visibility

- **Audited composer LLM calls** — composer requests use deterministic sampling
  where supported and record temperature/seed, provider cache counters,
  Anthropic-style prompt-cache markers, and provider-reported reasoning token
  counts in the LLM audit sidecar without fabricating missing values; normal chat
  history still hides those internals.
- **Advisor escalation contract** — the optional advisor tool is gated behind
  explicit trigger categories so the composer can ask for frontier-model help
  only under mechanically validated conditions.
- **Advisor-conditional skill markers** — composer prompt assembly strips
  `<!-- ADVISOR-ONLY -->` and `<!-- ADVISOR-DISABLED -->` regions depending on
  whether the advisor tool is enabled, so an advisor-disabled deployment can no
  longer leak `request_advisor_hint` guidance to the composer LLM.
- **Hard-mode evaluation harness** — reusable shell tooling and scenario fan-out
  capture validation transport failures, composer regressions, and per-row output
  evidence for demo-readiness checks.

#### Plugin and Contract Surface

- **Plugin-declared semantic contracts** — plugins can now publish semantic
  facts, requirements, comparison outcomes, and composer assistance text; the
  composer, MCP server, HTTP validation, and `/execute` surfaces expose the
  same contract shape.
- **Statistical batch plugin family** — added runnable local examples for
  `batch_distribution_profile`, `batch_experiment_compare`,
  `batch_classifier_metrics`, `batch_paired_preference`,
  `batch_drift_compare`, `batch_outlier_annotator`,
  `batch_data_quality_report`, `batch_top_k`, `batch_threshold_summary`, and
  `batch_effect_size`.
- **Value-source and schema-contract guidance** — plugin catalogs and composer
  guidance now carry richer missing-dependency, schema vocabulary, and repair
  hints so an invalid pipeline points at the contract it violated.

#### Audit and Accounting Model

- **Two-axis terminal outcome model** — RC5-UX separates lifecycle outcome from
  terminal path/provenance so success, failure, routed, discarded, fallback, and
  structural bookkeeping cases no longer overload a single row outcome.
- **Run accounting split** — routed success/failure, token lifecycle counts,
  source-row counts, discard summaries, and closure integrity are reported as
  distinct units instead of being inferred from one `rows_routed` number.
- **Per-tool-call composer audit** — both the web composer and the standalone
  composer MCP server persist the tool-call decision trail that produced a
  pipeline.

#### CI, Policy, and Test Surface

- **Policy gates expanded** — branch work added or tightened gates for
  component types, guard symmetry, audit evidence nominal typing, tier-1
  decoration, contract manifests, composer exception channels, composer catch
  order, and tier-model allowlists.
- **Frontend Playwright baseline** — a real browser E2E harness now boots the
  FastAPI backend and Vite frontend, with an initial smoke proof and fixme
  specs for composer-correctness acceptance paths.

#### Validator and Pipeline Authoring (RC-5.1)

- **`identity_node_advisory` validator check** — `validate_pipeline` now detects
  identity passthrough nodes wired between transforms and observed sinks, where
  the passthrough silently degrades observed-sink lineage. The check is gated by
  an exemption matrix locked in by tests and surfaces as an actionable composer
  repair hint.
- **Composer pipeline recipes** — `apply_pipeline_recipe` MCP tool plus two
  initial templates compose a multi-node pipeline from a single named recipe,
  with deep-frozen `RecipeSpec.slots` and slot defaults validated against
  `slot_type` at construction.
- **Source inspection MCP tool** — new `inspect_source` MCP tool surfaces
  external-data shape and silent-failure modes as warnings, supporting
  "look before you wire" composer sessions; hostile-input coverage included.
- **Forced-repair loop with proof diagnostics** — `preview_pipeline` runs a
  proof step emitting a `proof_diagnostics` array; the composer's forced-repair
  loop fires on the resumed-session first turn and plumbs `repair_turns_used`
  into `composition_states.composer_meta`. `compute_proof_diagnostics` verifies
  blob `content_hash` so a stale blob cannot pass the proof gate.
- **`<OPERATOR_REQUIRED>` sentinels for identity-bearing fields** — the composer
  skill replaces literal example values for `web_scrape.http.abuse_contact` and
  `scraping_reason` with angle-bracket sentinels and an explicit resolution order
  (operator-supplied → deployment-identity → ask). The angle-bracket form lets a
  placeholder validator mechanically reject any YAML that still carries a
  sentinel.
- **Hard rule against silent operator-input rewrites** — the composer skill now
  forbids silent normalisations of operator-supplied strings (e.g. prepending
  `https://`, lowercasing, trailing-slash strip); any rewrite must be confirmed
  by the operator or routed through a recorded normalisation step so it appears
  in the YAML and build summary.
- **Implicit-decision disclosure ("Decisions I made on your behalf")** — a new
  Build Summary Discipline subsection requires the composer to enumerate
  operator-invisible authoring decisions (identity headers, model/provider/
  temperature, output shape and routing, format choices, allowlist defaults,
  surviving operator-input rewrites) with explicit provenance markers
  (`default` / `picked` / `deployment-identity` / `operator-supplied`).

#### Run Evidence and Operator Visibility (RC-5.1)

- **Run outputs panel** — frontend `RunOutputsPanel` exposes the full
  audit-evidence manifest for a run, with downloadable artifacts gated by a
  per-artifact `downloadable` flag and backed by a new `/artifacts/preview`
  execution endpoint.
- **`data_dir` resolved to absolute path** — `WebSettings` resolves `data_dir`
  to an absolute path at validation time, eliminating ambiguity where relative
  paths were interpreted against different working directories at validate vs.
  run time.
- **Failure-sample aggregation in run-level errors** — a new
  `web/execution/failure_samples` module aggregates the top distinct
  `transform_errors` rows so a failed run's top-level error message carries
  actionable detail rather than a single bubble-up exception string.
- **Operator-visibility polish** — cancellation-requested runs carry a distinct
  badge separate from terminal `cancelled`, GraphView preserves pan/zoom across
  topology changes, and Tier-3 source inspection surfaces silent-failure modes
  (e.g. all-rows quarantined) as warnings.

#### Audit Integrity Test Coverage (RC-5.1)

- **Direct audit-integrity invariant coverage** — unit tests added for four
  ADR-019-family invariants that previously had none (deferred-invariant sweep,
  `_validate_token_row_ownership` lineage guard, `link_validation_error_to_row`
  quarantine branches, and all 12 `_REQUIRED_COMPOSITE_FOREIGN_KEYS` entries),
  plus residual SSRF blocklist boundary cases (the `::ffff:0:0/96` IPv4-mapped
  range and seven others) in `web_scrape`.

#### Contracts and Lifecycle (RC-5.1)

- **Lifecycle symmetry and contract-test derivation** — runtime-config protocols
  expose paired `_on_start_called` / `_on_complete_called` flags so asymmetric
  implementations become structurally detectable, and contract tests now derive
  the runtime-config inventory from introspection rather than a hand-maintained
  list.

### Changed

- **README front door refreshed for RC-5** — the README now leads with the
  high-assurance substrate, the gap ELSPETH closes, the two authoring audiences,
  parallel YAML/Web Composer start paths, the capability map, and the
  audit/assurance model, and the quickstart uses the `elspeth web` default port
  `8451`.
- **Batch-specific LLM transforms retired from the public story** —
  `azure_batch_llm` and `openrouter_batch_llm` are no longer advertised; use the
  regular `llm` transform with provider pooling/multi-query for throughput and
  the statistical batch transforms for audit-attributable local aggregation.
- **Default `on_validation_failure` is now `discard`** — the default per-source
  validation-failure behaviour changed to `discard` with documented quarantine
  semantics, replacing the prior implicit fall-through.
- **Unknown-plugin composer error is now actionable** —
  `_prevalidate_plugin_options` surfaces an unknown plugin id as a structured
  rejection instead of a silent fail-open, and rejected-mutation entries now lead
  `validation.errors` as the primary diagnostic.
- **Composer audit-trail honesty** — augment-shape preflight failures retain the
  model's original prose instead of a generic summary, and `apply_pipeline_recipe`
  explicitly reports a destructive replacement in the composer audit trail.

### Fixed

- **Wire-visible identity fabrication (Tier-1 audit-integrity defect)** — the
  composer skill previously carried `compliance@example.com` and a generic
  scraping reason as the canonical worked example for the
  `web_scrape.http.abuse_contact` / `scraping_reason` fields, and the LLM was
  copying those literals into generated YAML verbatim. A reproduced eval
  session shipped a fabricated abuse-contact email as an HTTP header to three
  external `.gov.au` sites — a confident wrong answer to a third party we have
  no relationship with. Closed by the `<OPERATOR_REQUIRED>` sentinel rewrite,
  the silent-rewrite hard rule, and the implicit-decision disclosure block
  (see *Validator and Pipeline Authoring (RC-5.1)*).
- **Composer skill correctness** — a multi-commit sweep closed skill-text
  fabrication and silent-shape-downgrade loopholes, widened the grounding
  detector, scoped state-claim grounding correction, forbade identity nodes
  between transforms and observed sinks, and narrowed the `ComposerResult`
  pairing invariant.
- **Composer frontend and audit-shape fixes** — a Tier-1 accessibility panel
  review fixed `aria-controls` / `aria-expanded`, `aria-live` scoping, and
  validation-colour decoupling; the SecretsPanel form recovers cleanly on a
  failed `createSecret`; and `augment` / `replace` audit shapes now use symmetric
  producer-side invariants.

---

## [0.5.0] (RC-5 — Web UX Composer + Systematic Hardening)

Full web application platform for chat-first pipeline composition, three-provider authentication, session management with versioning, blob storage, secret management, background pipeline execution with WebSocket progress, and a React frontend themed to DTA/AGDS guidelines. Also: sink failsink pattern for per-row write failure routing, pipeline composer MCP server, DAG schema propagation (`output_schema_config` as single source of truth), declaration-trust / compiler-boundary hardening for pre-data runtime guarantees, frontend UX refresh (A1-A7), composer agent tooling (B1-B5) and skill pack update (C1-D4), guard symmetry CI scanner, `TokenRef` type, exception hygiene with `TIER_1_ERRORS`, a 200+ bug closure campaign across all subsystems, and a comprehensive test hygiene sweep removing ~500 low-value tests while adding ~200 gap-filling tests.

### Added

#### Declaration-Trust / Compiler-Boundary Hardening

- **Declaration-trust framework** — generalized ELSPETH's config-to-execution contract into a first-class declaration-trust system. Declarations trusted during graph construction and web validation — including pass-through behavior, declared input/output fields, schema mode, source guaranteed fields, sink required fields, and empty-emission governance — are now explicit, validated at compile time where possible, enforced by CI manifest/scanner guardrails, recorded in the run header via the runtime VAL manifest, and re-verified at runtime with audit-complete violation reporting. This moves the web UX closer to a true compiler front-end: a configuration that validates is one that satisfies every pre-data runtime guarantee ELSPETH can assess before Tier 3 data arrives.

#### Web UX Composer Platform

- **`elspeth web` CLI command** — FastAPI app factory with `[webui]` extra, `WebSettings` config model, and default port 8451, serving the Vite-built React SPA from `src/elspeth/web/frontend/dist/` with `/api` and `/ws` proxying.
- **DTA/AGDS theming and accessibility** — deep teal, green accent, and GOLD semantic colours matching Australian Government Design System guidelines, plus skip-to-content links, reduced-motion support, touch-target sizing, session creation/archive guards, and destructive-action confirmation.

#### Authentication Subsystem

- **`AuthProvider` protocol** — pluggable identity model (`AuthenticationError` base) with three providers: `LocalAuthProvider` (bcrypt + JWT issuance), `OIDCAuthProvider` (OpenID Connect with JWKS discovery and key caching), and `EntraAuthProvider` (Microsoft Entra ID with tenant validation and group claims).
- **Auth surface** — `get_current_user` route-level middleware, login/token-refresh/profile/config routes, and a configurable registration endpoint (`open`, `email_verified`, `closed`).
- **python-jose → PyJWT migration** — replaced the unmaintained library across all auth code.

#### Plugin Catalog

- **`CatalogService` protocol and implementation** — plugin discovery service with REST API routes wired into the app factory.

#### Session Management

- **SQLAlchemy Core schema + `SessionServiceImpl`** — session database with migrations, and CRUD/versioning/run-enforcement (`RunAlreadyActiveError`) behind `SessionServiceProtocol`, with a full REST API (pagination, state pruning, upload hardening).
- **Fork-from-message** — create new session versions branching from specific conversation messages, with a text source plugin.
- **Concurrency and lifecycle hardening** — DB-level constraints replacing application-level TOCTOU checks, all DB calls moved off the async event loop via a thread-pool executor, and orphan cleanup wired into the FastAPI lifespan with UUID path parameters.

#### Blob Storage Manager

- **Blob storage subsystem** — data model, service, and migration; REST API and app wiring; frontend, composer-tool, and execution integration; schema inference; and upload dedup, quota enforcement, and file cleanup.

#### Secret Reference System

- **Historical `$secret{name}` reference resolution** — `SecretResolution` audit extension accepting `"env"` / `"user"` sources, a recursive `resolve_secret_refs()` config tree-walk, and `ServerSecretStore` / `WebSecretService` chained resolution with allowlist enforcement, env-var boundary, and fingerprint audit. Current Composer authoring uses `{secret_ref: NAME}` markers for new nodes and `wire_secret_ref(...)` for existing components.
- **Secret-system surface and hardening** — REST API, composer tools, execution integration, frontend wiring, plus audit trail, fingerprints, leakage prevention, and input validation.

#### Pipeline Execution Layer

- **Background pipeline runs** — `ExecutionServiceImpl` with WebSocket progress streaming, dry-run validation, and late-client seeding (clients connecting after run start receive current state).
- **Cancel-vs-execute race closure** — atomic state transition preventing concurrent execution attempts.

#### Pipeline Composer (LLM Tool-Use)

- **Frozen composition data models** — `SourceSpec`, `NodeSpec`, `EdgeSpec`, `OutputSpec`, `PipelineMetadata` with deep immutability.
- **`ComposerService` LLM tool-use loop** — composition tools and YAML generator, prompt/message management, and integration into the session API, with Stage 1 validation (warnings, suggestions, status tint).
- **Composer loop hardening** — dual-counter loop guard, discovery cache, partial state recovery, rate limiting, and tool registry.

#### Pipeline Inspector

- **Inspector UX overhaul** — EdgeSpec/NodeSpec fixes, graph readability improvements, version selector, catalog drawer.

#### Pipeline Composer MCP Server

- **`elspeth-composer` MCP server** — full pipeline composition toolset via Model Context Protocol (plugin discovery, state mutation, validation, YAML generation, session persistence), with a Claude Code pipeline-composer skill pack for interactive MCP-driven building.

#### Sink Failsink Pattern

- **Per-row write-failure routing contracts** — `RowDiversion` and `SinkWriteResult`, a new `DIVERTED` terminal row state with a `rows_diverted` counter, and a mandatory `on_write_failure` config field on `SinkSettings` (`route_to`, `discard`, `fail`).
- **Failsink DAG wiring and validation** — `BaseSink._divert_row()` (with `FrameworkBugError` guard), automatic `__failsink__` DIVERT edges in the DAG builder, construction-time `validate_sink_failsink_destinations()`, and `SinkExecutor.write()` failsink dispatch on per-row write failure.

#### DAG Schema Propagation

- **`output_schema_config` as single source of truth** — populated for all node types (source, transform, gate, aggregation, coalesce) at construction time. `_assign_schema` refactored to only set `output_schema_config`, dropping the parallel dict write.

#### Frontend UX Refresh (A1-A7)

- **Frontend UX refresh (A1-A7)** — categorized blob-manager folders, markdown/Mermaid rendering in chat (DOMPurify-sanitized), validation errors routed through chat, a default 50/50 graph/chat split, a relocated secrets button, per-node validation indicators, a three-state pipeline status indicator, and design-token / orchestration-extraction cleanup.

#### Composer Agent Tooling (B1-B5)

- **Composer agent tooling (B1-B5)** — blob CRUD, structured validation, path redaction, and pipeline diff for the composer MCP server.

#### Composer Skill Pack Update

- **Composer skill pack (C1-C8, D1-D4)** — expanded skill definitions plus the deployment skill layer for composer interactions.

#### Web Group E

- **Unified file storage** — blob refresh and inline source docs (file-handling consolidation).

#### Guard Symmetry Scanner

- **`enforce_guard_symmetry` CI tool** — detects write/read guard parity gaps (every Landscape write site must have a corresponding read guard), with a GitHub Actions workflow and allowlist support.

#### TokenRef Type

- **`TokenRef` frozen dataclass** — bundles `token_id + run_id` in `contracts/`, replacing loose 2-tuple passing; the Landscape API accepts `TokenRef` directly (`coalesce_tokens`, `_validate_token_run_ownership`) and read sites crash on corruption via `AuditIntegrityError` loader guards.

#### Exception Hygiene

- **`TIER_1_ERRORS` constant** — canonical tuple of exception types for Tier 1 catch sites, applied across all layers.

#### Server Configuration

- **Default port 8451** — server config design with skill restoration.

### Fixed

- **~100+ bug closure campaign (P1) across all subsystems** — Landscape/checkpoint/DAG integrity, plugin transform/source/sink contracts, engine orchestrator/processor/executor invariants, web execution races, and a large `cluster:null-check` sweep, including SSRF / auth-bypass / userinfo-leak hardening, with crash-on-invalid replacing silent-skip throughout. Per-bug detail in git history.
- **Web platform hardening** — successive code-review rounds across sessions, blobs, auth, and execution closed a blob IDOR guard, FK constraints, an Entra-issuer / SNI / regex / cancel-race set, OIDC flow and blob-quota atomicity, and startup/auth regressions.
- **Post-RC5-cut systematic bug sweep (~130 bugs)** — audit integrity, silent failure, security, race conditions, resource leaks, freeze gaps, validation-too-late, and type/contract safety across nine-plus clusters. Per-bug detail in git history.
- **Engine, contract, and tier-model hygiene** — deep-immutability frozen dataclasses, unjustified `.get()` elimination on Tier 1/2 data, narrowed broad exception handlers, non-finite float rejection, unconditional `validate_input`, and assorted infrastructure/contract fixes (canonical-JSON NumPy handling, HTTP-method serialization, TOCTOU guards, timezone-aware session columns).
- **Landscape audit-integrity guards** — `complete_node_state()` / `complete_batch()` terminal-overwrite guards, `DIVERTED` added to the Tier 1 outcome read guards, and terminal outcomes derived from the `RowOutcome` enum (closing the DIVERTED recovery gap).
- **Security** — DSN credential scrubbing (`_sanitize_dsn()` strips credentials from query parameters), Mermaid SVG DOMPurify sanitization, and composer path resolution against `data_dir` rather than CWD.

### Changed

- **README web startup docs** — explicit instructions for the `.[webui]` extra, building the frontend, `ELSPETH_WEB__SECRET_KEY`, creating a local auth user, and running the MVP locally.
- **Refactor and lint hygiene** — plugin-manager singleton extracted from `cli.py` to `manager.py`, 532 mypy/ruff errors resolved across the suite, and duplicated error-formatting / no-op-guard logic factored out (`_raise_if_invalid`, `_make_span`).
- **Stale schema-mode references** — `fields: dynamic` → `mode: observed` across docs.

### Removed

- **errorworks test suite** — tests belong in the standalone package.
- **`archive/` directory** — all content preserved in git history.

### Tests

- **Comprehensive test-hygiene sweep across all subsystems** — removed ~500 low-value tests and added ~200 behavioural gap-filling tests, plus new Azure Blob / DAG-validation / Web-Composer / Web-Auth coverage and post-cut gap closure (IDOR, timezone, Chroma metadata, mock-spec enforcement, audit error paths). Per-subsystem counts in git history.

## [0.4.1] (RC-4.1 — RAG Ingestion Pipeline)

Complete RAG ingestion story: ChromaSink for vector store population, pipeline `depends_on` for run sequencing, commencement gates for pre-flight go/no-go checks, and readiness contracts on retrieval providers. First pipeline-level orchestration primitives. Designed as a generic multi-stage pipeline pattern — RAG is the first consumer, but any plugin needing pre-populated external state can use the same mechanisms.

### Added

#### ChromaSink Plugin

- **ChromaSink** — new sink plugin writing pipeline rows into ChromaDB collections. Three `on_duplicate` modes: `overwrite` (upsert), `skip` (pre-filter existing IDs), `error` (pre-check and reject). Canonical content hash computed before write for audit integrity.
- **`FieldMappingConfig`** — explicit field mapping from row fields to ChromaDB concepts (`document_field`, `id_field`, `metadata_fields`). No convention-based defaults — operator declares exactly what goes where.
- **`DuplicateDocumentError`** — structured exception with `collection` and `duplicate_ids` (stored as immutable tuple) for `on_duplicate: error` mode.
- **ChromaDB metadata type validation** — metadata field values are validated as `str`, `int`, `float`, `bool`, or `None` at write time, before sending to ChromaDB. Invalid types (e.g. `dict`, `datetime`) crash with a `TypeError` naming the exact field, type, row index, and document ID.

#### Pipeline `depends_on` Mechanism

- **`depends_on` top-level config key** — declare pipelines that must run before the main pipeline starts. Each dependency is a fully independent pipeline run with its own `run_id`, Landscape records, and checkpoint stream.
- **`bootstrap_and_run()`** — reusable headless pipeline entry point in `cli_helpers.py` (L3). Handles secret resolution, passphrase handling, and directory creation. Injected into the dependency resolver via `PipelineRunner` protocol.
- **Circular dependency detection** — DFS cycle detector on canonicalized paths (`Path.resolve()`), with 3-level depth limit for nested dependencies.
- **Sequential execution** — dependencies run in declared order. `KeyboardInterrupt` propagates as-is (not wrapped in `DependencyFailedError`).
- **`DependencyRunResult`** — frozen dataclass with `run_id`, `settings_hash`, `duration_ms`, `indexed_at` for audit correlation.
- **Resume behaviour** — `elspeth resume` does NOT re-run dependencies. Fresh run required if dependencies need re-running.

#### Commencement Gates

- **`commencement_gates` top-level config key** — go/no-go conditions evaluated after dependencies complete, before the main pipeline starts.
- **`ExpressionParser` `allowed_names` extension** — gate expressions use the existing AST-whitelist parser with configurable namespace names (`collections`, `dependency_runs`, `env`). No `eval()`.
- **Pre-flight context** — assembled from dependency results, collection probes, and environment variables. Deep-frozen before gate evaluation (TOCTOU-safe). `env` excluded from Landscape audit snapshots to prevent secret leakage.
- **`collection_probes` explicit config** — operators declare which collections to probe. Probes assembled from explicit config, not auto-scanned from plugin configs.
- **`CommencementGateResult`** — frozen dataclass with `context_snapshot` deep-frozen via `freeze_fields()`.

#### Readiness Contract

- **`check_readiness()` on `RetrievalProvider` protocol** — returns `CollectionReadinessResult` (L0). Single-attempt, no retry. Called during `on_start()` after provider construction.
- **`ChromaSearchProvider.check_readiness()`** — collection count check with narrowed exception handling (connectivity errors only, not broad `except Exception`).
- **`AzureSearchProvider.check_readiness()`** — raw `httpx` count endpoint probe (not `AuditedHTTPClient`, which requires row-scoped `state_id`/`token_id` unavailable during `on_start()`).
- **RAG transform readiness guard** — `on_start()` checks both `reachable` and `count`. Raises `RetrievalNotReadyError` with `collection` and `reason` fields, with distinct messages for "empty" vs "unreachable".

#### Shared Infrastructure

- **`CollectionReadinessResult`** — unified frozen dataclass in L0 (`contracts/probes.py`) for all collection readiness checks. Used by probes, providers, and transforms.
- **`CollectionProbe` protocol** — L0 protocol for collection readiness probes, injectable into L2 engine without layer violations.
- **`ChromaConnectionConfig`** — shared Pydantic model for ChromaDB connection fields. Composed by `ChromaSinkConfig`, `ChromaSearchProviderConfig`, and `CollectionProbeConfig`. Collection name validated (min 3 chars, regex pattern).
- **`RetrievalNotReadyError`** — structured exception with keyword-only `collection` and `reason` fields.
- **`DependencyFailedError`** — structured exception with `dependency_name`, `run_id`, `reason`.
- **`CommencementGateFailedError`** — structured exception with deep-frozen `context_snapshot`.

#### Landscape Audit Trail

- **`preflight_results` table** — new Landscape table recording dependency runs and gate evaluations per pipeline run. `result_type` discriminator with `CheckConstraint`. Canonical JSON serialization via `deep_thaw()` + `canonical_json()`.
- **Readiness check outcomes recorded** — transform readiness results persisted alongside dependency and gate results.
- **Deferred recording pattern** — pre-flight results computed in `bootstrap_and_run()`, carried through `orchestrator.run()` as `PreflightResult`, recorded after `begin_run()`. Same pattern as `secret_resolutions`.
- **Dependency run correlation** — query run metadata links to indexing run via `run_id`, `settings_hash`, `indexed_at`. Auditor can trace: question → retrieved chunks → source documents → indexing decision → corpus state.

#### End-to-End Example

- **`examples/chroma_rag_indexed/`** — complete example: indexing pipeline (CSV → ChromaSink) + query pipeline (`depends_on` + commencement gate + RAG retrieval). Replaces the standalone `seed_collection.py` script with an audited pipeline.
- **CLI preflight wiring** — `elspeth run` now executes `depends_on` and commencement gates when configured.

### Fixed

#### Exception Hygiene Completion

- **3 overly-broad `except Exception` catches narrowed** to specific exception types in azure_batch, completing the exception narrowing sweep.
- **`batch_batch_timeout` double-prefix bug** — corrected in azure_batch per-row failure reason field. The prefix was applied twice, producing malformed reason strings.
- **Test asserting the bug** — corrected test that was asserting the buggy double-prefix behaviour rather than the correct single-prefix.
- **`TransformErrorCategory` and `TransformActionCategory` Literal gaps closed** — added missing category values that were valid at runtime but not in the type definitions.
- **HMAC-equivalent key pair filtering** — fingerprint property test now correctly filters HMAC-equivalent key pairs to avoid false failures.

#### Tier 1 Audit Integrity Hardening

- **`require_int()` utility** — Tier 1 int-field validator rejecting `bool` (Python's `isinstance(True, int)` footgun) and enforcing `min_value` bounds. Applied to 19 int fields across 13 audit dataclasses, plus `node_state_context`, `token_usage`, `batch_checkpoint`, `BufferEntry`, and `ResumePoint`.
- **TypedDict export records** — 15 typed shapes replacing `dict[str, Any]` in `LandscapeExporter._iter_records()`. `record_type` narrowed to `Literal` per record for mypy discriminated union support.
- **`CoalescePolicy` and `MergeStrategy` StrEnums** — replace bare strings in `CoalesceMetadata` and all call sites. Serialization-safe via `StrEnum`.
- **`Mapping[str, object]` write-path narrowing** — Tier 1 write paths in recorder and repositories narrowed from `dict[str, Any]` to `Mapping[str, object]` for tighter type safety.
- **`allow_nan=False`** — added to 6 `json.dumps()` calls in audit-path code, preventing NaN/Infinity from silently entering the Landscape.

### Changed

- **`ExpressionParser`** — now supports configurable `allowed_names` parameter (default `["row"]` for existing callers). Gate expressions use `["collections", "dependency_runs", "env"]`.
- **`ChromaSearchProviderConfig`** — refactored to compose `ChromaConnectionConfig` for shared validation. `to_connection_config()` method added.
- **`ElspethSettings`** — gains optional `depends_on`, `commencement_gates`, `collection_probes` fields.

---

## [0.4.0] (RC-4.0 — Plugins, Contracts, and Correctness)

Major feature release: Dataverse and RAG retrieval plugins, output schema contract enforcement, audit provenance boundary, freeze/serialize coherence, errorworks migration, and a 64-bug systematic sweep. Completes the RC-3.4 hardening sprint and delivers the first external-system plugin integrations. Also includes the agentic code threat model discussion paper (v0.1–v0.4) with MkDocs wiki and LaTeX build pipeline.

### Added

#### Dataverse Source and Sink Plugins

- **`DataverseSource`** — Microsoft Dataverse integration via OData v4 REST API. Supports structured OData queries and FetchXML with schema contracts. Pagination, SSRF validation, and rate limiting via the new `DataverseClient`.
- **`DataverseSink`** — upsert-only writes via PATCH with alternate key, idempotent for retries. Pre-processes all rows before HTTP calls.
- **`DataverseClient`** — pure protocol client handling authentication, pagination, SSRF validation, and rate limiting for the OData v4 API.
- **Shared utility extraction** — fingerprinting (`fingerprinting.py`) and strict JSON parsing (`json_utils.py`) extracted from `AuditedHTTPClient` into shared modules. 288 new tests.

#### RAG Retrieval Transform

- **`RAGRetrievalTransform`** — full retrieval-augmented generation transform with lifecycle management, process flow, and telemetry. Declared output schema config for downstream contract enforcement.
- **`RetrievalProvider` protocol** — L0 protocol with `RetrievalChunk` result type. Two implementations: `ChromaSearchProvider` (ephemeral/persistent/client modes, distance normalization) and `AzureSearchProvider` (score normalization, Tier 3 validation).
- **Query construction** — three modes: `field` (direct row field), `template` (string interpolation), `regex` (pattern extraction from row data).
- **Context formatting** — `numbered`, `separated`, and `raw` modes for assembling retrieved chunks into LLM context.
- **Shared template infrastructure** — extracted from LLM plugin for reuse by RAG query construction.
- **`PluginRetryableError`** — new base exception class. `LLMClientError` and `WebScrapeError` re-parented under it. Processor retry dispatch updated. New retrieval error categories added to `TransformErrorCategory`.
- **Example pipelines** — `examples/chroma_rag/` (standalone RAG) and `examples/chroma_rag_qa/` (RAG + LLM Q&A).

#### Output Schema Contract Enforcement

- **`_output_schema_config`** — new class attribute on `BaseTransform` with `_build_output_schema_config` helper. All field-adding transforms now declare their guaranteed output fields.
- **`FrameworkBugError` guard** — DAG builder crashes if a transform declares output fields but is missing `_output_schema_config`, preventing silent schema drift.
- **Integration tests** — full enforcement test and edge validation tests for output schema contracts.

#### Audit Provenance Boundary Enforcement

- **LLM audit metadata migration** — both batch and multi-query transforms now store audit fields in `success_reason` instead of polluting row data. Per-query provenance dicts collected and merged into `success_reason["metadata"]`.
- **`Call` return from `get_ssrf_safe()`** — `AuditedHTTPClient` now surfaces the `Call` object for audit correlation.
- **`payload_store` removed from `PluginContext`** — plugins access blob storage through `recorder.store_payload()` only, enforcing the provenance boundary.

#### Freeze/Serialize Coherence

- **Frozen container support in `contracts/hashing.py`** — canonical JSON serialization now handles `MappingProxyType`, `tuple`, and `frozenset` natively, resolving impedance mismatch between `deep_freeze` and hashing at L0.
- **Property tests** — hash equivalence (frozen == unfrozen), cross-module parity tests, frozen round-trip contract tests.
- **5 live bugs patched** — `MappingProxyType` NaN bypass, enum export, SSRF metadata, shallow thaw.
- **Thaw-refreeze elimination** — `plugin_context.record_call` no longer round-trips through thaw/refreeze. `ArtifactDescriptor` uses `deep_freeze` instead of shallow `MappingProxyType` wrap.

#### CI Enforcement

- **`enforce_freeze_guards.py`** — AST-based CI scanner detecting forbidden freeze patterns in `__post_init__`: bare `MappingProxyType` wraps (FG1) and `isinstance` type guards to skip freezing (FG2). Per-file allowlists with `max_hits`.
- **`enforce_mutable_annotations.py`** — CI linter detecting `list[]`/`dict[]`/`set[]` annotations on frozen dataclass fields. Allowlist for justified exceptions. *(Not yet merged — tracked for future implementation.)*
- **`freeze_fields()` promoted** to `contracts/freeze.py` as the canonical freeze utility. All freeze guards standardised.

#### web_scrape SSRF Allowlist

- **`allowed_hosts` configuration** — three-tier IP validation: `ALWAYS_BLOCKED_RANGES` (link-local, broadcast, multicast) → user allowlist (CIDR) → standard blocked ranges. Accepts `"public_only"` (default), `"allow_private"`, or explicit CIDR list. Threaded through redirect chain. 62 new tests.
- **Dual URL audit output** — both hostname and resolved IP URLs surfaced for audit comparison.

### Changed

- **`PipelineConfig` annotations** — `list`/`dict` fields changed to `Sequence`/`Mapping` to match frozen runtime types.
- **Field normalization mandatory** — removed `normalize_fields` toggle from `CSVSource`, `AzureBlobSource`, and `DataverseSource`. Header normalization always applied at the source boundary. Dunder name regression test added.
- **errorworks migration** — ChaosLLM, ChaosWeb, and ChaosEngine moved to external `errorworks` PyPI package (≥0.1.1). In-tree `chaosengine/`, `chaosllm/`, `chaosweb/`, `chaosllm_mcp/` directories deleted. Stale CLI subcommands removed.
- **Fabricated audit records removed** — `CallType.HTTP` records from `batch_replicate` validation and fabricated `variables_hash` sentinel from batch audit metadata.
- **Logger hygiene** — redundant logs removed, LLM `success_reason` audit metadata enriched.

### Fixed

#### RC4-Bugsweep (64 bugs across 13 clusters)

- **Broad-except swallowing framework errors** (6 bugs) — narrowed to specific exception types.
- **Exception type hygiene** (5 bugs) — replaced generic exceptions with domain-specific error types.
- **Dead code in exception handling** (4 bugs) — removed unreachable exception paths.
- **Missing audit on exception paths** (3 bugs) — added audit recording to previously silent failure paths.
- **Dataverse subsystem** (4 bugs) — sink pre-processing, source boundary validation.
- **Dataverse source Tier 3 boundary** (4 bugs) — trust boundary validation at OData response boundary.
- **`__post_init__` type guards** (4 bugs) — construction-time validation on checkpoint/engine dataclasses.
- **Freeze/immutability** (6 bugs) — `deep_freeze` Mapping support, shallow wrap elimination.
- **Tier 3 trust boundary validation** (5 bugs) — external system response validation.
- **LLM parallel execution audit integrity** (3 bugs) — concurrent audit recording correctness.
- **CI gate, retrieval, aggregation, verifier, CSV** (17 bugs) — cross-cutting correctness fixes.
- **IntegrityError propagation, `node_id` misattribution, buffer state corruption** (3 bugs).
- **Coalesce `rows_coalesced` double-increment** (1 bug) in timeout/flush path.
- **RAG subsystem** (3 bugs) — crash detection, resource cleanup, truncation budget.
- **ChromaDB distance type guard** (2 bugs) — crash on corrupt index instead of silent skip. Improved crash messages with collection name, doc ID, and remediation.

#### Additional Fixes

- **Plugin exception catch hygiene** (5 bugs) — pooling, search, sink, processor.
- **Exception type and chain hygiene** (5 bugs) — across contracts/engine/plugins.
- **Tier 1 checkpoint deserialization** — crash on corruption, don't coerce.
- **`AuditIntegrityError` misattribution** — prevented by outer exception handlers via dedicated error guard.
- **Unwrapped `record_call` SUCCESS paths** — all wrapped in `AuditIntegrityError`.
- **CLAUDE.md compliance** — defensive patterns, immutability, structlog, test types.

#### Mutation Testing

- **Checkpoint restore and WHERE clause exactness** — 71 new tests killing mutation survivors across 3 landscape repositories.
- **`canonical.py`** — 13 tests for None/NaT passthrough and numpy sanitization survivors.
- **`lineage.py`** — 3 tests for sink filter equality and terminal filtering survivors.

### Design Documentation

- Historical design specs and implementation plans from this era are no longer
  active docs. They are preserved in git history or the dated docs archives;
  start from the current ADRs, contracts, and archive manifests rather than old
  `docs/superpowers/` paths.

---

## [0.3.4] (RC-3.4 — Systematic Hardening)

Systematic hardening sprint driven by 191-bug triage, mutation testing, and code quality sweep. Focus: audit integrity, deep immutability, construction-time validation, exception hygiene, and elimination of defensive anti-patterns. No new features — pure correctness and reliability work.

### Fixed

#### Audit Integrity & Tier 1 Hardening

- **PayloadNotFoundError domain exception** — `PayloadStore` protocol, `FilesystemPayloadStore`, and `MockPayloadStore` now raise `PayloadNotFoundError` instead of generic `KeyError`, preventing accidental catch by `except KeyError:` dict-lookup handlers. All five caller sites updated. PURGED paths now emit debug logs with `content_hash` for operational visibility.
- **PayloadIntegrityError → AuditIntegrityError** — `get_call_response_data` now catches `PayloadIntegrityError` and translates to `AuditIntegrityError` with run/call context, instead of letting raw integrity errors escape the landscape layer.
- **AuditIntegrityError for Tier 1 corruption** — Lineage queries, edge lookups, and purge grade updates now raise `AuditIntegrityError` instead of generic `ValueError` when encountering corrupt audit data.
- **Silent default=str fallback removed** — Journal serialization no longer silently coerces unserializable types via `default=str`. Non-serializable data now crashes immediately, exposing the upstream bug.
- **BatchCheckpointState tuple restoration** — `from_dict()` now restores tuple types after JSON round-trip instead of leaving them as lists, preserving Tier 1 checkpoint invariants.
- **Null-content LLM responses recorded** — Null-content responses are now recorded in the audit trail before raising, closing an audit gap where failed LLM calls left no trace.
- **Exception type hygiene** — `ValueError` replaced with `AuditIntegrityError` or `OrchestrationInvariantError` at 12 sites where the generic type misrepresented the failure category.
- **Tier 1 invariants in graph.py** — DAG graph now crashes on invalid source count, missing route labels, and defensive `.get()` patterns that masked corruption.
- **Programming-error guards in exporters** — Exporters and journal now raise `AuditIntegrityError` or `FrameworkBugError` instead of silently continuing on corrupt state.
- **Dead `ExecutionError.from_dict()` deleted** — Removed dead deserialization method; `TokenUsage.from_dict()` now rejects bool values that would silently coerce to int.

#### Deep Immutability & Frozen Dataclass Hardening

- **Central freeze/thaw utilities** — New `deep_freeze()` and `deep_thaw()` functions standardize immutability across all frozen dataclasses, replacing ad-hoc `deepcopy` calls.
- **deep_freeze recursion** — Now recurses into tuples, frozensets, and `MappingProxyType` contents, closing gaps where nested mutable containers survived freezing.
- **Mutable dict fields frozen** — All frozen checkpoint dataclasses now freeze mutable dict fields at construction, preventing post-construction mutation of Tier 1 data.
- **Category A mutable-frozen bugs** — Enforced deep immutability on 5 frozen dataclasses where mutable fields were exposed.
- **Category B mutable-frozen bugs** — Froze 5 additional DTOs with mutable internal state.
- **`slots=True` on all frozen dataclasses** — Added `slots=True` to `ResumeCheck`, `ResumePoint`, `RowDataResult`, `_GateEntry`, and all remaining frozen dataclasses that lacked it.
- **Contracts layer hardened** — Frozen sets, `deep_freeze` over `deepcopy`, `AuditIntegrityError` for checkpoint corruption.
- **HTTP DTO headers copied before freezing** — Prevents shared mutable header dicts from being modified after DTO construction.
- **Frozen constants and LineageResult** — Immutability hardened across engine constants and lineage query results.
- **Frozen/shared data structure enforcement** — Cleared `cluster:mutable-frozen` bug cluster.

#### Construction-Time Validation (`__post_init__`)

- **12 frozen dataclass types validated** — Added `__post_init__` validation enforcing invariants at construction time across contracts and engine types.
- **Remaining `cluster:missing-post-init` types** — Completed validation coverage for all frozen dataclasses that lacked construction-time checks.
- **NaN bypass and generator truthiness** — Fixed `__post_init__` validators that failed to detect NaN values and generators that evaluated truthy regardless of content.
- **Coalesce checkpoint DTO validation** — `CoalesceTokenCheckpoint` and `CoalescePendingCheckpoint` now enforce non-empty identifiers, non-negative timing, dict types, and disjoint branch keys.
- **Config-time validation** — Added validation for free-string fields, encoding, delimiters, and cross-field invariants at settings load time, clearing `cluster:config-validation`.

#### Exception Handling Hygiene

- **Exception chains preserved** — Replaced `from None` with `from exc` across 16 files, preserving diagnostic context in exception chains.
- **5 broken exception chains repaired** — Fixed `raise X from None` patterns in engine, plugins, and CLI that destroyed root-cause information.
- **22 broad `except Exception` catches narrowed** — Replaced overly broad catches with specific exception types, clearing `cluster:broad-except`.
- **Missing programming-error re-raises** — Added `FrameworkBugError`/`AuditIntegrityError` re-raise guards to telemetry `except` blocks that swallowed system errors.
- **Generic exceptions replaced** — Domain-specific error types (`AuditIntegrityError`, `GraphValidationError`, `OrchestrationInvariantError`) replace generic `ValueError`/`RuntimeError`, clearing `cluster:wrong-exception-type`.
- **Silent skips → explicit crashes** — 6 audit-gap bugs where code silently returned on invalid state now crash with invariant violation messages.

#### Defensive Pattern Removal

- **`hasattr()` banned unconditionally** — All 3 occurrences replaced with type-safe alternatives. `hasattr` is no longer allowlistable in the tier model enforcer.
- **Defensive `.get()` → required-fields validation** — `AggregationNodeCheckpoint.from_dict()` and other Tier 1 deserializers now validate required fields explicitly instead of using `.get()` with defaults.
- **Defensive access patterns removed** — Two passes across typed and Tier 1 data, replacing `.get(key, default)` with direct access on data we own.
- **CUSTOM header mode fail-closed** — Sink header mode `CUSTOM` now raises on unmapped fields instead of silently falling back to normalized names.

#### Data Fabrication Elimination

- **10 `cluster:fabrication` bugs fixed** — Replaced fabricated defaults (`None` → `0`, missing → empty string) with explicit validation or propagation of absence.
- **LLM batch, DB sink, and coalesce fabrication** — Eliminated silent default injection in three additional paths where missing data was replaced with invented values.

#### Audit-Gap Bug Fixes

- **4 TOCTOU races, silent blank rows, checkpoint invariant** — Fixed race conditions in concurrent audit writes and checkpoint state transitions.
- **3 credential leak, buffer corruption, record ordering** — Closed credential exposure in error messages, buffer mutation after read, and out-of-order audit record insertion.
- **2 rowcount validation, dict mutation** — Added rowcount assertions for fork/coalesce/expand writes; fixed in-place dict mutation in sequential multi-query.
- **4 row_data serialization, replayer Tier 1 reads** — Protected row_data from mutation during serialization; hardened replayer reads to crash on corruption.
- **Audit-gap silent failures recorded** — Error file download, malformed JSONL, and batch quarantine now record failures instead of silently dropping them.

#### Plugin & Engine Fixes

- **`_prepare_call_payloads` extracted** — Deduplicates payload preparation between `record_call` and `record_operation_call`.
- **`_make_checkpoint_after_sink_factory` extracted** — Deduplicates checkpoint-after-sink closure across orchestrator paths.
- **`dataclass_to_dict` tuple handling** — Now handles tuples correctly; fixed `has_retries` off-by-one comparison.
- **Azure client close-before-null** — `CallDataResult` discriminated type replaces ambiguous `None` return from Azure client operations.
- **Missing fingerprint key crashes** — Auth header fingerprinting now crashes on missing key instead of silently skipping; non-finite row indices recorded in batch stats.
- **Broken output port and shutdown timeout** — Batch mixin now crashes on broken output ports and respects shutdown timeout instead of silently continuing.
- **Purge grade updates wrapped individually** — Prevents stale grades after partial payload deletion.
- **Absent finish_reason accepted** — LLM responses with no `finish_reason` field (distinct from non-STOP values) are now accepted; Azure batch file ID guards added.
- **Per-query templates pre-compiled at init** — Separates structural errors (config, caught at startup) from operational errors (per-row, caught at render).
- **Coalesce `_completed_keys` for late arrivals** — Late-arriving tokens after resume are now correctly detected via completed-keys tracking.
- **Route-label enforcement at construction** — Moved from runtime lookup to DAG construction time; `get_route_label()` simplified.
- **`select_branch` KeyError wrapped** — Now raises `GraphValidationError` with context instead of bare `KeyError`.
- **`_get_node()` extracted in AggregationExecutor** — Unifies node validation, removes dead code.

#### Type Design

- **22 type-design bugs fixed** — 14 across engine, contracts, and plugins; 8 across engine, landscape, plugins, and verifier. Tightened field types, added missing validation, removed dead fields.
- **Checkpoint and call_data contracts hardened** — Type design improvements in checkpoint and call_data contract types.

#### Logging & Telemetry

- **stdlib logging → structlog** — Replaced in batch mixin, multi_query, azure_blob_source, and azure_batch.
- **Telemetry emitted before null-content raise** — Telemetry events are now emitted before raising on null-content LLM responses, closing an observability gap.
- **Journal payload load errors caught** — Journal now catches and translates `PayloadNotFoundError` and `OSError` from payload store with diagnostic context.
- **LLM finish-reason fail-closed** — Restructured `_finish_reason_error` from blocklist to allowlist (accept only `STOP` and absent). Unknown finish reasons now rejected as non-retryable errors.
- **Shutdown checkpoint skip logging** — `_checkpoint_interrupted_progress` emits structured `shutdown_checkpoint_skipped` warning with diagnostic context instead of silently returning.
- **Schema epoch directional guard** — `_sync_sqlite_schema_epoch` raises `SchemaCompatibilityError` on future epochs instead of silently downgrading.
- **Checkpoint recovery type annotation** — `_get_buffered_checkpoint_token_ids` parameter typed as `Checkpoint` instead of `Any`.
- **SQLite read-only audit inspection** — `LandscapeDB.from_url(..., create_tables=False)` no longer stamps `PRAGMA user_version`, preserving forensic access.

#### Test Infrastructure (ChaosLLM/ChaosWeb)

- **Malformed header overrides** — ChaosLLM and ChaosWeb now handle malformed header overrides gracefully instead of crashing the test server.
- **ChaosLLM template pre-compilation** — Templates pre-compiled at init for faster test execution.
- **Flaky purge test fixed** — `test_grade_update_failures_logged` used `capsys` which is unreliable when prior tests reconfigure structlog; switched to `structlog.testing.capture_logs()`.

### Changed

- `isinstance` allowlist compacted from flat entries into per-file rules with `max_hits` caps
- Code review findings remediated — frozen field access, `deep_thaw` frozenset, DTO validation

### Added

- **Agentic code threat model discussion paper** — Comprehensive research paper covering forward analysis for agentic security, control strength hierarchy, incentive misalignment analysis, ISM control citations, and ACF framework. Multiple revisions through v0.3 with LaTeX build pipeline and DTA brand guidelines.
- **`PayloadNotFoundError`** — Domain exception in `PayloadStore` protocol, replacing generic `KeyError` for missing payload lookups.
- **`CallDataResult` discriminated type** — Replaces ambiguous `None` returns from Azure client data operations.
- DAG validation tests for route-label and sink-map invariants
- Coalesce checkpoint unit tests for `_get_buffered_checkpoint_token_ids` and `restore_from_checkpoint` rejection paths

### Removed

- Dead lifecycle hooks, stale comments, unused imports across engine and plugins
- Dead code and process-tracking comments in ChaosLLM and ChaosWeb
- Dead code, tombstone comments, and process-tracking prefixes across codebase

### Tests

- 6 P0 mutation survivors killed across canonical, lineage, tokens, triggers, and coalesce
- 15 P1 mutation survivors killed across topology, lineage, tokens, triggers, coalesce, payload, exporter, executors, and outcomes
- 9 test-gap bugs closed with 25 new tests across executors, coalesce, DAG, sinks, and plugins
- 2 remaining test-gap bugs closed — purge command and MCP analyzer queries

---

## [0.3.3] (RC-3.3 — Architectural Remediation)

4-phase remediation sprint driven by full architecture analysis. Focus: audit integrity hardening, layer enforcement, and elimination of defensive-pattern violations.

### T10: LLM Plugin Consolidation

Collapsed 6 LLM transform classes (~4,950 lines) into a unified `LLMTransform` with provider dispatch, eliminating ~3,300 lines of duplication. Strategy pattern: `LLMProvider` protocol handles transport (Azure SDK vs OpenRouter HTTP), two processing strategies (`SingleQueryStrategy` / `MultiQueryStrategy`) handle row logic, shared `LangfuseTracer` handles tracing.

- Extracted shared infrastructure: `LangfuseTracer` (~600 lines deduplicated), `PromptTemplate` system, validation utilities
- Created `LLMProvider` protocol with `AzureLLMProvider` and `OpenRouterLLMProvider` implementations
- Single plugin registration: `plugin: llm` + `provider: azure|openrouter`; old names raise `ValueError` with migration guidance
- Deleted 5 old source files, updated 16 example YAMLs and 10 documentation files

### T17: PluginContext Protocol Split

Decomposed the god-object `PluginContext` (20+ fields) into 4 phase-based protocols in `contracts/contexts.py` — `SourceContext`, `TransformContext`, `SinkContext`, `LifecycleContext` — narrowing each plugin method signature to only the fields that pipeline phase actually needs. Concrete `PluginContext` structurally satisfies all 4 protocols; engine executors mutate concrete fields between steps while plugins see read-only views via protocol typing. 23 plugin files updated.

### T18: Orchestrator/Processor Decomposition

Pure extract-method refactoring of the two largest engine files, reducing maximum method size to ≤150 lines with no behavior change. Extracted 7 methods from `orchestrator/core.py` and 3 from `processor.py`. Introduced typed parameter bundles (`GraphArtifacts`, `RunContext`, `LoopContext`) and discriminated union types for transform/gate outcomes.

### T19: Landscape Repository Pattern

Refactored `LandscapeRecorder` from 8 mixins into 4 composed domain repositories — `RunLifecycleRepository`, `ExecutionRepository`, `DataFlowRepository`, `QueryRepository` — split by pipeline-phase domain. `LandscapeRecorder` is now a pure delegation facade (~91 public methods, zero logic).

### Plugins Restructure (SDA Alignment)

Reorganized the flat `plugins/` directory into 4 SDA-aligned subfolders: `infrastructure/` (shared base classes, clients, batching, pooling), `sources/`, `transforms/`, `sinks/`. 247 files changed, ~200 imports rewritten.

### Protocol Relocation (L3→L0)

Moved `SourceProtocol`, `TransformProtocol`, `SinkProtocol`, `BatchTransformProtocol`, and `GateResult` from `plugins/infrastructure/` (L3) to `contracts/` (L0). Eliminates the engine→plugins layer violation that forced L2 code to import from L3.

### Fixed

- **Pending coalesce resume gaps** — Added typed coalesce checkpoint DTOs, persisted pending coalesce state in checkpoint records, restored coalesce barriers on resume, and taught recovery to exclude buffered coalesce tokens from replay. Graceful shutdown can now resume fork/join pipelines without losing pending joins or replaying already-buffered rows.
- **Interrupted resume checkpoint ordering** — Resumed runs now rebase checkpoint sequence numbers from the previous resume point before writing fresh checkpoints, so a second interrupted resume continues from the newest durable progress marker instead of falling back to an older checkpoint.
- **SQLite schema compatibility posture** — Replaced the ad hoc `checkpoints.coalesce_state_json` required-column gate with an explicit SQLite schema epoch stamp via `PRAGMA user_version`, preserving intentional pre-1.0 schema breaks while keeping a clear future migration seam.
- **Buffered-only resume shutdown semantics** — Resume now honors a pre-set shutdown signal before any end-of-source aggregation/coalesce flushes, so buffered-only checkpoints are re-checkpointed for another resume instead of being flushed to sinks.
- **Frozen audit records** — Added `frozen=True, slots=True` to all 16 mutable audit record dataclasses in `contracts/audit.py`. Mutations now crash at the mutation site instead of silently corrupting the Tier 1 audit trail.
- **FrameworkBugError/AuditIntegrityError re-raise** — Added explicit re-raise before all broad `except Exception` handlers (13 sites across 7 files). System-level errors now always propagate. Structural AST test enforces bare `raise` pattern at all 17 guard sites.
- **Silent failure remediation** — Comprehensive review of error handling across LLM plugins and plugin infrastructure. Silent fallbacks converted to proper exceptions or `TransformResult.error()` with diagnostic context. Missing optional packages now raise `RuntimeError` with install instructions instead of silently degrading.
- **azure_batch silent passthrough** — `_process_single` else branch now raises `RuntimeError` instead of silently passing through unprocessed rows as "processed", matching the hardened pattern in `openrouter_batch`.
- **Assert removal** — Replaced 18 `assert` statements across 10 plugin files with explicit `if/raise RuntimeError`. Asserts are stripped by `python -O`, silently removing safety checks.
- **Truthiness checks** — Fixed 21 `if x:` / `x or default` patterns across 8 files that silently excluded valid zero values and empty strings. All replaced with explicit `is not None` checks.
- **LLM transform bugs** — Fixed limiter dispatch using wrong config attribute, `response_format` not passed to provider, `output_fields` not extracted from multi-query responses, NaN/Infinity not rejected in LLM JSON responses
- **Layer violations resolved** — Moved `ExpressionParser` from `engine/` to `core/`, `MaxRetriesExceeded` and `BufferEntry` to `contracts/`, created `RuntimeServiceRateLimit` in `contracts/config/`. 10 upward import violations → 0.
- **OpenRouter parallel query client race** — Parallel multi-query runs shared a cached `AuditedHTTPClient` by `state_id`; first query to finish destroyed the transport for siblings. Added reference counting so client closes only when last query releases it.
- **Aggregation BUFFERED lifecycle gap** — Triggering token on count-threshold flush skipped `BUFFERED` and went directly to terminal. Moved `BUFFERED` recording before `should_flush()` check so every aggregation token follows `BUFFERED` → terminal.
- **BatchReplicate quarantine audit gap** — Buffer-time recording changed from `CONSUMED_IN_BATCH` (terminal) to `BUFFERED` (non-terminal) for transform-mode aggregation, enabling per-token `QUARANTINED` recording when batch transforms quarantine individual rows.
- **KeywordFilter fail-closed on non-string values** — Security transform was silently passing non-string values in explicitly configured fields (fail-open). Now returns error with `reason='non_string_field'`.
- **Multi-query regressions from T10** — Restored field type validation against declared `output_fields` type/enum constraints; restored pooled execution with AIMD capacity backoff; fixed Pydantic schema missing `output_fields`; fixed `_output_schema_config` using unprefixed single-query fields.
- **LLM empty/whitespace content detection** — Azure and OpenRouter providers now raise `ContentPolicyError` for empty or whitespace-only content before `LLMQueryResult` construction.
- **LLM content-filter finish reason fail-open** — Unified single-query and multi-query `LLMTransform` paths now treat `finish_reason=content_filter` as `reason='content_filtered'` instead of recording provider-filtered fallback text as successful output.
- **Telemetry/Landscape hash divergence** — Telemetry hashes now read from recorded `Call` object instead of recomputing independently, eliminating divergence for datetime/Decimal/bytes/numpy payloads.
- **URL password fingerprint encoding** — Fingerprinting now decodes percent-encoding before HMAC, so fingerprint represents the actual secret, not the URL-encoded form.
- **TUI coalesce error crash on older records** — `_validate_coalesce_error` crashed with `KeyError` on pre-RC3.3 records. Added schema shape detection; older records render with degraded-format note.
- **Graceful shutdown end-of-source synthesis** — Interrupted runs no longer force `END_OF_SOURCE` aggregation flushes or resolve pending coalesces just because shutdown arrived after the current row.
- **Graceful shutdown resumability for buffered pipelines** — Interrupted aggregation/coalesce runs now persist a shutdown checkpoint before raising, so buffered state remains resumable even when no sink token was written yet.
- **CLI explain passphrase silently swallowed** (T4) — YAML parse errors when `--settings` was explicitly provided now exit with code 1 and clear error message.
- **MCP `diagnose()` quarantine count unscoped** (T5) — Was counting all historical runs; now scoped to last 24 hours, matching "what's broken right now?" purpose.
- **ChaosLLM MCP CLI broken** (T27) — Called nonexistent `serve()` instead of `run_server()`, masked by `# type: ignore` comments.
- **Azure AI tracing silent no-op** — Wired `_configure_azure_monitor()` into `LLMTransform.on_start()` with provider compatibility validation (Azure-only); replaced broad `except TypeError` with explicit `None` check so real SDK errors propagate.
- **Contract-level fixes** — `Token.run_id` false optional removed; `CoalesceFailureReason` TypedDict replaced with frozen dataclass (3 dead fields deleted, 4 fields made required); dead `version` parameter removed from `stable_hash()`; Call XOR invariant (`state_id` vs `operation_id`) now enforced at construction; `RawCallPayload.to_dict()` returns shallow copy per immutability contract; `SanitizedDatabaseUrl` rewrote DSN handling to use `urllib.parse` (keeping `contracts/` a leaf layer).
- **Code review remediation** — 4 critical (provider key validation split, `REPR_FALLBACK` row data state, `AuditIntegrityError` in `ExecutionError.from_dict()`, type-narrow `_convert_retryable_to_error_result`), 8 important, 6 suggestion fixes.
- **CI/CD failures resolved** — ruff lint/format, mypy stale `type: ignore`, contracts allowlist, tier model (31 stale fingerprints refreshed).

### Changed

- Extracted `contracts/hashing.py` — primitive-only `canonical_json`, `stable_hash`, and `repr_hash` (RFC 8785 + hashlib, no pandas/numpy). Breaks circular dependency between `contracts/` and `core/canonical.py`.
- Aggregation `on_error` is now required for aggregation transforms
- DTO mapper classes renamed from `*Repository` to `*Loader` to avoid confusion with new domain repositories
- **Test infrastructure overhaul (P0.5a–P4)** — 6-phase systematic hardening of the test suite, eliminating brittle coupling to internal constructors:
  - P0.5a–b: New factories (`make_recorder_with_run()`, `register_test_node()`, etc.) and refactored existing factories to delegate through them
  - P1: Replaced ~350 direct `PluginContext(...)` constructions across 53 files with centralized `make_context()` factory
  - P2: Replaced ~452 inline `LandscapeDB.in_memory()`/`LandscapeRecorder(...)` constructions across 76 files with factory calls. Net −715 lines
  - P3: Replaced ~529 lines of duplicated inline test plugin classes across 10 files with shared `tests.fixtures.plugins` imports
  - P4: Re-raise guards in telemetry/orchestrator/operation tracking, frozen evidence types (`ExceptionResult`, `FailureInfo`), aggregation DRY via `accumulate_row_outcomes()` + `ExecutionCounters`
- Resolved all 401 mypy errors across test suite — removed ~74 stale `# type: ignore` comments, added union-type narrowing guards, fixed module re-exports, wrapped `NewType` constructors, fixed protocol signatures (103 files)
- `PluginBundle` frozen dataclass replaces `dict[str, Any]` return from `instantiate_plugins_from_config()`, enabling mypy checking on all access sites
- Fingerprint primitives (`get_fingerprint_key()`, `secret_fingerprint()`) moved to `contracts/security.py` as stdlib-only implementations
- Redundant `.value` on `StrEnum` usage removed across checkpoint, landscape repositories, MCP, and tests
- Removed file-path header comments from 128 source files
- Azure safety transform consolidation (T14) — extracted shared batch infrastructure into `BaseAzureSafetyTransform` and `safety_utils.py`

### Added

- Typed coalesce checkpoint contracts (`CoalesceCheckpointState`, `CoalescePendingCheckpoint`, `CoalesceTokenCheckpoint`) plus CLI resume visibility for whether a checkpoint carries coalesce state
- **ADR-006**: Layer Dependency Remediation — documents the strict 4-layer model and CI enforcement strategy
- Full architecture analysis (23 documents covering all 13 subsystems)
- **Security posture brief** — Comprehensive document covering threat model, security controls, assurance evidence, and residual risk for ELSPETH v0.3.0
- **TYPE_CHECKING layer import detection** — `enforce_tier_model.py` CI gate now detects `TYPE_CHECKING` imports crossing layer boundaries as allowlistable findings
- **MCP server `_ToolDef` registry** replacing if/elif dispatch chain (T15)
- ~150 new tests across hardening, code review, and infrastructure phases

### Removed

- Dead code: `BaseLLMTransform` (3,473 lines, zero subclasses), `RequestRecord` dataclass, `TokenManager.payload_store` parameter, `populate_run()` (raw SQL bypass of `LandscapeRecorder`), LLM validation utilities (`render_template_safe`, `check_truncation`)
- ~21 low-value tests (vacuous assertions, mock-testing, implementation coupling)
- Superseded aggregation helpers replaced by shared `accumulate_row_outcomes()`

### Tests

- Full suite: approximately 10,500 tests — mypy/ruff/contracts all clean
- P0.5a–P4 test infrastructure overhaul: centralized factories, shared fixtures, eliminated ~1,700 lines of duplicated test boilerplate

---

## [0.3.0] - 2026-02-22 (RC-3.2)

### Highlights

- **Schema Contracts** — First-row-inferred field contracts propagated through the DAG and recorded in the audit trail
- **Declarative DAG Wiring** — Every edge explicitly named and validated at construction time
- **PipelineRow** — Typed row wrapper replacing raw dicts throughout the pipeline
- **Strict Typing at Audit Boundaries** — Every `dict[str, Any]` crossing into the Landscape audit trail replaced with frozen dataclasses, eliminating an entire class of silent data-corruption bugs
- **Test Suite v2** — Complete rewrite with 8,000+ tests across unit, property, integration, E2E, and performance layers
- **178-Bug Triage** — Systematic closure of 160+ bugs across 8 hardening phases

### Added

- Schema contract system: inference, propagation, sink header modes, and audit recording
- Typed DTOs at audit boundaries: `BatchCheckpointState`, `WebOutcomeClassification`, `NodeStateContext`, `CoalesceMetadata`, `AggregationCheckpointState`, `TokenUsage`, `GateEvaluationContext`, `AggregationFlushContext`, `CallPayload` protocol with typed request/response pairs
- `NodeStateGuard` context manager enforcing terminal-state invariants in all executors
- `detect_field_collisions()` utility preventing silent data overwrites across all transforms
- Azure Key Vault secrets backend with audit trail
- SQLCipher encryption-at-rest for the Landscape database
- WebScrape transform with SSRF prevention and content fingerprinting
- ChaosWeb fake server for stress-testing HTTP transforms
- Langfuse v3 tracing for LLM plugins
- Per-branch transforms between fork and coalesce nodes
- Graceful shutdown (SIGINT/SIGTERM) for run and resume paths
- DIVERT routing for quarantine/error sink paths

### Fixed

- **P0:** DNS rebinding TOCTOU in SSRF, JSON sink data loss on crash, content safety / prompt shield fail-open
- Frozen dataclass DTOs replacing `dict[str, Any]` at 10+ audit trail boundaries — eliminates runtime KeyError risk and tier-model allowlist entries
- `PluginContext.update_checkpoint()` replaced with `set_checkpoint()` (replacement semantics) — fixes P1 bug where dict merge lost checkpoint updates on restored batch state
- NaN/Infinity rejection at JSON parse and schema validation boundaries
- Resume row-drop, batch adapter crash, gate-to-gate routing crash
- Telemetry DROP-mode evicting newest instead of oldest events
- SharedBatchAdapter duplicate-emit race condition (first-result-wins preserved)
- AzureBlobSink multi-batch overwrite, CSVSource multiline skip_rows, JSONL multibyte decoding

### Changed

- Orchestrator, LandscapeRecorder, MCP server, and executors decomposed from monoliths into focused modules
- Checkpoint API typed: `get_checkpoint()` returns `BatchCheckpointState | None`, `set_checkpoint()` accepts typed state
- Pre-commit hooks scan full codebase (12 hooks, check-only)
- docs/ restructured from 792 files to 62 files
- All Alembic migrations deleted (pre-release, no users)

### Removed

- Gate plugin subsystem — routing is now config-driven only
- Beads (bd) issue tracker — migrated to Filigree
- V1 test suite (7,487 tests, 222K lines) — replaced by v2
- Dead plugin protocols (CoalesceProtocol, GateProtocol, PluginProtocol)

---

## [0.1.0] - 2026-02-02 (RC-2)

Initial release candidate. Core SDA pipeline engine with audit trail,
plugin system, and CLI.

## Historical Changelogs

The root changelog is the visible release-history document. Earlier RC-1 and
RC-2 fragment files were removed during the repository cleanout because their
useful history is represented here and in git history.

<!-- Comparison links — tags created at release time -->
[0.5.1]: https://github.com/tachyon-beep/elspeth/compare/v0.5.0-rc5.0...v0.5.1-rc5.1
[0.5.0]: https://github.com/tachyon-beep/elspeth/compare/v0.4.1-rc4.1...v0.5.0-rc5.0
[0.4.1]: https://github.com/tachyon-beep/elspeth/compare/v0.4.0-rc4.0...v0.4.1-rc4.1
[0.4.0]: https://github.com/tachyon-beep/elspeth/compare/v0.3.4-rc3.4...v0.4.0-rc4.0
[0.3.4]: https://github.com/tachyon-beep/elspeth/compare/v0.3.3-rc3.3...v0.3.4-rc3.4
[0.3.3]: https://github.com/tachyon-beep/elspeth/compare/v0.3.0-rc3.2...v0.3.3-rc3.3
[0.3.0]: https://github.com/tachyon-beep/elspeth/compare/v0.1.0-phase1...v0.3.0-rc3.2
[0.1.0]: https://github.com/tachyon-beep/elspeth/releases/tag/v0.1.0-phase1

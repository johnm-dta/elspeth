# Elspeth Read-Only Codebase Audit

Generated: 2026-06-04

Repository: `/home/john/elspeth`

Observed branch: `RC5.3`

Observed worktree state: clean at audit start (`git status --short` and `git diff --stat` produced no output)

Audit mode: strictly read-only source review. No repository files were modified. No tests, builds, services, MCP tools, Filigree tools, Clarion tools, scanner commands, dependency installs, or live network checks were run.

Artifact path: `/tmp/elspeth-read-only-audit-2026-06-04.md`

## Read-Only Boundary

Seven specialized reviewers were dispatched with read-only prompts containing:

- `enable_write_tools=false`
- `enable_mcp_tools=false`
- explicit bans on `apply_patch`, shell writes, redirection writes, service/test/build/install commands, MCP, Filigree, Clarion, scanner, and reporting tools
- explicit instruction not to use backslash-escaped double quotes in tool arguments

The available subagent API exposed the `explorer` role rather than literal `enable_write_tools` or `enable_mcp_tools` parameters. The reports returned by the reviewers state that only read-only file and text inspection commands were used.

## Scope Notes

The requested `scanner/ast_primitives.py` and `src/elspeth/scanner` paths do not exist in the current checkout. Static-analysis code appears to live under `elspeth-lints/src/elspeth_lints`.

Literal `PY-WL-101` through `PY-WL-111` rule identifiers were not present in the current static-analysis tree. The active trust-tier implementation uses subrule IDs such as `R1` through `R9`, `TC`, `L1`, and `R_TB_*`.

## Reviewer Roster

- Architecture Critic: `019e8f4a-45ab-7b01-bb8d-72415bf6e333`
- Systems Thinker: `019e8f4a-4608-77e3-b0ac-5a375a6906ae`
- Python Engineer: `019e8f4a-4680-7e33-9e07-3cf3c4d33843`
- Quality Engineer: `019e8f4a-4710-7831-a472-76c61a5d9c51`
- Security Architect: `019e8f4a-47ae-77e0-85c6-6645ae3e831b`
- Static Tools Analyst: `019e8f4a-48cd-7201-bd1b-3bb5dcff2076`
- MCP and CLI Specialist: `019e8f50-caa5-78f2-909c-53e242dc4ef1`

## Executive Summary

Critical findings: 0

High findings: 7

Medium findings: 9

Low findings: 2

The highest-risk issues are security and audit-surface gaps: Azure Search RAG can perform credentialed requests through a non-SSRF-pinned POST path; the Landscape MCP server advertises read-only behavior but opens SQLite through a writable connection path; static whole-repo rules can silently skip unreadable or unparseable files; and several Tier 3 JSON paths split preview, audit, and runtime behavior.

## Critical Findings

No confirmed Critical findings.

## High Findings

### H-1: Azure Search RAG provider performs credentialed requests without SSRF-safe POST pinning

Locations:

- [azure_search.py](/home/john/elspeth/src/elspeth/plugins/infrastructure/clients/retrieval/azure_search.py:43) lines 43-51
- [azure_search.py](/home/john/elspeth/src/elspeth/plugins/infrastructure/clients/retrieval/azure_search.py:129) lines 129-150
- [azure_search.py](/home/john/elspeth/src/elspeth/plugins/infrastructure/clients/retrieval/azure_search.py:176) lines 176-190
- [azure_search.py](/home/john/elspeth/src/elspeth/plugins/infrastructure/clients/retrieval/azure_search.py:349) lines 349-379
- [http.py](/home/john/elspeth/src/elspeth/plugins/infrastructure/clients/http.py:367) lines 367-382
- [http.py](/home/john/elspeth/src/elspeth/plugins/infrastructure/clients/http.py:509) lines 509-545
- [config.py](/home/john/elspeth/src/elspeth/plugins/transforms/rag/config.py:138) lines 138-143

Evidence:

`AzureSearchProviderConfig.validate_endpoint()` only requires HTTPS and a hostname. RAG config validation instantiates this provider config, so that weak endpoint check is the effective boundary. `AzureSearchProvider` builds `_search_url` from the configured endpoint, attaches `api-key` when configured, and calls `AuditedHTTPClient.post()`. The generic POST path sends `full_url` directly through `self._client.post()`. A DNS/IP-pinned SSRF-safe helper exists, but only for GET. The readiness path calls `validate_url_for_ssrf(count_url)` and then performs `httpx.get(count_url)` using the original URL while attaching either `api-key` or a managed-identity bearer token.

Impact:

A user- or composer-authored RAG provider config can make the server send authenticated requests to arbitrary HTTPS endpoints reachable from the server network. This can expose credentials or enable internal network probing. The readiness path also has a validation-to-use DNS gap.

Remediation:

Add a method-general SSRF-safe request primitive, or at least a POST-capable variant, that connects to a validated/pinned address while preserving original Host and TLS SNI. Use it for Azure Search search and readiness calls. Revalidate redirect hops. At config time, reject literal private, loopback, link-local, and metadata endpoints by default; preferably require `*.search.windows.net` or an explicit operator allowlist for private Azure Search deployments.

Acceptance tests:

Add Azure Search tests proving `https://10.0.0.1`, `https://169.254.169.254`, and DNS-to-private fixtures are rejected or blocked before any request is sent. Add a regression proving `_execute_search()` uses the SSRF-safe POST path. Add a readiness test proving the actual request uses the pinned connection URL plus Host/SNI, not the original hostname lookup.

Confidence: High.

### H-2: Landscape MCP read-only server opens SQLite through a writable connection path

Locations:

- [server.py](/home/john/elspeth/src/elspeth/mcp/server.py:1) lines 1-5
- [analyzer.py](/home/john/elspeth/src/elspeth/mcp/analyzer.py:57) lines 57-65
- [database.py](/home/john/elspeth/src/elspeth/core/landscape/database.py:298) lines 298-306
- [database.py](/home/john/elspeth/src/elspeth/core/landscape/database.py:753) lines 753-760
- [database.py](/home/john/elspeth/src/elspeth/core/landscape/database.py:803) lines 803-841
- [queries.py](/home/john/elspeth/src/elspeth/mcp/analyzers/queries.py:57) lines 57-69
- [queries.py](/home/john/elspeth/src/elspeth/mcp/analyzers/queries.py:1012) lines 1012-1032

Evidence:

The MCP server describes itself as a lightweight read-only audit server. `LandscapeAnalyzer` initializes with `LandscapeDB.from_url(...)`; the SQLite path installs `_configure_sqlite`, which executes `PRAGMA journal_mode=WAL` on connect. Analyzer reads such as `list_runs` use `db.connection()`, while only the ad-hoc SQL `query` tool uses `db.read_only_connection()`.

Impact:

A read-only MCP audit surface can mutate SQLite database state or create WAL/SHM files during startup/read operations. It may also fail against a genuinely read-only mounted evidence database. This violates the expected security default for an audit inspection interface.

Remediation:

Add an explicit read-only database mode for MCP inspection. For SQLite, open with a read-only URI where appropriate and avoid writable PRAGMAs in this mode. Route analyzer SELECT operations through `read_only_connection()` or a read-only query facade. For PostgreSQL, use explicit read-only transactions.

Acceptance tests:

Create a SQLite landscape database, close it, make the DB path or containing directory read-only, then initialize the MCP server and call `list_tools` plus representative queries such as `list_runs`. Assert no `-wal` or `-shm` file is created and no write PRAGMA is issued. Add a regression proving analyzer reads use the read-only connection path.

Confidence: High.

### H-3: Whole-repo static rules silently drop unreadable or unparseable files

Locations:

- [cli.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/core/cli.py:895) lines 895-951
- [rule.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py:1488) lines 1488-1500
- [rule.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py:1554) lines 1554-1572
- [rule.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py:2467) lines 2467-2504
- [rule.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/trust_boundary/tests/rule.py:341) lines 341-365
- [rule.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/trust_boundary/scope/rule.py:170) lines 170-184

Evidence:

The core CLI converts `PythonSyntaxError` and `PythonFileReadError` into `parse-error` and `read-error` findings only for incremental rules. Whole-repo rules are invoked directly with an empty AST, bypassing that diagnostic path. The trust-tier scanner catches `OSError`, `UnicodeDecodeError`, and `SyntaxError`, prints to stderr, and returns empty findings. The trust-boundary whole-repo rules explicitly skip parse/read errors while comments say the CLI driver surfaces them.

Impact:

A file can be omitted from trust-tier and trust-boundary security checks without a first-class finding or failing result. This can hide forbidden defensive fallbacks, `hasattr`, upward imports, malformed trust-boundary metadata, or taint-scope violations.

Remediation:

Route whole-repo rules through the shared `walk_python_files` diagnostic path, or make every whole-repo scanner convert parse/read failures into first-class failing diagnostics. Prefer one central CLI mechanism so every rule family gets consistent coverage accounting.

Acceptance tests:

Create fixtures containing one valid Python file and one syntax-broken or unreadable Python file. Run the check path for `trust-tier-model` and trust-boundary whole-repo rules. Assert the result contains a `parse-error` or `read-error` finding and fails rather than reporting clean output.

Confidence: High.

### H-4: Broad-except checker treats nested-scope raises as handler re-raises

Locations:

- [rule.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py:1418) lines 1418-1451
- [ast_walker.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/core/ast_walker.py:35) lines 35-118
- [rule.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py:905) lines 905-918

Evidence:

`visit_ExceptHandler` marks a broad handler as safe if any `ast.Raise` appears under `ast.walk(node)`. `ast.walk(node)` descends into nested functions, lambdas, async functions, and classes. The repository already has `iter_own_scope()` to avoid this lexical-scope conflation, and the same rule uses it for specific exception swallowing.

Impact:

A broad handler can swallow an exception and avoid the `R4` finding by defining a nested helper that raises. That creates a false negative in the tier-model gate.

Remediation:

Replace the `ast.walk(node)` check with own-scope traversal over the handler body, using `iter_own_scope(statement)` as `_handler_is_silent()` already does. Only a raise in the handler's own lexical scope should suppress `R4`.

Acceptance tests:

Add a fixture where `except Exception:` defines a nested helper containing `raise RuntimeError()` and then swallows the exception; it should emit `R4`. Keep a companion fixture with direct `except Exception: raise`, which should not emit `R4`.

Confidence: High.

### H-5: Composer source inspection accepts non-finite JSON constants at the Tier 3 blob boundary

Locations:

- [_helpers.py](/home/john/elspeth/src/elspeth/web/sessions/routes/_helpers.py:2394) lines 2394-2415
- [source_inspection.py](/home/john/elspeth/src/elspeth/web/composer/source_inspection.py:450) lines 450-477
- [source_inspection.py](/home/john/elspeth/src/elspeth/web/composer/source_inspection.py:493) lines 493-505
- [source_inspection.py](/home/john/elspeth/src/elspeth/web/composer/source_inspection.py:553) lines 553-630
- [json_source.py](/home/john/elspeth/src/elspeth/plugins/sources/json_source.py:272) lines 272-314

Evidence:

`_inspect_latest_ready_session_blob()` documents blob bytes as Tier 3 and calls `inspect_blob_content()`. Source inspection parses JSON and JSONL with plain `json.loads`, which accepts `NaN`, `Infinity`, and `-Infinity` by default. `_facts_from_objects()` classifies any parsed `float` as `float` with no finiteness check. Runtime JSON sources separately reject non-finite constants via `parse_constant`.

Impact:

Composer preview can infer schema facts from invalid external JSON that runtime source loading will reject or quarantine. This creates a preview-to-runtime feedback split: generated guidance can be based on data that later fails the stricter execution boundary.

Remediation:

Parse JSON and JSONL in source inspection through the same strict JSON policy used by runtime sources. Prefer a shared strict parser. Preserve the inspection contract by returning warnings and excluding invalid objects rather than raising hard failures.

Acceptance tests:

Add JSON and JSONL inspection tests for `{"score": NaN}` and `{"score": Infinity}`. Assert warnings are returned, `score` is not inferred as `float`, and `SourceInspectionFacts` is still returned.

Confidence: High.

### H-6: Azure safety transforms re-parse audited HTTP responses with `response.json()`

Locations:

- [http.py](/home/john/elspeth/src/elspeth/plugins/infrastructure/clients/http.py:160) lines 160-181
- [http.py](/home/john/elspeth/src/elspeth/plugins/infrastructure/clients/http.py:218) lines 218-239
- [http.py](/home/john/elspeth/src/elspeth/plugins/infrastructure/clients/http.py:294) lines 294-306
- [json_utils.py](/home/john/elspeth/src/elspeth/plugins/infrastructure/clients/json_utils.py:1) lines 1-10
- [json_utils.py](/home/john/elspeth/src/elspeth/plugins/infrastructure/clients/json_utils.py:74) lines 74-99
- [prompt_shield.py](/home/john/elspeth/src/elspeth/plugins/transforms/azure/prompt_shield.py:236) lines 236-250
- [content_safety.py](/home/john/elspeth/src/elspeth/plugins/transforms/azure/content_safety.py:254) lines 254-263
- [test_content_safety.py](/home/john/elspeth/tests/unit/plugins/transforms/azure/test_content_safety.py:31) lines 31-39
- [test_prompt_shield.py](/home/john/elspeth/tests/unit/plugins/transforms/azure/test_prompt_shield.py:32) lines 32-40

Evidence:

`AuditedHTTPClient` parses JSON responses with strict NaN/Infinity and duplicate-key rejection, records that parsed payload to audit, and emits telemetry only after audit. Azure Prompt Shield and Content Safety then call `response.json()` after the audited call. Their tests use mocks with `response.json.return_value`, bypassing the audited parse path.

Impact:

Malformed Azure responses can be recorded in audit as strict-parse failures while transform behavior is driven by a second, looser parser. That breaks audit primacy and allows operational behavior and audit evidence to disagree about the same external response.

Remediation:

Make these transforms consume the strict parsed payload from `AuditedHTTPClient`, or parse `response.text` with the shared strict parser before validating response shape. Avoid production `response.json()` calls for external JSON.

Acceptance tests:

Add Azure Content Safety and Prompt Shield tests using raw response bodies containing `NaN`, `Infinity`, duplicate keys, and irrelevant extra fields. Assert malformed-response failure with no successful pass-through, and exercise the production audited HTTP response path or shared strict parser instead of only mocked `response.json.return_value`.

Confidence: High.

### H-7: Retry integration tests bypass the production retry path they claim to verify

Locations:

- [test_retry.py](/home/john/elspeth/tests/integration/pipeline/test_retry.py:2) lines 2-12
- [test_retry.py](/home/john/elspeth/tests/integration/pipeline/test_retry.py:229) lines 229-260
- [test_retry.py](/home/john/elspeth/tests/integration/pipeline/test_retry.py:266) lines 266-269
- [test_retry.py](/home/john/elspeth/tests/integration/pipeline/test_retry.py:330) lines 330-361
- [test_retry.py](/home/john/elspeth/tests/integration/pipeline/test_retry.py:370) lines 370-373
- [test_retry.py](/home/john/elspeth/tests/integration/pipeline/test_retry.py:441) lines 441-455

Evidence:

The module docstring claims to verify the full retry audit chain through settings, orchestrator, row processor, executor, and `node_states`. The tests instantiate `TransformExecutor` and `RetryManager` directly, manually track attempt numbers, and execute without the orchestrator/row-processor retry path. Queries filter `node_states` by `node_id` only.

Impact:

The test suite can pass while the production wiring from settings to row processor to audit attempt recording is broken. Since the asserted behavior is auditability of retries, this is a high-confidence false-positive coverage gap.

Remediation:

Replace or supplement the direct-executor tests with a production-path integration test that constructs a real runtime graph/config, runs the orchestrator, and lets row processing create retry attempts. Assert `node_states` using composite run/node scoping.

Acceptance tests:

Use a flaky transform that fails twice then succeeds. Run through the production orchestration path. Assert attempts 0 and 1 are failed, attempt 2 is completed, all rows are scoped by `run_id` plus `node_id`, and retry settings flow through `RuntimeRetryConfig.from_settings()`.

Confidence: High.

## Medium Findings

### M-1: OIDC issuer discovery fetch happens before issuer URL and SSRF validation

Locations:

- [config.py](/home/john/elspeth/src/elspeth/web/config.py:249) lines 249-262
- [config.py](/home/john/elspeth/src/elspeth/web/config.py:406) lines 406-427
- [app.py](/home/john/elspeth/src/elspeth/web/app.py:240) lines 240-263
- [oidc.py](/home/john/elspeth/src/elspeth/web/auth/oidc.py:43) lines 43-50
- [oidc.py](/home/john/elspeth/src/elspeth/web/auth/oidc.py:92) lines 92-110
- [oidc.py](/home/john/elspeth/src/elspeth/web/auth/oidc.py:253) lines 253-262

Evidence:

`WebSettings` rejects blank OIDC fields but does not validate `oidc_issuer` as an absolute HTTPS URL, reject embedded credentials, or block private/link-local/metadata hosts. During lifespan startup, the app builds `discovery_url = f"{issuer}/.well-known/openid-configuration"` and fetches it before validating the discovered authorization endpoint. `JWKSTokenValidator` similarly stores the raw issuer and fetches discovery before validating the discovery-provided `jwks_uri`.

Impact:

This is config-driven rather than request-driven, but if deployment config is compromised or faulty the web service can be made to fetch internal or metadata-network URLs at startup/auth time.

Remediation:

Add a `validate_oidc_issuer()` helper and apply it in `WebSettings` and `JWKSTokenValidator.__init__()`. Require absolute HTTPS, no embedded credentials, no query/fragment, and reject loopback, private, link-local, and metadata IPs by default. If private IdPs are supported, require an explicit allowlist/opt-in.

Acceptance tests:

Add config and startup/JWKS tests proving `http://issuer.example.com`, `https://user:pass@issuer.example.com`, `https://127.0.0.1`, and `https://169.254.169.254` fail before any network call. Keep a positive test for a valid HTTPS issuer and same-origin authorization endpoint.

Confidence: High.

### M-2: Web execution config loading diverges from file-backed CLI template expansion

Locations:

- [validation.py](/home/john/elspeth/src/elspeth/web/execution/validation.py:1) lines 1-14
- [validation.py](/home/john/elspeth/src/elspeth/web/execution/validation.py:1213) lines 1213-1240
- [service.py](/home/john/elspeth/src/elspeth/web/execution/service.py:1086) lines 1086-1090
- [config.py](/home/john/elspeth/src/elspeth/core/config.py:1772) lines 1772-1823
- [config.py](/home/john/elspeth/src/elspeth/core/config.py:1956) lines 1956-2014
- [config.py](/home/john/elspeth/src/elspeth/core/config.py:2088) lines 2088-2154
- [config.py](/home/john/elspeth/src/elspeth/core/config.py:2157) lines 2157-2183

Evidence:

Web validation and web execution use `load_settings_from_yaml_string()`. File-backed `load_settings()` expands `template_file`, `lookup_file`, and `system_prompt_file`; the string loader lowercases keys, validates top-level fields, expands environment variable patterns, and constructs `ElspethSettings` without expanding file-backed template options.

Impact:

A config valid through CLI file loading can fail or behave differently through web validation/execution. The validation docstring's parity claim is not mechanically true for file-backed template options.

Remediation:

Move shared config normalization into a single pipeline with an explicit source context. For web, either provide a safe base path and use the same template expansion, or reject file-backed options before Pydantic with a structured unsupported-feature error. Update validation comments to state the actual parity boundary.

Acceptance tests:

Add web validation/execution coverage for a pipeline containing `template_file`, `lookup_file`, and `system_prompt_file`. Assert either identical expansion to `load_settings()` with a supplied base path, or a clear structured validation error before plugin instantiation.

Confidence: High.

### M-3: Engine executor depends on plugin infrastructure internals

Locations:

- [transform.py](/home/john/elspeth/src/elspeth/engine/executors/transform.py:51) line 51
- [transform.py](/home/john/elspeth/src/elspeth/engine/executors/transform.py:162) lines 162-196
- [transform.py](/home/john/elspeth/src/elspeth/engine/executors/transform.py:262) lines 262-266
- [transform.py](/home/john/elspeth/src/elspeth/engine/executors/transform.py:389) lines 389-421
- [mixin.py](/home/john/elspeth/src/elspeth/plugins/infrastructure/batching/mixin.py:41) lines 41-105
- [engine.yaml](/home/john/elspeth/config/cicd/enforce_tier_model/engine.yaml:1) lines 1-15

Evidence:

`TransformExecutor` imports `BatchTransformMixin` from plugin infrastructure, narrows by concrete `isinstance`, casts to the mixin, reads `_pool_size` and `_batch_wait_timeout`, and calls `connect_output`. The tier-model allowlist documents this L1 violation and says protocol extraction is still pending.

Impact:

L2 engine behavior is coupled to an L3 plugin-infrastructure concrete class and private attributes. New batch implementations must inherit the mixin rather than satisfy an engine/contract protocol, and CI needs a targeted waiver.

Remediation:

Extract a `BatchTransformRuntimeProtocol` plus batch output/adapter contracts into `contracts` or an engine-owned package. Have the mixin implement that protocol, make `TransformExecutor` depend only on the protocol, and remove the L1 allowlist entry.

Acceptance tests:

Tier-model enforcement should report no L1 hit for `engine/executors/transform.py`. Batch transform execution tests should still cover timeout eviction, adapter registration, and LLM/Azure mixin transforms. Add protocol conformance coverage for `BatchTransformMixin`.

Confidence: High.

### M-4: SCC detection can report an acyclic graph when NetworkX is unavailable

Locations:

- [rule.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py:1992) lines 1992-2009
- [rule.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py:2042) lines 2042-2076
- [cli.py](/home/john/elspeth/elspeth-lints/src/elspeth_lints/core/cli.py:1168) lines 1168-1225
- [pyproject.toml](/home/john/elspeth/elspeth-lints/pyproject.toml:1) lines 1-8
- [test_trust_tier_dump_edges.py](/home/john/elspeth/tests/unit/elspeth_lints/test_trust_tier_dump_edges.py:293) lines 293-311

Evidence:

`scan_dump_edges()` imports `networkx` inside the function. On `ImportError`, it prints a warning and returns an empty SCC list. JSON rendering then reports `scc_count` from `len(sccs)`, and the CLI returns success after rendering. The standalone `elspeth-lints` package metadata declares only `pyyaml`, while tests cover the NetworkX-present happy path.

Impact:

An environment installed from `elspeth-lints` metadata alone can emit a cyclic graph with `scc_count: 0`, making "no SCCs" indistinguishable from "SCC computation skipped" unless a human notices stderr.

Remediation:

Implement deterministic local Tarjan SCC logic or declare NetworkX as a required dependency and make import failure a fatal machine-readable diagnostic. Do not encode missing SCC computation as an empty SCC list.

Acceptance tests:

Simulate NetworkX import failure against a fixture graph with a known cycle. Assert local Tarjan still reports the SCC, or the CLI returns nonzero with a machine-readable diagnostic. Add a package metadata test if NetworkX remains required.

Confidence: High for the fallback behavior; medium-high for operational likelihood.

### M-5: MCP `get_run_summary` token count ignores token run ownership

Locations:

- [reports.py](/home/john/elspeth/src/elspeth/mcp/analyzers/reports.py:62) lines 62-71
- [schema.py](/home/john/elspeth/src/elspeth/core/landscape/schema.py:196) lines 196-233
- [queries.py](/home/john/elspeth/src/elspeth/mcp/analyzers/queries.py:174) lines 174-180

Evidence:

`get_run_summary()` counts tokens by joining `tokens.row_id` to `rows.row_id` and filtering only `rows.run_id == run_id`. The `tokens` table also has its own `run_id`, explicitly documented as run ownership for cross-run contamination prevention. `list_tokens()` filters directly on `tokens.run_id`.

Impact:

If a stale, malformed, or contaminated token row has a `run_id` different from the referenced row, the MCP summary can count that token under the wrong run. In a valid database this should be rare, but MCP reporting is an audit surface.

Remediation:

Count tokens with `tokens_table.c.run_id == run_id`. If joining rows for integrity verification, join on both `row_id` and `run_id`, and surface an explicit audit integrity error when token ownership and row ownership disagree.

Acceptance tests:

Create two runs and inject a token row whose `run_id` is not the target run but whose `row_id` references a target-run row. `get_run_summary(target_run)` must not count the wrong-run token; preferably it should raise an explicit audit integrity error.

Confidence: Medium-high.

### M-6: PostgreSQL regression modules are fully skipped

Locations:

- [test_compose_loop_concurrent_sessions.py](/home/john/elspeth/tests/integration/web/test_compose_loop_concurrent_sessions.py:37) lines 37-44
- [test_blobs_ready_hash_postgres.py](/home/john/elspeth/tests/integration/web/test_blobs_ready_hash_postgres.py:54) lines 54-61

Evidence:

Both PostgreSQL regression modules are module-level skipped because session-DB CHECK constraints are SQLite-only. The tests' docstrings describe important PostgreSQL-specific behavior: advisory-lock isolation, connection pooling, and dialect equivalence for blob ready-hash constraints.

Impact:

If PostgreSQL session DB support or dialect parity is in scope for the release, these important regressions provide no protection. Because comments say PostgreSQL is deferred, this is Medium rather than High, but it remains a release-readiness gap.

Remediation:

Either formally mark PostgreSQL session DB support out of scope in product/release requirements, or make schema initialization portable enough for the Postgres tests to run. Do not leave parity tests in permanent skip state if the behavior is expected to ship.

Acceptance tests:

Re-enable the two modules behind a testcontainer marker that runs in an appropriate CI lane. Assert advisory-lock session sequencing and malformed hash rejection pass against PostgreSQL.

Confidence: High.

### M-7: External Azure E2E tests are permanent skip stubs despite an Azurite fixture

Locations:

- [test_blob_source.py](/home/john/elspeth/tests/e2e/external/test_blob_source.py:15) lines 15-38
- [test_blob_sink.py](/home/john/elspeth/tests/e2e/external/test_blob_sink.py:15) lines 15-38
- [test_keyvault.py](/home/john/elspeth/tests/e2e/external/test_keyvault.py:18) lines 18-41
- [azurite.py](/home/john/elspeth/tests/fixtures/azurite.py:58) lines 58-144

Evidence:

Blob source and sink modules use `pytest.mark.skipif(True)` and each test body calls `pytest.skip()`. Key Vault is environment-gated but every body still calls `pytest.skip()`. A session-scoped Azurite fixture exists and can create temporary containers, but the E2E tests do not use it.

Impact:

Azure Blob source/sink and Key Vault integration behavior have no executable E2E coverage in this suite. Failures in credential handling, emulator compatibility, source/sink IO, and secret resolution can survive CI.

Remediation:

Wire Blob source/sink tests to the Azurite fixtures and gate real-cloud tests on explicit environment variables. For Key Vault, either implement fixture-backed tests using a controlled test vault or mark the module as intentionally not implemented outside normal CI.

Acceptance tests:

Run Blob source/sink E2E tests against Azurite when Azurite is installed. Assert missing blob, empty blob, CSV/JSON reads, new blob writes, and overwrite behavior. For Key Vault, assert configured live tests execute when required env vars are present and otherwise skip once at module setup.

Confidence: High.

### M-8: Rate limiter property tests depend on wall-clock sleeps and disabled deadlines

Locations:

- [test_rate_limiter_state_machine.py](/home/john/elspeth/tests/property/core/test_rate_limiter_state_machine.py:47) lines 47-51
- [test_rate_limiter_state_machine.py](/home/john/elspeth/tests/property/core/test_rate_limiter_state_machine.py:166) lines 166-209
- [test_rate_limiter_state_machine.py](/home/john/elspeth/tests/property/core/test_rate_limiter_state_machine.py:277) lines 277-349
- [test_rate_limit_fairness.py](/home/john/elspeth/tests/property/core/test_rate_limit_fairness.py:61) lines 61-100
- [test_rate_limit_fairness.py](/home/john/elspeth/tests/property/core/test_rate_limit_fairness.py:107) lines 107-150
- [test_rate_limit_fairness.py](/home/john/elspeth/tests/property/core/test_rate_limit_fairness.py:152) lines 152-201

Evidence:

The state machine and fairness properties use real `time.sleep()`, multi-window waits, threads, and `deadline=None`.

Impact:

Hypothesis explores fewer cases per unit time, failures become scheduler/timing dependent, and CI runtime can grow unpredictably. Fairness failures may be flaky rather than reproducible.

Remediation:

Inject a controllable monotonic clock into `RateLimiter` or provide a test-only clock adapter. Advance virtual time in properties instead of sleeping. For thread fairness, separate deterministic scheduling/unit tests from a smaller stress test marked as slow.

Acceptance tests:

Property tests should run without `deadline=None` for deterministic clock-based cases. Stress tests may remain slow but should have fixed bounds and stable failure reproduction.

Confidence: High.

### M-9: Azure Blob source duplicates probative lifecycle facts into normal-path info logs

Locations:

- [azure_blob_source.py](/home/john/elspeth/src/elspeth/plugins/sources/azure_blob_source.py:467) lines 467-486
- [azure_blob_source.py](/home/john/elspeth/src/elspeth/plugins/sources/azure_blob_source.py:516) lines 516-526
- [azure_blob_source.py](/home/john/elspeth/src/elspeth/plugins/sources/azure_blob_source.py:788) lines 788-789
- [azure_blob_source.py](/home/john/elspeth/src/elspeth/plugins/sources/azure_blob_source.py:857) lines 857-858
- [azure_blob_source.py](/home/john/elspeth/src/elspeth/plugins/sources/azure_blob_source.py:962) line 962

Evidence:

Azure Blob download success is recorded through `ctx.record_call`, but the normal path also logs `blob_downloaded` with blob/container/size. CSV, JSON, and JSONL parse paths log row or line counts with blob paths.

Impact:

These logs are not row-content leaks, but they create a parallel operator-observability loop for source identifiers and parse counts. They can drift from audit/telemetry and encourage decisions from logger output rather than Landscape-backed facts.

Remediation:

Move source count/size facts into audit or telemetry after successful audit recording. Remove or demote normal-path info logs that carry probative source lifecycle facts. Keep logs for infrastructure lifecycle and telemetry/audit failure cases.

Acceptance tests:

Add a unit or static policy test proving normal Azure Blob success and parse paths do not emit info logs for blob identifiers/counts while equivalent facts remain available through audit or telemetry.

Confidence: Medium.

## Low Findings

### L-1: Shared runtime plugin construction is owned by a CLI-named module with CLI backedges

Locations:

- [preflight.py](/home/john/elspeth/src/elspeth/web/execution/preflight.py:1) lines 1-15
- [preflight.py](/home/john/elspeth/src/elspeth/web/execution/preflight.py:113) lines 113-142
- [service.py](/home/john/elspeth/src/elspeth/web/execution/service.py:1168) lines 1168-1172
- [cli_helpers.py](/home/john/elspeth/src/elspeth/cli_helpers.py:1) lines 1-72
- [cli_helpers.py](/home/john/elspeth/src/elspeth/cli_helpers.py:275) lines 275-331
- [preflight.py](/home/john/elspeth/src/elspeth/engine/orchestrator/preflight.py:146) lines 146-158

Evidence:

Web preflight and execution import runtime construction helpers from `cli_helpers.py`. That file is documented as CLI helper code but also hosts shared production plugin construction and a `bootstrap_and_run()` path that imports from `elspeth.cli`.

Impact:

Runtime assembly cohesion is weak. Future CLI-specific changes can unintentionally affect web/runtime behavior.

Remediation:

Move `PluginBundle`, plugin instantiation, sink factory construction, and shared runtime assembly into a neutral `engine` or `core` runtime factory module. Keep CLI-only command UX and secret loading in CLI modules.

Acceptance tests:

`rg -n "from elspeth.cli_helpers" src/elspeth/web src/elspeth/engine` should no longer find runtime construction imports. `cli_helpers.py` should no longer import `elspeth.cli`. Existing CLI and web plugin-instantiation tests should still pass.

Confidence: Medium-high.

### L-2: Composer MCP `new_session` accepts non-string names despite declaring a string schema

Locations:

- [server.py](/home/john/elspeth/src/elspeth/composer_mcp/server.py:108) lines 108-121
- [server.py](/home/john/elspeth/src/elspeth/composer_mcp/server.py:429) lines 429-440
- [session.py](/home/john/elspeth/src/elspeth/composer_mcp/session.py:129) lines 129-140
- [state.py](/home/john/elspeth/src/elspeth/web/composer/state.py:58) lines 58-74

Evidence:

The MCP tool schema declares `new_session.name` as a string. The dispatcher checks only whether `"name"` exists, then passes the value directly into `SessionManager.new_session()`. `PipelineMetadata` has type annotations but no runtime validation.

Impact:

Tier 3 MCP input can create composer session state that violates the advertised schema and internal metadata contract. Session IDs are separately generated/validated, so this is protocol drift rather than an obvious path traversal issue.

Remediation:

Validate composer MCP arguments before dispatch with Pydantic or explicit runtime checks. Reject present `name` unless it is exactly a string. Consider max-length and non-empty constraints if they match the web metadata contract.

Acceptance tests:

Call `new_session` with `{"name": 123}` and `{"name": {"x": "y"}}`. Assert `isError=True`, audit status `ARG_ERROR`, and no session file is created. Keep the positive test for a valid string name.

Confidence: High.

## Remediation Roadmap

Recommended first fixes:

1. Fix H-1 by adding method-general SSRF-safe HTTP calls and binding Azure Search POST/readiness to that path.
2. Fix H-2 by introducing read-only LandscapeDB/analyzer mode for MCP.
3. Fix H-3 and H-4 together in `elspeth-lints`, then add fixtures that prove parse/read failures and nested-scope raises are caught.
4. Fix H-5 and H-6 by centralizing strict external JSON parsing and removing production `response.json()` use from Azure safety transforms.
5. Replace H-7's direct executor retry coverage with a true orchestrator/row-processor integration path.

Recommended second wave:

1. Validate OIDC issuers before discovery network I/O.
2. Unify or explicitly split web/CLI config loading for file-backed template features.
3. Extract a batch-transform runtime protocol to remove the engine-to-plugin-infrastructure dependency.
4. Make SCC computation deterministic or fail closed when NetworkX is missing.
5. Tighten MCP summary ownership checks and Composer MCP argument validation.

Recommended quality cleanup:

1. Re-enable or formally retire PostgreSQL session DB regression modules.
2. Wire Azure Blob E2E tests to the Azurite fixture.
3. Replace wall-clock rate-limiter property tests with virtual-time tests.
4. Move Azure Blob lifecycle facts out of normal-path info logs and into audit/telemetry surfaces.

## Verification And Residual Risk

This audit verified source evidence by local read-only file inspection. It did not execute tests, run the analyzer, run MCP stdio end-to-end, start services, contact staging, or mutate databases.

Residual risk remains in paths that only manifest under live transport, live SQLite filesystem permissions, SQLCipher/PostgreSQL variants, external Azure services, OpenRouter/Azure provider responses, or staging configuration.

No repository files were changed to produce this report.

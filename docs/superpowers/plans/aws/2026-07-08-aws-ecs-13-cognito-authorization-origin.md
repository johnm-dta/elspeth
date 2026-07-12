# AWS ECS Cognito Authorization-Code And Exact-Origin Support Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:using-git-worktrees, superpowers:test-driven-development, and
> superpowers:executing-plans. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the design's recommended Cognito/OIDC browser login executable
without weakening ELSPETH's redirect, callback, or JWT claim-validation
boundaries, and replace the browser's implicit token grant with authorization
code + PKCE.

**Problem:** `validate_oidc_authorization_endpoint()` currently accepts only an
authorization endpoint whose origin exactly equals the issuer origin. That is a
safe default for generic OIDC/Entra, but a Cognito user-pool issuer and its
operator-selected hosted authorization domain are distinct origins. Cognito
access tokens also bind the app client in `client_id` rather than the generic
OIDC `aud` claim. Finally, `LoginPage` currently requests
`response_type=token`, receives a replayable access token in the URL fragment,
sends a nonce it never verifies, and accepts an undocumented bearer token in a
query parameter. OAuth 2.0 Security BCP (RFC 9700 section 2.1.2) and AWS's
current Cognito guidance both recommend authorization code + PKCE for browser
public clients. A production-recommended Cognito path cannot retain the
implicit grant.

**Architecture:** Add an optional, closed operator allowlist of exact HTTPS
browser-OIDC origins to `WebSettings`. The default remains empty and preserves
the current issuer-origin rule. Cross-origin authorization and token endpoints
are accepted only when their normalized `(scheme, ASCII hostname,
effective_port)` tuple exactly matches one explicitly configured origin; the
token endpoint must also share the accepted authorization endpoint's exact
origin. No wildcard, suffix, registrable-domain, redirect-chain, path-prefix,
or hostname inference is allowed. The allowlist is deployment configuration,
not caller input, and is never returned by the auth config API. The validated
authorization and token endpoints are returned because the browser needs both.

The browser uses a public client with no client secret and
`response_type=code`. It creates a transaction-specific, cryptographically
random state and PKCE verifier, sends only the S256 challenge, stores one
single-use transaction in `sessionStorage`, and removes that transaction plus
the callback query before accepting or rejecting the callback. It exchanges a
matching code directly with the validated token endpoint using
`credentials: "omit"`, `redirect: "error"`, `cache: "no-store"`, and
`referrerPolicy: "no-referrer"`; only a shaped bearer `access_token` is
adopted. The authorization code, PKCE verifier, ID/refresh tokens, and raw token
response are never intentionally logged or persisted. The accepted access
token is handed to the existing auth store and follows its current
`localStorage` session-persistence policy; changing that product-wide session
contract is not smuggled into this Cognito compatibility plan.
Cognito necessarily returns the short-lived PKCE-bound code in the callback
query; ELSPETH disables Uvicorn access logging so it is not copied into the
container log and scrubs the browser URL immediately. Any upstream ALB access
logging is therefore an explicit operator retention surface, not something the
application can redact. OAuth errors and malformed responses produce static
messages.

Add an explicit JWT audience-claim mode whose default remains `aud`; Cognito
operators select `client_id`, which additionally requires
`token_use == "access"`. Signature, issuer, expiry, algorithm, and key-ID
validation stay enabled. Because PyJWT validates `exp` only when present, both
modes explicitly require `exp`; the plan must not claim expiry protection while
accepting a token with no expiry.

**Depends on:** Plan 01 (both edit `WebSettings` and environment-ingestion
tests) and the shared signed-tier/Wardline verification baseline
`elspeth-8166b310e7` (Plan 13 cannot honestly close while mandatory gates are
known-broken). Plan 10 Task 3's Cognito browser harness and Plan 12 live
acceptance depend on this plan. Plan 13 must be committed before Plan 10 binds
its pre-packaging rollback baseline.

**Standards basis:** RFC 9700 section 2.1.2; RFC 7636; AWS Cognito
"Authorization endpoint", "Token endpoint", "Using PKCE in authorization code
grants", and "Verifying JSON web tokens". The app client must be a public
client with no secret, allow only the authorization-code OAuth flow for this
acceptance path, and support the exact callback URL and requested scopes.

**Tech Stack:** Pydantic v2, `urllib.parse`, FastAPI lifespan, PyJWT, React,
Web Crypto, Vitest, pytest, Wardline.

---

### Task 0: Atomic ownership and live preconditions

- [ ] From repository root, require `release/0.7.1`, inspect
  `git status --short`, and preserve all unrelated changes. Use the repository's
  worktree workflow to create a clean isolated implementation worktree at the
  live release tip; do not stash, reset, or stage another worker's files.
- [ ] Read `filigree show elspeth-5e729216f4 --json` and require Plan 01
  (`elspeth-b9e8b5d24b`) plus verification baseline
  (`elspeth-8166b310e7`) closed. Atomically start this exact issue:
  `filigree start-work elspeth-5e729216f4 --assignee <agent> --actor <agent>`.
  If it is already owned, continue only when the exact assignee is this worker;
  otherwise stop on `CONFLICT`. Do not use claim-plus-status.
- [ ] Record `BASE_SHA=$(git rev-parse HEAD)` and the pre-existing dirty path
  set. Re-read Plans 01, 10, and 12 before editing so the configuration names,
  browser harness, and rollback-baseline contract stay aligned.

---

### Task 1: Exact authorization/token endpoint validation

**Files:**

- Modify: `src/elspeth/web/auth/urls.py`
- Create or modify: `tests/unit/web/auth/test_urls.py`

**Interfaces:**

```python
validate_oidc_browser_endpoints(
    authorization_endpoint: str,
    token_endpoint: str,
    *,
    issuer: str,
    allowed_origins: tuple[str, ...] = (),
) -> tuple[str, str]
```

- [ ] RED: preserve same-origin acceptance and cross-origin rejection when
  `allowed_origins` is empty. Add exact-allowlist acceptance for issuer
  `https://cognito-idp.ap-southeast-2.amazonaws.com/pool-id` plus
  `https://example.auth.ap-southeast-2.amazoncognito.com/oauth2/authorize` and
  `/oauth2/token` endpoints.
- [ ] Add table-driven adversarial tests rejecting HTTP; blank/control-bearing
  values; backslashes; malformed percent escapes; invalid ports; embedded
  credentials; fragments or queries on either endpoint; an endpoint without a
  non-root path; literal loopback/link-local/private/metadata IPs; wildcard
  hosts; a trailing-dot host; raw Unicode hostnames (operators must configure
  the exact lower-case ASCII A-label); wildcard/underscore/empty DNS labels;
  legacy numeric IPv4 spellings such as `127.1`, octal/hex/integer forms; IPv6
  zone IDs; and an allowlist entry with a path beyond
  `/`, params/query/fragment, username, or password.
- [ ] Add equality-confusion tests for sibling/subdomain and suffix values,
  alternate ports, default `:443`, mixed host case, IPv6 brackets, Unicode
  versus punycode, user-info `@`, encoded separators, authorization/token
  origin mismatch, and an unallowlisted initial endpoint that embeds an allowed
  URL in its path/query. This pure validator performs no DNS or HTTP request;
  it authorizes only the initial browser destination and cannot constrain
  later redirects performed by Cognito or a federated IdP.
- [ ] Error assertions require static field/check names and forbid the full
  endpoint, query, credentials, code, verifier, or other raw input.
- [ ] Run `uv run pytest tests/unit/web/auth/test_urls.py -v` and require the
  new Cognito/paired-endpoint tests to fail before implementation.
- [ ] Implement one private bare-origin parser and one endpoint parser. Origins
  are absolute HTTPS URLs with no component beyond optional `/`; endpoints are
  absolute HTTPS URLs with a non-root path and no query/fragment/credentials.
  Reject non-ASCII/trailing-dot/wildcard/non-DNS hosts and browser/parser
  differential spellings; apply the existing literal-IP SSRF policy and
  canonicalize an accepted public IP with `ipaddress`; lowercase the ASCII
  hostname; treat omitted port and `:443` as equal. A non-default HTTPS port is
  accepted only when explicitly present and equal in the allowlist plus both
  endpoints; a mismatched or merely omitted alternate port rejects. Compare
  immutable origin tuples by equality. Do not resolve DNS or implement
  suffix/IDNA guessing. Parse and validate every allowlist entry, including
  when both endpoints already share the issuer origin, and reject normalized
  duplicates.
- [ ] Preserve issuer-origin acceptance. Otherwise require exact allowlist
  membership for both endpoints, then require authorization and token endpoint
  origins to equal one another. Return the stripped endpoint strings, never a
  redirect target.
- [ ] GREEN: `uv run pytest tests/unit/web/auth/test_urls.py -v` passes.

---

### Task 2: Settings, discovery, and public config wiring

**Files:**

- Modify: `src/elspeth/web/config.py`
- Modify: `src/elspeth/web/app.py`
- Modify: `src/elspeth/web/auth/routes.py`
- Modify: `src/elspeth/cli.py`
- Modify: `src/elspeth/web/frontend/index.html`
- Modify: `tests/unit/web/test_config.py`
- Modify: `tests/unit/web/test_app.py`
- Modify: `tests/unit/web/auth/test_routes.py`
- Modify: `tests/unit/cli/test_web_command.py`

**Interfaces:**

- `WebSettings.oidc_authorization_allowed_origins: tuple[str, ...] = ()`;
  environment input is a JSON array through
  `ELSPETH_WEB__OIDC_AUTHORIZATION_ALLOWED_ORIGINS` and the existing generic
  collection loader.
- `WebSettings.oidc_token_endpoint: str | None = None`.
- `_validate_authorization_endpoint_discovery_document(...)` becomes a
  browser-endpoint discovery validator that requires and returns both
  `authorization_endpoint` and `token_endpoint`.
- `AuthConfigResponse.token_endpoint: str | None = None`.

- [ ] RED settings/env tests: default empty; JSON array becomes an explicitly
  set tuple; malformed JSON, JSON `null`, non-list JSON, non-string elements,
  duplicates after normalization, and invalid entries fail with the
  environment variable/field name but no raw value. Keep the structural
  `_JSON_COLLECTION_FIELDS` completeness test green rather than adding a
  special-case parser.
- [ ] Add a collection validator in `urls.py` backed by the same private origin
  parser and a `WebSettings` field validator that runs at model construction.
  Discovery is not required to discover a malformed allowlist: every entry is
  validated and canonicalized even when no explicit endpoint is configured or
  when the endpoint is same-origin.
- [ ] RED applicability tests: the non-empty allowlist is accepted only for
  `auth_provider="oidc"`. Local rejects explicit browser endpoints. OIDC and
  Entra may configure an explicit pair, but Entra retains an empty allowlist
  and accepts only an issuer-origin pair. Preserve Entra's discovered
  same-origin behavior.
- [ ] RED pair tests: explicit OIDC configuration supplies both endpoints or
  neither; missing half fails. Same-origin pairs pass. Cross-origin pairs fail
  without the exact allowlist and pass with it. Keep `evil.example.com`
  rejection green.
- [ ] RED lifespan tests: discovery must provide nonblank string
  `issuer`, `authorization_endpoint`, and `token_endpoint`; the returned issuer
  must exactly equal the configured normalized issuer. Cognito cross-origin discovery
  fails without and succeeds with the exact allowlist; same-origin OIDC and
  Entra discovery stay green; either malformed/missing endpoint fails startup
  with static output. Mock redirects as failure—startup validation never
  follows an endpoint redirect.
- [ ] Validate explicit endpoint pairs in `WebSettings`, and pass the allowlist
  to the lifespan discovery validator. If an explicit pair is present, use it;
  otherwise discover both. Do not derive `/oauth2/token` from the authorization
  path or infer Cognito/custom domains.
- [ ] Return the validated authorization and token endpoints from
  `GET /api/auth/config`. Add exact response-schema tests proving the allowlist
  and `oidc_audience_claim` are absent. The endpoints and client ID are public
  OAuth metadata; operator policy fields are not. Remove raw collection values
  from `_settings_from_env()` errors and enable/wrap Pydantic's hide-input
  posture so malformed JSON, credentials, endpoint queries, and allowlist input
  never appear in exception/startup output; tests require only the environment
  variable, field, and failed check names.
- [ ] RED/green CLI test: `elspeth web` calls `uvicorn.run(...,
  access_log=False)` so Cognito's unavoidable `?code=...&state=...` callback is
  not copied into container stdout. Keep application security/audit events;
  this disables only Uvicorn's raw request-line access logger. Plan 10's
  runbook must warn that enabling upstream ALB access logs retains PKCE-bound
  callback codes and needs an approved short retention/access policy.
- [ ] Add an app/header regression requiring `Referrer-Policy: no-referrer` on
  the SPA callback document so same-origin subresource/API requests cannot copy
  `?code=...&state=...` into a `Referer` header before React scrubs the URL.
  Require callback responses to be non-cacheable. These are defense in depth;
  the code remains short-lived and verifier-bound, and the query is still
  removed synchronously by the client.
- [ ] Remove the static CSP meta tag from `frontend/index.html`; its current
  `connect-src 'self' ...` would block the planned cross-origin token POST and
  jsdom does not enforce it. Serve the equivalent CSP as an HTTP response
  header from `app.py`, preserving every existing directive and adding only
  the exact normalized validated token endpoint origin to `connect-src` when
  it differs from the app origin. Never widen to `https:`, a wildcard, the
  whole allowlist, or an unvalidated discovery value. Add local/OIDC/Entra/
  Cognito response-header tests and a built-index test proving the conflicting
  meta policy is absent.
- [ ] GREEN: run
  `uv run pytest tests/unit/web/auth/test_urls.py tests/unit/web/test_config.py tests/unit/web/test_app.py tests/unit/web/auth/test_routes.py -v`.

---

### Task 3: Authorization code + PKCE browser transaction

**Files:**

- Modify: `src/elspeth/web/frontend/src/types/index.ts`
- Modify: `src/elspeth/web/frontend/src/components/auth/LoginPage.tsx`
- Modify: `src/elspeth/web/frontend/src/components/auth/LoginPage.test.tsx`

**Transaction contract:** replace the loose `oidc_state` entry with one
versioned `oidc_transaction` JSON object containing only state, PKCE verifier,
and creation timestamp. It is session-scoped, expires after exactly 5 minutes,
and is removed synchronously before any match decision or network call. The
callback query is copied into bounded in-memory state and immediately scrubbed;
the exchange waits for a fresh, backend-validated `/api/auth/config` response
rather than trusting endpoint/client metadata from mutable browser storage.

- [ ] RED redirect tests: clicking SSO uses the validated endpoints/client ID,
  `response_type=code`, exact redirect URI, requested scopes, a random state,
  and PKCE `code_challenge_method=S256`; the challenge is base64url without
  padding and equals SHA-256 of a 43–128-character verifier. No access-token
  response type, client secret, raw verifier, or unverified nonce is sent.
- [ ] RED callback tests cover matching state, mismatching/missing state,
  missing/malformed transaction JSON, missing/malformed verifier, missing code,
  older-than-5-minute or more-than-30-second-future transaction timestamps,
  provider `error`/`error_description`,
  duplicate or mixed success/error callback parameters, a
  network failure, redirect response, non-2xx response, non-JSON/oversized or
  malformed token response, wrong/non-string `token_type`, and blank/non-string
  `access_token`.
- [ ] For every callback branch assert `oidc_transaction` is removed and the
  query/fragment is replaced before token exchange or adoption. A mismatch or
  malformed callback makes no token request. Static UI/operator errors never
  echo provider descriptions, code, state, verifier, token, response body, or
  endpoint query.
- [ ] The successful exchange is exactly one form-encoded POST containing
  `grant_type=authorization_code`, code, client ID, exact redirect URI, and
  verifier. Require `credentials: "omit"`, `redirect: "error"`,
  `cache: "no-store"`, and `referrerPolicy: "no-referrer"`. Do not send a
  client secret. Adopt only the access token; ignore and never persist an ID or
  refresh token. Bound the token response to 64 KiB before JSON parsing; an
  absent/invalid/oversized `Content-Length` is not trusted, so the reader also
  enforces the limit while consuming/cancelling the response stream. Bound the
  accepted `access_token` string to 16 KiB.
- [ ] Build the authorization request with `new URL()` and
  `URLSearchParams.set()` rather than string concatenation. The validated
  endpoint contract rejects pre-existing query/fragment components; tests pin
  that no duplicate security parameter can be smuggled into the request. The
  mocked unit exchange plus Plan 10's live Cognito browser lane must prove the
  public client's token endpoint permits the CORS request; if the browser
  cannot complete that real exchange, stop rather than falling back to implicit
  tokens or adding a client secret.
- [ ] Add a React `StrictMode` regression proving the single-use transaction
  produces exactly one exchange and one adoption despite development
  double-invocation. Preserve immediate cleanup while fresh auth-config loading
  is outstanding; retain the code/verifier only in memory, make no exchange if
  config loading fails, and use only the freshly returned token endpoint and
  client ID plus a recomputed exact redirect URI. Add valid-shaped tampered
  legacy transaction fields to prove they are ignored/rejected and can never
  redirect the exchange.
- [ ] Remove the legacy implicit-fragment and undocumented `?token=` bearer
  paths completely. Keep email-verification `?verify_token=` behavior separate
  and covered so the OIDC cleanup cannot consume or misclassify it.
- [ ] GREEN: run
  `npm --prefix src/elspeth/web/frontend test -- src/components/auth/LoginPage.test.tsx`
  and `npm --prefix src/elspeth/web/frontend run typecheck`.

---

### Task 4: Cognito access-token audience boundary

**Files:**

- Modify: `src/elspeth/web/config.py`
- Modify: `src/elspeth/web/app.py`
- Modify: `src/elspeth/web/auth/oidc.py`
- Modify: `docs/release/guarantees.md`
- Modify: `tests/unit/web/test_config.py`
- Modify: `tests/unit/web/test_app.py`
- Modify: `tests/unit/web/auth/test_oidc_provider.py`

**Interface:**

- `WebSettings.oidc_audience_claim: Literal["aud", "client_id"] = "aud"`;
  Cognito uses `ELSPETH_WEB__OIDC_AUDIENCE_CLAIM=client_id`.
- `JWKSTokenValidator(..., audience_claim: Literal["aud", "client_id"] = "aud")`.
- `OIDCAuthProvider` threads the same closed mode; Entra does not expose or
  select `client_id` mode.

- [ ] RED settings tests: `client_id` is OIDC-only. Local/Entra reject an
  explicitly configured non-default; default `aud` remains accepted. Invalid
  strings fail Pydantic's closed literal boundary. Assert the mode is not
  present in the public auth-config response.
- [ ] RED token tests: default `aud` matching behavior remains unchanged except
  that missing `exp` is now rejected. `client_id` mode accepts only a
  signed/unexpired/issuer-matching token whose `client_id` is an exact string
  equal to the configured audience and whose `token_use` is exactly `access`.
- [ ] Reject missing/null/bool/list/object/blank/mismatched `client_id`, missing
  or wrong/non-string `token_use`, an `aud`-only token, missing/non-numeric or
  expired `exp`, wrong issuer, wrong signature, wrong/unknown key ID, and
  algorithm mismatch. Add a token containing both claims to prove each mode
  consults only its selected claim and never falls back.
- [ ] In Cognito `client_id` mode require `exp`, `iat`, `iss`, `sub`,
  `client_id`, and `token_use`; require Cognito's documented `RS256` algorithm
  independently of the untrusted JWT header and reject another algorithm even
  when a supplied JWK is key-compatible. Generic `aud` mode retains the
  existing provider/JWK algorithm compatibility boundary.
- [ ] Put the manual `client_id`/`token_use` shape and equality checks in a
  small `@trust_boundary` helper with an exact invariant and regression test.
  All errors must be static/class-only and must not echo JWT segments, `kid`,
  claim values, expected audience, or token contents.
- [ ] In `aud` mode retain `jwt.decode(..., audience=..., issuer=...)`. In
  `client_id` mode call the same verified decode with
  `options={"verify_aud": False, "require": ["exp", "iat", "iss", "sub", "client_id", "token_use"]}`
  and `issuer=...`, then call the helper. Add `require: ["exp"]` to `aud` mode
  too. Do not disable
  signature, issuer, expiry, `nbf`/`iat`-when-present, algorithm, key parsing,
  or key-ID selection.
- [ ] Thread the mode through `create_app()` only for `OIDCAuthProvider`; keep
  `EntraAuthProvider` on the default `aud` path. Add `audience_claim` as a
  keyword-only parameter after the existing cache arguments and pass it by
  keyword so the current positional TTL/retry forwarding cannot shift. Add
  constructor-wiring tests so the setting cannot stop at the model.
- [ ] Strengthen both app-level and JWKS discovery boundaries to require the
  discovery document's exact `issuer` match before accepting browser endpoints
  or `jwks_uri`. Missing, wrong-type, and mismatched issuer tests fail with
  static errors.
- [ ] Update `OIDCAuthProvider.get_user_info` trust-boundary metadata and
  `docs/release/guarantees.md` to describe the actual bearer access-token/JWT
  boundary, configured `aud` versus Cognito `client_id` binding, required
  expiry, and PKCE/state callback controls. Remove the false ID-token/verified-
  nonce promise: this flow intentionally adopts an access token and sends no
  nonce it cannot verify.
- [ ] GREEN: run
  `uv run pytest tests/unit/web/auth/test_oidc_provider.py tests/unit/web/test_config.py tests/unit/web/test_app.py -v`.

---

### Task 5: Static, trust-boundary, regression, and handoff gates

- [ ] Run the complete affected regression set:

  ```bash
  uv run pytest \
    tests/unit/web/auth/test_urls.py \
    tests/unit/web/auth/test_oidc_provider.py \
    tests/unit/web/auth/test_entra_provider.py \
    tests/unit/web/auth/test_routes.py \
    tests/unit/web/test_config.py \
    tests/unit/web/test_app.py \
    tests/unit/cli/test_web_command.py -v
  npm --prefix src/elspeth/web/frontend test -- src/components/auth/LoginPage.test.tsx
  npm --prefix src/elspeth/web/frontend run typecheck
  npm --prefix src/elspeth/web/frontend run lint
  npm --prefix src/elspeth/web/frontend run build
  ```

- [ ] Run repository gates exactly; scoped-only Ruff/mypy is insufficient for
  a new trust boundary:

  ```bash
  uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run mypy src/ elspeth-lints/src/
  uv run python scripts/check_contracts.py
  ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing uv run elspeth-lints diagnose-judge-signatures --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model --format text
  ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing uv run elspeth-lints check --rules trust_tier.tier_model --root src/elspeth
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_boundary.tests,trust_boundary.scope,trust_boundary.tier --root src/elspeth
  git diff --check
  ```

- [ ] Run `wardline scan . --fail-on ERROR`. Exit 1 requires the full
  explain/fix/rescan loop at the configuration, endpoint, callback, or claim
  boundary; exit 2 blocks. Do not baseline or waive a new auth finding.
- [ ] Review `git diff --name-only` against the Task 1–4 allowlist. Stage only
  paths actually changed by this issue; in particular, do not stage
  `LoginPage.tsx` merely because it appears in the plan if another worker owns
  an unrelated edit. In the isolated worktree, require the NUL-delimited
  `BASE_SHA..worktree` changed-path set to equal the staged path set, and
  require both to be subsets of the closed Task 1–4 allowlist; a missing staged
  issue-owned file is a hard stop. Inspect `git diff --cached --check` and
  `git diff --cached --name-status`, then run
  `git diff --cached --name-only -z | xargs -0 uv run pre-commit run --files`.
  Commit atomically with
  `feat(web): use Cognito authorization code with PKCE`. Hooks must pass; never
  bypass them.
- [ ] Add a Filigree comment containing the commit SHA, exact test/gate results,
  and any operator migration note (public client, code grant, no secret, exact
  callback/origin). Close `elspeth-5e729216f4` only after the comment and clean
  staged diff. If any required gate cannot run or fails, comment the exact
  blocker and leave the issue open; do not claim completion. Require
  `git status --short` empty after commit. Finally reread the
  issue and require terminal status, the implementation owner, and the exact
  close commit before handing off.

**Definition of Done:**

- [ ] Default generic OIDC/Entra browser endpoints remain issuer-origin and
  fail closed; a real operator-declared Cognito hosted origin is accepted by
  exact equality only.
- [ ] The allowlist cannot express wildcards, suffixes, paths, unsafe literal
  IPs, Unicode/trailing-dot aliases, or credential-bearing URLs and is absent
  from public auth config.
- [ ] The browser uses authorization code + S256 PKCE with a public client,
  never places an access token or verifier in the authorization request/URL,
  and removes its transaction and callback URL before all decisions.
- [ ] Only one matching callback can exchange/adopt a shaped bearer access
  token; mismatches, OAuth errors, duplicate inputs, redirects, malformed
  responses, and React StrictMode replay fail closed without secret-bearing
  output.
- [ ] Default `aud` validation has an explicit expiry requirement. Cognito
  `client_id` mode additionally requires exact app client ID and
  `token_use=access`; neither mode falls back to the other claim.
- [ ] Local and Entra cannot select OIDC-only origin/audience modes. Direct
  configuration, discovery, generic env ingestion, public config, frontend,
  static/contracts/trust-boundary gates, and Wardline are green.

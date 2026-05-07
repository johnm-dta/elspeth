# RC4.2 UX Remediation — Secret References Implementation Plan

Date: 2026-03-30
Status: Draft
Parent: `docs/plans/rc4.2-ux-remediation/2026-03-30-rc4.2-ux-implementation-plan.md`

---

## 1. Scope

This plan covers assistant-safe secret references, dedicated write-only secret
entry UX, and runtime secret resolution across user, org, and server scopes.

Included requirement:

- `REQ-API-02` — secret references

Primary surfaces:

- secret-resolution contract
- storage backends
- REST API
- composer tool exposure
- execution-time resolution
- system-status and validation integration
- settings/profile UX

This plan does not cover blob storage, session forking, or broader workspace
membership features beyond the minimum org-scope contract needed for future
compatibility.

---

## 2. Current State

The web app currently has no user-facing secret management at all.

What exists today:

- `WebSettings.secret_key` for JWT signing and server safety
- boot-time composer availability checks via LiteLLM environment validation
- core secret-loading utilities for environment variables and Azure Key Vault
- typed audit records for secret resolution provenance in Landscape

What does not exist today:

- user-entered secret inventory
- write-only secret submission UX
- secret references in web-authored pipeline state
- runtime resolution of `{ "secret_ref": "NAME" }` inside web execution
- scoped lookup across user, org, and server layers

The current composer-availability banner is driven entirely by environment
validation for the configured model. It can say "missing OPENAI_API_KEY," but
it cannot yet account for a world where the runtime may satisfy that need from
user/profile or org-backed secret stores.

---

## 3. Goals

- Never expose plaintext secret values to the browser or assistant after
  submission.
- Let users see which secret references are available without seeing values.
- Support write-only submission: once entered, the value is handed off to
  server-side storage and disappears from the web interface.
- Resolve secrets server-side at execution time through scoped lookup.
- Preserve or extend the existing audit story so secret provenance is recorded
  without exposing secret contents.

Non-goals for this phase:

- allowing secrets to be pasted into normal chat as a product pattern
- returning secret values from any read API
- building full workspace/org administration UX beyond the minimum inventory
  and resolution contract
- replacing the existing core Key Vault / env secret infrastructure

---

## 4. Architecture Decisions

### AD-1: Contract-first design, not web-first design

Secret resolution is a cross-cutting runtime concern, not just a web feature.
The web layer should depend on a shared resolution contract rather than invent
its own secret semantics.

Recommended layering:

- protocol in `contracts/` or an equivalent shared contract layer
- reusable resolution helper in `core/`
- web-specific backends and routes in `web/`

This keeps CLI, web execution, and future non-web runners from drifting apart.

The implementation should prefer extending or wrapping the existing
`SecretLoader` / `CompositeSecretLoader` model in `core.security`, not creating
an unrelated parallel hierarchy with overlapping names and semantics.

### AD-2: Secret references are inventory-visible but value-invisible

The browser may list that a secret reference is available, but it must never
receive the corresponding value.

Implications:

- list APIs return names, scopes, capability metadata, and availability
- write APIs accept plaintext only on submission
- after submission, the input is cleared and cannot be read back
- no “show secret” or “copy secret” affordance exists

### AD-3: Secret submission is write-only and immediately externalized

User-submitted secret values should be accepted by the web UI, transmitted to a
server-side secret backend, and then discarded from browser-visible state.

The UI may continue to show:

- secret name
- scope
- status such as available / missing / invalid

The UI may not show:

- the current value
- any masked derivative implying the value can be recovered

### AD-4: Scoped lookup is server-side and policy-driven

Secret resolution must support the following logical scopes:

- `user`
- `org`
- `server`

Recommended default lookup order:

- `user -> org -> server`

However, the policy should live in the resolver, not be duplicated in route
handlers or the frontend.

### AD-5: Secret refs are opaque values in composition state

Pipeline state may contain:

```json
{ "secret_ref": "OPENROUTER_API_KEY" }
```

Composer tools and persisted composition state should treat this as an opaque
reference. They should not attempt to inline or resolve the secret value.

### AD-6: Resolution occurs at runtime immediately before execution

The correct place to resolve secret refs is the server-side execution path
after composition state is loaded and before the pipeline is instantiated or
run.

This keeps plaintext values out of:

- chat messages
- composer tool results
- persisted composition state
- REST responses
- websocket events

### AD-7: Audit extends existing secret provenance patterns

The codebase already has typed secret-resolution audit records and core secret
loader primitives. The web implementation should extend that story rather than
inventing a parallel audit format.

The audit record should capture:

- the logical reference name
- which scope satisfied the lookup
- the backend/source used
- a fingerprint when available

It should not capture plaintext secret values.

### AD-8: v1 implements user + server scopes cleanly and models org scope now

Org scope should exist in the contract and API metadata now, but the first full
implementation can phase delivery as:

- v1: `user` and `server`
- v2: real `org` storage and admin UX

That keeps the runtime contract future-proof without forcing workspace
infrastructure into RC4.2.

---

## 5. Proposed Secret Model

### Secret reference shape

Assistant-safe config shape:

```json
{ "secret_ref": "OPENROUTER_API_KEY" }
```

Optional future metadata shape if needed:

```json
{
  "secret_ref": "OPENROUTER_API_KEY",
  "scope_hint": "user"
}
```

The base model should stay minimal unless a real routing need appears.

### Secret inventory record shape

Minimum browser-safe metadata:

- `name`
- `scope`
- `available`
- `usable_by`
- `source_kind`
- `last_updated_at`

Optional:

- `validation_status`
- `reason_unavailable`

Forbidden fields:

- plaintext value
- masked value
- encrypted blob content

### Secret submission shape

Write-only request shape:

- `name`
- `value`
- `scope`

Write response shape:

- `name`
- `scope`
- `available: true`

The response should behave like an acknowledgement, not a read-back.

---

## 6. Runtime Resolution Design

### 6.1 Resolver contract

Introduce a shared secret-resolution contract that can:

- list available secret references for a principal/context
- validate whether a secret ref is resolvable
- resolve a secret ref to a runtime secret value plus provenance metadata

Recommended conceptual operations:

- `list_refs(principal, scopes)`
- `has_ref(principal, name)`
- `resolve(principal, name)`

`resolve()` should return:

- plaintext value for in-process runtime use
- metadata for audit recording

### 6.2 Backend composition

The resolver should compose multiple backends, for example:

- user-secret backend
- org-secret backend
- server/env backend
- optional Key Vault backend later

This mirrors the existing `CompositeSecretLoader` pattern rather than replacing
it.

Naming/shape constraint:

- avoid introducing a second conflicting `SecretRef` abstraction unless the
  distinction is explicit and necessary
- if separate browser-inventory and runtime-resolution records are needed, they
  should have distinct names

### 6.3 Execution integration

During web execution:

1. load the current composition state
2. generate pipeline YAML or config representation
3. walk the config tree for `secret_ref` objects
4. resolve them through the secret resolver using the current principal
5. inject resolved values only into the execution-local config
6. record audit provenance
7. continue with pipeline validation/instantiation/run

The resolved config must be treated as ephemeral runtime material, not
persisted session state.

### 6.4 Audit integration

The current audit contract `SecretResolutionInput` only allows `source="keyvault"`.
The secret-reference rollout needs a compatible extension before any web secret
resolution work can be considered executable.

This is not a follow-up nicety; it is a phase-1 blocker.

Expected early change:

- extend allowed audit sources to include at least `env`, `user`, and later
  `org`
- keep fingerprint semantics intact
- record scope/backend without ever storing the raw value

This is a shared contract change and should be completed before runtime secret
resolution is wired into execution.

---

## 7. Storage Model

### 7.1 User-scoped secrets

User secrets need a server-side persistence backend that supports:

- write-only creation/update
- list-by-name metadata
- deletion
- server-side resolution

The exact storage backend may be:

- encrypted database records
- OS keychain / secret manager integration
- external vault adapter

For RC4.2, an encrypted server-side persistence layer is acceptable if it is
clearly isolated behind the resolver interface.

Execution seam note:

- because `_run_pipeline()` executes in a worker thread, the user-secret
  backend used there should present a synchronous access surface or an explicit
  thread-bridge design
- do not implicitly assume async request-handler patterns are safe in the
  runtime resolution hot path

### 7.2 Server-scoped secrets

Server secrets already conceptually exist as environment/config-provided
capabilities.

For v1:

- expose inventory metadata for a curated allowlist of server-available secret
  refs
- resolve via environment or configured backend

Do not dump arbitrary environment variable names to the browser.

### 7.3 Org-scoped secrets

Org scope should be modeled in contracts and API shapes now, but actual storage
and administration can be deferred until workspace/team support exists.

The plan should explicitly mark org storage as deferred rather than leaving it
ambiguous.

---

## 8. API Surface

### REST endpoints

Recommended initial endpoints:

- `GET /api/secrets`
  - list visible secret references and scopes for the current user
- `POST /api/secrets`
  - create or update a write-only user secret
- `DELETE /api/secrets/{name}`
  - delete a user-scoped secret
- `POST /api/secrets/{name}/validate`
  - confirm the ref exists and is accessible

Optional later endpoints:

- admin-only server secret inventory refresh
- org secret management endpoints

### Composer tools

Recommended initial tools:

- `list_secret_refs`
- `validate_secret_ref`
- `wire_secret_ref`

Rules:

- no tool may return plaintext secret values
- `wire_secret_ref` should set an opaque ref in pipeline state, not write a
  literal secret
- secret-creation itself is better handled by dedicated UI/API than by chat
- user-secret storage is a dedicated REST/UI operation, not a composer/chat tool

---

## 9. System Status and Validation Integration

### 9.1 Boot-time system status

The current `/api/system/status` endpoint reports composer availability based on
LiteLLM environment validation only.

With secret references, the system-status story should become:

- if the configured composer can be satisfied by available server/runtime
  secret sources, report available
- if not, return a precise reason such as missing required secret ref

This is especially important once the model may be satisfiable via a secret
store rather than a raw environment variable.

### 9.2 Composition validation

Validation should be able to report:

- secret ref exists
- secret ref is missing
- secret ref exists but is unavailable to the current user/context

This should remain metadata-only validation, not a value fetch exposed to the
browser.

### 9.3 Execution-time failure shape

If a run reaches execution and a secret still cannot be resolved, the failure
should be explicit and user-readable:

- which secret ref could not be resolved
- which scope/policy was attempted
- that no secret value was exposed

---

## 10. Frontend Design

### 10.1 Dedicated secrets UI

The secret-entry experience should live in a dedicated settings/profile/admin
surface, not the main chat flow.

Required UX:

- enter secret name and value
- submit once
- clear the value field immediately
- continue showing the secret name as available

### 10.2 Inventory view

The inventory should show:

- secret name
- scope
- availability state
- optional “used by provider” metadata

The inventory should not imply that values can be retrieved.

### 10.3 Browser-state discipline

Frontend state stores must not retain plaintext values after submission.

Implications:

- do not persist secret values in Zustand beyond the live form state
- clear the form immediately after successful submission
- do not include values in optimistic caches or debug logs

---

## 11. Implementation Phases

### Phase 1: Shared contract and audit groundwork

- define resolver protocol in shared layers
- define browser-safe inventory models
- extend audit source model for non-Key Vault secret references
- decide explicitly whether the resolver is an extension of existing
  `SecretLoader` semantics or a superset wrapper around them

Deliverable:

- a single secret-resolution contract the web app can build on safely

### Phase 2: Server and user backend implementation

- implement user-secret persistence backend
- implement curated server-secret inventory/resolution backend
- compose them behind a resolver service
- ensure the execution-facing user-secret backend is safe to call from the
  worker-thread execution path
- define chained inventory behavior:
  - deduplicate by secret name
  - highest-priority scope wins when the same name exists in multiple backends

Deliverable:

- the backend can list and resolve secrets without exposing values

### Phase 3: Web API and settings/profile UX

- add secret routes and schemas
- add dedicated UI for write-only entry and inventory display
- ensure the browser clears secret inputs after submission
- keep secret creation/update on REST/UI only; never route plaintext secret
  entry through chat/composer tooling

Deliverable:

- users can submit and manage secret references safely from the web UI

### Phase 4: Composer and validation integration

- add secret-ref composer tools
- add validation checks for secret refs
- update system-status readiness logic to account for resolver-backed secrets
- define `resolve_secret_refs()` failure semantics:
  - collect all missing/unresolvable refs in one pass where practical
  - raise one clear error listing the missing refs rather than failing one name
    at a time

Deliverable:

- the assistant can reference secrets safely and the UI can explain readiness

### Phase 5: Execution integration

- resolve secret refs in the web execution path
- feed provenance into audit recording
- ensure resolved values remain execution-local and ephemeral

Deliverable:

- web-authored pipelines can execute with scoped secret resolution

### Phase 6: Org-scope follow-up

- implement real org secret backend and admin model when workspace/team support
  exists

Deliverable:

- scoped resolution contract extends cleanly without rewriting the earlier work

---

## 12. File-Level Work

### Shared/core

Expected modified files:

- `src/elspeth/contracts/audit.py`
- `src/elspeth/core/security/secret_loader.py`
- new shared secret-resolution contract/helper modules as needed

Potential new files:

- `src/elspeth/contracts/secrets.py`
- `src/elspeth/core/security/runtime_secret_resolver.py`

Implementation notes:

- any runtime `ResolvedSecret` type must override `__repr__`/debug rendering so
  plaintext values cannot leak via logs or tracebacks
- audit extension tests should verify existing Key Vault sources still validate
  while new `env`/`user` sources are accepted

### Web backend

Expected new files:

- `src/elspeth/web/secrets/routes.py`
- `src/elspeth/web/secrets/protocol.py`
- `src/elspeth/web/secrets/service.py`
- `src/elspeth/web/secrets/schemas.py`

Expected modified files:

- `src/elspeth/web/app.py`
- `src/elspeth/web/config.py`
- `src/elspeth/web/execution/service.py`
- `src/elspeth/web/composer/tools.py`
- `src/elspeth/web/execution/validation.py`
- session DB models/protocols if user-secret persistence lives there

### Frontend

Expected new files:

- `src/elspeth/web/frontend/src/components/settings/SecretsPanel.tsx`
- `src/elspeth/web/frontend/src/stores/secretsStore.ts`

Expected modified files:

- `src/elspeth/web/frontend/src/api/client.ts`
- `src/elspeth/web/frontend/src/types/index.ts`
- top-level settings/profile layout that hosts the panel
- `src/elspeth/web/frontend/src/App.tsx` if system-status messaging changes

---

## 13. Testing Plan

### Backend tests

- secret creation/update does not expose value in response bodies
- list endpoints return metadata only
- validation correctly distinguishes available vs missing refs
- resolution follows the configured scope order
- execution receives resolved secrets without persisting them
- audit records capture provenance without plaintext values
- `resolve_secret_refs()` tree-walk unit tests cover nested structures and
  aggregate missing-ref errors
- chained inventory tests verify dedup by name with highest-priority scope
  winning
- worker-thread safety tests cover execution-path reads from the user-secret
  backend
- audit contract regression tests verify old and new secret source types

### Frontend tests

- secret value field clears immediately after successful submission
- inventory continues to show secret availability after submission
- browser state does not retain the value once the request completes
- system-status banner updates correctly when resolver-backed availability
  changes

### Manual checks

1. Create a user secret and confirm the value disappears from the UI
   immediately after submission.
2. Refresh the page and confirm the secret still appears as available by name
   only.
3. Use a pipeline with `{ "secret_ref": "..." }` and confirm execution works
   without exposing the value in chat, YAML, or inspector surfaces.
4. Remove the secret and confirm readiness/validation surfaces report the
   missing reference clearly.

---

## 14. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Secret values accidentally leak into browser-visible state | High | Make write-only handling and response-shape discipline explicit from the start |
| Web invents a resolver path that diverges from CLI/core secret handling | High | Put the contract in shared layers and reuse core loader patterns |
| Server-secret inventory becomes an environment-variable leak | High | Use a curated allowlist of exposed server refs, not arbitrary env introspection |
| Audit contract changes ripple into existing secret-resolution logic | Medium | Extend existing typed audit records carefully rather than replacing them |
| Org scope creates design churn before workspace support exists | Low | Model org in contracts now, defer storage/admin implementation cleanly |

---

## 15. Sequencing

Recommended order inside this subplan:

1. Shared contract and audit groundwork
2. User/server resolver backends
3. Web API and write-only UI
4. Composer and validation integration
5. Execution-time resolution
6. Org-scope follow-up

That order keeps the most security-sensitive decisions first and prevents the
web UX from racing ahead of the runtime and audit contract.

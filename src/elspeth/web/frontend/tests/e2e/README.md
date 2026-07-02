# Frontend E2E tests (Playwright)

End-to-end tests for the ELSPETH composer UI. Boots the FastAPI backend +
Vite dev server, exercises real browser flows, and verifies invariants
that single-component (Vitest + jsdom) tests cannot.

## Running locally

From `src/elspeth/web/frontend/`:

```bash
# One-time: install the chromium binary and its system deps.
npm run test:e2e:install

# Run the suite headless.
npm run test:e2e

# Interactive UI mode for debugging.
npm run test:e2e:ui

# A single spec.
npx playwright test smoke.spec.ts
```

Playwright owns the lifecycle of both servers via `webServer:` in
`playwright.config.ts`. The backend is always started by Playwright so the
E2E-only auth policy and `.e2e-data` store are applied. Local runs may still
reuse an existing Vite frontend dev server.

## Layout

```
tests/e2e/
├── README.md                    (this file)
├── setup/
│   └── global-setup.ts          registers a test user, writes storageState
├── helpers/
│   └── api.ts                   typed REST helpers (no UI driving)
├── page-objects/
│   ├── composer-page.ts         left/main panel (chat + empty state)
│   └── catalog-page.ts          plugin catalog drawer
└── *.spec.ts                    one file per logical scenario
```

## What's working today

`smoke.spec.ts` exercises the full boot path:

1. globalSetup ran and obtained a JWT.
2. Both webServers came up healthy.
3. SPA loads, restores auth, renders empty composer.
4. `/api/sessions` accepts the bearer token (round-trip create+delete).
5. Hash routes resolve to a session id.

This is the proof-of-life that the harness itself works. If `smoke.spec.ts`
fails, fix the harness before debugging any other spec.

## What's stubbed (tracked test.skip)

The remaining specs target the composer-correctness epic
[`elspeth-e1ab67e55a`](../../../../../../) and its sub-issues. Five of them
are gated with a describe-level `test.skip(true, "…tracked as
elspeth-3a7df642c5")` — NOT `test.fixme`, which silently reports as passing;
a tracked skip is CI-visible in the run summary — and detailed step-by-step
bodies so a contributor can pick them up without re-deriving intent:

| Spec                              | Targets                                | Blocker (tracked)                                            |
|-----------------------------------|-----------------------------------------|---------------------------------------------------------------|
| `topology.spec.ts`                | `elspeth-3724f02de9` (closed)          | `elspeth-3a7df642c5` state-seed                               |
| `mandatory-fields.spec.ts`        | `elspeth-39089c98ee` (closed)          | `elspeth-3a7df642c5` state-seed                               |
| `schema-preview-parity.spec.ts`   | `elspeth-87f6d5dea5` (open, P2)        | `elspeth-3a7df642c5` + expected-fail on `elspeth-87f6d5dea5`  |
| `yaml-export-roundtrip.spec.ts`   | parent epic acceptance criterion       | `elspeth-3a7df642c5` state-seed                               |
| `compose-happy-path.spec.ts`      | through-UI proof; `elspeth-528bde62bb` | `elspeth-3a7df642c5` LLM stub server                          |

`llm-provider-schema.spec.ts` is different: its describe-level `test.fixme()`
gate was already removed (per plan 16c Task 6 Step 3a) — most of its tests run
on every CI pass, with a single per-test `test.skip(!hasStateSeed, "state-seed
gap — see elspeth-dcf12c061b")` for the one path still needing the
add-node-without-LLM affordance.

### The two unblockers

1. **Direct state-mutation REST endpoint.** The composer's `upsert_node`,
   `upsert_edge`, `set_source`, etc. are LLM tools, not REST endpoints.
   For E2E seeding, we need either a `POST /api/sessions/{id}/state`
   accepting canonical state JSON, OR a test-only endpoint behind a
   feature flag. Either way the seam lives in the engine, not in the
   frontend tests.

2. **JS-side LLM stub server.** The pytest suite uses the `ChaosLLM`
   fixture (under `tests/`) — a real HTTP server impersonating an LLM
   provider that returns scripted responses. Porting that to a JS shape
   the Playwright `webServer` can dial via `ELSPETH_WEB__composer_model`
   env override unblocks `compose-happy-path.spec.ts` and would also let
   the seeding-blocked specs drive through the LLM as a fallback.

## Selectors

These specs prefer Playwright's accessibility-first locators —
`getByRole`, `getByLabel`, `getByText` — over `data-testid`. The composer
components are already richly aria-labelled (`role="tab"` with explicit
`aria-label`s, `aria-haspopup`, etc.), so role-based selectors disambiguate
without brittle CSS coupling. Only add `data-testid` when no semantic
selector can disambiguate.

## CI

A `e2e-frontend` GitHub Actions job runs the suite on every PR with
`continue-on-error: true` for the first 2 weeks (informational baseline).
After ≥99% green pass-rate, flip the gate. See `.github/workflows/ci.yaml`
and the follow-up filigree task created during Step 4 of the install plan.

## Debugging a CI failure

1. Download the `playwright-report/` artifact from the failed job.
2. Locally: `npx playwright show-trace path/to/trace.zip`.
3. The trace replays the full DOM, network, and console — frame-by-frame.

// Client-side abort budget for composer requests — the freeform
// POST /api/sessions/{id}/messages call and the guided step-chat call.
//
// INVARIANT: this ceiling sits ABOVE the backend compose wall clock
// (ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS) plus grace, so the backend's
// discriminated 422 (convergence_wall_clock_timeout) always arrives before
// the client aborts. The client abort is a last-resort guard against a
// truly dead connection, not a UX pacing mechanism — the composer progress
// panel is what tells the user the turn is alive.
//
// The wall clock is deployment-configurable with NO fixed maximum (only
// transport-ceiling headroom — see WebSettings._validate_composer_timeout_
// transport_headroom), so a hard-coded constant can only satisfy the
// invariant for the checked-in defaults. The SPA therefore derives the
// ceiling at boot from GET /api/system/status (composer_timeout_seconds)
// via applyServerComposerTimeout; the default below covers the window
// before that fetch lands and deployments running the checked-in 270s.
//
// History: 295s (backend 270s + 25s grace) for big-bang compose; lowered
// to 90s on 2026-07-03 on the premise that subphase-era turns "never
// legitimately run that long". The 2026-07-10 web eval battery falsified
// that premise 4/4: freeform multi-tool builds (11-22 tool calls) run
// minutes while healthy, and the 90s abort killed every one of them —
// then left the session wedged behind a zombie server turn
// (elspeth-e08063c3a5). The server now also cancels the turn when the
// client disconnects, so an abort at this ceiling (or Stop) actually
// stops the work instead of orphaning it.
export const COMPOSE_CLIENT_GRACE_MS = 25_000;
export const DEFAULT_COMPOSE_TIMEOUT_MS = 270_000 + COMPOSE_CLIENT_GRACE_MS;

let composeTimeoutMs = DEFAULT_COMPOSE_TIMEOUT_MS;

/** Current compose abort ceiling. Read at CALL time (useComposer), never
 * cached at module load, so the boot-applied server value governs every
 * send that starts after it lands. */
export function getComposeTimeoutMs(): number {
  return composeTimeoutMs;
}

/**
 * Derive the abort ceiling from the backend's configured compose wall
 * clock (seconds, from GET /api/system/status). Non-finite or non-positive
 * values are ignored — the current ceiling (default 295s) is a safe floor,
 * and a garbage value must not shrink the guard below the backend wall.
 */
export function applyServerComposerTimeout(
  backendTimeoutSeconds: number,
): void {
  if (
    !Number.isFinite(backendTimeoutSeconds) ||
    backendTimeoutSeconds <= 0
  ) {
    return;
  }
  composeTimeoutMs =
    Math.round(backendTimeoutSeconds * 1000) + COMPOSE_CLIENT_GRACE_MS;
}

// Abort reasons are internal frontend control-plane values. They let
// sessionStore distinguish a user-requested stop from the timeout guard while
// still using the browser's native AbortController path.
export const COMPOSE_TIMEOUT_ABORT_REASON = "compose_timeout";
export const COMPOSE_USER_CANCEL_ABORT_REASON = "compose_user_cancel";

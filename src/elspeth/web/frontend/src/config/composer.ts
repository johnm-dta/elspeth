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

// User-facing reasons for the two compose-gate closed states. Exported so
// every Send affordance (ChatInput, side-rail Apply, the guided Explain button)
// renders identical copy and the two states can never drift apart per surface.
/** Transient boot window: the backend wall clock has not landed yet. */
export const COMPOSE_CONNECTING_MESSAGE = "Connecting to the composer…";
/** Stuck state: backend reachable but it reported no usable compose timeout. */
export const COMPOSE_UNAVAILABLE_MESSAGE =
  "Composer unavailable — the server did not report a compose timeout.";

let composeTimeoutMs = DEFAULT_COMPOSE_TIMEOUT_MS;
// Whether a backend wall clock has been applied to the ceiling above. Set
// together with composeTimeoutMs in applyServerComposerTimeout so readiness
// and the ceiling can never disagree — a ready gate that outran the ceiling
// would schedule an abort from the stale default, the exact bug this guards.
let composeTimeoutApplied = false;

/** Current compose abort ceiling. Read at CALL time (useComposer), never
 * cached at module load, so the boot-applied server value governs every
 * send that starts after it lands. */
export function getComposeTimeoutMs(): number {
  return composeTimeoutMs;
}

/**
 * Whether a valid backend wall clock has been applied, i.e. a compose-abort
 * timer scheduled now would use the deployment's configured ceiling rather
 * than the boot default. Because this flag is set in the SAME
 * applyServerComposerTimeout call that sets the ceiling, readiness can never
 * outrun the ceiling. sessionStore.composeTimeoutReady is the reactive mirror
 * of this predicate (latched by App.checkHealth) — components subscribe to the
 * store; this module function is the non-reactive source of truth it mirrors.
 */
export function isComposeTimeoutReady(): boolean {
  return composeTimeoutApplied;
}

/**
 * Derive the abort ceiling from the backend's configured compose wall
 * clock (seconds, from GET /api/system/status). Non-finite or non-positive
 * values are ignored — the current ceiling (default 295s) is a safe floor,
 * and a garbage value must not shrink the guard below the backend wall.
 *
 * Returns whether a valid value was applied, and sets isComposeTimeoutReady()
 * (this module's authoritative "a known-good ceiling exists" predicate) in the
 * same call. App.checkHealth mirrors that predicate into
 * sessionStore.composeTimeoutReady — the reactive gate the UI subscribes to —
 * so a garbage value leaves BOTH the ceiling and readiness untouched.
 */
export function applyServerComposerTimeout(
  backendTimeoutSeconds: number,
): boolean {
  if (
    !Number.isFinite(backendTimeoutSeconds) ||
    backendTimeoutSeconds <= 0
  ) {
    return false;
  }
  composeTimeoutMs =
    Math.round(backendTimeoutSeconds * 1000) + COMPOSE_CLIENT_GRACE_MS;
  composeTimeoutApplied = true;
  return true;
}

/**
 * Restore the ceiling AND readiness to their boot defaults, together. Called on
 * logout (sessionStore.reset) so the module predicate and the store gate reset
 * in lockstep — otherwise the module would retain readiness the reset store no
 * longer reflects — and by tests for isolation.
 */
export function resetComposeTimeout(): void {
  composeTimeoutMs = DEFAULT_COMPOSE_TIMEOUT_MS;
  composeTimeoutApplied = false;
}

/**
 * Run a compose request under the client abort ceiling — the single shared
 * primitive for BOTH freeform (useComposer) and guided (ChatPanel) sends, so
 * the two paths cannot drift the timer/guard logic apart again.
 *
 * INVARIANT: a compose-abort setTimeout is only ever scheduled from a
 * known-good ceiling. `ready` is sessionStore.composeTimeoutReady, the single
 * reactive source of truth (set true by App.checkHealth once the backend wall
 * clock lands). While NOT ready the runner is not invoked at all (no API
 * request, no controller, no timer) — the Send affordances are disabled until
 * readiness, so reaching here un-ready is the programmatic-caller (e.g.
 * SideRailValidationBanner) defense-in-depth path. Readiness is a parameter,
 * not a module read, so the primitive stays pure and the caller owns the one
 * source.
 *
 * The controllerRef is assigned BEFORE the runner awaits (so a concurrent
 * Stop can abort the in-flight fetch) and released with an identity check
 * after it settles (so a later send's controller is never clobbered).
 */
export async function runComposeWithTimeout(
  controllerRef: { current: AbortController | null },
  ready: boolean,
  runner: (signal: AbortSignal) => Promise<void>,
): Promise<void> {
  if (!ready) {
    return;
  }
  const controller = new AbortController();
  controllerRef.current = controller;
  const timer = setTimeout(
    () => controller.abort(COMPOSE_TIMEOUT_ABORT_REASON),
    // Read at call time: the ceiling is derived from the backend's configured
    // wall clock once /api/system/status lands at boot, and readiness above
    // guarantees that has happened before we reach here.
    getComposeTimeoutMs(),
  );
  try {
    await runner(controller.signal);
  } finally {
    clearTimeout(timer);
    if (controllerRef.current === controller) {
      controllerRef.current = null;
    }
  }
}

// Abort reasons are internal frontend control-plane values. They let
// sessionStore distinguish a user-requested stop from the timeout guard while
// still using the browser's native AbortController path.
export const COMPOSE_TIMEOUT_ABORT_REASON = "compose_timeout";
export const COMPOSE_USER_CANCEL_ABORT_REASON = "compose_user_cancel";

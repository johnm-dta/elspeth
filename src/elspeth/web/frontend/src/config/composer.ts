// Client-side abort budget for composer requests — the freeform
// POST /api/sessions/{id}/messages call and the guided step-chat call.
//
// INVARIANT: this ceiling sits ABOVE the backend compose wall clock
// (ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS = 270s, deploy/elspeth-web.env)
// plus grace, so the backend's discriminated 422
// (convergence_wall_clock_timeout) always arrives before the client
// aborts. The client abort is a last-resort guard against a truly dead
// connection, not a UX pacing mechanism — the composer progress panel is
// what tells the user the turn is alive.
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
export const COMPOSE_TIMEOUT_MS = 295_000;

// Abort reasons are internal frontend control-plane values. They let
// sessionStore distinguish a user-requested stop from the timeout guard while
// still using the browser's native AbortController path.
export const COMPOSE_TIMEOUT_ABORT_REASON = "compose_timeout";
export const COMPOSE_USER_CANCEL_ABORT_REASON = "compose_user_cancel";

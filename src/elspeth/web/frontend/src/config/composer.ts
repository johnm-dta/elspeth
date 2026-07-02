// Client-side abort budget for composer requests — the freeform
// POST /api/sessions/{id}/messages call and the guided step-chat call.
//
// Sized for SUBPHASE-era compose (elspeth-b189b5b3b8): the compose flow is
// broken into bounded subphases and live turns settle in seconds to ~30s.
// 90s is ~3x that observed worst case — past it the request reads as
// stalled, not slow, and holding the abort any longer only delays failure
// surfacing by minutes.
//
// History: this was 295s (backend budget 270s + 25s client grace), sized for
// the old big-bang "compose the whole pipeline in one call" flow so the
// backend's discriminated 422 (e.g. convergence_wall_clock_timeout) always
// arrived before the client aborted. With subphase compose this ceiling
// deliberately sits BELOW the backend's 270s wall
// (ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS, deploy/elspeth-web.env): a call
// living in the 90–270s band now surfaces the client timeout copy instead of
// a structured 422 — accepted, because subphase turns never legitimately run
// that long.
export const COMPOSE_TIMEOUT_MS = 90_000;

// Abort reasons are internal frontend control-plane values. They let
// sessionStore distinguish a user-requested stop from the timeout guard while
// still using the browser's native AbortController path.
export const COMPOSE_TIMEOUT_ABORT_REASON = "compose_timeout";
export const COMPOSE_USER_CANCEL_ABORT_REASON = "compose_user_cancel";

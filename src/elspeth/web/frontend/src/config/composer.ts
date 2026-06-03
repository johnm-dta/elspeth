// Client-side abort budget for the POST /api/sessions/{id}/messages call.
//
// MUST sit between the backend's ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS and the
// practical browser/proxy idle ceiling so the server reaches its own deadline
// first and emits a discriminated 422 (e.g. convergence_wall_clock_timeout).
// If the client or transport aborts first, the structured response is lost and
// the user only sees a generic AbortError fallback.
//
// Staging backend budget: 270.0 s (see deploy/elspeth-web.env). That leaves
// 30 s below the ~5 minute idle ceiling. The client waits 25 s beyond the
// backend deadline so the 422 has room to arrive while the client abort still
// fires before the transport cliff.
export const COMPOSE_BACKEND_TIMEOUT_MS = 270_000;
export const COMPOSE_TRANSPORT_IDLE_CEILING_MS = 300_000;
export const COMPOSE_SERVER_TRANSPORT_HEADROOM_MS = 30_000;
export const COMPOSE_CLIENT_GRACE_MS = 25_000;
export const COMPOSE_TIMEOUT_MS = COMPOSE_BACKEND_TIMEOUT_MS + COMPOSE_CLIENT_GRACE_MS;

// Abort reasons are internal frontend control-plane values. They let
// sessionStore distinguish a user-requested stop from the timeout guard while
// still using the browser's native AbortController path.
export const COMPOSE_TIMEOUT_ABORT_REASON = "compose_timeout";
export const COMPOSE_USER_CANCEL_ABORT_REASON = "compose_user_cancel";

// Client-side abort budget for the POST /api/sessions/{id}/messages call.
//
// MUST exceed the backend's ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS so the
// server reaches its own deadline first and emits a discriminated 422
// (e.g. convergence_wall_clock_timeout). If the client aborts first, the
// browser kills the request before the structured error can be sent and
// the user only sees a generic AbortError fallback.
//
// Backend default: 300.0 s (see deploy/elspeth-web.env). Grace: 30 s for
// network + envelope. Browser fetch-side ceiling on Chrome is ~5 min idle,
// so 330 s stays well below it.
export const COMPOSE_TIMEOUT_MS = 330_000;

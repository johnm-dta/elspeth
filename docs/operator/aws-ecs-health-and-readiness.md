# AWS ECS health and readiness

ELSPETH exposes two deliberately different unauthenticated health surfaces for AWS ECS:

- The container health check calls `GET /api/health`. This is a shallow liveness check and always returns `{"status":"ok"}` after the application has bound its socket.
- The ALB target group calls `GET /api/ready`. This is the dependency-aware traffic gate. The target group must use the exact path `/api/ready`, enable health checks, accept only HTTP 200, and set a timeout strictly above the endpoint's five-second ceiling.
- `elspeth doctor aws-ecs` is a one-shot pre-traffic deployment task. It validates the deployment contract and dependencies before traffic is shifted; it is not a recurring health check.
- `elspeth health` is never wired to the container or ALB health checks.

Plan 04 performs its database/schema and mounted-directory gate before the process binds its socket. During that startup window, connection refusal is expected and is not a liveness failure. Configure both the container health-check `startPeriod` and the ECS service `healthCheckGracePeriodSeconds` to approximately 150 seconds so startup validation can finish before replacement logic begins.

Each readiness dependency group has a two-second wall-clock budget. The whole report, including cache wait, has a five-second budget. Successful reports are cached for two seconds. Probe admission is bounded to one unresolved worker per closed dependency label and five workers total. If a probe is already running or saturated, readiness returns a static not-ready result; fix the slow or unavailable dependency and allow the registered probe to drain rather than increasing worker count.

`/api/ready` is unauthenticated because an ALB health checker cannot present application credentials. Cancellation-safe single-flight caching and bounded per-label admission prevent request stampedes from creating unbounded dependency work. Responses and logs are redacted to closed check names, static remedies, schema state names, and exception class names.

## Orphan Landscape reconciliation

Session runs cancelled by startup or periodic orphan cleanup carry an exact durable `[landscape-reconciliation:pending]` suffix until the corresponding Landscape audit outcome is resolved. A running Landscape row becomes `INTERRUPTED`; an existing terminal row or a null anchor closes with `[landscape-reconciliation:complete]`.

A non-null anchor with no Landscape row closes with `[landscape-reconciliation:absent]`. This is a durable audit exception, not successful audit reconciliation: it can represent a crash before `begin_run` or later audit-row loss. The service emits the static `orphan_landscape_run_absent` event with `operator_action="investigate audit-row absence"`. Operators must investigate the missing audit row; the closed marker only prevents automatic retry from looping forever.

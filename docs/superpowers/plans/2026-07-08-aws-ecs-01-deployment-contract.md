# AWS ECS Deployment Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Give `WebSettings` a `deployment_target` mode and a pure validator that checks a settings object against the strict AWS ECS deployment contract.

**Architecture:** `WebSettings` gains a `deployment_target: Literal["default", "aws-ecs"]` field (default `"default"`), picked up automatically by the existing generic `ELSPETH_WEB__*` env-var loader (`web/app.py:547-594`) — no loader change needed. A new pure module, `web/deployment_contract.py`, checks the strict aws-ecs rules against any `WebSettings` instance and returns `list[ContractCheck]`. It reuses — never re-derives — the secret-placeholder heuristics `WebSettings` already enforces at boot, extracted into three predicate functions in `config.py`. Unlike `WebSettings`'s own boot guard, the aws-ecs checks have no localhost/pytest bypass.

**Tech Stack:** Python 3.12/3.13, Pydantic v2, SQLAlchemy 2.x `make_url`, pytest.

**Depends on:** None — foundational. Sibling plans (schema-probe, `/api/ready`, `doctor aws-ecs`, S3 plugins, Bedrock provider) depend on this plan's outputs.

**Global Constraints** (verbatim spec, Deployment Contract section):
- `SESSION_DB_URL`/`LANDSCAPE_URL` required; both must use a PostgreSQL SQLAlchemy scheme, incl. driver variants like `postgresql+psycopg`; no silent SQLite fallback.
- Session/landscape URLs should point to separate logical Aurora targets, either separate databases or schemas, even sharing one cluster (spec's "should", not "must"). `ContractCheck.ok` is binary with no severity tier, so an `ok=False` would over-enforce a soft "should" — deliberately **not** a check in this plan. Deferred to `doctor aws-ecs`'s advisory output (doctor CLI sibling plan owns it), not dropped.
- `DATA_DIR`/`PAYLOAD_STORE_PATH` required. Writability is I/O, out of scope (doctor/readiness plans own it) — this function checks shape (present, non-blank) **and** explicit configuration. `data_dir` has a non-`None`, non-blank class default (`Path("data")`), unlike `payload_store_path` (`None` default), so a bare None/blank check can never fail for `data_dir` — it would silently pass an omitted `ELSPETH_WEB__DATA_DIR`. Task 2 closes this via `field_name in settings.model_fields_set`.
- Enumerated placeholder set only: `SECRET_KEY` (rejects `change-me-in-production` + existing length floor) and `SHAREABLE_LINK_SIGNING_KEY` (rejects uniform-byte/degenerate keys, existing heuristic). No other secret-backed setting is placeholder-checked. `landscape_passphrase` deliberately absent (SQLite-only). OIDC carries no client secret.
- Web host must be suitable for container serving, normally `0.0.0.0`. This plan checks exact equality with `"0.0.0.0"` (single-container Fargate/ALB v4 posture, this milestone's scope) rather than excluding loopback/private addresses generally; an IPv6 bind-all (`"::"`) is out of scope.
- Batch/CLI use remains flexible; strict contract applies only to the web deployment target.

---

### Task 1: `deployment_target` field + extract reusable secret-placeholder predicates

**Files:**
- Modify: `src/elspeth/web/config.py:37` (insert 3 predicate functions after `_allow_insecure_test_keys`)
- Modify: `src/elspeth/web/config.py:83` (insert `deployment_target` field after `auth_provider`)
- Modify: `src/elspeth/web/config.py:602-618` (`_enforce_secret_key_in_production` — call predicates, keep exact error text)
- Modify: `src/elspeth/web/config.py:620-644` (`_reject_known_weak_signing_key` — call predicate, keep exact error text)
- Test: `tests/unit/web/test_config.py`

**Interfaces:**
- Consumes: `_MIN_NON_LOCAL_JWT_SECRET_KEY_BYTES` (config.py:25), `_allow_insecure_test_keys` (config.py:36-37).
- Produces: `is_default_secret_key_placeholder/is_undersized_secret_key(secret_key: str) -> bool`, `is_uniform_byte_key(key_bytes: bytes) -> bool` in `elspeth.web.config` (consumed by Task 2); `WebSettings.deployment_target: Literal["default", "aws-ecs"] = "default"`.

```python
def is_default_secret_key_placeholder(secret_key: str) -> bool:
    return secret_key == "change-me-in-production"


def is_undersized_secret_key(secret_key: str) -> bool:
    return len(secret_key.encode("utf-8")) < _MIN_NON_LOCAL_JWT_SECRET_KEY_BYTES


def is_uniform_byte_key(key_bytes: bytes) -> bool:
    return len(set(key_bytes)) == 1
```

In `_enforce_secret_key_in_production`, replace the two inline conditions with `if is_default_secret_key_placeholder(self.secret_key):` and `if is_undersized_secret_key(self.secret_key):`, keeping both raised messages byte-for-byte identical. In `_reject_known_weak_signing_key`, replace `if len(set(raw_key)) == 1:` with `if is_uniform_byte_key(raw_key):`, same message.

Field, inserted after `auth_provider: AuthProviderType = "local"`:
```python
    # "default" preserves current non-ECS behavior; "aws-ecs" is validated
    # strictly by web/deployment_contract.py::validate_aws_ecs_settings.
    deployment_target: Literal["default", "aws-ecs"] = "default"
```

**Steps:**
- [ ] Add `TestDeploymentTarget` to `tests/unit/web/test_config.py`: `test_defaults_to_default` (no kwarg → `settings.deployment_target == "default"`), `test_accepts_aws_ecs` (`deployment_target="aws-ecs"` round-trips), `test_rejects_unknown_value` (`deployment_target="azure-aca"` → `pytest.raises(ValidationError, match=r"'default' or 'aws-ecs'")`). Reuse the minimal-kwargs pattern at `test_config.py:32-39` (`composer_max_composition_turns=15, composer_max_discovery_turns=10, composer_timeout_seconds=85.0, composer_rate_limit_per_minute=10, shareable_link_signing_key=b"\x00" * 32`).
- [ ] Run `pytest tests/unit/web/test_config.py -k TestDeploymentTarget -v` → fails: `AttributeError: 'WebSettings' object has no attribute 'deployment_target'` on test 1; unexpected `ValidationError` (`extra_forbidden`) on the other two.
- [ ] Add the field and the three predicates; wire the two model_validators to call them.
- [ ] Run `pytest tests/unit/web/test_config.py -k TestDeploymentTarget -v` → 3 passed.
- [ ] Run `pytest tests/unit/web/test_config.py tests/unit/web/test_config_shareable_link.py -v` → all pass, confirming the extraction preserved behavior.
- [ ] `git add src/elspeth/web/config.py tests/unit/web/test_config.py && git commit -m "feat(web): add WebSettings.deployment_target and extract secret-placeholder heuristics"`

### Task 2: `deployment_contract.py` pure validator

**Files:**
- Create: `src/elspeth/web/deployment_contract.py`
- Test: `tests/unit/web/test_deployment_contract.py`

**Interfaces:**
- Consumes: `WebSettings` + the three Task 1 predicates; `make_url` — safe unguarded since `WebSettings._validate_db_url` (config.py:334-342) already guarantees any non-`None` `session_db_url`/`landscape_url` is parseable with a drivername. Also consumes `WebSettings.model_fields_set` (pydantic v2: fields explicitly passed to the constructor, unaffected by `validate_default=True` — confirmed against installed pydantic 2.13.4) to detect an operator-omitted `data_dir`/`payload_store_path`, since `_settings_from_env()` (web/app.py:562-594) only passes a kwarg for `ELSPETH_WEB__*` vars present in `os.environ`.
- Produces (pinned): `DEPLOYMENT_TARGET_AWS_ECS = "aws-ecs"`; `@dataclass(frozen=True) class ContractCheck: name: str; ok: bool; detail: str`; `validate_aws_ecs_settings(settings: WebSettings) -> list[ContractCheck]`.

```python
"""Pure AWS ECS deployment contract validator. No I/O, no network, no
filesystem. ContractCheck.detail is pre-redacted -- never a URL, path,
or secret value."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.engine.url import make_url

from elspeth.web.config import (
    WebSettings,
    is_default_secret_key_placeholder,
    is_undersized_secret_key,
    is_uniform_byte_key,
)

DEPLOYMENT_TARGET_AWS_ECS = "aws-ecs"
_POSTGRES_DRIVER = "postgresql"
_CONTAINER_SERVING_HOST = "0.0.0.0"  # this IS the container-serving check, not a bind-all bug;
# exact-match only -- IPv6 dual-stack ("::") is out of scope for this milestone


@dataclass(frozen=True)
class ContractCheck:
    name: str
    ok: bool
    detail: str


def _check_postgres_url(name: str, env_var: str, url: str | None) -> ContractCheck:
    if url is None:
        return ContractCheck(name, False, f"{env_var} is required in aws-ecs deployment mode")
    driver = make_url(url).drivername.split("+")[0]
    if driver != _POSTGRES_DRIVER:
        return ContractCheck(
            name, False,
            f"{env_var} must use a PostgreSQL SQLAlchemy scheme (e.g. postgresql+psycopg); "
            f"no SQLite fallback is permitted in aws-ecs mode, got driver '{driver}'",
        )
    return ContractCheck(name, True, f"{env_var} uses a PostgreSQL scheme")


def _check_required_path(
    name: str, env_var: str, value: Path | None, *, explicitly_set: bool
) -> ContractCheck:
    # `explicitly_set` is `field_name in settings.model_fields_set` at the
    # call site. `payload_store_path` truly defaults to None, so a bare
    # None-check would already catch its omission -- but `data_dir` defaults
    # to `Path("data")` (never None, never blank), so without this flag an
    # omitted ELSPETH_WEB__DATA_DIR would silently report ok=True. Checking
    # explicit-construction presence closes that gap for both fields
    # uniformly.
    if not explicitly_set or value is None or not str(value).strip():
        return ContractCheck(name, False, f"{env_var} is required and must not be blank in aws-ecs deployment mode")
    return ContractCheck(name, True, f"{env_var} is set")


def validate_aws_ecs_settings(settings: WebSettings) -> list[ContractCheck]:
    """Run every strict check unconditionally; ignores deployment_target --
    callers (doctor CLI, web startup) decide when to invoke it."""
    checks = [
        _check_postgres_url("session_db_url", "ELSPETH_WEB__SESSION_DB_URL", settings.session_db_url),
        _check_postgres_url("landscape_url", "ELSPETH_WEB__LANDSCAPE_URL", settings.landscape_url),
        _check_required_path(
            "data_dir", "ELSPETH_WEB__DATA_DIR", settings.data_dir,
            explicitly_set="data_dir" in settings.model_fields_set,
        ),
        _check_required_path(
            "payload_store_path", "ELSPETH_WEB__PAYLOAD_STORE_PATH", settings.payload_store_path,
            explicitly_set="payload_store_path" in settings.model_fields_set,
        ),
    ]
    host_ok = settings.host == _CONTAINER_SERVING_HOST
    checks.append(ContractCheck(
        "host", host_ok,
        "host is suitable for container serving" if host_ok
        else f"ELSPETH_WEB__HOST must be {_CONTAINER_SERVING_HOST} for container/ALB reachability",
    ))
    # Outside pytest-local-host construction, WebSettings' own boot guards
    # already reject a weak secret_key/signing_key, so these two checks can
    # only report ok=False in a local-host test fixture -- kept for
    # doctor-output completeness, not because production reaches False.
    secret_ok = not (is_default_secret_key_placeholder(settings.secret_key) or is_undersized_secret_key(settings.secret_key))
    checks.append(ContractCheck(
        "secret_key", secret_ok,
        "secret_key is production-shaped" if secret_ok
        else "ELSPETH_WEB__SECRET_KEY must not be the default placeholder and must meet the length floor",
    ))
    key_ok = not is_uniform_byte_key(settings.shareable_link_signing_key.get_secret_value())
    checks.append(ContractCheck(
        "shareable_link_signing_key", key_ok,
        "shareable_link_signing_key is production-shaped" if key_ok
        else "ELSPETH_WEB__SHAREABLE_LINK_SIGNING_KEY is a known-weak uniform-byte placeholder",
    ))
    return checks
```

**Steps:**
- [ ] Write `tests/unit/web/test_deployment_contract.py`. `_base_kwargs()` returns Task 1's required-fields pattern (composer_* fields per `test_config.py:32-39`) **plus** a production-shaped `secret_key` (32+ bytes, non-placeholder, e.g. `"a" * 40`) and a non-uniform 32-byte `shareable_link_signing_key` (e.g. `bytes(range(32))`). It does **not** set `host` (leaves the class default `"127.0.0.1"`) and does **not** set `data_dir` or `payload_store_path` (leaves `Path("data")` / `None` respectively, so both are absent from `model_fields_set` unless a test overrides them). The production-shaped secret/signing keys are needed unconditionally because the two full-pass tests below override `host` to `"0.0.0.0"`, outside `_LOCAL_HOSTS` (config.py:24) — `WebSettings`'s own boot guards run unconditionally for a non-local host and would reject a weak default at *construction*, before `validate_aws_ecs_settings` runs. Tests exercising the aws-ecs secret/signing-key checks instead override `secret_key`/`shareable_link_signing_key` directly while leaving `host="127.0.0.1"` (the pytest-local-host construction bypass).
- [ ] One test per check, indexing `{c.name: c for c in validate_aws_ecs_settings(settings)}` and asserting `.ok`: `test_missing_session_db_url_fails`, `test_missing_landscape_url_fails`, `test_sqlite_session_db_url_rejected` (`session_db_url="sqlite:///x.db"`), `test_sqlite_landscape_url_rejected`, `test_postgresql_psycopg_driver_accepted` (`"postgresql+psycopg://u:p@host/db"` → ok), `test_non_postgres_driver_rejected` (`session_db_url="mysql://user:hunter2@host/db"` → `checks["session_db_url"].ok is False`; also assert `"hunter2" not in checks["session_db_url"].detail` — locks in the allowlist-not-denylist shape of `_check_postgres_url` and the `.detail` redaction contract pinned by the shared brief), `test_missing_payload_store_path_fails` (default `None`), `test_missing_data_dir_fails` (base kwargs unchanged — `data_dir` absent from `model_fields_set`, falls back to `Path("data")` — asserts `checks["data_dir"].ok is False`; regression test for the omitted-`ELSPETH_WEB__DATA_DIR` gap, uncatchable by a bare None/blank check since `data_dir` is never `None`), `test_non_container_host_fails` (default `host="127.0.0.1"`), `test_container_host_passes` (`host="0.0.0.0"`, base kwargs unchanged), `test_placeholder_secret_key_fails` (override `secret_key="change-me-in-production"`, keep `host="127.0.0.1"`), `test_undersized_secret_key_fails` (override `secret_key="short"`, `host="127.0.0.1"`), `test_uniform_byte_signing_key_fails` (override `shareable_link_signing_key=b"\x00" * 32`, `host="127.0.0.1"`), `test_all_checks_pass_for_fully_valid_ecs_settings` (`host="0.0.0.0"`, both URLs `postgresql://`, `data_dir` **and** `payload_store_path` both explicitly overridden — e.g. `tmp_path / "data"`, `tmp_path / "payloads"` — so both are in `model_fields_set`, base secret/signing keys unchanged → `all(c.ok for c in checks)` **and** `all(c.detail for c in checks)`, the latter locking in a non-empty `.detail` per check).
- [ ] Run `pytest tests/unit/web/test_deployment_contract.py -v` → `ModuleNotFoundError: No module named 'elspeth.web.deployment_contract'`.
- [ ] Create `src/elspeth/web/deployment_contract.py` as above.
- [ ] Run `pytest tests/unit/web/test_deployment_contract.py -v` → 14 passed.
- [ ] `git add src/elspeth/web/deployment_contract.py tests/unit/web/test_deployment_contract.py && git commit -m "feat(web): add pure AWS ECS deployment contract validator"`

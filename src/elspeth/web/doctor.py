"""Safe, redacted AWS ECS deployment preflight checks.

The default doctor path creates no persistent application state.  Its only
filesystem write is an unlinked temporary probe operated through one file
descriptor.  Database inspection and optional initialization are added by the
second Plan 03 task.
"""

from __future__ import annotations

import importlib
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.engine import make_url

from elspeth.core.landscape.database import SchemaCompatibilityError
from elspeth.web.config import WebSettings
from elspeth.web.deployment_contract import ContractCheck, validate_aws_ecs_settings
from elspeth.web.paths import allowed_source_directories
from elspeth.web.schema_probe import (
    DatabaseTargetConflictError,
    SchemaInitBusyError,
    SchemaLockCleanupError,
    SchemaState,
    init_landscape_schema,
    init_session_schema,
    postgres_engine_kwargs,
    probe_landscape_schema,
    probe_session_schema,
    require_distinct_postgres_targets,
)
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import SessionSchemaError

_PROBE_SENTINEL = b"elspeth-doctor-probe"


def sanitize_error(context: str, exc: BaseException) -> str:
    """Return static context plus exception class, never exception content."""
    return f"{context} ({type(exc).__name__})"


def probe_directory_writable(label: str, path: Path | None) -> ContractCheck:
    """Actively prove an existing directory is writable without a named residue."""
    name = f"{label}_writable"
    if path is None:
        return ContractCheck(name, False, f"{label} directory is required and must already exist")
    try:
        if not path.is_dir():
            return ContractCheck(name, False, f"{label} directory is required and must already exist")
    except Exception as exc:
        return ContractCheck(name, False, sanitize_error(f"{label} directory validation failed", exc))

    fd: int | None = None
    probe_name: str | None = None
    unlinked = False
    probe_error: Exception | None = None
    cleanup_error: Exception | None = None
    try:
        fd, probe_name = tempfile.mkstemp(prefix=".doctor-probe-", dir=path)
        os.unlink(probe_name)
        unlinked = True
        with os.fdopen(fd, "w+b") as probe:
            fd = None  # ownership transferred to ``probe``
            probe.write(_PROBE_SENTINEL)
            probe.flush()
            os.fsync(probe.fileno())
            probe.seek(0)
            if probe.read() != _PROBE_SENTINEL:
                raise OSError("doctor probe readback did not match")
    except Exception as exc:
        probe_error = exc
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception as exc:
                cleanup_error = exc
        if probe_name is not None and not unlinked:
            try:
                os.unlink(probe_name)
            except FileNotFoundError:
                pass
            except Exception as exc:
                cleanup_error = cleanup_error or exc

    if cleanup_error is not None:
        return ContractCheck(name, False, sanitize_error(f"{label} directory probe cleanup failed", cleanup_error))
    if probe_error is not None:
        return ContractCheck(name, False, sanitize_error(f"{label} directory probe failed", probe_error))
    return ContractCheck(name, True, f"{label} directory is writable")


def _probe_payload_store(path: Path | None) -> ContractCheck:
    """Apply the payload-store root contract before the active write probe."""
    name = "payload_store_writable"
    if path is None:
        return probe_directory_writable("payload_store", None)
    try:
        path_stat = path.lstat()
        if stat.S_ISLNK(path_stat.st_mode):
            return ContractCheck(name, False, "payload_store directory must not be a symlink")
        if not stat.S_ISDIR(path_stat.st_mode):
            return ContractCheck(name, False, "payload_store path must be an existing directory")
        if path_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            return ContractCheck(name, False, "payload_store group/world-writable directory is not allowed")
        resolved = path.resolve(strict=True)
    except Exception as exc:
        return ContractCheck(name, False, sanitize_error("payload_store directory validation failed", exc))
    return probe_directory_writable("payload_store", resolved)


def _aws_s3_plugin_check() -> ContractCheck:
    name = "aws_s3_plugin"
    try:
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

        manager = get_shared_plugin_manager()
        sources = {plugin.name for plugin in manager.get_sources()}
        sinks = {plugin.name for plugin in manager.get_sinks()}
        ok = "aws_s3" in sources and "aws_s3" in sinks
    except Exception as exc:
        return ContractCheck(name, False, sanitize_error("AWS S3 plugin discovery failed", exc))
    return ContractCheck(
        name,
        ok,
        "aws_s3 source and sink are registered" if ok else "aws_s3 source and sink must both be registered",
    )


def _bedrock_provider_check() -> ContractCheck:
    name = "bedrock_provider"
    try:
        provider_module = importlib.import_module("elspeth.plugins.transforms.llm.transform")
        providers = provider_module._PROVIDERS  # deliberate narrow private-registry read
        ok = "bedrock" in providers
    except Exception as exc:
        return ContractCheck(name, False, sanitize_error("Bedrock provider discovery failed", exc))
    return ContractCheck(name, ok, "bedrock provider is registered" if ok else "bedrock provider must be registered")


def _aws_operator_telemetry_check() -> ContractCheck:
    name = "aws_operator_telemetry"
    try:
        # Plan 14 owns the validators and immutable policy builder.  At this
        # earlier plan boundary their published WebSettings surface is the
        # static evidence that AWS mode can resolve the fixed task-local OTLP
        # policy rather than accept an authored endpoint or headers.
        required_fields = {
            "operator_telemetry",
            "operator_telemetry_service_name",
            "operator_telemetry_environment",
            "operator_telemetry_release",
            "operator_telemetry_ecs_cluster",
            "operator_telemetry_ecs_service",
            "operator_telemetry_task_definition_family",
            "operator_telemetry_task_definition_revision",
            "operator_telemetry_export_interval_seconds",
            "operator_pipeline_telemetry_granularity",
        }
        ok = required_fields <= set(WebSettings.model_fields)
    except Exception as exc:
        return ContractCheck(name, False, sanitize_error("AWS operator telemetry policy discovery failed", exc))
    return ContractCheck(
        name,
        ok,
        "AWS task-local OTLP operator policy is registered" if ok else "AWS task-local OTLP operator policy must be registered",
    )


def _capability_value(value: object) -> object:
    return getattr(value, "value", value)


def _bedrock_guardrail_plugins_check() -> ContractCheck:
    name = "bedrock_guardrail_plugins"
    try:
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

        transforms = {plugin.name: plugin for plugin in get_shared_plugin_manager().get_transforms()}
        prompt = transforms["aws_bedrock_prompt_shield"]
        content = transforms["aws_bedrock_content_safety"]

        prompt_declarations = cast(Any, prompt).policy_capabilities
        content_declarations = cast(Any, content).policy_capabilities
        prompt_ok = any(
            _capability_value(declaration.capability) == "prompt_shield" and _capability_value(declaration.control_role) == "input"
            for declaration in prompt_declarations
        )
        content_ok = any(
            _capability_value(declaration.capability) == "content_safety" and _capability_value(declaration.control_role) == "output"
            for declaration in content_declarations
        )
        ok = prompt_ok and content_ok
    except Exception as exc:
        return ContractCheck(name, False, sanitize_error("Bedrock Guardrail plugin discovery failed", exc))
    return ContractCheck(
        name,
        ok,
        "Bedrock Guardrail plugins and typed capabilities are registered"
        if ok
        else "Bedrock Guardrail plugins require provider-neutral typed capabilities",
    )


def _dependency_check(module_name: str, check_name: str) -> ContractCheck:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return ContractCheck(check_name, False, sanitize_error(f"{module_name} dependency import failed", exc))
    return ContractCheck(check_name, True, f"{module_name} dependency is importable")


def plugin_and_dependency_checks() -> list[ContractCheck]:
    """Return isolated capability checks in stable report order."""
    return [
        _aws_s3_plugin_check(),
        _bedrock_provider_check(),
        _aws_operator_telemetry_check(),
        _bedrock_guardrail_plugins_check(),
        _dependency_check("psycopg", "psycopg_dependency"),
        _dependency_check("boto3", "boto3_dependency"),
        _dependency_check("ijson", "ijson_dependency"),
        _dependency_check("jinja2", "jinja2_dependency"),
    ]


def database_target_check(session_url: str | None, landscape_url: str | None) -> ContractCheck:
    """Prove the two PostgreSQL URLs resolve to distinct logical targets."""
    try:
        require_distinct_postgres_targets(session_url, landscape_url)  # type: ignore[arg-type]
    except DatabaseTargetConflictError:
        return ContractCheck(
            "separate_db_targets",
            False,
            "session and Landscape PostgreSQL targets must be provably distinct",
        )
    return ContractCheck("separate_db_targets", True, "session and Landscape targets are provably distinct")


def schema_check(label: str, state: SchemaState) -> ContractCheck:
    """Render one schema state using static, redacted guidance."""
    if state is SchemaState.CURRENT:
        return ContractCheck(label, True, "current")
    if state is SchemaState.MISSING:
        return ContractCheck(label, False, "missing; rerun with --init-schema")
    if state is SchemaState.PARTIAL and label == "landscape_schema":
        return ContractCheck(label, False, "partial; rerun with --init-schema")
    database = "session" if label == "session_schema" else "Landscape"
    return ContractCheck(label, False, f"stale; drop and recreate the {database} database, then rerun doctor")


def _human_schema_label(label: str) -> str:
    return "session schema" if label == "session_schema" else "Landscape schema"


def _build_engine(label: str, raw_url: str) -> Engine:
    parsed = make_url(raw_url)
    kwargs: dict[str, Any] = dict(postgres_engine_kwargs(raw_url))
    if parsed.drivername.split("+", 1)[0] == "postgresql":
        kwargs["connect_args"] = {"connect_timeout": 10}
    if label == "session_schema":
        return create_session_engine(raw_url, **kwargs)
    if label == "landscape_schema":
        return create_engine(raw_url, **kwargs)
    raise ValueError("unknown doctor schema label")


def _inspect_database(
    label: str,
    raw_url: str,
    probe_fn: Callable[[Engine | Connection], SchemaState],
) -> tuple[SchemaState | None, ContractCheck]:
    """Inspect connectivity and schema state through one one-shot engine."""
    engine: Engine | None = None
    result: tuple[SchemaState | None, ContractCheck]
    try:
        engine = _build_engine(label, raw_url)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            state = probe_fn(connection)
        result = (state, schema_check(label, state))
    except (SessionSchemaError, SchemaCompatibilityError):
        result = (SchemaState.STALE, schema_check(label, SchemaState.STALE))
    except Exception as exc:
        result = (
            None,
            ContractCheck(label, False, sanitize_error(f"{_human_schema_label(label)} inspection failed", exc)),
        )
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception as exc:
                result = (
                    None,
                    ContractCheck(
                        label,
                        False,
                        sanitize_error(f"{_human_schema_label(label)} engine disposal failed", exc),
                    ),
                )
    return result


def _initialize_database(
    label: str,
    raw_url: str,
    probe_fn: Callable[[Engine | Connection], SchemaState],
    init_fn: Callable[[Engine], None],
) -> ContractCheck:
    """Initialize one eligible schema and independently verify it afterward."""
    engine: Engine | None = None
    result: ContractCheck
    try:
        engine = _build_engine(label, raw_url)
        init_fn(engine)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            final_state = probe_fn(connection)
        if final_state is SchemaState.CURRENT:
            result = ContractCheck(
                label,
                True,
                "current; initialization completed or was already completed",
            )
        else:
            result = ContractCheck(
                label,
                False,
                f"{_human_schema_label(label)} final verification did not report current",
            )
    except SchemaInitBusyError:
        result = ContractCheck(
            label,
            False,
            "another schema initialization is in progress; wait for it to finish and rerun",
        )
    except SchemaLockCleanupError:
        result = ContractCheck(
            label,
            False,
            "initialization may have completed but lock cleanup was not verified; investigate the database connection and rerun",
        )
    except (SessionSchemaError, SchemaCompatibilityError):
        result = schema_check(label, SchemaState.STALE)
    except Exception as exc:
        result = ContractCheck(
            label,
            False,
            sanitize_error(f"{_human_schema_label(label)} initialization failed", exc),
        )
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception as exc:
                result = ContractCheck(
                    label,
                    False,
                    sanitize_error(f"{_human_schema_label(label)} engine disposal failed", exc),
                )
    return result


def collect_checks(settings: WebSettings, *, init_schema: bool = False) -> list[ContractCheck]:
    """Collect the ordered report and optionally initialize eligible schemas."""
    checks = list(validate_aws_ecs_settings(settings))
    by_name = {check.name: check for check in checks}
    url_eligible = by_name["session_db_url"].ok and by_name["landscape_url"].ok
    if url_eligible:
        target_check = database_target_check(settings.session_db_url, settings.landscape_url)
    else:
        target_check = ContractCheck(
            "separate_db_targets",
            False,
            "database target comparison was not attempted because the database URL contract failed",
        )
    checks.append(target_check)
    checks.extend(
        [
            probe_directory_writable("data_dir", settings.data_dir),
            _probe_payload_store(settings.payload_store_path),
            probe_directory_writable("blob", allowed_source_directories(str(settings.data_dir))[0]),
        ]
    )
    checks.extend(plugin_and_dependency_checks())

    database_prerequisites_pass = url_eligible and by_name["deployment_target"].ok and target_check.ok
    if not database_prerequisites_pass:
        blocked_detail = "schema inspection was not attempted because the AWS ECS database prerequisites failed"
        checks.extend(
            [
                ContractCheck("session_schema", False, blocked_detail),
                ContractCheck("landscape_schema", False, blocked_detail),
            ]
        )
        return checks

    session_url = cast(str, settings.session_db_url)
    landscape_url = cast(str, settings.landscape_url)
    session_state, session_result = _inspect_database("session_schema", session_url, probe_session_schema)
    landscape_state, landscape_result = _inspect_database("landscape_schema", landscape_url, probe_landscape_schema)

    if not init_schema:
        checks.extend([session_result, landscape_result])
        return checks

    session_repairable = session_state is SchemaState.MISSING
    landscape_repairable = landscape_state in (SchemaState.MISSING, SchemaState.PARTIAL)
    states_eligible = session_state in (SchemaState.MISSING, SchemaState.CURRENT) and landscape_state in (
        SchemaState.MISSING,
        SchemaState.PARTIAL,
        SchemaState.CURRENT,
    )
    complete_preflight_passed = all(check.ok for check in checks) and states_eligible
    if not complete_preflight_passed:
        blocked_init_detail = "not initialized because the complete preflight failed"
        if session_repairable:
            session_result = ContractCheck("session_schema", False, blocked_init_detail)
        if landscape_repairable:
            landscape_result = ContractCheck("landscape_schema", False, blocked_init_detail)
        checks.extend([session_result, landscape_result])
        return checks

    if session_repairable:
        session_result = _initialize_database(
            "session_schema",
            session_url,
            probe_session_schema,
            init_session_schema,
        )
    if landscape_repairable:
        landscape_result = _initialize_database(
            "landscape_schema",
            landscape_url,
            probe_landscape_schema,
            init_landscape_schema,
        )
    checks.extend([session_result, landscape_result])
    return checks

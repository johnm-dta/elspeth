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
from pathlib import Path
from typing import Any, cast

from elspeth.web.config import WebSettings
from elspeth.web.deployment_contract import ContractCheck, validate_aws_ecs_settings
from elspeth.web.paths import allowed_source_directories

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


def collect_checks(settings: WebSettings, *, init_schema: bool = False) -> list[ContractCheck]:
    """Collect the complete ordered Task 1 report without persistent state."""
    del init_schema  # Task 2 gives this flag its sole mutating behavior.
    checks = list(validate_aws_ecs_settings(settings))
    checks.append(
        ContractCheck(
            "separate_db_targets",
            False,
            "database target comparison is not implemented until Plan 03 Task 2",
        )
    )
    checks.extend(
        [
            probe_directory_writable("data_dir", settings.data_dir),
            _probe_payload_store(settings.payload_store_path),
            probe_directory_writable("blob", allowed_source_directories(str(settings.data_dir))[0]),
        ]
    )
    checks.extend(plugin_and_dependency_checks())
    checks.extend(
        [
            ContractCheck("session_schema", False, "session schema inspection is not implemented until Plan 03 Task 2"),
            ContractCheck("landscape_schema", False, "Landscape schema inspection is not implemented until Plan 03 Task 2"),
        ]
    )
    return checks

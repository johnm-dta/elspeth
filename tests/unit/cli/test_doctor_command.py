"""CLI rendering and error-boundary tests for ``doctor aws-ecs``."""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from elspeth.cli import app
from elspeth.web.deployment_contract import ContractCheck

runner = CliRunner()


def _patch_doctor(
    monkeypatch: pytest.MonkeyPatch,
    checks: list[ContractCheck],
    *,
    settings: object | None = None,
) -> list[bool]:
    import elspeth.web.app as web_app
    import elspeth.web.doctor as doctor

    marker = object() if settings is None else settings
    init_values: list[bool] = []

    monkeypatch.setattr(web_app, "_settings_from_env", lambda: marker)

    def fake_collect(actual_settings: object, *, init_schema: bool = False) -> list[ContractCheck]:
        assert actual_settings is marker
        init_values.append(init_schema)
        return checks

    monkeypatch.setattr(doctor, "collect_checks", fake_collect)
    return init_values


def test_json_output_is_bare_ordered_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_doctor(
        monkeypatch,
        [ContractCheck("first", True, "ready"), ContractCheck("second", True, "also ready")],
    )

    result = runner.invoke(app, ["--no-dotenv", "doctor", "aws-ecs", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == [
        {"name": "first", "ok": True, "detail": "ready"},
        {"name": "second", "ok": True, "detail": "also ready"},
    ]
    assert result.stderr == ""


def test_text_output_is_deterministic_and_failure_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_doctor(
        monkeypatch,
        [ContractCheck("first", True, "ready"), ContractCheck("second", False, "blocked")],
    )

    first = runner.invoke(app, ["--no-dotenv", "doctor", "aws-ecs"])
    second = runner.invoke(app, ["--no-dotenv", "doctor", "aws-ecs"])

    assert first.exit_code == 1
    assert first.stdout == "OK first: ready\nFAIL second: blocked\n"
    assert second.stdout == first.stdout


def test_no_dotenv_skips_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.cli as cli

    _patch_doctor(monkeypatch, [ContractCheck("ready", True, "ready")])

    def unexpected_loader(*_args: Any, **_kwargs: Any) -> bool:
        raise AssertionError("dotenv must not be loaded")

    monkeypatch.setattr(cli, "_load_dotenv", unexpected_loader)

    result = runner.invoke(app, ["--no-dotenv", "doctor", "aws-ecs", "--json"])

    assert result.exit_code == 0


def test_settings_load_failure_is_one_sanitized_check(monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.web.app as web_app

    def fail_settings() -> object:
        raise RuntimeError("postgresql://user:hunter2@private/db /secret/path")  # secret-scan: allow-this-line

    monkeypatch.setattr(web_app, "_settings_from_env", fail_settings)

    result = runner.invoke(app, ["--no-dotenv", "doctor", "aws-ecs", "--json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == [
        {"name": "settings_load", "ok": False, "detail": "web settings could not be loaded (RuntimeError)"}
    ]
    assert "hunter2" not in result.output
    assert "/secret/path" not in result.output


def test_last_resort_internal_error_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.web.app as web_app
    import elspeth.web.doctor as doctor

    monkeypatch.setattr(web_app, "_settings_from_env", lambda: object())

    def fail_collection(_settings: object, *, init_schema: bool = False) -> list[ContractCheck]:
        del init_schema
        raise RuntimeError("postgresql://user:hunter2@private/db")  # secret-scan: allow-this-line

    monkeypatch.setattr(doctor, "collect_checks", fail_collection)

    result = runner.invoke(app, ["--no-dotenv", "doctor", "aws-ecs", "--json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == [
        {"name": "doctor_internal_error", "ok": False, "detail": "doctor collection failed (RuntimeError)"}
    ]
    assert "hunter2" not in result.output


@pytest.mark.parametrize("init_schema", [False, True])
def test_init_schema_flag_is_propagated(monkeypatch: pytest.MonkeyPatch, init_schema: bool) -> None:
    observed = _patch_doctor(monkeypatch, [ContractCheck("ready", True, "ready")])
    args = ["--no-dotenv", "doctor", "aws-ecs", "--json"]
    if init_schema:
        args.insert(-1, "--init-schema")

    result = runner.invoke(app, args)

    assert result.exit_code == 0
    assert observed == [init_schema]


@pytest.mark.parametrize(
    ("detail", "raw_cause"),
    [
        (
            "another schema initialization is in progress; wait for it to finish and rerun",
            "private busy-lock cause",
        ),
        (
            "initialization may have completed but lock cleanup was not verified; investigate the database connection and rerun",
            "private cleanup cause",
        ),
    ],
)
def test_lock_domain_failures_render_complete_redacted_json(
    detail: str,
    raw_cause: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = [
        ContractCheck("deployment_target", True, "ready"),
        ContractCheck("session_schema", False, detail),
        ContractCheck("landscape_schema", True, "current"),
    ]
    _patch_doctor(monkeypatch, checks)

    result = runner.invoke(app, ["--no-dotenv", "doctor", "aws-ecs", "--init-schema", "--json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == [
        {"name": "deployment_target", "ok": True, "detail": "ready"},
        {"name": "session_schema", "ok": False, "detail": detail},
        {"name": "landscape_schema", "ok": True, "detail": "current"},
    ]
    assert raw_cause not in result.output

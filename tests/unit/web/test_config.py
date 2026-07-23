"""Tests for WebSettings configuration model."""

from __future__ import annotations

import re
import sys
import types
import typing
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from elspeth.web.config import WebSettings
from elspeth.web.deployment_contract import validate_aws_ecs_settings


def test_playwright_local_backend_secret_key_satisfies_non_pytest_guard() -> None:
    """The Playwright-managed backend runs outside pytest and needs a 32-byte key."""
    config_path = Path(__file__).parents[3] / "src/elspeth/web/frontend/playwright.config.ts"
    match = re.search(r'ELSPETH_WEB__secret_key:\s*"([^"]+)"', config_path.read_text(encoding="utf-8"))

    assert match is not None
    assert len(match.group(1).encode("utf-8")) >= 32


class TestWebSettingsValidation:
    """Tests for field validation."""

    def test_unknown_setting_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            WebSettings(
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
                composer_expose_provder_errors=True,  # type: ignore[call-arg]
            )

    def test_bedrock_guardrail_settings_round_trip_without_repr_leak(self) -> None:
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
            bedrock_guardrail_profiles=(
                {
                    "alias": "prompt-default",
                    "plugin": "aws_bedrock_prompt_shield",
                    "guardrail_identifier": "privateguardrail",
                    "guardrail_version": "7",
                    "region": "us-east-1",
                },
            ),
        )

        assert settings.bedrock_guardrail_profiles[0].alias == "prompt-default"
        assert "privateguardrail" not in repr(settings)

    def test_invalid_auth_provider_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                auth_provider="invalid",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_invalid_auth_provider_kerberos_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                auth_provider="kerberos",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_port_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                port=0,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_port_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                port=-1,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_port_above_65535_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                port=65536,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_composer_max_composition_turns_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                composer_max_composition_turns=0,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_composer_max_discovery_turns_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                composer_max_composition_turns=15,
                composer_max_discovery_turns=0,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_composer_max_tool_calls_per_turn_defaults_to_16(self) -> None:
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )

        assert settings.composer_max_tool_calls_per_turn == 16

    def test_pipeline_planner_budgets_are_explicit_and_validated(self) -> None:
        base = {
            "composer_max_composition_turns": 15,
            "composer_max_discovery_turns": 10,
            "composer_timeout_seconds": 85.0,
            "composer_rate_limit_per_minute": 10,
            "shareable_link_signing_key": b"\x00" * 32,
        }

        settings = WebSettings(**base)
        assert settings.composer_planner_max_provider_calls > 0
        assert settings.composer_planner_max_request_bytes > 0
        assert settings.composer_planner_max_completion_tokens > 0
        assert settings.composer_planner_max_cumulative_provider_cost == Decimal("5.00")
        assert settings.composer_planner_repair_budget == 2

        for field in (
            "composer_planner_max_provider_calls",
            "composer_planner_max_request_bytes",
            "composer_planner_max_completion_tokens",
        ):
            with pytest.raises(ValidationError):
                WebSettings(**base, **{field: 0})
        with pytest.raises(ValidationError):
            WebSettings(**base, composer_planner_max_cumulative_provider_cost=Decimal("-0.01"))
        with pytest.raises(ValidationError):
            WebSettings(**base, composer_planner_repair_budget=-1)

    def test_composer_max_tool_calls_per_turn_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_max_tool_calls_per_turn=0,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_composer_timeout_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_composer_timeout_must_leave_transport_headroom(self) -> None:
        with pytest.raises(ValidationError, match="transport idle ceiling"):
            WebSettings(
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=300.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_composer_timeout_allows_explicit_larger_transport_ceiling(self) -> None:
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=300.0,
            composer_transport_idle_ceiling_seconds=360.0,
            composer_transport_headroom_seconds=30.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )

        assert settings.composer_timeout_seconds == 300.0

    def test_composer_rate_limit_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=0,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_composer_fields_required_no_defaults(self) -> None:
        """Regression: composer fields must be explicitly configured — no silent defaults."""
        with pytest.raises(ValidationError):
            WebSettings(shareable_link_signing_key=b"\x00" * 32)  # type: ignore[call-arg]  # intentionally omitted to test validation

    def test_composer_model_default_uses_current_openai_frontier_model(self) -> None:
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )

        assert settings.composer_model == "gpt-5.5"

    def test_composer_sampling_defaults_to_none_and_probe_enabled(self) -> None:
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )

        assert settings.composer_temperature is None
        assert settings.composer_seed is None
        assert settings.composer_boot_probe_enabled is True

    def test_composer_temperature_accepts_in_range_and_rejects_out_of_range(self) -> None:
        base = {
            "composer_max_composition_turns": 15,
            "composer_max_discovery_turns": 10,
            "composer_timeout_seconds": 85.0,
            "composer_rate_limit_per_minute": 10,
            "shareable_link_signing_key": b"\x00" * 32,
        }

        assert WebSettings(**base, composer_temperature=0.0).composer_temperature == 0.0
        assert WebSettings(**base, composer_temperature=1.5).composer_temperature == 1.5
        with pytest.raises(ValidationError):
            WebSettings(**base, composer_temperature=2.5)
        with pytest.raises(ValidationError):
            WebSettings(**base, composer_temperature=-0.1)

    def test_composer_seed_accepts_int_and_none(self) -> None:
        base = {
            "composer_max_composition_turns": 15,
            "composer_max_discovery_turns": 10,
            "composer_timeout_seconds": 85.0,
            "composer_rate_limit_per_minute": 10,
            "shareable_link_signing_key": b"\x00" * 32,
        }

        assert WebSettings(**base, composer_seed=42).composer_seed == 42
        assert WebSettings(**base, composer_seed=None).composer_seed is None

    def test_max_upload_bytes_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                max_upload_bytes=0,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_payload_store_retention_days_zero_rejected(self) -> None:
        """The audit-readiness retention row reads this setting and surfaces
        it to operators ("System retention: N days"). An accidental ``ge=0``
        relaxation would render as "System retention: 0 days" — wire-valid
        but semantically meaningless. The ``ge=1`` floor is load-bearing.
        """
        with pytest.raises(ValidationError):
            WebSettings(
                payload_store_retention_days=0,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )


class TestDeploymentTarget:
    def test_defaults_to_default(self) -> None:
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )

        assert settings.deployment_target == "default"

    def test_accepts_aws_ecs(self) -> None:
        settings = WebSettings(
            deployment_target="aws-ecs",
            operator_telemetry="aws-otlp",
            operator_telemetry_environment="production",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )

        assert settings.deployment_target == "aws-ecs"


class TestOperatorTelemetrySettings:
    def test_local_defaults_remain_prometheus_only(self) -> None:
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )

        assert settings.operator_telemetry == "prometheus"
        assert settings.operator_telemetry_service_name == "elspeth-web"
        assert settings.operator_telemetry_environment is None
        assert settings.operator_telemetry_release is None
        assert settings.operator_telemetry_ecs_cluster is None
        assert settings.operator_telemetry_ecs_service is None
        assert settings.operator_telemetry_task_definition_family is None
        assert settings.operator_telemetry_task_definition_revision is None
        assert settings.operator_telemetry_export_interval_seconds == 60
        assert settings.operator_pipeline_telemetry_granularity == "lifecycle"
        assert settings.operator_metrics_bearer_token is None

    def test_operator_metrics_bearer_token_is_masked(self) -> None:
        raw_token = "operator-metrics-token-0123456789abcdef"

        settings = WebSettings(
            operator_metrics_bearer_token=raw_token,
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )

        assert settings.operator_metrics_bearer_token is not None
        assert settings.operator_metrics_bearer_token.get_secret_value() == raw_token
        assert raw_token not in repr(settings)

    @pytest.mark.parametrize("raw_token", ["short", "x" * 513, "x" * 31 + " ", "x" * 31 + "ñ"])
    def test_operator_metrics_bearer_token_rejects_weak_or_non_header_safe_values(self, raw_token: str) -> None:
        with pytest.raises(ValidationError, match="operator_metrics_bearer_token"):
            WebSettings(
                operator_metrics_bearer_token=raw_token,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    @pytest.mark.parametrize(
        ("field", "raw_value"),
        [
            ("operator_telemetry_service_name", "arn:aws:ecs:ap-southeast-2:123456789012:service/elspeth-web"),
            ("operator_telemetry_service_name", "123456789012"),
            ("operator_telemetry_service_name", "elspeth-123456789012-web"),
            ("operator_telemetry_environment", "arn:aws:ecs:ap-southeast-2:123456789012:cluster/production"),
            ("operator_telemetry_environment", "123456789012"),
            ("operator_telemetry_environment", "prod-123456789012-blue"),
        ],
    )
    def test_aws_mode_rejects_arn_and_account_resource_labels(self, field: str, raw_value: str) -> None:
        overrides = {
            "operator_telemetry": "aws-otlp",
            "operator_telemetry_environment": "production",
            field: raw_value,
        }
        with pytest.raises(ValidationError, match=field) as caught:
            WebSettings(
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
                **overrides,  # type: ignore[arg-type]
            )

        assert raw_value not in str(caught.value)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("operator_telemetry_service_name", "elspeth-web"),
            ("operator_telemetry_service_name", "orders.api_v2"),
            pytest.param("operator_telemetry_service_name", "s" * 128, id="service-128-char-boundary"),
            ("operator_telemetry_environment", "production"),
            ("operator_telemetry_environment", "prod-blue"),
            pytest.param("operator_telemetry_environment", "e" * 128, id="environment-128-char-boundary"),
        ],
    )
    def test_aws_mode_accepts_safe_resource_labels(self, field: str, value: str) -> None:
        overrides = {
            "operator_telemetry": "aws-otlp",
            "operator_telemetry_environment": "production",
            field: value,
        }
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
            **overrides,  # type: ignore[arg-type]
        )

        assert getattr(settings, field) == value

    def test_local_mode_preserves_generic_operator_resource_labels(self) -> None:
        settings = WebSettings(
            operator_telemetry_service_name="team/service:blue",
            operator_telemetry_environment="staging/eu:1",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )

        assert settings.operator_telemetry_service_name == "team/service:blue"
        assert settings.operator_telemetry_environment == "staging/eu:1"

    @pytest.mark.parametrize(
        ("overrides", "field"),
        [
            ({"operator_telemetry": "prometheus", "operator_telemetry_environment": "production"}, "operator_telemetry"),
            ({"operator_telemetry": "aws-otlp"}, "operator_telemetry_environment"),
            (
                {
                    "operator_telemetry": "aws-otlp",
                    "operator_telemetry_environment": "production",
                    "operator_telemetry_export_interval_seconds": 0,
                },
                "operator_telemetry_export_interval_seconds",
            ),
            (
                {
                    "operator_telemetry": "aws-otlp",
                    "operator_telemetry_environment": "production",
                    "operator_telemetry_export_interval_seconds": 3601,
                },
                "operator_telemetry_export_interval_seconds",
            ),
            (
                {"operator_telemetry": "aws-otlp", "operator_telemetry_environment": "production", "operator_telemetry_service_name": " "},
                "operator_telemetry_service_name",
            ),
            (
                {"operator_telemetry": "aws-otlp", "operator_telemetry_environment": "production\nsecret"},
                "operator_telemetry_environment",
            ),
            (
                {"operator_telemetry": "aws-otlp", "operator_telemetry_environment": "p" * 129},
                "operator_telemetry_environment",
            ),
        ],
    )
    def test_aws_ecs_rejects_invalid_operator_telemetry(self, overrides: dict[str, object], field: str) -> None:
        contract_only = field == "operator_telemetry" or (field == "operator_telemetry_environment" and field not in overrides)
        if contract_only:
            settings = WebSettings(
                deployment_target="aws-ecs",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
                **overrides,
            )
            checks = {check.name: check for check in validate_aws_ecs_settings(settings)}
            assert checks[field].ok is False
            return
        with pytest.raises(ValidationError, match=field) as caught:
            WebSettings(
                deployment_target="aws-ecs",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
                **overrides,
            )

        assert "production\nsecret" not in str(caught.value)
        assert "p" * 129 not in str(caught.value)

    @pytest.mark.parametrize(
        ("field", "raw_value"),
        [
            ("operator_telemetry_release", "arn:aws:ecr:ap-southeast-2:123456789012:repository/elspeth"),
            ("operator_telemetry_ecs_cluster", "arn:aws:ecs:ap-southeast-2:123456789012:cluster/elspeth-production"),
            ("operator_telemetry_ecs_service", "elspeth-123456789012-service"),
            ("operator_telemetry_task_definition_family", "task-definition/elspeth-web"),
            ("operator_telemetry_task_definition_revision", "123456789012"),
        ],
    )
    def test_aws_deployment_identity_rejects_arns_accounts_and_non_names(self, field: str, raw_value: str) -> None:
        with pytest.raises(ValidationError, match=field) as caught:
            WebSettings(
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
                **{field: raw_value},
            )

        assert raw_value not in str(caught.value)

    def test_aws_release_accepts_valid_digest_with_numeric_run(self) -> None:
        digest = "sha256:" + "a" + "123456789012" + ("b" * 51)

        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
            operator_telemetry_release=digest,
        )

        assert settings.operator_telemetry_release == digest

    @pytest.mark.parametrize(
        "field",
        [
            "operator_telemetry_endpoint",
            "operator_telemetry_headers",
            "operator_telemetry_credentials",
            "operator_telemetry_aws_access_key_id",
        ],
    )
    def test_operator_telemetry_has_no_egress_or_credential_settings(self, field: str) -> None:
        with pytest.raises(ValidationError, match="extra") as caught:
            WebSettings(
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
                **{field: "secret-remote-value"},
            )

        assert "secret-remote-value" not in str(caught.value)

    def test_rejects_unknown_value(self) -> None:
        with pytest.raises(ValidationError, match="'default' or 'aws-ecs'"):
            WebSettings(
                deployment_target="azure-aca",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )


class TestWebSettingsDerivedAccessors:
    """Tests for get_landscape_url() and get_payload_store_path()."""

    def test_get_landscape_url_default_derives_from_data_dir(self) -> None:
        settings = WebSettings(
            data_dir=Path("/app/data"),
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        url = settings.get_landscape_url()
        assert url == "sqlite:////app/data/runs/audit.db"

    def test_get_landscape_url_explicit_value_returned(self) -> None:
        settings = WebSettings(
            landscape_url="postgresql://db/audit",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        url = settings.get_landscape_url()
        assert url == "postgresql://db/audit"

    def test_get_payload_store_path_default_derives_from_data_dir(self) -> None:
        settings = WebSettings(
            data_dir=Path("/app/data"),
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        path = settings.get_payload_store_path()
        assert path == Path("/app/data/payloads")

    def test_get_payload_store_path_explicit_value_returned(self) -> None:
        settings = WebSettings(
            payload_store_path=Path("/mnt/payloads"),
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        path = settings.get_payload_store_path()
        assert path == Path("/mnt/payloads")

    def test_default_data_dir_landscape_url(self) -> None:
        """Default data_dir='data' is resolved to an absolute path at
        validation time, so derived URLs are anchored to a fixed
        location regardless of later os.chdir calls."""
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        url = settings.get_landscape_url()
        expected_data_dir = Path("data").resolve()
        assert url == f"sqlite:///{expected_data_dir / 'runs' / 'audit.db'}"

    def test_default_data_dir_payload_store_path(self) -> None:
        """Default data_dir='data' is resolved at validation time;
        payload path inherits that absolute prefix."""
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        path = settings.get_payload_store_path()
        assert path == Path("data").resolve() / "payloads"

    def test_get_session_db_url_default_derives_from_data_dir(self) -> None:
        settings = WebSettings(
            data_dir=Path("/app/data"),
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        url = settings.get_session_db_url()
        assert url == "sqlite:////app/data/sessions.db"

    def test_get_session_db_url_explicit_value_returned(self) -> None:
        settings = WebSettings(
            session_db_url="postgresql://db/sessions",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        url = settings.get_session_db_url()
        assert url == "postgresql://db/sessions"

    def test_default_data_dir_session_db_url(self) -> None:
        """Default data_dir='data' is resolved to an absolute path at
        validation time; session DB URL inherits that prefix."""
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        url = settings.get_session_db_url()
        expected_data_dir = Path("data").resolve()
        assert url == f"sqlite:///{expected_data_dir / 'sessions.db'}"


class TestSecretKeyGuard:
    """Tests for the secret_key production guard validator."""

    def test_default_secret_key_rejected_on_non_local_host(self) -> None:
        with pytest.raises(ValidationError, match="secret_key must be set"):
            WebSettings(
                host="0.0.0.0",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_default_secret_key_allowed_on_localhost(self) -> None:
        # Should not raise
        settings = WebSettings(
            host="127.0.0.1",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.secret_key == "change-me-in-production"

    def test_default_secret_key_allowed_on_localhost_name(self) -> None:
        settings = WebSettings(
            host="localhost",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.secret_key == "change-me-in-production"

    def test_default_secret_key_allowed_on_ipv6_loopback(self) -> None:
        settings = WebSettings(
            host="::1",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.secret_key == "change-me-in-production"

    def test_default_secret_key_rejected_on_loopback_outside_test_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "pytest", raising=False)
        monkeypatch.delenv("ELSPETH_ENV", raising=False)

        with pytest.raises(ValidationError, match="secret_key must be set"):
            WebSettings(
                host="127.0.0.1",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\xab\xcd" * 16,
            )

    def test_custom_secret_key_allowed_on_any_host(self) -> None:
        # DC-2 FIX-L: non-loopback hosts also need a non-weak signing key.
        # Uniform-byte placeholders are rejected by the weak-key validator.
        settings = WebSettings(
            host="0.0.0.0",
            secret_key="this-non-loopback-secret-is-long-enough",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\xab\xcd" * 16,
        )
        assert settings.secret_key == "this-non-loopback-secret-is-long-enough"

    def test_short_secret_key_rejected_on_non_local_host(self) -> None:
        with pytest.raises(ValidationError, match="secret_key must be at least 32 bytes"):
            WebSettings(
                host="0.0.0.0",
                secret_key="short-secret",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\xab\xcd" * 16,
            )

    def test_short_secret_key_allowed_on_localhost(self) -> None:
        settings = WebSettings(
            host="127.0.0.1",
            secret_key="short-secret",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.secret_key == "short-secret"

    @pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
    @pytest.mark.parametrize("secret_key", ["", "   "])
    def test_blank_secret_key_rejected_on_all_hosts(self, host: str, secret_key: str) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                host=host,
                secret_key=secret_key,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )


class TestAuthFieldValidation:
    """Tests for OIDC/Entra conditional field requirements."""

    def test_local_provider_no_oidc_fields_required(self) -> None:
        """Local auth (default) should work without any OIDC fields."""
        settings = WebSettings(
            auth_provider="local",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.auth_provider == "local"

    @pytest.mark.parametrize(
        "field_name",
        [
            "oidc_issuer",
            "oidc_audience",
            "oidc_client_id",
            "oidc_authorization_endpoint",
            "entra_tenant_id",
        ],
    )
    def test_local_provider_rejects_oidc_entra_fields(self, field_name: str) -> None:
        """Local auth must not accept inert OIDC/Entra configuration."""
        with pytest.raises(ValidationError, match=field_name):
            WebSettings(
                auth_provider="local",
                **{field_name: "https://issuer.example.com"},
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_oidc_provider_missing_fields_raises(self) -> None:
        """OIDC provider without required fields should raise."""
        with pytest.raises(ValidationError, match="OIDC auth requires"):
            WebSettings(
                auth_provider="oidc",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )


class TestPathFieldValidation:
    """Tests for path normalization and blank-path rejection."""

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("data_dir", ""),
            ("data_dir", "   "),
            ("payload_store_path", ""),
            ("payload_store_path", "   "),
        ],
    )
    def test_blank_path_fields_rejected(self, field_name: str, value: str) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                **{
                    field_name: value,
                    "composer_max_composition_turns": 15,
                    "composer_max_discovery_turns": 10,
                    "composer_timeout_seconds": 85.0,
                    "composer_rate_limit_per_minute": 10,
                    "shareable_link_signing_key": b"\x00" * 32,
                }
            )

    def test_data_dir_expands_user_home(self) -> None:
        settings = WebSettings(
            data_dir="~/data",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        expected = Path("~/data").expanduser().resolve()
        assert settings.data_dir == expected
        assert settings.data_dir.is_absolute()
        assert settings.get_landscape_url() == f"sqlite:///{expected / 'runs' / 'audit.db'}"
        assert settings.get_session_db_url() == f"sqlite:///{expected / 'sessions.db'}"

    def test_payload_store_path_expands_user_home(self) -> None:
        settings = WebSettings(
            payload_store_path="~/payloads",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.get_payload_store_path() == Path("~/payloads").expanduser().resolve()
        assert settings.get_payload_store_path().is_absolute()

    def test_preexisting_payload_symlink_preserves_configured_lexical_path(self, tmp_path: Path) -> None:
        target = tmp_path / "payload-target"
        target.mkdir()
        configured_path = tmp_path / "payload-link"
        configured_path.symlink_to(target, target_is_directory=True)

        settings = WebSettings(
            payload_store_path=configured_path,
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )

        assert settings.payload_store_path == configured_path.absolute()
        assert settings.payload_store_path != target.resolve()

    def test_relative_data_dir_resolved_at_validation_immune_to_chdir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The download/preview path-allowlist check used to silently
        depend on the running process CWD: a relative data_dir was
        resolved at every callsite, so a chdir between sink-write and
        download-time would move the allowlist with the process. After
        validation-time resolve(), the data_dir is pinned for the
        process lifetime regardless of later chdir.
        """
        monkeypatch.chdir(tmp_path)
        settings = WebSettings(
            data_dir="data",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        # Captured at validation time, anchored to tmp_path.
        expected = (tmp_path / "data").resolve()
        assert settings.data_dir == expected
        # Now chdir somewhere else — settings.data_dir MUST NOT move.
        other_dir = tmp_path / "elsewhere"
        other_dir.mkdir()
        monkeypatch.chdir(other_dir)
        assert settings.data_dir == expected


class TestServerSecretAllowlistValidation:
    """Tests for server_secret_allowlist field validation."""

    def test_reserved_elspeth_server_secret_names_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ELSPETH_"):
            WebSettings(
                server_secret_allowlist=("ELSPETH_FINGERPRINT_KEY",),
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )


class TestAuthFieldValidationContinued:
    """Additional OIDC/Entra field requirement coverage."""

    def test_oidc_provider_with_all_fields_valid(self) -> None:
        """OIDC provider with all required fields should succeed."""
        settings = WebSettings(
            auth_provider="oidc",
            oidc_issuer="https://issuer.example.com",
            oidc_audience="my-audience",
            oidc_client_id="my-client-id",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.auth_provider == "oidc"
        assert settings.oidc_issuer == "https://issuer.example.com"

    def test_oidc_provider_partial_fields_raises(self) -> None:
        """OIDC provider with only some fields should name the missing ones."""
        with pytest.raises(ValidationError, match="oidc_audience"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="https://issuer.example.com",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_entra_provider_missing_fields_raises(self) -> None:
        """Entra provider without required fields should raise."""
        with pytest.raises(ValidationError, match="Entra auth requires"):
            WebSettings(
                auth_provider="entra",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_entra_provider_missing_tenant_id_raises(self) -> None:
        """Entra with OIDC fields but no tenant_id should raise."""
        with pytest.raises(ValidationError, match="entra_tenant_id"):
            WebSettings(
                auth_provider="entra",
                oidc_issuer="https://login.microsoftonline.com/t/v2.0",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_entra_provider_with_all_fields_valid(self) -> None:
        """Entra provider with all required fields should succeed."""
        settings = WebSettings(
            auth_provider="entra",
            oidc_issuer="https://login.microsoftonline.com/t/v2.0",
            oidc_audience="my-audience",
            oidc_client_id="my-client-id",
            entra_tenant_id="my-tenant-id",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.auth_provider == "entra"
        assert settings.entra_tenant_id == "my-tenant-id"


class TestOIDCIssuerValidation:
    """OIDC issuer config must be safe before startup discovery can fetch it."""

    _COMPOSER_DEFAULTS: typing.ClassVar[dict[str, object]] = {
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }

    @pytest.mark.parametrize(
        "issuer",
        [
            "http://issuer.example.com",
            "https://user:pass@issuer.example.com",
            "https://127.0.0.1",
            "https://169.254.169.254",
            "https://issuer.example.com?tenant=default",
            "https://issuer.example.com#fragment",
        ],
    )
    def test_oidc_provider_rejects_unsafe_issuer(self, issuer: str) -> None:
        with pytest.raises(ValidationError):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer=issuer,
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                **self._COMPOSER_DEFAULTS,
            )

    def test_oidc_provider_accepts_path_issuer_and_same_origin_authorization_endpoint(self) -> None:
        settings = WebSettings(
            auth_provider="oidc",
            oidc_issuer="https://issuer.example.com/tenant/v2.0/",
            oidc_audience="my-audience",
            oidc_client_id="my-client-id",
            oidc_authorization_endpoint="https://issuer.example.com/oauth2/authorize",
            oidc_token_endpoint="https://issuer.example.com/oauth2/token",
            **self._COMPOSER_DEFAULTS,
        )

        assert settings.oidc_issuer == "https://issuer.example.com/tenant/v2.0"
        assert settings.oidc_authorization_endpoint == "https://issuer.example.com/oauth2/authorize"
        assert settings.oidc_token_endpoint == "https://issuer.example.com/oauth2/token"


class TestOIDCBlankStringRejection:
    """Blank/whitespace-only OIDC/Entra fields must be rejected at config time."""

    _COMPOSER_DEFAULTS: typing.ClassVar[dict[str, object]] = {
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }

    def test_oidc_empty_string_issuer_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                **self._COMPOSER_DEFAULTS,
            )

    def test_oidc_whitespace_issuer_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="   ",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                **self._COMPOSER_DEFAULTS,
            )

    def test_oidc_empty_audience_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="https://issuer.example.com",
                oidc_audience="",
                oidc_client_id="my-client-id",
                **self._COMPOSER_DEFAULTS,
            )

    def test_oidc_empty_client_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="https://issuer.example.com",
                oidc_audience="my-audience",
                oidc_client_id="",
                **self._COMPOSER_DEFAULTS,
            )

    def test_entra_empty_tenant_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="entra",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                entra_tenant_id="",
                **self._COMPOSER_DEFAULTS,
            )

    def test_entra_whitespace_tenant_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="entra",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                entra_tenant_id="   ",
                **self._COMPOSER_DEFAULTS,
            )

    def test_oidc_empty_authorization_endpoint_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="https://issuer.example.com",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                oidc_authorization_endpoint="",
                **self._COMPOSER_DEFAULTS,
            )

    def test_oidc_http_authorization_endpoint_rejected(self) -> None:
        with pytest.raises(ValidationError, match="HTTPS"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="https://issuer.example.com",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                oidc_authorization_endpoint="http://issuer.example.com/oauth2/authorize",
                oidc_token_endpoint="https://issuer.example.com/oauth2/token",
                **self._COMPOSER_DEFAULTS,
            )

    def test_oidc_cross_origin_authorization_endpoint_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not allowed"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="https://issuer.example.com",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                oidc_authorization_endpoint="https://evil.example.com/oauth2/authorize",
                oidc_token_endpoint="https://evil.example.com/oauth2/token",
                **self._COMPOSER_DEFAULTS,
            )

    def test_entra_cross_origin_authorization_endpoint_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not allowed"):
            WebSettings(
                auth_provider="entra",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                entra_tenant_id="test-tenant-id",
                oidc_authorization_endpoint="https://evil.example.com/oauth2/authorize",
                oidc_token_endpoint="https://evil.example.com/oauth2/token",
                **self._COMPOSER_DEFAULTS,
            )

    def test_oidc_browser_fields_have_closed_defaults(self) -> None:
        settings = WebSettings(**self._COMPOSER_DEFAULTS)
        assert settings.oidc_authorization_allowed_origins == ()
        assert settings.oidc_token_endpoint is None
        assert settings.oidc_audience_claim == "aud"

    def test_client_id_audience_claim_is_oidc_only(self) -> None:
        settings = WebSettings(
            auth_provider="oidc",
            oidc_issuer="https://issuer.example.com",
            oidc_audience="client",
            oidc_client_id="client",
            oidc_audience_claim="client_id",
            **self._COMPOSER_DEFAULTS,
        )
        assert settings.oidc_audience_claim == "client_id"
        for provider, fields in (
            ("local", {}),
            (
                "entra",
                {
                    "oidc_audience": "client",
                    "oidc_client_id": "client",
                    "entra_tenant_id": "tenant",
                },
            ),
        ):
            with pytest.raises(ValidationError, match="audience claim"):
                WebSettings(
                    auth_provider=provider,  # type: ignore[arg-type]
                    oidc_audience_claim="client_id",
                    **fields,
                    **self._COMPOSER_DEFAULTS,
                )

    def test_client_id_audience_claim_rejects_browser_backend_client_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="oidc_audience must match oidc_client_id"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="https://issuer.example.com",
                oidc_audience="backend-client",
                oidc_client_id="browser-client",
                oidc_audience_claim="client_id",
                **self._COMPOSER_DEFAULTS,
            )

    def test_invalid_audience_claim_mode_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"aud|client_id"):
            WebSettings(
                oidc_audience_claim="fallback",  # type: ignore[arg-type]
                **self._COMPOSER_DEFAULTS,
            )

    @pytest.mark.parametrize(
        ("authorization_endpoint", "token_endpoint"),
        [
            ("https://issuer.example.com/oauth2/authorize", None),
            (None, "https://issuer.example.com/oauth2/token"),
        ],
    )
    def test_oidc_explicit_browser_endpoints_are_both_or_neither(
        self,
        authorization_endpoint: str | None,
        token_endpoint: str | None,
    ) -> None:
        with pytest.raises(ValidationError, match="both or neither"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="https://issuer.example.com/pool",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                oidc_authorization_endpoint=authorization_endpoint,
                oidc_token_endpoint=token_endpoint,
                **self._COMPOSER_DEFAULTS,
            )

    def test_cognito_cross_origin_pair_requires_exact_allowlist(self) -> None:
        values = {
            "auth_provider": "oidc",
            "oidc_issuer": "https://cognito-idp.ap-southeast-2.amazonaws.com/pool-id",
            "oidc_audience": "client-id",
            "oidc_client_id": "client-id",
            "oidc_authorization_endpoint": "https://example.auth.ap-southeast-2.amazoncognito.com/oauth2/authorize",
            "oidc_token_endpoint": "https://example.auth.ap-southeast-2.amazoncognito.com/oauth2/token",
            **self._COMPOSER_DEFAULTS,
        }
        with pytest.raises(ValidationError, match="not allowed"):
            WebSettings(**values)
        settings = WebSettings(
            **values,
            oidc_authorization_allowed_origins=("https://example.auth.ap-southeast-2.amazoncognito.com",),
        )
        assert settings.oidc_authorization_endpoint is not None
        assert settings.oidc_token_endpoint is not None

    @pytest.mark.parametrize("provider", ["local", "entra"])
    def test_allowlist_is_oidc_only(self, provider: str) -> None:
        provider_fields: dict[str, object] = {}
        if provider == "entra":
            provider_fields = {
                "oidc_audience": "audience",
                "oidc_client_id": "client",
                "entra_tenant_id": "tenant",
            }
        with pytest.raises(ValidationError, match="allowlist"):
            WebSettings(
                auth_provider=provider,  # type: ignore[arg-type]
                oidc_authorization_allowed_origins=("https://login.example.com",),
                **provider_fields,
                **self._COMPOSER_DEFAULTS,
            )

    def test_allowlist_is_validated_without_explicit_endpoints(self) -> None:
        with pytest.raises(ValidationError, match="bare-origin"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="https://issuer.example.com",
                oidc_audience="audience",
                oidc_client_id="client",
                oidc_authorization_allowed_origins=("https://host.example.com/not-an-origin",),
                **self._COMPOSER_DEFAULTS,
            )

    def test_local_auth_blank_oidc_field_still_rejected(self) -> None:
        """Field validator fires regardless of auth_provider — blank is always invalid."""
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="local",
                oidc_issuer="",
                **self._COMPOSER_DEFAULTS,
            )


class TestDBURLValidation:
    """Database URL format and passphrase compatibility validation."""

    _COMPOSER_DEFAULTS: typing.ClassVar[dict[str, object]] = {
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }

    def test_landscape_url_empty_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(landscape_url="", **self._COMPOSER_DEFAULTS)

    def test_landscape_url_whitespace_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(landscape_url="   ", **self._COMPOSER_DEFAULTS)

    def test_landscape_url_malformed_rejected(self) -> None:
        with pytest.raises(ValidationError, match="invalid database URL"):
            WebSettings(landscape_url="not-a-url", **self._COMPOSER_DEFAULTS)

    def test_landscape_url_valid_sqlite_accepted(self) -> None:
        settings = WebSettings(landscape_url="sqlite:///path/audit.db", **self._COMPOSER_DEFAULTS)
        assert settings.landscape_url == "sqlite:///path/audit.db"

    def test_landscape_url_valid_postgresql_accepted(self) -> None:
        settings = WebSettings(landscape_url="postgresql://host/db", **self._COMPOSER_DEFAULTS)
        assert settings.landscape_url == "postgresql://host/db"

    def test_session_db_url_empty_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(session_db_url="", **self._COMPOSER_DEFAULTS)

    def test_session_db_url_whitespace_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(session_db_url="   ", **self._COMPOSER_DEFAULTS)

    def test_session_db_url_malformed_rejected(self) -> None:
        with pytest.raises(ValidationError, match="invalid database URL"):
            WebSettings(session_db_url="garbage", **self._COMPOSER_DEFAULTS)

    def test_session_db_url_valid_accepted(self) -> None:
        settings = WebSettings(session_db_url="sqlite:///sessions.db", **self._COMPOSER_DEFAULTS)
        assert settings.session_db_url == "sqlite:///sessions.db"

    def test_passphrase_with_postgresql_rejected(self) -> None:
        with pytest.raises(ValidationError, match="requires a SQLite"):
            WebSettings(
                landscape_url="postgresql://host/db",
                landscape_passphrase="secret",
                **self._COMPOSER_DEFAULTS,
            )

    def test_passphrase_with_sqlite_accepted(self) -> None:
        settings = WebSettings(
            landscape_url="sqlite:///audit.db",
            landscape_passphrase="secret",
            **self._COMPOSER_DEFAULTS,
        )
        assert settings.landscape_passphrase == "secret"

    def test_passphrase_without_explicit_url_accepted(self) -> None:
        """When landscape_url is None, default is SQLite — passphrase is valid."""
        settings = WebSettings(landscape_passphrase="secret", **self._COMPOSER_DEFAULTS)
        assert settings.landscape_passphrase == "secret"
        assert settings.landscape_url is None

    def test_passphrase_empty_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(landscape_passphrase="", **self._COMPOSER_DEFAULTS)

    def test_passphrase_whitespace_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(landscape_passphrase="   ", **self._COMPOSER_DEFAULTS)


def _optional_string_field_names() -> list[str]:
    """Enumerate all str | None fields on WebSettings for parametrized testing."""
    fields: list[str] = []
    for name, field_info in WebSettings.model_fields.items():
        # field_info.annotation is typed as `type[Any] | None` by pydantic,
        # but at runtime a `str | None` field is stored as a types.UnionType
        # instance.  Widen to Any so the isinstance narrowing is honest —
        # mypy otherwise treats the branch body as unreachable.
        ann: Any = field_info.annotation
        if isinstance(ann, types.UnionType):
            args = typing.get_args(ann)
            if str in args and type(None) in args:
                fields.append(name)
    return fields


class TestFieldValidatorCoverage:
    """Structural test: every str | None field must reject blank strings.

    Prevents the 'Drifting Goals' pattern — new optional string fields added
    without blank-string validators silently accumulate validation gaps.
    """

    _COMPOSER_DEFAULTS: typing.ClassVar[dict[str, object]] = {
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }

    @pytest.mark.parametrize("field_name", _optional_string_field_names())
    def test_blank_string_rejected_for_all_optional_string_fields(self, field_name: str) -> None:
        """Every str | None field must reject empty strings at config time."""
        with pytest.raises(ValidationError):
            WebSettings(**{field_name: "", **self._COMPOSER_DEFAULTS})


class TestJWKSFailureRetryFloor:
    """Regression guard for the cold-start DoS shield (elspeth-32982f17cf).

    ``jwks_failure_retry_seconds`` has a schema floor of ``ge=10``, not
    ``ge=1``.  The floor is load-bearing: during an IdP outage the FIRST
    caller pays the httpx timeout (~15s worst case) while subsequent
    callers short-circuit on the cached negative result.  A configured
    1-second retry collapses that shield — concurrent auth requests
    re-hit the dead IdP almost immediately, reinstating a partial DoS.
    Ten seconds is tight enough for fixtures to advance the window
    deliberately and loose enough that operators cannot configure the
    throttle away.  These tests fail loudly if the floor is silently
    relaxed to ``ge=1`` again.
    """

    _COMPOSER_DEFAULTS: typing.ClassVar[dict[str, object]] = {
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }

    @pytest.mark.parametrize("below_floor", [1, 2, 5, 9])
    def test_below_floor_rejected(self, below_floor: int) -> None:
        """Any value below 10 must raise ValidationError at schema time."""
        with pytest.raises(ValidationError):
            WebSettings(jwks_failure_retry_seconds=below_floor, **self._COMPOSER_DEFAULTS)

    def test_floor_value_accepted(self) -> None:
        """The boundary value ``10`` is the minimum legal configuration."""
        settings = WebSettings(jwks_failure_retry_seconds=10, **self._COMPOSER_DEFAULTS)
        assert settings.jwks_failure_retry_seconds == 10

    def test_above_floor_accepted(self) -> None:
        """Operators may raise the retry window (trading stale-serve risk for safety)."""
        settings = WebSettings(jwks_failure_retry_seconds=600, **self._COMPOSER_DEFAULTS)
        assert settings.jwks_failure_retry_seconds == 600

    def test_zero_rejected(self) -> None:
        """Zero would disable the throttle entirely — must be rejected."""
        with pytest.raises(ValidationError):
            WebSettings(jwks_failure_retry_seconds=0, **self._COMPOSER_DEFAULTS)

    def test_negative_rejected(self) -> None:
        """Negative values are schema-illegal."""
        with pytest.raises(ValidationError):
            WebSettings(jwks_failure_retry_seconds=-1, **self._COMPOSER_DEFAULTS)


class TestJWKSMaxStaleAge:
    """The operator-configured JWKS hard lifetime must be finite and positive."""

    _COMPOSER_DEFAULTS: typing.ClassVar[dict[str, object]] = {
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }

    def test_default_is_one_day(self) -> None:
        settings = WebSettings(**self._COMPOSER_DEFAULTS)
        assert settings.jwks_max_stale_seconds == 86_400

    def test_positive_override_accepted(self) -> None:
        settings = WebSettings(jwks_max_stale_seconds=1, **self._COMPOSER_DEFAULTS)
        assert settings.jwks_max_stale_seconds == 1

    @pytest.mark.parametrize("invalid", [0, -1])
    def test_non_positive_rejected(self, invalid: int) -> None:
        with pytest.raises(ValidationError):
            WebSettings(jwks_max_stale_seconds=invalid, **self._COMPOSER_DEFAULTS)


def _settings(**overrides: Any) -> WebSettings:
    """Construct WebSettings with required no-default fields + overrides.

    Mirrors the per-test construction boilerplate elsewhere in this module
    (the existing tests inline the same required-field set); collapsed here
    into a module-level helper for the advisor-config tests.
    """
    base: dict[str, Any] = {
        "composer_max_composition_turns": 20,
        "composer_max_discovery_turns": 20,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 30,
        "shareable_link_signing_key": b"x" * 32,
    }
    base.update(overrides)
    return WebSettings(**base)


class TestPublicBaseUrlValidation:
    """Tests for the trusted external origin used in generated email links."""

    def test_public_base_url_accepts_https_and_strips_trailing_slash(self) -> None:
        settings = _settings(public_base_url="https://composer.example.test/")

        assert settings.public_base_url == "https://composer.example.test"

    @pytest.mark.parametrize(
        "url",
        [
            "http://composer.example.test",
            "https://user:pass@composer.example.test",
            "https://127.0.0.1",
            "https://169.254.169.254",
        ],
    )
    def test_public_base_url_rejects_unsafe_origin(self, url: str) -> None:
        with pytest.raises(ValidationError):
            _settings(public_base_url=url)

    def test_public_base_url_allows_http_loopback_for_local_development(self) -> None:
        settings = _settings(public_base_url="http://127.0.0.1:8451/")

        assert settings.public_base_url == "http://127.0.0.1:8451"

    def test_email_verified_non_local_host_requires_public_base_url(self) -> None:
        with pytest.raises(ValidationError, match="public_base_url"):
            _settings(
                host="0.0.0.0",
                registration_mode="email_verified",
                secret_key="this-non-loopback-secret-is-long-enough",
                shareable_link_signing_key=b"\xab\xcd" * 16,
            )

    def test_email_verified_non_local_host_allows_configured_public_base_url(self) -> None:
        settings = _settings(
            host="0.0.0.0",
            registration_mode="email_verified",
            public_base_url="https://composer.example.test",
            secret_key="this-non-loopback-secret-is-long-enough",
            shareable_link_signing_key=b"\xab\xcd" * 16,
        )

        assert settings.public_base_url == "https://composer.example.test"


def test_advisor_must_differ_from_primary_exact() -> None:
    with pytest.raises(ValidationError, match="composer_advisor_model must differ from composer_model"):
        _settings(composer_model="gpt-5.5", composer_advisor_model="gpt-5.5")


def test_advisor_distinct_normalizes_provider_prefix() -> None:
    # openrouter/openai/gpt-5.5 and gpt-5.5 denote the same model -> reject.
    with pytest.raises(ValidationError, match="must differ"):
        _settings(composer_model="openrouter/openai/gpt-5.5", composer_advisor_model="gpt-5.5")


def test_advisor_distinct_accepts_different_models() -> None:
    s = _settings(composer_model="claude-sonnet-4-6", composer_advisor_model="claude-opus-4-7")
    assert s.composer_advisor_model == "claude-opus-4-7"


def test_advisor_checkpoint_budget_default_and_floor() -> None:
    assert _settings().composer_advisor_checkpoint_max_passes == 2
    with pytest.raises(ValidationError):
        _settings(composer_advisor_checkpoint_max_passes=0)


def test_settings_from_env_coerces_numeric_strings_for_strict_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """ELSPETH_WEB__* values arrive as strings; strict int/float fields must
    still be settable from the environment (live regression 2026-07-22: setting
    ELSPETH_WEB__COMPOSER_PLANNER_MAX_COMPLETION_TOKENS crash-looped the
    service with int_type because settings_from_env passed the raw string to a
    strict field)."""
    import base64
    import secrets as _secrets

    from elspeth.web.config import settings_from_env

    for key, value in {
        "ELSPETH_WEB__COMPOSER_MAX_COMPOSITION_TURNS": "30",
        "ELSPETH_WEB__COMPOSER_MAX_DISCOVERY_TURNS": "10",
        "ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS": "20.0",
        "ELSPETH_WEB__COMPOSER_RATE_LIMIT_PER_MINUTE": "10",
        "ELSPETH_WEB__COMPOSER_PLANNER_MAX_COMPLETION_TOKENS": "32768",
        "ELSPETH_WEB__COMPOSER_PLANNER_MAX_PROVIDER_CALLS": "80",
        "ELSPETH_WEB__SHAREABLE_LINK_SIGNING_KEY": base64.b64encode(_secrets.token_bytes(32)).decode(),
        "ELSPETH_WEB__OPERATOR_METRICS_BEARER_TOKEN": "operator-metrics-token-from-environment-0001",
    }.items():
        monkeypatch.setenv(key, value)

    settings = settings_from_env()

    assert settings.composer_planner_max_completion_tokens == 32768
    assert settings.composer_planner_max_provider_calls == 80
    assert settings.composer_max_composition_turns == 30
    assert settings.composer_timeout_seconds == 20.0
    assert settings.operator_metrics_bearer_token is not None
    assert settings.operator_metrics_bearer_token.get_secret_value() == "operator-metrics-token-from-environment-0001"

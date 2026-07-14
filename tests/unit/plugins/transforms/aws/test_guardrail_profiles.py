from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError

from elspeth.plugins.transforms.aws.guardrail_profiles import BedrockGuardrailProfileSettings


def _profile(**overrides: object) -> BedrockGuardrailProfileSettings:
    values: dict[str, object] = {
        "alias": "prompt-default",
        "plugin": "aws_bedrock_prompt_shield",
        "guardrail_identifier": "abc123guardrail",
        "guardrail_version": "7",
        "region": "us-east-1",
    }
    values.update(overrides)
    return BedrockGuardrailProfileSettings.model_validate(values)


def test_bedrock_profile_requires_numeric_version_and_closed_control() -> None:
    profile = _profile(guardrail_identifier="privateguardrail7")

    assert profile.guardrail_version == "7"
    assert "privateguardrail7" not in repr(profile)
    assert "us-east-1" not in repr(profile)


@pytest.mark.parametrize("version", ["DRAFT", "0", "1.2", "latest", ""])
def test_profile_rejects_non_numeric_immutable_version(version: str) -> None:
    with pytest.raises(ValidationError):
        _profile(guardrail_version=version)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alias", "Not Valid"),
        ("plugin", "azure_prompt_shield"),
        ("guardrail_identifier", ""),
        ("guardrail_identifier", "guardrail with spaces"),
        ("region", "not-a-region"),
    ],
)
def test_profile_rejects_invalid_closed_bindings(field: str, value: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        _profile(**{field: value})

    if value:
        assert value not in str(exc_info.value)


def test_profile_contains_no_credentials_or_endpoint_fields() -> None:
    fields = BedrockGuardrailProfileSettings.model_fields

    assert not ({"access_key", "secret_key", "session_token", "endpoint", "endpoint_url"} & fields.keys())


def test_local_requirement_check_is_lazy_and_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _profile()
    real_import = importlib.import_module
    imported: list[str] = []

    def tracking_import(name: str, package: str | None = None) -> object:
        imported.append(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", tracking_import)
    result = profile.check_local_requirements()

    assert result.available is True
    assert "boto3" in imported
    assert "botocore" in imported

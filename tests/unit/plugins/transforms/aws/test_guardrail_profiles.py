from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from elspeth.plugins.transforms.aws import guardrail_profiles
from elspeth.plugins.transforms.aws.guardrail_profiles import BedrockGuardrailProfileSettings, validate_guardrail_identifier


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


@pytest.mark.parametrize("version", ["DRAFT", "0", "1.2", "latest", "", "00000001", "123456789"])
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


@pytest.mark.parametrize(
    "identifier",
    [
        "abc123",
        "arn:aws:bedrock:us-east-1:123456789012:guardrail/abc123",
        "arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:guardrail/abc123",
        "arn:aws-iso:bedrock:us-iso-east-1:123456789012:guardrail/abc123",
        "arn:aws-iso-b:bedrock:us-isob-east-1:123456789012:guardrail/abc123",
    ],
)
def test_profile_accepts_installed_guardrail_identifier_grammar(identifier: str) -> None:
    assert validate_guardrail_identifier(identifier) == identifier


def test_profile_accepts_guardrail_arn_in_configured_region() -> None:
    identifier = "arn:aws:bedrock:ap-southeast-2:123456789012:guardrail/abc123"

    assert _profile(guardrail_identifier=identifier, region="ap-southeast-2").guardrail_identifier == identifier


def test_profile_rejects_guardrail_arn_from_different_region() -> None:
    identifier = "arn:aws:bedrock:us-east-1:123456789012:guardrail/abc123"

    with pytest.raises(ValidationError) as exc_info:
        _profile(guardrail_identifier=identifier, region="ap-southeast-2")

    assert identifier not in str(exc_info.value)


def test_profile_rejects_noncommercial_guardrail_arn_even_when_region_matches() -> None:
    identifier = "arn:aws-us-gov:bedrock:us-east-1:123456789012:guardrail/abc123"

    with pytest.raises(ValidationError) as exc_info:
        _profile(guardrail_identifier=identifier, region="us-east-1")

    assert identifier not in str(exc_info.value)


@pytest.mark.parametrize(
    "identifier",
    [
        "ABC123",
        "arn:aws_:bedrock:us-east-1:123456789012:guardrail/abc123",
        "arn:aws:bedrock:us-east-1:123:guardrail/abc123",
        "arn:aws:bedrock:us-east-1234567890123:123456789012:guardrail/abc123",
        "arn:aws:bedrock:us-east-1:123456789012:guardrail/abc-123",
        "a" * 2049,
    ],
)
def test_profile_rejects_identifier_outside_installed_guardrail_grammar(identifier: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        _profile(guardrail_identifier=identifier)

    assert identifier not in str(exc_info.value)


EXPECTED_BEDROCK_REGIONS = frozenset(
    {
        "af-south-1",
        "ap-east-2",
        "ap-northeast-1",
        "ap-northeast-2",
        "ap-northeast-3",
        "ap-south-1",
        "ap-south-2",
        "ap-southeast-1",
        "ap-southeast-2",
        "ap-southeast-3",
        "ap-southeast-4",
        "ap-southeast-5",
        "ap-southeast-6",
        "ap-southeast-7",
        "ca-central-1",
        "ca-west-1",
        "eu-central-1",
        "eu-central-2",
        "eu-north-1",
        "eu-south-1",
        "eu-south-2",
        "eu-west-1",
        "eu-west-2",
        "eu-west-3",
        "il-central-1",
        "me-central-1",
        "me-south-1",
        "mx-central-1",
        "sa-east-1",
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
    }
)


def test_profile_region_vocabulary_is_pinned_to_installed_boto3_metadata() -> None:
    assert guardrail_profiles.BEDROCK_GUARDRAIL_REGIONS == EXPECTED_BEDROCK_REGIONS
    assert len(EXPECTED_BEDROCK_REGIONS) == 33
    for region in EXPECTED_BEDROCK_REGIONS:
        assert _profile(region=region).region == region

    with pytest.raises(ValidationError):
        _profile(region="us-private-1")


def test_documented_guardrail_identifiers_are_valid_configuration_examples() -> None:
    document = (Path(__file__).parents[5] / "docs/reference/configuration.md").read_text()

    for invalid in ("operator-prompt-guardrail", "operator-content-guardrail"):
        assert invalid not in document
    for identifier in ("operatorpromptguardrail", "operatorcontentguardrail"):
        assert document.count(identifier) >= 2
        assert _profile(guardrail_identifier=identifier).guardrail_identifier == identifier


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

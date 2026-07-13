"""Closed validators for operator-controlled AWS resource identity."""

from __future__ import annotations

import re

_AWS_ACCOUNT_ID = re.compile(r"(?<![A-Za-z0-9])[0-9]{12}(?![A-Za-z0-9])")
_AWS_ECS_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_AWS_TASK_REVISION = re.compile(r"[1-9][0-9]{0,9}\Z")
_RELEASE_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}\Z")


def is_aws_ecs_name(value: str) -> bool:
    """Return whether a value is a bounded ECS name, never an ARN/account."""

    return _AWS_ACCOUNT_ID.search(value) is None and _AWS_ECS_NAME.fullmatch(value) is not None


def is_aws_task_revision(value: str) -> bool:
    """Return whether a value is a positive bounded task-definition revision."""

    return _AWS_TASK_REVISION.fullmatch(value) is not None


def is_release_identity(value: str) -> bool:
    """Return whether a value is a SHA/version/digest, not a resource locator."""

    return (
        not value.casefold().startswith("arn:") and _AWS_ACCOUNT_ID.search(value) is None and _RELEASE_IDENTITY.fullmatch(value) is not None
    )


__all__ = ["is_aws_ecs_name", "is_aws_task_revision", "is_release_identity"]

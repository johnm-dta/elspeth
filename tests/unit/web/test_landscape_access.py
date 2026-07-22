from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from elspeth.web.deployment_contract import DEPLOYMENT_TARGET_AWS_ECS


class _FakeLandscapeDB:
    calls: ClassVar[list[tuple[str, dict[str, object]]]] = []
    sentinel = object()

    @classmethod
    def from_url(cls, url: str, **kwargs: object) -> object:
        cls.calls.append((url, kwargs))
        return cls.sentinel


def _settings(
    deployment_target: str,
    *,
    url: str = "sqlite:///landscape.db",
    passphrase: str | None = None,
) -> Any:
    return SimpleNamespace(
        deployment_target=deployment_target,
        landscape_passphrase=passphrase,
        get_landscape_url=lambda: url,
    )


@pytest.fixture(autouse=True)
def _patch_landscape_db(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeLandscapeDB.calls = []
    monkeypatch.setattr("elspeth.web.landscape_access.LandscapeDB", _FakeLandscapeDB)


def test_aws_ecs_disables_create_tables() -> None:
    from elspeth.web.landscape_access import open_landscape_db

    result = open_landscape_db(_settings(DEPLOYMENT_TARGET_AWS_ECS))

    assert result is _FakeLandscapeDB.sentinel
    assert _FakeLandscapeDB.calls[0][1]["create_tables"] is False


def test_local_default_keeps_create_tables() -> None:
    from elspeth.web.landscape_access import open_landscape_db

    open_landscape_db(_settings("default"))

    assert _FakeLandscapeDB.calls[0][1]["create_tables"] is True


def test_unknown_deployment_target_fails_before_url_or_db_open(monkeypatch: pytest.MonkeyPatch) -> None:
    from elspeth.web.landscape_access import open_landscape_db

    url_was_read = False

    def _get_url() -> str:
        nonlocal url_was_read
        url_was_read = True
        return "sqlite:///must-not-open.db"

    settings = SimpleNamespace(
        deployment_target="future-target",
        landscape_passphrase=None,
        get_landscape_url=_get_url,
    )

    with pytest.raises(ValueError, match="unsupported deployment_target"):
        open_landscape_db(settings)

    assert url_was_read is False
    assert _FakeLandscapeDB.calls == []


def test_forwards_url_and_passphrase() -> None:
    from elspeth.web.landscape_access import open_landscape_db

    url = "sqlite:///specific.db"
    passphrase = "passphrase-sentinel"

    open_landscape_db(_settings("default", url=url, passphrase=passphrase))

    assert _FakeLandscapeDB.calls == [
        (url, {"passphrase": passphrase, "create_tables": True}),
    ]


def test_postgres_url_gets_pool_kwargs() -> None:
    from elspeth.web.landscape_access import open_landscape_db
    from elspeth.web.schema_probe import AWS_ECS_POOL_KWARGS

    open_landscape_db(_settings(DEPLOYMENT_TARGET_AWS_ECS, url="postgresql+psycopg://u@h/db"))

    _, kwargs = _FakeLandscapeDB.calls[0]
    assert kwargs.items() >= AWS_ECS_POOL_KWARGS.items()


def test_sqlite_url_gets_no_pool_kwargs() -> None:
    from elspeth.web.landscape_access import open_landscape_db

    open_landscape_db(_settings("default", url="sqlite:///x.db"))

    _, kwargs = _FakeLandscapeDB.calls[0]
    assert "pool_size" not in kwargs

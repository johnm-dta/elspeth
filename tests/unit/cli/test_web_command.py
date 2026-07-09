"""Tests for the elspeth web CLI command."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from typer.testing import CliRunner

from elspeth.cli import app

runner = CliRunner()


@dataclass(frozen=True)
class UvicornRunCall:
    args: tuple[object, ...]
    kwargs: dict[str, object]


@dataclass
class UvicornRunRecorder:
    side_effect: Callable[..., None] | None = None
    calls: list[UvicornRunCall] = field(default_factory=list)

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.calls.append(UvicornRunCall(args=args, kwargs=dict(kwargs)))
        if self.side_effect is not None:
            self.side_effect(*args, **kwargs)


class FakeUvicornModule(ModuleType):
    run: UvicornRunRecorder

    def __init__(self, side_effect: Callable[..., None] | None = None) -> None:
        super().__init__("uvicorn")
        self.run = UvicornRunRecorder(side_effect=side_effect)


class TestWebCommandImportGuard:
    """Tests for the [webui] import guard."""

    def test_missing_uvicorn_prints_install_instruction(self) -> None:
        with patch.dict("sys.modules", {"uvicorn": None}):
            result = runner.invoke(app, ["web"])
        assert result.exit_code == 1
        assert "[webui]" in result.output

    def test_missing_uvicorn_exits_code_1(self) -> None:
        with patch.dict("sys.modules", {"uvicorn": None}):
            result = runner.invoke(app, ["web"])
        assert result.exit_code == 1


class TestWebCommandHappyPath:
    """Tests for the web command when [webui] is installed.

    Note: WebSettings validates that non-local hosts require a non-default
    secret_key (_enforce_secret_key_in_production). Tests using --host 0.0.0.0
    must also set ELSPETH_WEB__SECRET_KEY to pass validation.
    """

    def test_calls_uvicorn_run_with_factory_true(self) -> None:
        """Use default host (127.0.0.1) with custom port — avoids secret_key guard."""
        uvicorn = FakeUvicornModule()
        with patch.dict("sys.modules", {"uvicorn": uvicorn}):
            runner.invoke(app, ["web", "--port", "9999"])

        assert len(uvicorn.run.calls) == 1
        assert uvicorn.run.calls[0].kwargs["factory"] is True

    def test_passes_correct_host_and_port_to_uvicorn(self) -> None:
        """Use localhost with custom port — verifies host and port forwarding."""
        uvicorn = FakeUvicornModule()
        with patch.dict("sys.modules", {"uvicorn": uvicorn}):
            runner.invoke(app, ["web", "--port", "9999", "--host", "127.0.0.1"])

        call = uvicorn.run.calls[0]
        assert call.kwargs["host"] == "127.0.0.1"
        assert call.kwargs["port"] == 9999

    def test_non_local_host_with_default_secret_key_fails(self) -> None:
        """0.0.0.0 with default secret_key triggers the production guard."""
        import os

        uvicorn = FakeUvicornModule()
        # --no-dotenv prevents .env loading (which sets ELSPETH_WEB__SECRET_KEY).
        # Also scrub the key from env in case prior tests leaked it via the
        # web command's env-var bridging (cli.py:2324).
        env_without_key = {k: v for k, v in os.environ.items() if k != "ELSPETH_WEB__SECRET_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True), patch.dict("sys.modules", {"uvicorn": uvicorn}):
            result = runner.invoke(app, ["--no-dotenv", "web", "--host", "0.0.0.0"])

        assert result.exit_code != 0
        assert uvicorn.run.calls == []

    def test_uses_create_app_factory_string(self) -> None:
        uvicorn = FakeUvicornModule()
        with patch.dict("sys.modules", {"uvicorn": uvicorn}):
            runner.invoke(app, ["web"])

        assert uvicorn.run.calls[0].args[0] == "elspeth.web.app:create_app"

    def test_reload_flag_forwarded(self) -> None:
        uvicorn = FakeUvicornModule()
        with patch.dict("sys.modules", {"uvicorn": uvicorn}):
            runner.invoke(app, ["web", "--reload"])

        assert uvicorn.run.calls[0].kwargs["reload"] is True

    def test_default_host_requires_no_secret_key(self) -> None:
        """Default host (127.0.0.1) should work with the default secret_key."""
        uvicorn = FakeUvicornModule()
        with patch.dict("sys.modules", {"uvicorn": uvicorn}):
            result = runner.invoke(app, ["web"])

        assert result.exit_code == 0
        assert len(uvicorn.run.calls) == 1


class TestWebCommandAuthBridging:
    """Tests for --auth env var bridging.

    The CLI bridges --auth to ELSPETH_WEB__AUTH_PROVIDER for create_app().
    Full auth provider validation (OIDC required fields, invalid providers)
    is tested in tests/unit/web/test_config.py at the WebSettings level.
    """

    def test_auth_provider_bridged_to_env_var(self) -> None:
        """--auth=oidc sets ELSPETH_WEB__AUTH_PROVIDER for create_app()."""
        import os

        captured_env: dict[str, str] = {}

        def capture_env(*args: object, **kwargs: object) -> None:
            captured_env["auth"] = os.environ.get("ELSPETH_WEB__AUTH_PROVIDER", "")

        uvicorn = FakeUvicornModule(side_effect=capture_env)
        with patch.dict("sys.modules", {"uvicorn": uvicorn}):
            runner.invoke(app, ["web", "--auth", "oidc"])

        assert captured_env["auth"] == "oidc"

    def test_default_auth_bridged_as_local(self) -> None:
        """Default --auth=local sets ELSPETH_WEB__AUTH_PROVIDER=local."""
        import os

        captured_env: dict[str, str] = {}

        def capture_env(*args: object, **kwargs: object) -> None:
            captured_env["auth"] = os.environ.get("ELSPETH_WEB__AUTH_PROVIDER", "")

        uvicorn = FakeUvicornModule(side_effect=capture_env)
        with patch.dict("sys.modules", {"uvicorn": uvicorn}):
            result = runner.invoke(app, ["web", "--auth", "local"])

        assert result.exit_code == 0
        assert captured_env["auth"] == "local"


def _auth_user_row(auth_db: Path, username: str) -> tuple[str, str | None, int] | None:
    with closing(sqlite3.connect(str(auth_db))) as conn:
        row = conn.execute(
            "SELECT display_name, email, email_verified FROM users WHERE user_id = ?",
            (username,),
        ).fetchone()
    return row


class TestComposerUsersCommand:
    """Tests for local composer user management commands."""

    def test_add_user_creates_verified_local_auth_user(self, tmp_path: Path) -> None:
        auth_db = tmp_path / "auth.db"

        result = runner.invoke(
            app,
            [
                "--no-dotenv",
                "composer",
                "users",
                "add",
                "alice",
                "--password",
                "password123",
                "--display-name",
                "Alice Smith",
                "--email",
                "alice@example.com",
                "--auth-db",
                str(auth_db),
            ],
        )

        assert result.exit_code == 0
        assert "Added composer user alice" in result.output
        assert _auth_user_row(auth_db, "alice") == (
            "Alice Smith",
            "alice@example.com",
            1,
        )

    def test_remove_user_deletes_local_auth_user(self, tmp_path: Path) -> None:
        auth_db = tmp_path / "auth.db"
        add_result = runner.invoke(
            app,
            [
                "--no-dotenv",
                "composer",
                "users",
                "add",
                "alice",
                "--password",
                "password123",
                "--auth-db",
                str(auth_db),
            ],
        )
        assert add_result.exit_code == 0

        result = runner.invoke(
            app,
            [
                "--no-dotenv",
                "composer",
                "users",
                "remove",
                "alice",
                "--auth-db",
                str(auth_db),
                "--yes",
            ],
        )

        assert result.exit_code == 0
        assert "Removed composer user alice" in result.output
        assert _auth_user_row(auth_db, "alice") is None

    def test_remove_missing_auth_db_does_not_create_new_database(self, tmp_path: Path) -> None:
        auth_db = tmp_path / "missing" / "auth.db"

        result = runner.invoke(
            app,
            [
                "--no-dotenv",
                "composer",
                "users",
                "remove",
                "alice",
                "--auth-db",
                str(auth_db),
                "--yes",
            ],
        )

        assert result.exit_code == 1
        assert "auth database not found" in result.output
        assert not auth_db.exists()

    def test_add_unverified_option_is_not_supported(self, tmp_path: Path) -> None:
        auth_db = tmp_path / "auth.db"

        result = runner.invoke(
            app,
            [
                "--no-dotenv",
                "composer",
                "users",
                "add",
                "alice",
                "--password",
                "password123",
                "--auth-db",
                str(auth_db),
                "--unverified",
            ],
        )

        assert result.exit_code != 0
        assert not auth_db.exists()

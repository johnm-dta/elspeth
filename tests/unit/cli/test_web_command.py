"""Tests for the elspeth web CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from elspeth.cli import app

runner = CliRunner()


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
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            mock_uvicorn.run = MagicMock()
            runner.invoke(app, ["web", "--port", "9999"])

        mock_uvicorn.run.assert_called_once()
        call_kwargs = mock_uvicorn.run.call_args
        assert call_kwargs.kwargs.get("factory") is True or call_kwargs[1].get("factory") is True

    def test_passes_correct_host_and_port_to_uvicorn(self) -> None:
        """Use localhost with custom port — verifies host and port forwarding."""
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            mock_uvicorn.run = MagicMock()
            runner.invoke(app, ["web", "--port", "9999", "--host", "127.0.0.1"])

        call_kwargs = mock_uvicorn.run.call_args
        assert call_kwargs.kwargs.get("host") == "127.0.0.1" or call_kwargs[1].get("host") == "127.0.0.1"
        assert call_kwargs.kwargs.get("port") == 9999 or call_kwargs[1].get("port") == 9999

    def test_non_local_host_with_default_secret_key_fails(self) -> None:
        """0.0.0.0 with default secret_key triggers the production guard."""
        import os

        mock_uvicorn = MagicMock()
        # --no-dotenv prevents .env loading (which sets ELSPETH_WEB__SECRET_KEY).
        # Also scrub the key from env in case prior tests leaked it via the
        # web command's env-var bridging (cli.py:2324).
        env_without_key = {k: v for k, v in os.environ.items() if k != "ELSPETH_WEB__SECRET_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True), patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            mock_uvicorn.run = MagicMock()
            result = runner.invoke(app, ["--no-dotenv", "web", "--host", "0.0.0.0"])

        assert result.exit_code != 0
        mock_uvicorn.run.assert_not_called()

    def test_uses_create_app_factory_string(self) -> None:
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            mock_uvicorn.run = MagicMock()
            runner.invoke(app, ["web"])

        call_args = mock_uvicorn.run.call_args
        assert call_args[0][0] == "elspeth.web.app:create_app"

    def test_reload_flag_forwarded(self) -> None:
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            mock_uvicorn.run = MagicMock()
            runner.invoke(app, ["web", "--reload"])

        call_kwargs = mock_uvicorn.run.call_args
        assert call_kwargs.kwargs.get("reload") is True or call_kwargs[1].get("reload") is True

    def test_default_host_requires_no_secret_key(self) -> None:
        """Default host (127.0.0.1) should work with the default secret_key."""
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            mock_uvicorn.run = MagicMock()
            result = runner.invoke(app, ["web"])

        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once()


class TestWebCommandAuthBridging:
    """Tests for --auth env var bridging.

    The CLI bridges --auth to ELSPETH_WEB__AUTH_PROVIDER for create_app().
    Full auth provider validation (OIDC required fields, invalid providers)
    is tested in tests/unit/web/test_config.py at the WebSettings level.
    """

    def test_auth_provider_bridged_to_env_var(self) -> None:
        """--auth=oidc sets ELSPETH_WEB__AUTH_PROVIDER for create_app()."""
        import os

        mock_uvicorn = MagicMock()
        captured_env: dict[str, str] = {}
        original_run = MagicMock()

        def capture_env(*args: object, **kwargs: object) -> None:
            captured_env["auth"] = os.environ.get("ELSPETH_WEB__AUTH_PROVIDER", "")
            original_run(*args, **kwargs)

        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            mock_uvicorn.run = MagicMock(side_effect=capture_env)
            runner.invoke(app, ["web", "--auth", "oidc"])

        assert captured_env["auth"] == "oidc"

    def test_default_auth_bridged_as_local(self) -> None:
        """Default --auth=local sets ELSPETH_WEB__AUTH_PROVIDER=local."""
        import os

        mock_uvicorn = MagicMock()
        captured_env: dict[str, str] = {}

        def capture_env(*args: object, **kwargs: object) -> None:
            captured_env["auth"] = os.environ.get("ELSPETH_WEB__AUTH_PROVIDER", "")

        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            mock_uvicorn.run = MagicMock(side_effect=capture_env)
            result = runner.invoke(app, ["web", "--auth", "local"])

        assert result.exit_code == 0
        assert captured_env["auth"] == "local"

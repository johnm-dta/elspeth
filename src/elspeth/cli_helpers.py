"""CLI helper functions for database resolution and audit passphrases."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from elspeth.plugins.infrastructure.runtime_factory import (
    PluginBundle as PluginBundle,
)
from elspeth.plugins.infrastructure.runtime_factory import (
    instantiate_plugins_from_config as instantiate_plugins_from_config,
)
from elspeth.plugins.infrastructure.runtime_factory import (
    make_sink_factory as _make_sink_factory,
)

__all__ = [
    "PluginBundle",
    "_make_sink_factory",
    "instantiate_plugins_from_config",
    "resolve_audit_passphrase",
    "resolve_database_url",
    "resolve_latest_run_id",
    "resolve_run_id",
]

if TYPE_CHECKING:
    from elspeth.core.config import ElspethSettings, LandscapeSettings
    from elspeth.core.landscape.factory import RecorderFactory


def resolve_database_url(
    database: str | None,
    settings_path: Path | None,
) -> tuple[str, "ElspethSettings | None"]:
    """Resolve database URL from CLI option or settings file.

    Priority: CLI --database > explicit --settings > settings.yaml landscape.url

    Args:
        database: Explicit database path from CLI (optional)
        settings_path: Path to settings.yaml file (optional)

    Returns:
        Tuple of (database_url, config_or_none)

    Raises:
        ValueError: If database file not found, settings invalid, or neither provided
    """
    from elspeth.core.config import load_settings

    config: ElspethSettings | None = None

    if database:
        db_path = Path(database).expanduser().resolve()
        # Fail fast with clear error if file doesn't exist
        if not db_path.exists():
            raise ValueError(f"Database file not found: {db_path}")
        if settings_path is not None:
            normalized_settings = settings_path.expanduser().resolve()
            if not normalized_settings.exists():
                raise ValueError(f"Settings file not found: {normalized_settings}")
        return f"sqlite:///{db_path}", None

    # Try explicit settings file
    if settings_path is not None:
        normalized_settings = settings_path.expanduser().resolve()
        if not normalized_settings.exists():
            raise ValueError(f"Settings file not found: {normalized_settings}")
        try:
            config = load_settings(normalized_settings)
        except Exception as e:
            raise ValueError(f"Error loading settings from {settings_path}: {e}") from e

    if config is not None:
        return config.landscape.url, config

    # Try default settings.yaml - DO NOT silently swallow errors
    default_settings = Path("settings.yaml")
    if default_settings.exists():
        try:
            config = load_settings(default_settings)
            return config.landscape.url, config
        except Exception as e:
            # Don't silently fall through - user should know why settings.yaml failed
            raise ValueError(f"Error loading default settings.yaml: {e}") from e

    raise ValueError("No database specified. Provide --database or ensure settings.yaml exists with landscape.url configured.")


def resolve_latest_run_id(factory: "RecorderFactory") -> str | None:
    """Get the most recently started run ID.

    Args:
        factory: RecorderFactory with database connection

    Returns:
        Run ID of most recent run, or None if no runs exist
    """
    runs = factory.run_lifecycle.list_runs()
    if not runs:
        return None
    # list_runs returns ordered by started_at DESC
    return runs[0].run_id


def resolve_run_id(run_id: str, factory: "RecorderFactory") -> str | None:
    """Resolve run_id, handling 'latest' keyword.

    Args:
        run_id: Explicit run ID or 'latest'
        factory: RecorderFactory for looking up latest

    Returns:
        Resolved run ID, or None if 'latest' requested but no runs exist
    """
    if run_id.lower() == "latest":
        return resolve_latest_run_id(factory)
    return run_id


def resolve_audit_passphrase(
    settings: "LandscapeSettings | None",
) -> str | None:
    """Resolve the SQLCipher passphrase from the environment.

    The passphrase is always read from an environment variable (never from config
    files or URLs) to prevent it from appearing in logs, tracebacks, or the audit
    trail itself.

    When settings is None (e.g. ad-hoc CLI access via ``--database``), returns
    None — encryption requires explicit ``backend: sqlcipher`` configuration.
    This prevents ELSPETH_AUDIT_KEY from accidentally opening plain SQLite
    databases through SQLCipher.

    Args:
        settings: LandscapeSettings determining which env var to read.
            If None, returns None (no encryption without explicit config).

    Returns:
        Passphrase string if backend is sqlcipher, None otherwise.

    Raises:
        RuntimeError: If backend is sqlcipher but the env var is not set.
    """
    if settings is not None and settings.backend == "sqlcipher":
        env_var = settings.encryption_key_env
        passphrase = os.environ.get(env_var)
        if passphrase is None or not passphrase.strip():
            raise RuntimeError(
                f'SQLCipher backend requires a non-empty encryption passphrase.\nSet the environment variable: export {env_var}="your-passphrase"'
            )
        return passphrase

    # No settings, or settings.backend is not sqlcipher → no encryption.
    # We intentionally do NOT fall back to ELSPETH_AUDIT_KEY when settings
    # is None — that env var may be set for a different pipeline, and passing
    # a passphrase to a plain SQLite database causes "file is not a database".
    return None

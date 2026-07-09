from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_graph_audit_repository_uses_public_config_audit_boundary() -> None:
    graph_path = REPO_ROOT / "src/elspeth/core/landscape/data_flow/graph.py"
    tree = ast.parse(graph_path.read_text(), filename=str(graph_path))

    private_config_imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "elspeth.core.config"
        for alias in node.names
        if alias.name.startswith("_")
    ]

    assert private_config_imports == []


def test_public_node_config_audit_sanitizer_fingerprints_database_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elspeth.core.config import sanitize_node_config_for_audit

    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
    monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

    config = {
        "url": "postgresql://user:node-secret@host/db",  # secret-scan: allow-this-line
        "api_key": "sk-node-secret",
        "table": "results",
    }

    sanitized = sanitize_node_config_for_audit(config, plugin_name="database")
    sanitized_url = sanitized["url"]

    assert isinstance(sanitized_url, str)
    assert "node-secret" not in sanitized_url
    assert "api_key" not in sanitized
    assert "api_key_fingerprint" in sanitized
    assert "url_password_fingerprint" in sanitized
    assert sanitized["table"] == "results"

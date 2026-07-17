"""Cross-surface Node.js and npm runtime contract.

The project builds JavaScript in local development, CI, Docker, and the
source-checkout deployment runbooks.  Keep those surfaces on one supported
major line so a green local build cannot hide an older production toolchain.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

NODE_VERSION = "24.13.0"
NODE_ENGINE = ">=24 <25"
NPM_ENGINE = ">=11 <12"
PACKAGE_MANAGER = "npm@11.6.2"
SETUP_NODE_REVISION = "48b55a011bda9f5d6aeb4c2d9c7362e8dae4041e"


def _json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_manifests_and_locks_publish_the_node_24_contract() -> None:
    version_file = REPO_ROOT / ".node-version"
    assert version_file.is_file(), "repository must publish a root .node-version"
    assert version_file.read_text(encoding="utf-8").strip() == NODE_VERSION

    pairs = (
        (REPO_ROOT / "package.json", REPO_ROOT / "package-lock.json"),
        (
            REPO_ROOT / "src/elspeth/web/frontend/package.json",
            REPO_ROOT / "src/elspeth/web/frontend/package-lock.json",
        ),
    )
    expected_engines = {"node": NODE_ENGINE, "npm": NPM_ENGINE}
    for manifest_path, lock_path in pairs:
        manifest = _json(manifest_path)
        lock_root = _json(lock_path)["packages"][""]
        assert manifest["engines"] == expected_engines
        assert manifest["packageManager"] == PACKAGE_MANAGER
        assert lock_root["engines"] == expected_engines


def test_frontend_types_track_the_node_24_line() -> None:
    manifest = _json(REPO_ROOT / "src/elspeth/web/frontend/package.json")
    lock = _json(REPO_ROOT / "src/elspeth/web/frontend/package-lock.json")

    assert manifest["devDependencies"]["@types/node"].startswith("^24.")
    assert lock["packages"][""]["devDependencies"]["@types/node"].startswith("^24.")
    assert lock["packages"]["node_modules/@types/node"]["version"].startswith("24.")


def test_ci_and_release_image_build_with_node_24() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yaml").read_text(encoding="utf-8"))
    setup_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", ())
        if str(step.get("uses", "")).startswith("actions/setup-node@")
    ]

    assert len(setup_steps) == 2
    assert {step["uses"] for step in setup_steps} == {f"actions/setup-node@{SETUP_NODE_REVISION}"}
    assert {step["with"]["node-version"] for step in setup_steps} == {"24"}

    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(
        rf"^FROM node:{re.escape(NODE_VERSION)}-bookworm-slim@sha256:[0-9a-f]{{64}} AS frontend-builder$",
        dockerfile,
        flags=re.MULTILINE,
    )
    assert "FROM node:22" not in dockerfile


def test_active_deployment_runbooks_require_node_24() -> None:
    aws = (REPO_ROOT / "docs/runbooks/aws-ecs-deployment.md").read_text(encoding="utf-8")
    ansible = (REPO_ROOT / "docs/runbooks/ansible-ubuntu-deployment.md").read_text(encoding="utf-8")

    assert "Node 24/npm 11" in aws
    assert "Node 22/npm" not in aws

    assert "Node.js 24" in ansible
    assert "NodeSource Node 24.x" in ansible
    assert "https://deb.nodesource.com/node_24.x" in ansible
    assert "Node.js 20.19" not in ansible
    assert "NodeSource Node 20.x" not in ansible
    assert "https://deb.nodesource.com/node_20.x" not in ansible

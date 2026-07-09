"""Regression guards for deleted CI script references in active docs."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ANSIBLE_RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "ansible-ubuntu-deployment.md"
LINT_MIGRATION_STATUS = REPO_ROOT / "config" / "cicd" / "lint_migration_status.yaml"
DELETED_TIER_MODEL_SCRIPT = "scripts/cicd/enforce_tier_model.py"
SUPPORTED_TIER_MODEL_GATE = (
    "`elspeth-lints check --rules trust_tier.tier_model --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model`"
)
HISTORICAL_DOC_PREFIXES = (
    "docs-archive/",
    "docs/superpowers/plans/",
    "docs/superpowers/specs/",
)
HISTORICAL_DOC_PATHS = ("CHANGELOG.md",)
SOURCE_PREFIXES_OUT_OF_SCOPE = ("src/elspeth/web/",)


def _active_markdown_paths() -> list[Path]:
    paths = [*REPO_ROOT.glob("*.md"), *REPO_ROOT.glob("docs/**/*.md")]
    return [
        path
        for path in sorted(paths)
        if not any(path.relative_to(REPO_ROOT).as_posix().startswith(prefix) for prefix in HISTORICAL_DOC_PREFIXES)
    ]


def _deleted_migrated_scripts() -> set[str]:
    scripts: set[str] = set()
    text = LINT_MIGRATION_STATUS.read_text(encoding="utf-8")
    for block in text.split("\n  - old_script: ")[1:]:
        old_script = block.splitlines()[0].strip()
        if re.search(r"(?m)^    status: deleted$", block):
            scripts.add(old_script)
    return scripts


def _active_reference_paths() -> list[Path]:
    paths = [
        *REPO_ROOT.glob("*.md"),
        *REPO_ROOT.glob("docs/**/*.md"),
        *REPO_ROOT.glob("src/elspeth/**/*.py"),
    ]
    return [
        path
        for path in sorted(paths)
        if not any(path.relative_to(REPO_ROOT).as_posix().startswith(prefix) for prefix in HISTORICAL_DOC_PREFIXES)
        and not any(path.relative_to(REPO_ROOT).as_posix().startswith(prefix) for prefix in SOURCE_PREFIXES_OUT_OF_SCOPE)
        and path.relative_to(REPO_ROOT).as_posix() not in HISTORICAL_DOC_PATHS
    ]


def test_active_docs_do_not_reference_deleted_tier_model_script() -> None:
    offenders: list[str] = []

    for path in _active_markdown_paths():
        text = path.read_text(encoding="utf-8")
        if DELETED_TIER_MODEL_SCRIPT in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []


def test_active_docs_and_non_web_source_do_not_reference_deleted_migrated_ci_scripts() -> None:
    offenders: list[str] = []

    for path in _active_reference_paths():
        text = path.read_text(encoding="utf-8")
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        for script in _deleted_migrated_scripts():
            for needle in (script, Path(script).name):
                if needle in text:
                    offenders.append(f"{rel_path}: {needle}")

    assert offenders == []


def test_ansible_runbook_documents_supported_tier_model_gate() -> None:
    text = ANSIBLE_RUNBOOK.read_text(encoding="utf-8")

    assert SUPPORTED_TIER_MODEL_GATE in text
    assert "ELSPETH_JUDGE_METADATA_HMAC_KEY" in text

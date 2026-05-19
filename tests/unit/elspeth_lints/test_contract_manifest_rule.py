"""Tests for the contract-manifest elspeth-lints rule."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from elspeth_lints.core.protocols import RuleContext
from elspeth_lints.rules.manifest.contract_manifest import RULE


def test_contract_manifest_reports_extra_registration(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "elspeth"
    _write_manifest(source_root, {})
    _write_contract(
        source_root / "contracts" / "ghost.py",
        """
        from elspeth.contracts.declaration_contracts import (
            DeclarationContract,
            implements_dispatch_site,
            register_declaration_contract,
        )


        class GhostContract(DeclarationContract):
            name = "ghost_not_in_manifest"

            @implements_dispatch_site("post_emission_check")
            def post_emission_check(self, inputs, outputs):
                raise NotImplementedError


        register_declaration_contract(GhostContract())
        """,
    )

    findings = list(RULE.analyze(ast.Module(body=[], type_ignores=[]), source_root, RuleContext(root=source_root)))

    assert [finding.rule_id for finding in findings] == ["MC1"]
    assert findings[0].file_path == "src/elspeth/contracts/ghost.py"
    assert findings[0].fingerprint == "src/elspeth/contracts/ghost.py:MC1:ghost_not_in_manifest"


def test_contract_manifest_passes_matching_manifest_and_marker(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "elspeth"
    _write_manifest(source_root, {"passes_through_input": ["post_emission_check"]})
    _write_contract(
        source_root / "contracts" / "pass_through.py",
        """
        from elspeth.contracts.declaration_contracts import (
            DeclarationContract,
            implements_dispatch_site,
            register_declaration_contract,
        )


        class PassThroughContract(DeclarationContract):
            name = "passes_through_input"

            @implements_dispatch_site("post_emission_check")
            def post_emission_check(self, inputs, outputs):
                raise NotImplementedError


        register_declaration_contract(PassThroughContract())
        """,
    )

    findings = list(RULE.analyze(ast.Module(body=[], type_ignores=[]), source_root, RuleContext(root=source_root)))

    assert findings == []


def _write_manifest(source_root: Path, entries: dict[str, list[str]]) -> None:
    manifest_path = source_root / "contracts" / "declaration_contracts.py"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_entries: list[str] = []
    for name, sites in entries.items():
        rendered_sites = ", ".join(repr(site) for site in sites)
        rendered_entries.append(f"{name!r}: frozenset({{{rendered_sites}}})")
    body = "{" + ", ".join(rendered_entries) + "}"
    manifest_path.write_text(
        textwrap.dedent(
            f"""
            from types import MappingProxyType

            EXPECTED_CONTRACT_SITES = MappingProxyType({body})
            """
        ),
        encoding="utf-8",
    )


def _write_contract(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")

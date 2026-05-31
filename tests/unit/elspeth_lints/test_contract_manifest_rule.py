"""Tests for the contract-manifest elspeth-lints rule."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from elspeth_lints.core.protocols import RuleContext
from elspeth_lints.rules.manifest.contract_manifest import RULE
from elspeth_lints.rules.manifest.contract_manifest.metadata import RULE_MC2, RULE_MC3A, RULE_MC3B, RULE_MC3C


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


def test_contract_manifest_reports_missing_registration(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "elspeth"
    _write_manifest(
        source_root,
        {
            "present_contract": ["post_emission_check"],
            "missing_contract": ["post_emission_check"],
        },
    )
    _write_registration(source_root, "contracts/present.py", "PresentContract", "present_contract", marker_sites=["post_emission_check"])

    findings = list(RULE.analyze(ast.Module(body=[], type_ignores=[]), source_root, RuleContext(root=source_root)))

    assert [finding.rule_id for finding in findings] == [RULE_MC2]
    assert findings[0].file_path == "src/elspeth/contracts/declaration_contracts.py"
    assert "missing_contract" in findings[0].fingerprint


def test_contract_manifest_reports_marker_without_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "elspeth"
    _write_manifest(source_root, {"declared_output_fields": ["post_emission_check"]})
    _write_registration(
        source_root,
        "contracts/declared_output_fields.py",
        "DeclaredOutputFieldsContract",
        "declared_output_fields",
        marker_sites=["post_emission_check", "batch_flush_check"],
    )

    findings = list(RULE.analyze(ast.Module(body=[], type_ignores=[]), source_root, RuleContext(root=source_root)))

    assert [finding.rule_id for finding in findings] == [RULE_MC3A]
    assert "declared_output_fields::batch_flush_check" in findings[0].fingerprint


def test_contract_manifest_reports_manifest_without_marker(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "elspeth"
    _write_manifest(source_root, {"schema_config_mode": ["post_emission_check", "batch_flush_check"]})
    _write_registration(
        source_root,
        "contracts/schema_config_mode.py",
        "SchemaConfigModeContract",
        "schema_config_mode",
        marker_sites=["post_emission_check"],
    )

    findings = list(RULE.analyze(ast.Module(body=[], type_ignores=[]), source_root, RuleContext(root=source_root)))

    assert [finding.rule_id for finding in findings] == [RULE_MC3B]
    assert "schema_config_mode::batch_flush_check" in findings[0].fingerprint


def test_contract_manifest_reports_trivial_marker_body(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "elspeth"
    _write_manifest(source_root, {"can_drop_rows": ["post_emission_check"]})
    _write_registration(
        source_root,
        "contracts/can_drop_rows.py",
        "CanDropRowsContract",
        "can_drop_rows",
        marker_sites=["post_emission_check"],
        trivial_body_sites=["post_emission_check"],
    )

    findings = list(RULE.analyze(ast.Module(body=[], type_ignores=[]), source_root, RuleContext(root=source_root)))

    assert [finding.rule_id for finding in findings] == [RULE_MC3C]
    assert "can_drop_rows::post_emission_check" in findings[0].fingerprint


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


def _write_registration(
    source_root: Path,
    relative_path: str,
    class_name: str,
    contract_name: str,
    *,
    marker_sites: list[str],
    trivial_body_sites: list[str] | None = None,
) -> None:
    trivial_body_sites = trivial_body_sites or []
    method_defs: list[str] = []
    for site in marker_sites:
        body = "pass" if site in trivial_body_sites else "raise NotImplementedError"
        method_defs.append(f"    @implements_dispatch_site({site!r})\n    def {site}(self, inputs, outputs):\n        {body}\n")
    methods_block = "\n".join(method_defs)
    header = textwrap.dedent(
        f"""
        from elspeth.contracts.declaration_contracts import (
            DeclarationContract,
            implements_dispatch_site,
            register_declaration_contract,
        )


        class {class_name}(DeclarationContract):
            name = {contract_name!r}
        """
    )
    footer = textwrap.dedent(
        f"""


        register_declaration_contract({class_name}())
        """
    )
    _write_contract(
        source_root / relative_path,
        header + methods_block + footer,
    )

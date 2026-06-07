"""Tests for the contract-manifest elspeth-lints rule."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from elspeth_lints.core.protocols import RuleContext
from elspeth_lints.rules.manifest.contract_manifest import RULE
from elspeth_lints.rules.manifest.contract_manifest.metadata import RULE_MC1, RULE_MC2, RULE_MC3A, RULE_MC3B, RULE_MC3C
from elspeth_lints.rules.manifest.contract_manifest.rule import load_contract_allowlist


def test_load_contract_allowlist_rejects_malformed_expiry(tmp_path: Path) -> None:
    """A typoed ``expires`` must fail closed, not silently drop the time bound.

    Regression for elspeth-99ae5c0991: ``_parse_date`` swallowed malformed
    dates and returned ``None``, so ``fail_on_expired`` could never enforce a
    typoed expiry and a one-character diff silently disabled the bound.
    """
    (tmp_path / "rules.yaml").write_text(
        """
allow_contracts:
  - key: ghost_not_in_manifest
    owner: tests
    reason: synthetic fixture
    task: tests
    expires: not-a-date
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expires"):
        load_contract_allowlist(tmp_path)


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


def test_contract_manifest_rejects_shadowed_register_call(tmp_path: Path) -> None:
    # elspeth-1e8f4ece9a: a locally-defined no-op register_declaration_contract
    # (NOT imported from the canonical module) must not be counted as a real
    # registration, so the manifest entry is left unresolved (MC2).
    source_root = tmp_path / "src" / "elspeth"
    _write_manifest(source_root, {"shadowed": ["post_emission_check"]})
    _write_contract(
        source_root / "contracts" / "shadowed.py",
        """
        from elspeth.contracts.declaration_contracts import (
            DeclarationContract,
            implements_dispatch_site,
        )


        def register_declaration_contract(contract):
            return None


        class ShadowContract(DeclarationContract):
            name = "shadowed"

            @implements_dispatch_site("post_emission_check")
            def post_emission_check(self, inputs, outputs):
                raise NotImplementedError


        register_declaration_contract(ShadowContract())
        """,
    )

    findings = list(RULE.analyze(ast.Module(body=[], type_ignores=[]), source_root, RuleContext(root=source_root)))

    assert [finding.rule_id for finding in findings] == [RULE_MC2]
    assert "shadowed" in findings[0].fingerprint


def test_contract_manifest_rejects_shadowed_dispatch_marker(tmp_path: Path) -> None:
    # elspeth-487dfef2ce: a locally-defined no-op implements_dispatch_site must
    # not be counted as a real marker, so the manifested site is reported missing
    # (MC3b) rather than silently satisfied.
    source_root = tmp_path / "src" / "elspeth"
    _write_manifest(source_root, {"shadow_marker": ["post_emission_check"]})
    _write_contract(
        source_root / "contracts" / "shadow_marker.py",
        """
        from elspeth.contracts.declaration_contracts import (
            DeclarationContract,
            register_declaration_contract,
        )


        def implements_dispatch_site(site):
            def decorate(fn):
                return fn

            return decorate


        class ShadowMarkerContract(DeclarationContract):
            name = "shadow_marker"

            @implements_dispatch_site("post_emission_check")
            def post_emission_check(self, inputs, outputs):
                raise NotImplementedError


        register_declaration_contract(ShadowMarkerContract())
        """,
    )

    findings = list(RULE.analyze(ast.Module(body=[], type_ignores=[]), source_root, RuleContext(root=source_root)))

    assert [finding.rule_id for finding in findings] == [RULE_MC3B]
    assert "shadow_marker::post_emission_check" in findings[0].fingerprint


def test_contract_manifest_reports_duplicate_registration(tmp_path: Path) -> None:
    # elspeth-07d9f8a619: two registrations sharing a contract name must be
    # flagged (runtime register_declaration_contract raises ValueError on a dup),
    # not silently deduped into a set.
    source_root = tmp_path / "src" / "elspeth"
    _write_manifest(source_root, {"dup": ["post_emission_check"]})
    _write_contract(
        source_root / "contracts" / "dup.py",
        """
        from elspeth.contracts.declaration_contracts import (
            DeclarationContract,
            implements_dispatch_site,
            register_declaration_contract,
        )


        class DupOne(DeclarationContract):
            name = "dup"

            @implements_dispatch_site("post_emission_check")
            def post_emission_check(self, inputs, outputs):
                raise NotImplementedError


        class DupTwo(DeclarationContract):
            name = "dup"

            @implements_dispatch_site("post_emission_check")
            def post_emission_check(self, inputs, outputs):
                raise NotImplementedError


        register_declaration_contract(DupOne())
        register_declaration_contract(DupTwo())
        """,
    )

    findings = list(RULE.analyze(ast.Module(body=[], type_ignores=[]), source_root, RuleContext(root=source_root)))

    assert [finding.rule_id for finding in findings] == [RULE_MC1]
    assert "dup::duplicate@" in findings[0].fingerprint


def test_contract_manifest_accepts_keyword_form_marker(tmp_path: Path) -> None:
    # elspeth-2b5edd369e: the runtime decorator accepts site_name as a keyword,
    # so @implements_dispatch_site(site_name="...") is a valid marker shape and
    # must not raise a spurious MC3b.
    source_root = tmp_path / "src" / "elspeth"
    _write_manifest(source_root, {"kw": ["post_emission_check"]})
    _write_contract(
        source_root / "contracts" / "kw.py",
        """
        from elspeth.contracts.declaration_contracts import (
            DeclarationContract,
            implements_dispatch_site,
            register_declaration_contract,
        )


        class KeywordContract(DeclarationContract):
            name = "kw"

            @implements_dispatch_site(site_name="post_emission_check")
            def post_emission_check(self, inputs, outputs):
                raise NotImplementedError


        register_declaration_contract(KeywordContract())
        """,
    )

    findings = list(RULE.analyze(ast.Module(body=[], type_ignores=[]), source_root, RuleContext(root=source_root)))

    assert findings == []


def test_contract_manifest_json_mode_succeeds_on_current_codebase(
    elspeth_lints_subprocess_env: dict[str, str],
) -> None:
    # Real-codebase finding-set invariance: the provenance/dup/keyword tightenings
    # must not change the verdict on the live tree (zero blast radius).
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "elspeth_lints.core.cli",
            "check",
            "--rules",
            "manifest.contract_manifest",
            "--root",
            "src/elspeth",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
        env=elspeth_lints_subprocess_env,
    )

    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert json.loads(result.stdout) == []


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

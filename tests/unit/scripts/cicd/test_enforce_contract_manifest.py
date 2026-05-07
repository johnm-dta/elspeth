"""Tests for scripts/cicd/enforce_contract_manifest.py.

Post-H2 (ADR-010 §Semantics amendment 2026-04-20): the scanner parses the
new ``EXPECTED_CONTRACT_SITES: Mapping[str, frozenset[DispatchSiteName]] =
MappingProxyType({...})`` manifest shape and checks five rules:

- MC1 (extra): register call with no manifest entry.
- MC2 (missing): manifest entry with no register call.
- MC3a (marker-without-manifest): class carries @implements_dispatch_site
  for a site not in the manifest.
- MC3b (manifest-without-marker): manifest names a site not marked on the
  concrete class.
- MC3c (trivial-body): marked method body is structurally trivial
  (pass / ... / bare return / literal-only).

Tests construct synthetic source trees in ``tmp_path`` so the scanner
exercises its real filesystem walk + AST parse + manifest-extract path.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from scripts.cicd.enforce_contract_manifest import (
    RULE_ID_EXTRA,
    RULE_ID_MANIFEST_WITHOUT_MARKER,
    RULE_ID_MARKER_WITHOUT_MANIFEST,
    RULE_ID_MISSING,
    RULE_ID_TRIVIAL_BODY,
    compute_findings,
    extract_manifest,
    scan_source_tree,
)

# ---------------------------------------------------------------------------
# Synthetic project-root helpers
# ---------------------------------------------------------------------------


def _write_manifest(root: Path, entries: dict[str, list[str]]) -> Path:
    """Write a manifest file with ``EXPECTED_CONTRACT_SITES`` in the new shape.

    ``entries`` is ``{contract_name: [site1, site2, ...]}``. Each contract's
    sites are rendered as ``frozenset({"site1", "site2"})``.
    """
    manifest_path = root / "declaration_contracts.py"

    if not entries:
        body = "{}"
    else:
        dict_entries = []
        for name, sites in entries.items():
            if sites:
                sites_repr = ", ".join(repr(s) for s in sites)
                rendered_sites = f"frozenset({{{sites_repr}}})"
            else:
                rendered_sites = "frozenset([])"
            dict_entries.append(f"{name!r}: {rendered_sites}")
        body = "{" + ", ".join(dict_entries) + "}"

    manifest_path.write_text(
        textwrap.dedent(
            f"""\
            from __future__ import annotations

            from types import MappingProxyType

            EXPECTED_CONTRACT_SITES = MappingProxyType({body})
            """
        )
    )
    return manifest_path


def _write_registration(
    root: Path,
    filename: str,
    class_name: str,
    contract_name: str,
    *,
    marker_sites: list[str] | None = None,
    trivial_body_sites: list[str] | None = None,
) -> Path:
    """Write a module that defines ``class_name`` (inheriting DeclarationContract)
    and registers an instance.

    ``marker_sites`` — sites the class decorates with @implements_dispatch_site.
    ``trivial_body_sites`` — subset of marker_sites whose method body is a
    trivial ``pass``. Sites in ``marker_sites`` but NOT in ``trivial_body_sites``
    get a non-trivial body (``raise NotImplementedError``).
    """
    marker_sites = marker_sites or []
    trivial_body_sites = trivial_body_sites or []

    method_defs: list[str] = []
    for site in marker_sites:
        if site in trivial_body_sites:
            body = "pass"
        else:
            body = "raise NotImplementedError"
        # Each method is indented 4 spaces (one class level) and the body
        # indented 8 spaces to land inside the class definition.
        method_defs.append(f"    @implements_dispatch_site({site!r})\n    def {site}(self, *args, **kwargs):\n        {body}\n")
    methods_block = "\n".join(method_defs)

    header = textwrap.dedent(
        f"""\
        from __future__ import annotations

        from declaration_contracts import (
            DeclarationContract,
            implements_dispatch_site,
            register_declaration_contract,
        )


        class {class_name}(DeclarationContract):
            name = {contract_name!r}
            payload_schema = dict

            def applies_to(self, plugin):
                return False

            @classmethod
            def negative_example(cls):
                raise NotImplementedError

            @classmethod
            def positive_example_does_not_apply(cls):
                raise NotImplementedError
        """
    )

    footer = textwrap.dedent(
        f"""


        register_declaration_contract({class_name}())
        """
    )

    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + methods_block + footer)
    return path


# ---------------------------------------------------------------------------
# Manifest extraction
# ---------------------------------------------------------------------------


class TestManifestExtraction:
    """extract_manifest parses EXPECTED_CONTRACT_SITES into per-contract site
    frozensets and records key line numbers."""

    def test_single_entry_extracted(self, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path, {"passes_through_input": ["post_emission_check"]})
        name_to_sites, name_to_line, _assign_line = extract_manifest(manifest)
        assert name_to_sites == {"passes_through_input": frozenset({"post_emission_check"})}
        assert "passes_through_input" in name_to_line

    def test_multi_site_entry_extracted(self, tmp_path: Path) -> None:
        manifest = _write_manifest(
            tmp_path,
            {"passes_through_input": ["post_emission_check", "batch_flush_check"]},
        )
        name_to_sites, _line, _al = extract_manifest(manifest)
        assert name_to_sites["passes_through_input"] == frozenset({"post_emission_check", "batch_flush_check"})

    def test_multiple_contracts_extracted(self, tmp_path: Path) -> None:
        manifest = _write_manifest(
            tmp_path,
            {
                "a_contract": ["post_emission_check"],
                "b_contract": ["pre_emission_check", "post_emission_check"],
            },
        )
        name_to_sites, _line, _al = extract_manifest(manifest)
        assert set(name_to_sites.keys()) == {"a_contract", "b_contract"}

    def test_missing_symbol_exits(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "declaration_contracts.py"
        manifest_path.write_text("SOMETHING_ELSE = 1\n")
        with pytest.raises(SystemExit) as exc_info:
            extract_manifest(manifest_path)
        assert exc_info.value.code == 1

    def test_unknown_site_name_exits(self, tmp_path: Path) -> None:
        """Sites must be drawn from the DispatchSite enum."""
        manifest_path = tmp_path / "declaration_contracts.py"
        manifest_path.write_text(
            textwrap.dedent(
                """\
                from types import MappingProxyType

                EXPECTED_CONTRACT_SITES = MappingProxyType(
                    {"contract": frozenset({"not_a_real_site"})}
                )
                """
            )
        )
        with pytest.raises(SystemExit):
            extract_manifest(manifest_path)

    def test_non_string_key_exits(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "declaration_contracts.py"
        manifest_path.write_text(
            textwrap.dedent(
                """\
                from types import MappingProxyType

                EXPECTED_CONTRACT_SITES = MappingProxyType(
                    {42: frozenset({"post_emission_check"})}
                )
                """
            )
        )
        with pytest.raises(SystemExit):
            extract_manifest(manifest_path)


# ---------------------------------------------------------------------------
# Happy path (MC1/MC2/MC3a/b/c all clean)
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_single_contract_no_findings(self, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path, {"passes_through_input": ["post_emission_check"]})
        _write_registration(
            tmp_path,
            "pass_through.py",
            "PassThroughContract",
            "passes_through_input",
            marker_sites=["post_emission_check"],
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        assert findings == []

    def test_multi_site_contract_no_findings(self, tmp_path: Path) -> None:
        manifest = _write_manifest(
            tmp_path,
            {"passes_through_input": ["post_emission_check", "batch_flush_check"]},
        )
        _write_registration(
            tmp_path,
            "pass_through.py",
            "PassThroughContract",
            "passes_through_input",
            marker_sites=["post_emission_check", "batch_flush_check"],
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        assert findings == []


# ---------------------------------------------------------------------------
# MC1 extra-registration
# ---------------------------------------------------------------------------


class TestMC1ExtraRegistration:
    def test_extra_registered_contract_detected(self, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path, {"passes_through_input": ["post_emission_check"]})
        _write_registration(
            tmp_path,
            "pass_through.py",
            "PassThroughContract",
            "passes_through_input",
            marker_sites=["post_emission_check"],
        )
        _write_registration(
            tmp_path,
            "ghost.py",
            "GhostContract",
            "ghost_not_in_manifest",
            marker_sites=["post_emission_check"],
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        extras = [f for f in findings if f.rule_id == RULE_ID_EXTRA and f.contract_name == "ghost_not_in_manifest"]
        assert len(extras) == 1
        assert "ghost.py" in extras[0].file_path


# ---------------------------------------------------------------------------
# MC2 missing-registration
# ---------------------------------------------------------------------------


class TestMC2MissingRegistration:
    def test_missing_registered_contract_detected(self, tmp_path: Path) -> None:
        manifest = _write_manifest(
            tmp_path,
            {"present_contract": ["post_emission_check"], "missing_contract": ["post_emission_check"]},
        )
        _write_registration(
            tmp_path,
            "present.py",
            "PresentContract",
            "present_contract",
            marker_sites=["post_emission_check"],
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        missing = [f for f in findings if f.rule_id == RULE_ID_MISSING]
        assert len(missing) == 1
        assert missing[0].contract_name == "missing_contract"


# ---------------------------------------------------------------------------
# MC3a / MC3b / MC3c — per-site rules (N1 acceptance)
# ---------------------------------------------------------------------------


class TestMC3aMarkerWithoutManifest:
    """MC3a: contract's @implements_dispatch_site marker names a site NOT
    listed in the manifest."""

    def test_extra_site_marker_detected(self, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path, {"c": ["post_emission_check"]})
        _write_registration(
            tmp_path,
            "c.py",
            "CContract",
            "c",
            marker_sites=["post_emission_check", "batch_flush_check"],  # batch_flush is extra
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        mc3a = [f for f in findings if f.rule_id == RULE_ID_MARKER_WITHOUT_MANIFEST]
        assert len(mc3a) == 1
        assert "batch_flush_check" in mc3a[0].contract_name

    def test_declared_output_fields_marker_without_manifest_detected(self, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path, {"declared_output_fields": ["post_emission_check"]})
        _write_registration(
            tmp_path,
            "declared_output_fields.py",
            "DeclaredOutputFieldsContract",
            "declared_output_fields",
            marker_sites=["post_emission_check", "batch_flush_check"],
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        mc3a = [f for f in findings if f.rule_id == RULE_ID_MARKER_WITHOUT_MANIFEST]
        assert len(mc3a) == 1
        assert "declared_output_fields" in mc3a[0].contract_name
        assert "batch_flush_check" in mc3a[0].contract_name

    def test_declared_required_fields_marker_without_manifest_detected(self, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path, {"declared_required_fields": []})
        _write_registration(
            tmp_path,
            "declared_required_fields.py",
            "DeclaredRequiredFieldsContract",
            "declared_required_fields",
            marker_sites=["pre_emission_check"],
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        mc3a = [f for f in findings if f.rule_id == RULE_ID_MARKER_WITHOUT_MANIFEST]
        assert len(mc3a) == 1
        assert "declared_required_fields" in mc3a[0].contract_name
        assert "pre_emission_check" in mc3a[0].contract_name

    def test_schema_config_mode_marker_without_manifest_detected(self, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path, {"schema_config_mode": ["post_emission_check"]})
        _write_registration(
            tmp_path,
            "schema_config_mode.py",
            "SchemaConfigModeContract",
            "schema_config_mode",
            marker_sites=["post_emission_check", "batch_flush_check"],
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        mc3a = [f for f in findings if f.rule_id == RULE_ID_MARKER_WITHOUT_MANIFEST]
        assert len(mc3a) == 1
        assert "schema_config_mode" in mc3a[0].contract_name
        assert "batch_flush_check" in mc3a[0].contract_name

    def test_can_drop_rows_marker_without_manifest_detected(self, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path, {"can_drop_rows": ["post_emission_check"]})
        _write_registration(
            tmp_path,
            "can_drop_rows.py",
            "CanDropRowsContract",
            "can_drop_rows",
            marker_sites=["post_emission_check", "batch_flush_check"],
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        mc3a = [f for f in findings if f.rule_id == RULE_ID_MARKER_WITHOUT_MANIFEST]
        assert len(mc3a) == 1
        assert "can_drop_rows" in mc3a[0].contract_name
        assert "batch_flush_check" in mc3a[0].contract_name


class TestMC3bManifestWithoutMarker:
    """MC3b: manifest names a site with no @implements_dispatch_site marker
    on the concrete class."""

    def test_missing_marker_for_manifest_site_detected(self, tmp_path: Path) -> None:
        manifest = _write_manifest(
            tmp_path,
            {"c": ["post_emission_check", "batch_flush_check"]},
        )
        _write_registration(
            tmp_path,
            "c.py",
            "CContract",
            "c",
            marker_sites=["post_emission_check"],  # batch_flush_check missing
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        mc3b = [f for f in findings if f.rule_id == RULE_ID_MANIFEST_WITHOUT_MARKER]
        assert len(mc3b) == 1
        assert "batch_flush_check" in mc3b[0].contract_name

    def test_declared_output_fields_missing_marker_detected(self, tmp_path: Path) -> None:
        manifest = _write_manifest(
            tmp_path,
            {"declared_output_fields": ["post_emission_check", "batch_flush_check"]},
        )
        _write_registration(
            tmp_path,
            "declared_output_fields.py",
            "DeclaredOutputFieldsContract",
            "declared_output_fields",
            marker_sites=["post_emission_check"],
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        mc3b = [f for f in findings if f.rule_id == RULE_ID_MANIFEST_WITHOUT_MARKER]
        assert len(mc3b) == 1
        assert "declared_output_fields" in mc3b[0].contract_name
        assert "batch_flush_check" in mc3b[0].contract_name

    def test_schema_config_mode_missing_marker_detected(self, tmp_path: Path) -> None:
        manifest = _write_manifest(
            tmp_path,
            {"schema_config_mode": ["post_emission_check", "batch_flush_check"]},
        )
        _write_registration(
            tmp_path,
            "schema_config_mode.py",
            "SchemaConfigModeContract",
            "schema_config_mode",
            marker_sites=["post_emission_check"],
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        mc3b = [f for f in findings if f.rule_id == RULE_ID_MANIFEST_WITHOUT_MARKER]
        assert len(mc3b) == 1
        assert "schema_config_mode" in mc3b[0].contract_name
        assert "batch_flush_check" in mc3b[0].contract_name

    def test_declared_required_fields_missing_marker_detected(self, tmp_path: Path) -> None:
        manifest = _write_manifest(
            tmp_path,
            {"declared_required_fields": ["pre_emission_check"]},
        )
        _write_registration(
            tmp_path,
            "declared_required_fields.py",
            "DeclaredRequiredFieldsContract",
            "declared_required_fields",
            marker_sites=[],
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        mc3b = [f for f in findings if f.rule_id == RULE_ID_MANIFEST_WITHOUT_MARKER]
        assert len(mc3b) == 1
        assert "declared_required_fields" in mc3b[0].contract_name
        assert "pre_emission_check" in mc3b[0].contract_name

    def test_can_drop_rows_missing_marker_detected(self, tmp_path: Path) -> None:
        manifest = _write_manifest(
            tmp_path,
            {"can_drop_rows": ["post_emission_check", "batch_flush_check"]},
        )
        _write_registration(
            tmp_path,
            "can_drop_rows.py",
            "CanDropRowsContract",
            "can_drop_rows",
            marker_sites=["post_emission_check"],
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        mc3b = [f for f in findings if f.rule_id == RULE_ID_MANIFEST_WITHOUT_MARKER]
        assert len(mc3b) == 1
        assert "can_drop_rows" in mc3b[0].contract_name
        assert "batch_flush_check" in mc3b[0].contract_name


class TestMC3cTrivialBody:
    """MC3c: a marked site's method body is structurally trivial
    (pass / ... / bare return / literal-only)."""

    def test_pass_only_body_detected(self, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path, {"c": ["post_emission_check"]})
        _write_registration(
            tmp_path,
            "c.py",
            "CContract",
            "c",
            marker_sites=["post_emission_check"],
            trivial_body_sites=["post_emission_check"],  # body is bare ``pass``
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        mc3c = [f for f in findings if f.rule_id == RULE_ID_TRIVIAL_BODY]
        assert len(mc3c) == 1

    def test_ellipsis_only_body_detected(self, tmp_path: Path) -> None:
        """W5 expansion: ``...`` (Ellipsis literal) is structurally trivial."""
        manifest = _write_manifest(tmp_path, {"c": ["post_emission_check"]})
        contract_path = tmp_path / "c.py"
        contract_path.write_text(
            textwrap.dedent(
                """\
                from declaration_contracts import (
                    DeclarationContract,
                    implements_dispatch_site,
                    register_declaration_contract,
                )


                class CContract(DeclarationContract):
                    name = "c"
                    payload_schema = dict

                    def applies_to(self, plugin):
                        return False

                    @implements_dispatch_site("post_emission_check")
                    def post_emission_check(self, inputs, outputs):
                        ...

                    @classmethod
                    def negative_example(cls):
                        raise NotImplementedError

                    @classmethod
                    def positive_example_does_not_apply(cls):
                        raise NotImplementedError


                register_declaration_contract(CContract())
                """
            )
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        mc3c = [f for f in findings if f.rule_id == RULE_ID_TRIVIAL_BODY]
        assert len(mc3c) == 1

    def test_non_trivial_body_passes(self, tmp_path: Path) -> None:
        """A body with a ``raise`` statement is non-trivial."""
        manifest = _write_manifest(tmp_path, {"c": ["post_emission_check"]})
        _write_registration(
            tmp_path,
            "c.py",
            "CContract",
            "c",
            marker_sites=["post_emission_check"],
            trivial_body_sites=[],  # body raises NotImplementedError
        )
        name_to_sites, name_to_line, assign_line = extract_manifest(manifest)
        registrations = scan_source_tree(tmp_path, tmp_path, manifest)
        findings = compute_findings(name_to_sites, name_to_line, registrations, "declaration_contracts.py", assign_line)
        assert findings == []

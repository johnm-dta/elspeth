"""Tests for plugin_hash.py — shared hash computation for plugin source files.

Tests cover:
1. Hash computation with self-referential normalization
2. AST extraction of plugin class attributes
3. In-place hash line rewriting

Uses tmp_path fixtures for file-based operations.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from scripts.cicd.plugin_hash import (
    compute_source_file_hash,
    extract_plugin_attributes,
    fix_source_file_hash,
)

# =============================================================================
# Helper: write a plugin file
# =============================================================================


def _write_plugin(tmp_path: Path, source: str, filename: str = "plugin.py") -> Path:
    """Write dedented source to a temp file and return the path."""
    file_path = tmp_path / filename
    file_path.write_text(textwrap.dedent(source), encoding="utf-8")
    return file_path


# =============================================================================
# compute_source_file_hash
# =============================================================================


class TestComputeHash:
    """Tests for compute_source_file_hash."""

    def test_compute_hash_excludes_own_value(self, tmp_path: Path) -> None:
        """Same hash regardless of declared source_file_hash value.

        The self-referential normalization must ensure the hash doesn't
        change when only the hash value itself differs.
        """
        source_a = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash = "sha256:aaaaaaaaaaaaaaaa"
        """
        source_b = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash = "sha256:bbbbbbbbbbbbbbbb"
        """
        file_a = _write_plugin(tmp_path, source_a, "plugin_a.py")
        file_b = _write_plugin(tmp_path, source_b, "plugin_b.py")

        hash_a = compute_source_file_hash(file_a)
        hash_b = compute_source_file_hash(file_b)

        assert hash_a == hash_b
        assert hash_a.startswith("sha256:")
        assert len(hash_a) == len("sha256:") + 16  # sha256: + 16 hex chars

    def test_compute_hash_changes_on_content_change(self, tmp_path: Path) -> None:
        """Different hash when actual content (not just hash value) changes."""
        source_a = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash = "sha256:aaaaaaaaaaaaaaaa"

                def process(self):
                    return "version_a"
        """
        source_b = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash = "sha256:aaaaaaaaaaaaaaaa"

                def process(self):
                    return "version_b"
        """
        file_a = _write_plugin(tmp_path, source_a, "plugin_a.py")
        file_b = _write_plugin(tmp_path, source_b, "plugin_b.py")

        hash_a = compute_source_file_hash(file_a)
        hash_b = compute_source_file_hash(file_b)

        assert hash_a != hash_b

    def test_compute_hash_uses_raw_bytes(self, tmp_path: Path) -> None:
        """Non-ASCII content works — hashing uses raw bytes, not decoded text."""
        source = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                description = "Ünïcödé — «fancy» plugin™"
        """
        file_path = _write_plugin(tmp_path, source)
        result = compute_source_file_hash(file_path)

        assert result.startswith("sha256:")
        assert len(result) == len("sha256:") + 16

    def test_compute_hash_normalizes_non_hex_placeholder(self, tmp_path: Path) -> None:
        """Non-hex placeholders like "sha256:stale_stale_stale" must normalize.

        A developer seeding a new plugin with a non-hex placeholder should
        still get a stable hash — the placeholder must be normalized before
        hashing, just like a real hex hash is.
        """
        source_placeholder = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash = "sha256:stale_stale_stale"
        """
        source_hex = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash = "sha256:aaaaaaaaaaaaaaaa"
        """
        file_placeholder = _write_plugin(tmp_path, source_placeholder, "plugin_placeholder.py")
        file_hex = _write_plugin(tmp_path, source_hex, "plugin_hex.py")

        hash_placeholder = compute_source_file_hash(file_placeholder)
        hash_hex = compute_source_file_hash(file_hex)

        # Both should produce the same hash since only the hash value differs
        assert hash_placeholder == hash_hex

    def test_compute_hash_normalizes_angle_bracket_placeholder(self, tmp_path: Path) -> None:
        """Placeholders like "sha256:<computed>" must also normalize."""
        source_a = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash = "sha256:<computed>"
        """
        source_b = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash = "sha256:abcdef0123456789"
        """
        file_a = _write_plugin(tmp_path, source_a, "plugin_a.py")
        file_b = _write_plugin(tmp_path, source_b, "plugin_b.py")

        assert compute_source_file_hash(file_a) == compute_source_file_hash(file_b)

    def test_compute_hash_normalizes_annotated_non_hex_placeholder(self, tmp_path: Path) -> None:
        """Non-hex placeholders with type annotations must also normalize."""
        source_a = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash: str | None = "sha256:PLACEHOLDER"
        """
        source_b = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash: str | None = "sha256:abcdef0123456789"
        """
        file_a = _write_plugin(tmp_path, source_a, "plugin_a.py")
        file_b = _write_plugin(tmp_path, source_b, "plugin_b.py")

        assert compute_source_file_hash(file_a) == compute_source_file_hash(file_b)

    def test_compute_hash_file_without_hash_line(self, tmp_path: Path) -> None:
        """File with no source_file_hash line still hashes correctly."""
        source = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"

                def process(self):
                    pass
        """
        file_path = _write_plugin(tmp_path, source)
        result = compute_source_file_hash(file_path)

        assert result.startswith("sha256:")
        assert len(result) == len("sha256:") + 16

    def test_crlf_and_lf_produce_same_hash(self, tmp_path: Path) -> None:
        """CRLF line endings (Windows checkout) must hash identically to LF."""
        content_lf = b'class MyPlugin:\n    name = "my-plugin"\n    plugin_version = "1.0.0"\n'
        content_crlf = b'class MyPlugin:\r\n    name = "my-plugin"\r\n    plugin_version = "1.0.0"\r\n'

        file_lf = tmp_path / "plugin_lf.py"
        file_crlf = tmp_path / "plugin_crlf.py"
        file_lf.write_bytes(content_lf)
        file_crlf.write_bytes(content_crlf)

        assert compute_source_file_hash(file_lf) == compute_source_file_hash(file_crlf)

    def test_lone_cr_normalized_to_lf(self, tmp_path: Path) -> None:
        """Lone CR (old Mac) must hash identically to LF."""
        content_lf = b'class MyPlugin:\n    name = "my-plugin"\n    plugin_version = "1.0.0"\n'
        content_cr = b'class MyPlugin:\r    name = "my-plugin"\r    plugin_version = "1.0.0"\r'

        file_lf = tmp_path / "plugin_lf.py"
        file_cr = tmp_path / "plugin_cr.py"
        file_lf.write_bytes(content_lf)
        file_cr.write_bytes(content_cr)

        assert compute_source_file_hash(file_lf) == compute_source_file_hash(file_cr)

    def test_utf8_bom_stripped(self, tmp_path: Path) -> None:
        """UTF-8 BOM prefix must not affect the hash."""
        content = b'class MyPlugin:\n    name = "my-plugin"\n    plugin_version = "1.0.0"\n'
        content_bom = b"\xef\xbb\xbf" + content

        file_plain = tmp_path / "plugin_plain.py"
        file_bom = tmp_path / "plugin_bom.py"
        file_plain.write_bytes(content)
        file_bom.write_bytes(content_bom)

        assert compute_source_file_hash(file_plain) == compute_source_file_hash(file_bom)

    def test_crlf_with_bom_matches_plain_lf(self, tmp_path: Path) -> None:
        """CRLF + BOM (common Windows editor output) must match plain LF."""
        content_lf = b'class MyPlugin:\n    name = "my-plugin"\n    plugin_version = "1.0.0"\n'
        content_crlf_bom = b"\xef\xbb\xbf" + b'class MyPlugin:\r\n    name = "my-plugin"\r\n    plugin_version = "1.0.0"\r\n'

        file_lf = tmp_path / "plugin_lf.py"
        file_crlf_bom = tmp_path / "plugin_crlf_bom.py"
        file_lf.write_bytes(content_lf)
        file_crlf_bom.write_bytes(content_crlf_bom)

        assert compute_source_file_hash(file_lf) == compute_source_file_hash(file_crlf_bom)


# =============================================================================
# extract_plugin_attributes
# =============================================================================


class TestExtractPluginAttributes:
    """Tests for extract_plugin_attributes."""

    def test_extract_class_attribute_simple(self, tmp_path: Path) -> None:
        """Basic extraction of name, plugin_version, source_file_hash."""
        source = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash = "sha256:abcdef0123456789"
        """
        file_path = _write_plugin(tmp_path, source)
        results = extract_plugin_attributes(file_path)

        assert len(results) == 1
        attrs = results[0]
        assert attrs.class_name == "MyPlugin"
        assert attrs.plugin_version == "1.0.0"
        assert attrs.source_file_hash == "sha256:abcdef0123456789"
        assert attrs.hash_line_number is not None
        assert attrs.hash_line_number > 0

    def test_extract_class_attribute_annotated(self, tmp_path: Path) -> None:
        """Annotated assignment form: source_file_hash: str = "sha256:..."."""
        source = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version: str = "2.0.0"
                source_file_hash: str = "sha256:1111222233334444"
        """
        file_path = _write_plugin(tmp_path, source)
        results = extract_plugin_attributes(file_path)

        assert len(results) == 1
        attrs = results[0]
        assert attrs.class_name == "MyPlugin"
        assert attrs.plugin_version == "2.0.0"
        assert attrs.source_file_hash == "sha256:1111222233334444"

    def test_extract_class_attribute_none_default(self, tmp_path: Path) -> None:
        """source_file_hash = None is detected (value is None)."""
        source = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash = None
        """
        file_path = _write_plugin(tmp_path, source)
        results = extract_plugin_attributes(file_path)

        assert len(results) == 1
        attrs = results[0]
        assert attrs.class_name == "MyPlugin"
        assert attrs.source_file_hash is None

    def test_extract_ignores_non_plugin_classes(self, tmp_path: Path) -> None:
        """Classes without a `name` class attribute are not plugin classes."""
        source = """\
            class NotAPlugin:
                plugin_version = "1.0.0"
                source_file_hash = "sha256:abcdef0123456789"

            class AlsoNotAPlugin:
                def name(self):
                    return "method, not attribute"

            class IsAPlugin:
                name = "real-plugin"
                plugin_version = "1.0.0"
                source_file_hash = "sha256:abcdef0123456789"
        """
        file_path = _write_plugin(tmp_path, source)
        results = extract_plugin_attributes(file_path)

        assert len(results) == 1
        assert results[0].class_name == "IsAPlugin"


# =============================================================================
# fix_source_file_hash
# =============================================================================


class TestFixSourceFileHash:
    """Tests for fix_source_file_hash."""

    def test_fix_updates_hash_in_place(self, tmp_path: Path) -> None:
        """fix_source_file_hash rewrites the hash line with correct value."""
        source = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash = "sha256:0000000000000000"

                def process(self):
                    return 42
        """
        file_path = _write_plugin(tmp_path, source)
        new_hash = "sha256:abcdef0123456789"

        fix_source_file_hash(file_path, "MyPlugin", new_hash)

        content = file_path.read_text(encoding="utf-8")
        assert new_hash in content
        # The stale placeholder should be gone
        assert "sha256:0000000000000000" not in content
        # Rest of the file is preserved
        assert 'name = "my-plugin"' in content
        assert "def process(self):" in content
        assert "return 42" in content

    def test_fix_is_idempotent(self, tmp_path: Path) -> None:
        """Fixing twice with the same hash produces the same result."""
        source = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash = "sha256:0000000000000000"
        """
        file_path = _write_plugin(tmp_path, source)
        new_hash = "sha256:abcdef0123456789"

        fix_source_file_hash(file_path, "MyPlugin", new_hash)
        content_after_first = file_path.read_text(encoding="utf-8")

        fix_source_file_hash(file_path, "MyPlugin", new_hash)
        content_after_second = file_path.read_text(encoding="utf-8")

        assert content_after_first == content_after_second

    def test_fix_preserves_union_type_annotation(self, tmp_path: Path) -> None:
        """fix must preserve `source_file_hash: str | None = ...` annotations.

        All real plugin declarations use `str | None`. The fixer must not
        strip the annotation when rewriting the hash value.
        """
        source = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash: str | None = "sha256:0000000000000000"

                def process(self):
                    return 42
        """
        file_path = _write_plugin(tmp_path, source)
        new_hash = "sha256:abcdef0123456789"

        fix_source_file_hash(file_path, "MyPlugin", new_hash)

        content = file_path.read_text(encoding="utf-8")
        # The annotation MUST be preserved
        assert "source_file_hash: str | None =" in content
        assert new_hash in content
        # Rest of file preserved
        assert "def process(self):" in content

    def test_fix_preserves_simple_type_annotation(self, tmp_path: Path) -> None:
        """fix must preserve `source_file_hash: str = ...` annotations."""
        source = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash: str = "sha256:0000000000000000"
        """
        file_path = _write_plugin(tmp_path, source)
        new_hash = "sha256:abcdef0123456789"

        fix_source_file_hash(file_path, "MyPlugin", new_hash)

        content = file_path.read_text(encoding="utf-8")
        assert "source_file_hash: str =" in content
        assert new_hash in content

    def test_fix_converges_in_one_pass_for_non_hex_placeholder(self, tmp_path: Path) -> None:
        """check --fix must converge in one pass even for non-hex placeholders.

        If the normalization regex doesn't match non-hex placeholders,
        the first fix writes a hash computed over the un-normalized file,
        and the second check will compute a different hash over the now-
        normalized file. This test verifies single-pass convergence.
        """
        source = """\
            class MyPlugin:
                name = "my-plugin"
                plugin_version = "1.0.0"
                source_file_hash = "sha256:stale_stale_stale"

                def process(self):
                    return 42
        """
        file_path = _write_plugin(tmp_path, source)

        # Compute the correct hash and fix
        correct_hash = compute_source_file_hash(file_path)
        fix_source_file_hash(file_path, "MyPlugin", correct_hash)

        # After fix, re-computing should produce the SAME hash (convergence)
        recomputed = compute_source_file_hash(file_path)
        assert recomputed == correct_hash, f"Fix did not converge: first pass computed {correct_hash}, second pass computed {recomputed}"


class TestBOMHandling:
    """BOM-prefixed files must work through the full extract/fix workflow."""

    def _write_bom_plugin(self, tmp_path: Path) -> Path:
        file_path = tmp_path / "bom_plugin.py"
        content = b"\xef\xbb\xbf" + textwrap.dedent("""\
            class BomPlugin:
                name = "bom"
                plugin_version = "1.0.0"
                source_file_hash: str | None = "sha256:0000000000000000"
        """).encode("utf-8")
        file_path.write_bytes(content)
        return file_path

    def test_extract_plugin_attributes_with_bom(self, tmp_path: Path) -> None:
        file_path = self._write_bom_plugin(tmp_path)
        attrs_list = extract_plugin_attributes(file_path)
        assert len(attrs_list) == 1
        assert attrs_list[0].class_name == "BomPlugin"
        assert attrs_list[0].source_file_hash == "sha256:0000000000000000"

    def test_fix_source_file_hash_with_bom(self, tmp_path: Path) -> None:
        file_path = self._write_bom_plugin(tmp_path)
        computed = compute_source_file_hash(file_path)
        fix_source_file_hash(file_path, "BomPlugin", computed)

        # After fix, extract and compute should agree
        attrs_list = extract_plugin_attributes(file_path)
        recomputed = compute_source_file_hash(file_path)
        assert attrs_list[0].source_file_hash == recomputed

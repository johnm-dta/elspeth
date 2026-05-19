"""Tests for enforce_frozen_annotations.py CI linter."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from scripts.cicd.enforce_frozen_annotations import find_violations


def test_enforce_frozen_annotations_json_mode_succeeds_on_current_codebase() -> None:
    """Legacy frozen-annotations script emits JSON findings for shadow-mode parity."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/cicd/enforce_frozen_annotations.py",
            "check",
            "--root",
            "src/elspeth",
            "--allowlist",
            "config/cicd/enforce_frozen_annotations",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[4],
    )

    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert json.loads(result.stdout) == []


def _parse(source: str) -> list[dict[str, str]]:
    """Run the linter on a source string and return violations."""
    return find_violations(textwrap.dedent(source), filename="test.py")


class TestFrozenDataclassDetection:
    """Linter must detect frozen=True with various keyword combinations."""

    def test_frozen_true_detected(self) -> None:
        violations = _parse("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Foo:
                items: list[int]
        """)
        assert len(violations) == 1
        assert "list[" in violations[0]["annotation"]

    def test_frozen_true_slots_true_detected(self) -> None:
        violations = _parse("""
            from dataclasses import dataclass

            @dataclass(frozen=True, slots=True)
            class Foo:
                items: list[int]
        """)
        assert len(violations) == 1

    def test_non_frozen_ignored(self) -> None:
        violations = _parse("""
            from dataclasses import dataclass

            @dataclass
            class Foo:
                items: list[int]
        """)
        assert len(violations) == 0

    def test_frozen_false_ignored(self) -> None:
        violations = _parse("""
            from dataclasses import dataclass

            @dataclass(frozen=False)
            class Foo:
                items: list[int]
        """)
        assert len(violations) == 0


class TestAnnotationDetection:
    """Linter must detect mutable container annotations."""

    def test_list_detected(self) -> None:
        violations = _parse("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Foo:
                items: list[int]
        """)
        assert len(violations) == 1

    def test_dict_detected(self) -> None:
        violations = _parse("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Foo:
                mapping: dict[str, int]
        """)
        assert len(violations) == 1

    def test_set_detected(self) -> None:
        violations = _parse("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Foo:
                unique: set[str]
        """)
        assert len(violations) == 1

    def test_union_with_none_detected(self) -> None:
        violations = _parse("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Foo:
                items: list[int] | None
        """)
        assert len(violations) == 1

    def test_sequence_clean(self) -> None:
        violations = _parse("""
            from dataclasses import dataclass
            from collections.abc import Sequence

            @dataclass(frozen=True)
            class Foo:
                items: Sequence[int]
        """)
        assert len(violations) == 0

    def test_mapping_clean(self) -> None:
        violations = _parse("""
            from dataclasses import dataclass
            from collections.abc import Mapping

            @dataclass(frozen=True)
            class Foo:
                mapping: Mapping[str, int]
        """)
        assert len(violations) == 0

    def test_tuple_clean(self) -> None:
        violations = _parse("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Foo:
                items: tuple[int, ...]
        """)
        assert len(violations) == 0


class TestFutureAnnotations:
    """Linter must work with from __future__ import annotations (PEP 563)."""

    def test_stringified_list_detected(self) -> None:
        violations = _parse("""
            from __future__ import annotations
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Foo:
                items: list[int]
        """)
        assert len(violations) == 1

    def test_stringified_sequence_clean(self) -> None:
        violations = _parse("""
            from __future__ import annotations
            from dataclasses import dataclass
            from collections.abc import Sequence

            @dataclass(frozen=True)
            class Foo:
                items: Sequence[int]
        """)
        assert len(violations) == 0


class TestMultipleFields:
    """Linter reports all violations, not just the first."""

    def test_multiple_violations_reported(self) -> None:
        violations = _parse("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Foo:
                items: list[int]
                mapping: dict[str, str]
                unique: set[float]
        """)
        assert len(violations) == 3

    def test_mixed_clean_and_violations(self) -> None:
        violations = _parse("""
            from dataclasses import dataclass
            from collections.abc import Sequence

            @dataclass(frozen=True)
            class Foo:
                items: list[int]
                safe: Sequence[int]
                name: str
        """)
        assert len(violations) == 1
        assert "list[" in violations[0]["annotation"]

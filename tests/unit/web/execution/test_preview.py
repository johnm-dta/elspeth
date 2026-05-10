"""Tests for ``elspeth.web.execution.preview.build_artifact_preview``.

These exercise the text/binary classifier and the
byte-cap/row-cap interplay independently of FastAPI plumbing.
The endpoint-level tests live in ``test_outputs_routes.py``.
"""

from __future__ import annotations

from pathlib import Path

from elspeth.web.execution.preview import (
    _classify_text_or_binary,
    build_artifact_preview,
)


class TestClassifyTextOrBinary:
    def test_pure_ascii_is_text(self) -> None:
        is_text, decoded = _classify_text_or_binary(b"hello\nworld\n")
        assert is_text is True
        assert decoded == "hello\nworld\n"

    def test_complete_utf8_multibyte_is_text(self) -> None:
        # "héllo wörld" — multi-byte chars throughout, decodes cleanly.
        payload = "héllo wörld\n".encode()
        is_text, decoded = _classify_text_or_binary(payload)
        assert is_text is True
        assert decoded == "héllo wörld\n"

    def test_truncated_utf8_codepoint_in_tail_is_still_text(self) -> None:
        # Four-byte emoji '🦀' (\xf0\x9f\xa6\x80) sliced after first 2 bytes:
        # the trailing partial sequence is in the last 3 bytes — must be
        # treated as text via errors='ignore', not flipped to binary.
        full = "abc🦀def".encode()
        # Cut so the emoji is partial at the END of the buffer
        truncated = full[: full.index(b"\xf0") + 2]  # keep first 2 bytes of the emoji
        is_text, decoded = _classify_text_or_binary(truncated)
        assert is_text is True
        # 'abc' survives; the partial emoji bytes are dropped.
        assert decoded == "abc"

    def test_genuinely_binary_bytes_are_binary(self) -> None:
        # JPEG SOI marker followed by APP0 — invalid UTF-8 from the very start.
        payload = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 100
        is_text, decoded = _classify_text_or_binary(payload)
        assert is_text is False
        assert decoded == ""

    def test_binary_byte_in_middle_is_binary_not_text(self) -> None:
        # Lone continuation byte (0x80) far from the tail — real binary.
        payload = b"text " + b"\x80" + b" more text plus padding " + b"x" * 100
        is_text, decoded = _classify_text_or_binary(payload)
        assert is_text is False
        assert decoded == ""

    def test_empty_buffer_is_text(self) -> None:
        is_text, decoded = _classify_text_or_binary(b"")
        assert is_text is True
        assert decoded == ""


class TestBuildArtifactPreview:
    def test_csv_under_caps_returns_full_content(self, tmp_path: Path) -> None:
        f = tmp_path / "small.csv"
        f.write_text("col1,col2\n1,2\n3,4\n")

        preview = build_artifact_preview(f, artifact_id="art-1")

        assert preview.content_type == "csv"
        assert preview.preview_text == "col1,col2\n1,2\n3,4\n"
        assert preview.truncated is False
        assert preview.total_size_bytes == f.stat().st_size
        assert preview.row_count_preview == 3  # header + 2 data rows

    def test_csv_over_row_cap_truncates_to_row_cap(self, tmp_path: Path) -> None:
        f = tmp_path / "many_rows.csv"
        # 200 rows of "x" — well under the byte cap, but over the row cap.
        f.write_text("\n".join(f"row{i}" for i in range(200)) + "\n")

        preview = build_artifact_preview(f, artifact_id="art-2", row_cap=50)

        assert preview.content_type == "csv"
        assert preview.row_count_preview == 50
        assert preview.truncated is True
        assert preview.preview_text.count("\n") == 49  # 50 lines joined by 49 newlines

    def test_text_under_byte_cap_returns_full_content(self, tmp_path: Path) -> None:
        f = tmp_path / "log.txt"
        f.write_text("line one\nline two\n")

        preview = build_artifact_preview(f, artifact_id="art-3")

        assert preview.content_type == "text"
        assert preview.preview_text == "line one\nline two\n"
        assert preview.truncated is False
        assert preview.row_count_preview is None  # not tabular

    def test_text_over_byte_cap_marks_truncated(self, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        # Use bytes well beyond the 1 KiB cap.
        f.write_bytes(b"a" * 5000)

        preview = build_artifact_preview(f, artifact_id="art-4", byte_cap=1024)

        assert preview.content_type == "text"
        assert preview.truncated is True
        assert preview.total_size_bytes == 5000
        assert len(preview.preview_text) <= 1024

    def test_binary_extension_returns_binary_with_no_preview(self, tmp_path: Path) -> None:
        f = tmp_path / "image.bin"
        f.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 200)

        preview = build_artifact_preview(f, artifact_id="art-5")

        assert preview.content_type == "binary"
        assert preview.preview_text == ""
        assert preview.row_count_preview is None

    def test_extension_says_csv_but_bytes_are_binary(self, tmp_path: Path) -> None:
        # Defence: a sink mis-extensioned its output. Binary detection
        # must override the format hint from extension.
        f = tmp_path / "lying.csv"
        f.write_bytes(b"\xff\xd8" + b"\x80\x80\x80" + b"row1\n" + b"x" * 200)

        preview = build_artifact_preview(f, artifact_id="art-6")

        assert preview.content_type == "binary"
        assert preview.preview_text == ""

    def test_jsonl_extension_uses_jsonl_content_type(self, tmp_path: Path) -> None:
        f = tmp_path / "events.jsonl"
        f.write_text('{"a":1}\n{"a":2}\n')

        preview = build_artifact_preview(f, artifact_id="art-7")

        assert preview.content_type == "jsonl"
        assert preview.row_count_preview == 2

    def test_json_extension_uses_json_content_type(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.json"
        f.write_text('{"key": "value"}')

        preview = build_artifact_preview(f, artifact_id="art-8")

        assert preview.content_type == "json"
        assert preview.row_count_preview is None

    def test_unknown_extension_with_text_bytes_falls_back_to_text(self, tmp_path: Path) -> None:
        f = tmp_path / "what.xyz"
        f.write_text("hello\n")

        preview = build_artifact_preview(f, artifact_id="art-9")

        assert preview.content_type == "text"
        assert preview.preview_text == "hello\n"

    def test_csv_with_byte_cap_below_row_cap_marks_truncated(self, tmp_path: Path) -> None:
        # The byte cap fires before the row cap: even though we have
        # fewer than row_cap full lines, the final line may itself be
        # cut. Truncated must be True.
        f = tmp_path / "small_rows_huge_cells.csv"
        f.write_text("a,b\n" + ("x" * 100 + ",y\n") * 20)  # ~2 KiB

        preview = build_artifact_preview(f, artifact_id="art-10", byte_cap=200)

        assert preview.truncated is True
        assert preview.total_size_bytes == f.stat().st_size

    def test_total_size_bytes_uses_live_filesystem_size(self, tmp_path: Path) -> None:
        # build_artifact_preview reads stat() at call time — caller can
        # cross-check against the manifest's recorded size.
        f = tmp_path / "out.txt"
        f.write_bytes(b"a" * 42)
        preview = build_artifact_preview(f, artifact_id="art-11")
        assert preview.total_size_bytes == 42

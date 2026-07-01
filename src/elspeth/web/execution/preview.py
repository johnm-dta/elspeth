"""Bounded artifact-preview reader.

Companion to ``elspeth.web.execution.outputs``: where ``outputs``
returns the audit-evidence manifest, this module reads a head-of-file
preview so the operator UI can show the first N rows / first N bytes
of a sink-write artefact without forcing a full download.

Bounded reads are the explicit design:
* ``_DEFAULT_BYTE_CAP = 256 KiB`` — cap on bytes read from disk per
  request. Caps memory and IO blast radius.
* ``_DEFAULT_ROW_CAP = 100`` — for tabular formats (``.csv``, ``.tsv``,
  ``.jsonl``), the additional row-count cap below the byte cap.

UTF-8 truncation discipline
---------------------------
A naive "read N bytes, then ``decode('utf-8')``" would routinely raise
on legitimate text files: any multi-byte codepoint sliced by the byte
cap looks like a malformed sequence. Treating that as "binary" would
be wrong (the file is perfectly good UTF-8; the cap landed mid-codepoint).

We classify text vs binary with a probe:

* If ``head_bytes.decode('utf-8', errors='strict')`` succeeds → text.
* Else, if the failure position is within the **last 3 bytes** of the
  buffer → it's a truncation artifact (UTF-8 codepoints are at most 4
  bytes; any partial cut is in the tail). Re-decode with
  ``errors='ignore'`` to drop the partial sequence and treat as text.
* Else → genuine binary (a malformed sequence appears far from the cap
  boundary, e.g., a JPEG header byte at offset 0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from elspeth.web.execution.schemas import RunOutputArtifactPreview

_DEFAULT_BYTE_CAP = 256 * 1024
_DEFAULT_ROW_CAP = 100
DEFAULT_ARTIFACT_PREVIEW_BYTE_CAP = _DEFAULT_BYTE_CAP

# Extensions we render as a parsed-row table on the frontend.
_CSV_EXTENSIONS = frozenset({".csv", ".tsv"})
_JSONL_EXTENSIONS = frozenset({".jsonl", ".ndjson"})
# Extensions we render as monospace text but not a table.
_JSON_EXTENSIONS = frozenset({".json"})
_PLAIN_TEXT_EXTENSIONS = frozenset(
    {".txt", ".log", ".md", ".markdown", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sql", ".html", ".htm", ".xml", ".tsv"}
)

PreviewContentType = Literal["text", "csv", "jsonl", "json", "binary"]


def _classify_text_or_binary(head_bytes: bytes) -> tuple[bool, str]:
    """Return ``(is_text, decoded_text)``.

    A trailing partial UTF-8 sequence (cut by the byte cap) is treated
    as text — silently dropped via ``errors='ignore'``. A malformed
    sequence anywhere else is treated as genuine binary.
    """
    try:
        return True, head_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        # UTF-8 codepoints are at most 4 bytes. A truncation artifact
        # always lies within the last 3 bytes of the buffer.
        if exc.start >= len(head_bytes) - 3:
            return True, head_bytes.decode("utf-8", errors="ignore")
        return False, ""


def _select_content_type(suffix: str) -> PreviewContentType:
    suffix = suffix.lower()
    if suffix in _CSV_EXTENSIONS:
        return "csv"
    if suffix in _JSONL_EXTENSIONS:
        return "jsonl"
    if suffix in _JSON_EXTENSIONS:
        return "json"
    if suffix in _PLAIN_TEXT_EXTENSIONS:
        return "text"
    # Unknown extension but UTF-8 decodable: still serve as text.
    return "text"


def build_artifact_preview(
    fs_path: Path,
    *,
    artifact_id: str,
    byte_cap: int = _DEFAULT_BYTE_CAP,
    row_cap: int = _DEFAULT_ROW_CAP,
) -> RunOutputArtifactPreview:
    """Build a bounded preview for ``fs_path``.

    Caller is responsible for verifying ``fs_path`` is in the sink
    allowlist and exists. This function reads at most ``byte_cap`` bytes
    and, for tabular content types, additionally caps at ``row_cap`` rows.
    """
    total_size_bytes = fs_path.stat().st_size
    with fs_path.open("rb") as f:
        head_bytes = f.read(byte_cap)
    return _build_artifact_preview_from_head(
        fs_path,
        artifact_id=artifact_id,
        total_size_bytes=total_size_bytes,
        head_bytes=head_bytes,
        byte_cap=byte_cap,
        row_cap=row_cap,
    )


def build_artifact_preview_from_head(
    fs_path: Path,
    *,
    artifact_id: str,
    total_size_bytes: int,
    head_bytes: bytes,
    byte_cap: int = _DEFAULT_BYTE_CAP,
    row_cap: int = _DEFAULT_ROW_CAP,
) -> RunOutputArtifactPreview:
    """Build a bounded preview from a previously verified head-of-file snapshot."""
    return _build_artifact_preview_from_head(
        fs_path,
        artifact_id=artifact_id,
        total_size_bytes=total_size_bytes,
        head_bytes=head_bytes,
        byte_cap=byte_cap,
        row_cap=row_cap,
    )


def _build_artifact_preview_from_head(
    fs_path: Path,
    *,
    artifact_id: str,
    total_size_bytes: int,
    head_bytes: bytes,
    byte_cap: int,
    row_cap: int,
) -> RunOutputArtifactPreview:
    bytes_read = len(head_bytes)
    truncated_by_bytes = bytes_read < total_size_bytes

    is_text, head_text = _classify_text_or_binary(head_bytes)
    if not is_text:
        return RunOutputArtifactPreview(
            artifact_id=artifact_id,
            content_type="binary",
            preview_text="",
            truncated=truncated_by_bytes,
            total_size_bytes=total_size_bytes,
            row_count_preview=None,
        )

    content_type = _select_content_type(fs_path.suffix)
    is_tabular = content_type in {"csv", "jsonl"}

    if is_tabular:
        # ``splitlines`` is tolerant: malformed CSV rows render as raw
        # text rather than crashing the request. The frontend gets the
        # first ``row_cap`` lines and decides how to render them.
        lines = head_text.splitlines()
        if len(lines) > row_cap:
            preview_text = "\n".join(lines[:row_cap])
            row_count_preview: int | None = row_cap
            truncated = True
        else:
            preview_text = head_text
            row_count_preview = len(lines)
            # If the byte cap fired, the final line may itself be a
            # partial row — we don't know how many full rows remain on
            # disk, so the preview is honestly truncated.
            truncated = truncated_by_bytes
    else:
        preview_text = head_text
        row_count_preview = None
        truncated = truncated_by_bytes

    return RunOutputArtifactPreview(
        artifact_id=artifact_id,
        content_type=content_type,
        preview_text=preview_text,
        truncated=truncated,
        total_size_bytes=total_size_bytes,
        row_count_preview=row_count_preview,
    )

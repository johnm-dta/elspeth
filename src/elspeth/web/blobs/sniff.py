"""Lightweight MIME content detection for uploaded blobs.

Inspects the first bytes of content to detect the actual data format.
Uses only stdlib (csv module for quote-aware field counting) — no
third-party dependencies. Detects the data-oriented types ELSPETH
accepts: CSV, JSON, JSONL, plain text.

Encoding handling
-----------------

Pre-BOM dispatch recognises UTF-16 LE/BE and UTF-32 LE/BE.  These
encodings naturally contain NUL bytes (one per ASCII code point for
UTF-16, three for UTF-32), so the previous "NUL byte => binary" guard
misclassified valid CSV/JSON/text as ``application/octet-stream`` and
caused the upload route to reject them with HTTP 415 even though the
ELSPETH source plugins (csv_source, json_source, text_source) explicitly
support an ``encoding`` option.  When a BOM is present we decode with
the matching codec and run the same heuristics on the decoded text.

Without a BOM we cannot reliably distinguish UTF-16 text from binary
content whose byte distribution happens to contain NUL — the safer
default is still to classify the content as binary in that case, which
the upload route will reject as outside the allowlist.
"""

from __future__ import annotations

import csv
import io

# Minimum field (column) count to classify content as CSV. At 2 fields
# (1 delimiter per line), natural-language prose with a single comma
# ("Hello, world!\nGoodbye, world!") is indistinguishable from a
# 2-column CSV. Requiring 3+ fields eliminates this class of false
# positives. 2-column CSVs rely on the browser-declared MIME type.
_MIN_CSV_FIELDS = 3

# Bytes inspected for sniffing.  Widened from 4K to 8K so that after
# stripping a UTF-32 BOM (4 bytes) we still have enough payload left to
# evaluate multi-line heuristics for sparse encodings (UTF-32 consumes
# 4 bytes per ASCII codepoint).
_SAMPLE_BYTES = 8192

# Ordered longest-BOM-first.  The UTF-32 LE BOM (``\xff\xfe\x00\x00``)
# shares its first two bytes with the UTF-16 LE BOM (``\xff\xfe``), so
# the 4-byte markers must be checked before the 2-byte ones or UTF-32
# content would be misdecoded as UTF-16 and produce garbage text.
_BOM_CODECS: tuple[tuple[bytes, str], ...] = (
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\xfe\xff", "utf-16-be"),
    (b"\xff\xfe", "utf-16-le"),
)


def detect_mime_type(content: bytes) -> str | None:
    """Detect MIME type from content bytes. Returns None if uncertain.

    Only detects data-oriented formats relevant to ELSPETH:
    - JSON (starts with ``[`` or ``{`` after whitespace)
    - JSONL (multiple lines starting with ``{``)
    - CSV (2+ lines with consistent field counts >= 3, RFC 4180-aware)
    - Plain text (valid UTF-8 / UTF-16 / UTF-32 with no binary bytes)

    Returns None when content is ambiguous (e.g. delimiter-containing
    text with fewer than 3 columns) so the caller can fall back to the
    browser-declared MIME type.

    Returns ``application/octet-stream`` when the content positively
    identifies as binary — including the case where a BOM is present
    but the following bytes do not decode with the declared codec.  A
    lying BOM is two Tier 3 signals in conflict (the BOM itself and the
    bytes that should follow it); returning ``octet-stream`` forces the
    upload route's allowlist check to engage rather than silently
    deferring to the browser-declared type.
    """
    if not content:
        return None

    sample = content[:_SAMPLE_BYTES]

    # Multi-byte BOM dispatch — recognised codecs supply their own
    # textual semantics.  A BOM whose following bytes do not decode is
    # a corrupted or deceptive upload; we classify as binary so the
    # route layer's ALLOWED_MIME_TYPES check rejects it, rather than
    # falling back to the browser-declared type that this same request
    # cannot be trusted on.
    for bom, codec in _BOM_CODECS:
        if sample.startswith(bom):
            try:
                text = sample[len(bom) :].decode(codec, errors="strict")
            except UnicodeDecodeError:
                return "application/octet-stream"
            return _detect_from_text(text)

    # UTF-8 BOM: strip and fall through to the standard UTF-8 path.
    if sample.startswith(b"\xef\xbb\xbf"):
        sample = sample[3:]

    # Outside the BOM'd encodings above, NUL bytes reliably indicate
    # binary content (images, archives, executables).  Keep the hard
    # reject so malformed uploads declared as text/csv do not slip
    # through the content-vs-declaration check in the route layer.
    if b"\x00" in sample:
        return "application/octet-stream"

    try:
        text = sample.decode("utf-8")
    except UnicodeDecodeError:
        return "application/octet-stream"

    return _detect_from_text(text)


def _detect_from_text(text: str) -> str | None:
    """Apply data-format heuristics to already-decoded text.

    Shared between the UTF-8 and BOM'd-encoding paths so both see
    identical CSV/JSON/JSONL/plain-text classification logic.
    """
    stripped = text.lstrip()
    if not stripped:
        return "text/plain"

    # JSON: starts with [ or {
    if stripped[0] == "[":
        return "application/json"
    if stripped[0] == "{":
        # Could be JSON object or JSONL — check if multiple lines start with {
        lines = stripped.split("\n", 3)
        json_lines = sum(1 for line in lines if line.strip().startswith("{"))
        if json_lines > 1:
            return "application/x-jsonlines"
        return "application/json"

    # CSV heuristic: require 2+ non-empty lines with consistent field
    # counts and at least 3 fields per row.  Uses csv.reader so that
    # quoted fields containing commas (RFC 4180) don't inflate the count.
    lines = stripped.split("\n", 10)
    non_empty = [line for line in lines if line.strip()]
    first_line = non_empty[0] if non_empty else ""
    has_delimiters = "," in first_line or "\t" in first_line

    if len(non_empty) >= 2 and has_delimiters:
        csv_sample = "\n".join(non_empty)
        for delimiter in (",", "\t"):
            try:
                reader = csv.reader(io.StringIO(csv_sample), delimiter=delimiter)
                field_counts = [len(row) for row in reader]
            except csv.Error:
                continue
            if field_counts and field_counts[0] >= _MIN_CSV_FIELDS and all(c == field_counts[0] for c in field_counts):
                return "text/csv"

    # Content has delimiters but didn't meet the CSV threshold (single-line,
    # <3 columns, or inconsistent counts).  Return None so the caller can
    # fall back to the browser-declared MIME type — a legitimate 2-column
    # CSV declared as text/csv should not be overridden.
    if has_delimiters:
        return None

    # No delimiters — definitively plain text.
    return "text/plain"

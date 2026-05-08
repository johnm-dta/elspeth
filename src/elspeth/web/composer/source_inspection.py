"""Bounded inspection of blob-backed source content.

Step 2 of the composer simple-pipeline-convergence program.

Surfaces structured facts to the model and to ``preview_pipeline`` so the
LLM no longer has to guess at CSV column names or numeric types when the
operator has already supplied an inline blob.

Contract:
  * **Bounded reads.** At most 8 KiB or 100 rows, whichever comes first.
    Inspection MUST be cheap — it runs on every preview_pipeline call.
  * **No row-level logging.** Sensitive data may be present in the blob.
    Logger is reserved for inspection-success/decline summaries; raw row
    content never leaves this module.
  * **Redacted identity only.** Filename, MIME, byte size, and content
    hash prefix are safe to surface; storage paths and full content
    hashes are not.
  * **Coerce, don't fabricate.** Per CLAUDE.md tier model, source-level
    inspection MAY coerce ``"42"`` → int hint and ``"true"`` → bool hint
    because we are observing what *would* be coerced when the source
    plugin runs. We never fabricate a value the blob did not contain;
    if a column is empty in every sampled row, we record ``"null"``,
    not a guessed type.

Facts surfaced:
  * ``source_kind`` — csv / jsonl / json / text / unknown.
  * ``observed_headers`` — CSV column names (None for non-CSV).
  * ``inferred_types`` — column → ``int`` | ``float`` | ``str`` | ``bool``
    | ``null`` based on the sampled rows.
  * ``sample_row_count`` — how many rows we actually read.
  * ``url_candidates`` — for text or single-line content, any URLs found
    (a strong hint that ``web_scrape`` is needed).
  * ``warnings`` — observations that should bubble up as proof
    diagnostics in Step 3 (e.g., "first row looks like data, not
    headers", "field 'price' is numeric-shaped but typed str").
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal
from uuid import UUID

from elspeth.contracts.freeze import freeze_fields

_MAX_BYTES: Final[int] = 8 * 1024
_MAX_ROWS: Final[int] = 100

SourceKind = Literal["csv", "jsonl", "json", "text", "unknown"]

InferredType = Literal["int", "float", "bool", "str", "null"]


_URL_PATTERN: Final[re.Pattern[str]] = re.compile(r"\bhttps?://[^\s<>\"']+")
_INT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^-?\d+$")
_FLOAT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^-?\d+\.\d+([eE][+-]?\d+)?$|^-?\d+[eE][+-]?\d+$")
_BOOL_LITERALS: Final[frozenset[str]] = frozenset({"true", "false", "yes", "no"})


@dataclass(frozen=True, slots=True)
class SourceInspectionFacts:
    """Bounded inspection of a blob-backed source.

    Frozen and deeply immutable so the same facts can be safely cached
    and shared across the composer service and preview_pipeline call sites.
    """

    source_kind: SourceKind
    redacted_identity: Mapping[str, str]
    byte_range_inspected: tuple[int, int]
    sample_row_count: int
    observed_headers: tuple[str, ...] | None
    inferred_types: Mapping[str, InferredType] | None
    url_candidates: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        freeze_fields(self, "redacted_identity")
        if self.inferred_types is not None:
            freeze_fields(self, "inferred_types")


def inspect_blob_content(
    *,
    content: bytes,
    filename: str,
    mime_type: str,
    blob_id: UUID | None = None,
    content_hash: str | None = None,
) -> SourceInspectionFacts:
    """Inspect raw blob bytes and return bounded structural facts.

    Cheap and deterministic. Reads at most ``_MAX_BYTES`` and parses at
    most ``_MAX_ROWS``. Returns facts even on parse error — partial
    inspection beats no inspection.
    """
    inspected = content[:_MAX_BYTES]
    byte_range = (0, len(inspected))

    redacted_identity: dict[str, str] = {
        "filename": filename,
        "mime_type": mime_type,
        "byte_size": str(len(content)),
    }
    if blob_id is not None:
        redacted_identity["blob_id"] = str(blob_id)
    if content_hash:
        # Surface only the prefix so identity is verifiable without leaking the full hash.
        redacted_identity["content_hash_prefix"] = content_hash[:8]

    kind = _detect_kind(filename, mime_type, inspected)

    if kind == "csv":
        return _inspect_csv(inspected, redacted_identity, byte_range)
    if kind == "jsonl":
        return _inspect_jsonl(inspected, redacted_identity, byte_range)
    if kind == "json":
        return _inspect_json(inspected, redacted_identity, byte_range)
    if kind == "text":
        return _inspect_text(inspected, redacted_identity, byte_range)

    return SourceInspectionFacts(
        source_kind="unknown",
        redacted_identity=redacted_identity,
        byte_range_inspected=byte_range,
        sample_row_count=0,
        observed_headers=None,
        inferred_types=None,
        url_candidates=tuple(_URL_PATTERN.findall(_safe_decode(inspected))),
        warnings=(f"unrecognised mime_type {mime_type!r} and filename {filename!r}",),
    )


def _detect_kind(filename: str, mime_type: str, sample: bytes) -> SourceKind:
    """Detect source kind from MIME first, filename next, content peek last."""
    mime = mime_type.lower()
    name = filename.lower()
    if mime == "text/csv" or name.endswith((".csv", ".tsv")):
        return "csv"
    if mime in ("application/x-jsonlines", "application/jsonl") or name.endswith((".jsonl", ".ndjson")):
        return "jsonl"
    if mime == "application/json" or name.endswith(".json"):
        # Decide json vs jsonl from content shape — repeated single-line objects
        # separated by newlines is jsonl, even when the file ends in .json.
        decoded = _safe_decode(sample).strip()
        if decoded.startswith("{") and "\n{" in decoded:
            return "jsonl"
        return "json"
    if mime.startswith("text/") or name.endswith((".txt", ".log", ".md")):
        return "text"
    return "unknown"


def _safe_decode(content: bytes) -> str:
    """Decode bytes as utf-8 with replacement; never raises."""
    return content.decode("utf-8", errors="replace")


def _infer_scalar_type(value: str) -> InferredType:
    """Infer a single scalar's likely type from its string form."""
    if value == "":
        return "null"
    stripped = value.strip()
    if stripped == "":
        return "str"
    if stripped.lower() in _BOOL_LITERALS:
        return "bool"
    if _INT_PATTERN.match(stripped):
        return "int"
    if _FLOAT_PATTERN.match(stripped):
        return "float"
    return "str"


def _merge_types(types: list[InferredType]) -> InferredType:
    """Merge per-row type observations for a single column.

    Conservative ladder: any non-null str → str. Mixed int/float → float.
    All-null → null. Bool requires unanimity; mixing bool with int/str
    falls back to str because bool literals like ``"yes"`` are
    indistinguishable from generic str otherwise.
    """
    seen = {t for t in types if t != "null"}
    if not seen:
        return "null"
    if "str" in seen:
        return "str"
    if seen == {"bool"}:
        return "bool"
    if seen <= {"int", "float"}:
        return "float" if "float" in seen else "int"
    # Mixed (e.g., int + bool). Generalize to str — the caller can add a
    # warning if one of these columns is then used in a numeric op.
    return "str"


def _inspect_csv(
    sample: bytes,
    redacted_identity: dict[str, str],
    byte_range: tuple[int, int],
) -> SourceInspectionFacts:
    text = _safe_decode(sample)
    reader = csv.reader(io.StringIO(text))
    rows: list[list[str]] = []
    try:
        for i, row in enumerate(reader):
            if i >= _MAX_ROWS:
                break
            rows.append(row)
    except csv.Error as exc:
        return SourceInspectionFacts(
            source_kind="csv",
            redacted_identity=redacted_identity,
            byte_range_inspected=byte_range,
            sample_row_count=len(rows),
            observed_headers=None,
            inferred_types=None,
            url_candidates=(),
            warnings=(f"csv parse error: {exc.__class__.__name__}",),
        )

    if not rows:
        return SourceInspectionFacts(
            source_kind="csv",
            redacted_identity=redacted_identity,
            byte_range_inspected=byte_range,
            sample_row_count=0,
            observed_headers=None,
            inferred_types=None,
            url_candidates=(),
            warnings=("csv content is empty",),
        )

    headers = tuple(h.strip() for h in rows[0])
    data_rows = rows[1:]
    warnings: list[str] = []

    if not all(headers):
        warnings.append("csv has empty header cells; consider field_mapping")

    # If the first row looks like data (every cell parseable as int/float/bool),
    # the file probably has no headers.
    headerless = all(_infer_scalar_type(cell) in {"int", "float", "bool"} for cell in rows[0] if cell.strip())
    if headerless and rows[0]:
        warnings.append("first row looks like data, not headers — consider explicit columns or field_mapping")

    types_per_column: dict[str, list[InferredType]] = {h: [] for h in headers}
    for row in data_rows:
        for col_idx, header in enumerate(headers):
            value = row[col_idx] if col_idx < len(row) else ""
            types_per_column[header].append(_infer_scalar_type(value))

    inferred = {h: _merge_types(types_per_column[h]) for h in headers}

    # URL hints inside data cells — sometimes a CSV has a URL column that
    # downstream needs to feed web_scrape.
    url_candidates: list[str] = []
    for row in data_rows:
        for cell in row:
            url_candidates.extend(_URL_PATTERN.findall(cell))
    # Deduplicate while preserving order.
    url_candidates = list(dict.fromkeys(url_candidates))

    return SourceInspectionFacts(
        source_kind="csv",
        redacted_identity=redacted_identity,
        byte_range_inspected=byte_range,
        sample_row_count=len(data_rows),
        observed_headers=headers,
        inferred_types=inferred,
        url_candidates=tuple(url_candidates),
        warnings=tuple(warnings),
    )


def _inspect_jsonl(
    sample: bytes,
    redacted_identity: dict[str, str],
    byte_range: tuple[int, int],
) -> SourceInspectionFacts:
    text = _safe_decode(sample)
    objects: list[dict[str, Any]] = []
    warnings: list[str] = []
    parse_failures = 0
    for i, raw_line in enumerate(text.splitlines()):
        if i >= _MAX_ROWS:
            break
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            parse_failures += 1
            continue
        if not isinstance(value, dict):
            parse_failures += 1
            continue
        objects.append(value)
    if parse_failures:
        warnings.append(f"{parse_failures} jsonl line(s) failed to parse as objects")
    return _facts_from_objects(
        objects=objects,
        kind="jsonl",
        redacted_identity=redacted_identity,
        byte_range=byte_range,
        extra_warnings=warnings,
        sample_text=text,
    )


def _inspect_json(
    sample: bytes,
    redacted_identity: dict[str, str],
    byte_range: tuple[int, int],
) -> SourceInspectionFacts:
    text = _safe_decode(sample)
    warnings: list[str] = []
    objects: list[dict[str, Any]] = []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        # Sample may be truncated in the middle of an object; that's the
        # normal case for an 8 KiB peek into a 50 MiB file. Record and move on.
        warnings.append(f"json parse error (sample may be truncated): {exc.msg}")
        loaded = None

    if isinstance(loaded, list):
        for item in loaded[:_MAX_ROWS]:
            if isinstance(item, dict):
                objects.append(item)
    elif isinstance(loaded, dict):
        # Wrapped data_key shapes — look for the first list-of-dicts value.
        for value in loaded.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                for item in value[:_MAX_ROWS]:
                    if isinstance(item, dict):
                        objects.append(item)
                warnings.append("json appears to be a wrapped object — set data_key on the source plugin")
                break
        if not objects:
            # Single object — still useful as a row of one.
            objects.append(loaded)

    return _facts_from_objects(
        objects=objects,
        kind="json",
        redacted_identity=redacted_identity,
        byte_range=byte_range,
        extra_warnings=warnings,
        sample_text=text,
    )


def _facts_from_objects(
    *,
    objects: list[dict[str, Any]],
    kind: SourceKind,
    redacted_identity: dict[str, str],
    byte_range: tuple[int, int],
    extra_warnings: list[str],
    sample_text: str,
) -> SourceInspectionFacts:
    warnings = list(extra_warnings)
    if not objects:
        return SourceInspectionFacts(
            source_kind=kind,
            redacted_identity=redacted_identity,
            byte_range_inspected=byte_range,
            sample_row_count=0,
            observed_headers=None,
            inferred_types=None,
            url_candidates=tuple(_URL_PATTERN.findall(sample_text)),
            warnings=tuple(warnings) if warnings else ("no parseable rows in sample",),
        )

    # Union of keys across sampled objects, preserving first-seen order.
    seen_keys: dict[str, None] = {}
    for obj in objects:
        for k in obj:
            seen_keys.setdefault(k, None)
    headers = tuple(seen_keys.keys())

    # Infer types from observed values.
    types_per_column: dict[str, list[InferredType]] = {h: [] for h in headers}
    for obj in objects:
        for header in headers:
            if header not in obj:
                types_per_column[header].append("null")
                continue
            value = obj[header]
            if value is None:
                types_per_column[header].append("null")
            elif isinstance(value, bool):
                types_per_column[header].append("bool")
            elif isinstance(value, int):
                types_per_column[header].append("int")
            elif isinstance(value, float):
                types_per_column[header].append("float")
            elif isinstance(value, str):
                types_per_column[header].append(_infer_scalar_type(value))
            else:
                # Nested structures (list/dict) — not a scalar; treat as str
                # for downstream-typing purposes and warn once.
                types_per_column[header].append("str")
    if any("str" in types_per_column[h] for h in headers):
        # Warn if the underlying value was a list/dict (vs a string scalar).
        for obj in objects:
            for h in headers:
                v = obj.get(h)
                if isinstance(v, (list, dict)):
                    warnings.append(f"field {h!r} contains nested structures; consider json_explode")
                    break
            else:
                continue
            break

    inferred = {h: _merge_types(types_per_column[h]) for h in headers}

    url_candidates: list[str] = []
    for obj in objects:
        for v in obj.values():
            if isinstance(v, str):
                url_candidates.extend(_URL_PATTERN.findall(v))
    url_candidates = list(dict.fromkeys(url_candidates))

    return SourceInspectionFacts(
        source_kind=kind,
        redacted_identity=redacted_identity,
        byte_range_inspected=byte_range,
        sample_row_count=len(objects),
        observed_headers=headers,
        inferred_types=inferred,
        url_candidates=tuple(url_candidates),
        warnings=tuple(warnings),
    )


def _inspect_text(
    sample: bytes,
    redacted_identity: dict[str, str],
    byte_range: tuple[int, int],
) -> SourceInspectionFacts:
    text = _safe_decode(sample)
    lines = [line for line in text.splitlines() if line.strip()][:_MAX_ROWS]
    url_candidates = list(dict.fromkeys(_URL_PATTERN.findall(text)))
    warnings: list[str] = []

    if len(lines) == 1 and url_candidates and url_candidates[0] == lines[0].strip():
        warnings.append(
            "text content is a single URL — pipeline must wire web_scrape to fetch the URL "
            "(text source emits the URL string itself, not the URL's content)"
        )
    elif url_candidates:
        warnings.append("text content contains URL(s); consider web_scrape downstream if URL fetch is intended")

    return SourceInspectionFacts(
        source_kind="text",
        redacted_identity=redacted_identity,
        byte_range_inspected=byte_range,
        sample_row_count=len(lines),
        observed_headers=None,
        inferred_types=None,
        url_candidates=tuple(url_candidates),
        warnings=tuple(warnings),
    )


def facts_to_dict(facts: SourceInspectionFacts) -> dict[str, Any]:
    """Serialize facts into a JSON-safe dict for tool results / proof diagnostics.

    Used by ``inspect_source`` MCP tool and by Step 3's ``preview_pipeline``
    so consumers can iterate without depending on the dataclass shape.
    """
    return {
        "source_kind": facts.source_kind,
        "redacted_identity": dict(facts.redacted_identity),
        "byte_range_inspected": list(facts.byte_range_inspected),
        "sample_row_count": facts.sample_row_count,
        "observed_headers": list(facts.observed_headers) if facts.observed_headers is not None else None,
        "inferred_types": dict(facts.inferred_types) if facts.inferred_types is not None else None,
        "url_candidates": list(facts.url_candidates),
        "warnings": list(facts.warnings),
    }


def derive_extra_column_risk(
    facts: SourceInspectionFacts,
    declared_fields: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Return observed headers absent from a declared fixed schema.

    Returns an empty tuple when the schema is observed/flexible (caller
    passes ``None``) or when every observed header is declared. Used by
    Step 3 (``preview_pipeline`` proof) to flag the all-row-discard
    hazard before the pipeline runs.
    """
    if declared_fields is None or facts.observed_headers is None:
        return ()
    declared_lower = {f.split(":", 1)[0].strip().lower() for f in declared_fields}
    missing = tuple(h for h in facts.observed_headers if h.lower() not in declared_lower)
    return missing


# Re-export the immutable identity used by callers that want to project the
# read-only dict view without re-copying.
def freeze_facts_identity(redacted_identity: Mapping[str, str]) -> Mapping[str, str]:
    """Return an immutable view of the identity mapping for caching."""
    return MappingProxyType(dict(redacted_identity))

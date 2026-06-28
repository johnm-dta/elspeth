#!/bin/bash
# Build the composer walkthrough as a professional PDF.
#
# Sibling to build-briefing.sh.  Both drive the same single-source
# pipeline (preprocess → pandoc → postprocess → typst), reuse the
# same Typst template, and differ only in their pinned defaults
# for source markdown, metadata YAML, and output PDF path.
#
# Pipeline: source walkthrough markdown → strip self-contained
# metadata header (since the title page is template-driven) →
# preprocess.py (mermaid render, hrule strip) → pandoc (typst
# output with template + walkthrough metadata) → postprocess.py
# (cell-alignment strip) → typst compile → PDF.
#
# Requirements: pandoc >= 3.0, typst >= 0.14, mermaid-cli (mmdc),
# python3.  mmdc is required by the shared toolchain check even
# though the walkthrough has no mermaid blocks.
#
# Usage:
#   ./build-walkthrough.sh                          # Generate .typ intermediate only
#   ./build-walkthrough.sh --pdf                    # Generate .typ and compile to PDF
#   ./build-walkthrough.sh --pdf --source PATH      # Override source markdown
#   ./build-walkthrough.sh --pdf --metadata PATH    # Override metadata YAML
#   ./build-walkthrough.sh --pdf --output PATH      # Override output PDF path
#
# Environment:
#   ELSPETH_WALKTHROUGH_SOURCE     Override source markdown path.
#   ELSPETH_WALKTHROUGH_METADATA   Override metadata YAML path.
#   ELSPETH_WALKTHROUGH_OUTPUT     Override output PDF path.
#   FORCE_DATE                     Override the title-page date (default: today).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# ─────────────────────────────────────────────────────────────
# Defaults — can be overridden via env or flag.
# ─────────────────────────────────────────────────────────────
DEFAULT_SOURCE="$PROJECT_ROOT/docs/composer/evidence/composer-walkthrough-2026-05-03.md"
DEFAULT_METADATA="$SCRIPT_DIR/walkthrough-metadata.yaml"
DEFAULT_OUTPUT="$PROJECT_ROOT/tools/pdf/out/composer-walkthrough-2026-05-03.pdf"

SOURCE="${ELSPETH_WALKTHROUGH_SOURCE:-$DEFAULT_SOURCE}"
METADATA="${ELSPETH_WALKTHROUGH_METADATA:-$DEFAULT_METADATA}"
OUTPUT_PDF="${ELSPETH_WALKTHROUGH_OUTPUT:-$DEFAULT_OUTPUT}"
COMPILE_PDF=false

# ─────────────────────────────────────────────────────────────
# Argument parsing.
# ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pdf)      COMPILE_PDF=true; shift ;;
        --source)   SOURCE="$2"; shift 2 ;;
        --metadata) METADATA="$2"; shift 2 ;;
        --output)   OUTPUT_PDF="$2"; shift 2 ;;
        *) echo "[error] unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "$SOURCE" ]]; then
    echo "[error] source not found: $SOURCE" >&2
    exit 1
fi
if [[ ! -f "$METADATA" ]]; then
    echo "[error] metadata not found: $METADATA" >&2
    exit 1
fi

OUTPUT_TYP="$SCRIPT_DIR/$(basename "${SOURCE%.md}").typ"
MERMAID_DIR="$SCRIPT_DIR/.mermaid-tmp"

echo "Source:   $SOURCE"
echo "Metadata: $METADATA"
echo "Output:   $OUTPUT_PDF"

els_check_toolchain

# ─────────────────────────────────────────────────────────────
# Strip the markdown's self-contained metadata header.  The
# walkthrough is designed to stand alone as markdown, so its first
# block is a title (H1) + a list of bold metadata fields + a
# free-text framing paragraph + a horizontal-rule divider.  For
# PDF rendering the template provides the title page from the
# metadata YAML, so the markdown's equivalent block must be
# removed to avoid duplication.
#
# The strip is bounded: from the first H1 down through the
# first standalone "---" line, inclusive.  Subsequent "---"
# dividers (between sections) are preserved at this stage and
# stripped later by preprocess.py.
# ─────────────────────────────────────────────────────────────
COMBINED=$(mktemp)
PROCESSED=$(mktemp)
STAMPED_METADATA=$(mktemp --suffix=.yaml)
trap 'rm -f "$COMBINED" "$PROCESSED" "$STAMPED_METADATA"; rm -rf "$MERMAID_DIR"' EXIT

echo "Stripping markdown title block..."
awk '
    BEGIN { in_header = 1 }
    in_header && /^---$/ { in_header = 0; next }
    !in_header { print }
' "$SOURCE" > "$COMBINED"

echo "Preprocessing markdown..."
python3 "$SCRIPT_DIR/preprocess.py" \
    --input="$COMBINED" \
    --output="$PROCESSED" \
    --mermaid-dir="$MERMAID_DIR" \
    --mermaid-rel-base="$SCRIPT_DIR"

echo "Stamping build date..."
els_stamp_date "$METADATA" "$STAMPED_METADATA"

echo "Generating Typst intermediate..."
els_run_pandoc "$PROCESSED" "$OUTPUT_TYP" "$STAMPED_METADATA"

echo "Post-processing Typst output..."
python3 "$SCRIPT_DIR/postprocess.py" "$OUTPUT_TYP" "$OUTPUT_TYP"
echo "  -> $OUTPUT_TYP"

if $COMPILE_PDF; then
    echo "Compiling PDF..."
    els_compile_pdf "$OUTPUT_TYP" "$OUTPUT_PDF"
    echo "  -> $OUTPUT_PDF"
    echo "  $(wc -c < "$OUTPUT_PDF" | xargs) bytes"
fi

echo "Done."

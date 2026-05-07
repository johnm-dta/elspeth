#!/bin/bash
# Build the ELSPETH Architecture Pack as a single professional PDF.
#
# Pipeline: discover arch-pack source dir → concatenate chapters with
# synthetic Part dividers → preprocess.py (mermaid render, hrule strip)
# → pandoc (typst output) → postprocess.py (cell-alignment strip) →
# typst compile → PDF.
#
# Requirements: pandoc >= 3.0, typst >= 0.14, mermaid-cli (mmdc), python3.
#
# Usage:
#   ./build-arch-pack.sh              # Generate .typ intermediate only
#   ./build-arch-pack.sh --pdf        # Generate .typ and compile to PDF
#
# Environment:
#   ELSPETH_ARCH_PACK_DIR   Override the source directory (default: latest
#                           docs/arch-pack-* by lexicographic order).
#   FORCE_DATE              Override the title-page date (default: today).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

ARCH_PACK_DIR="$(els_discover_arch_pack)"
echo "Source: $ARCH_PACK_DIR"

OUTPUT_TYP="$SCRIPT_DIR/elspeth-arch-pack.typ"
OUTPUT_PDF="$PROJECT_ROOT/docs/assets/elspeth-arch-pack.pdf"
MERMAID_DIR="$SCRIPT_DIR/.mermaid-tmp"

# ─────────────────────────────────────────────────────────────
# Chapter ordering.  Ordering matches the on-disk arrangement of
# the arch-pack:
#
#   Narrative track:    00..08 chapters at the top of the pack
#   Subsystem track:    subsystems/* (one chapter per cluster)
#   Appendix track:     appendix/A,B,C
#   Reference track:    reference/* prose (binary oracles excluded)
#
# Each track's first file already provides an H1 introduction
# heading ("# Subsystem Reference", "# Reference Data") or a
# clearly-labelled appendix start ("# Appendix A — Glossary"),
# so no synthetic Part divider is injected.  The chapters carry
# their own chapter breaks; the build script's job is just to
# concatenate in the right order.
#
# Excluded from concatenation:
#   reference/l3-import-graph.{dot,json,mmd}  (deterministic oracles)
#   reference/tier-model-oracle.txt           (deterministic oracle)
#   README.md (top-level)                     (navigation aid; absorbed
#                                              into 00-executive-summary)
# ─────────────────────────────────────────────────────────────
NARRATIVE_CHAPTERS=(
    00-executive-summary.md
    01-system-context.md
    02-architecture-overview.md
    03-container-view.md
    04-component-view.md
    05-cross-cutting-concerns.md
    06-quality-assessment.md
    07-improvement-roadmap.md
    08-known-gaps.md
)

SUBSYSTEM_CHAPTERS=(
    subsystems/README.md
    subsystems/contracts.md
    subsystems/core.md
    subsystems/engine.md
    subsystems/plugins.md
    subsystems/web-composer.md
    subsystems/leaf-subsystems.md
)

APPENDIX_CHAPTERS=(
    appendix/A-glossary.md
    appendix/B-methodology.md
    appendix/C-provenance.md
)

REFERENCE_CHAPTERS=(
    reference/README.md
    reference/adr-index.md
    reference/re-derive.md
)

els_check_toolchain

COMBINED=$(mktemp)
PROCESSED=$(mktemp)
STAMPED_METADATA=$(mktemp --suffix=.yaml)
trap 'rm -f "$COMBINED" "$PROCESSED" "$STAMPED_METADATA"; rm -rf "$MERMAID_DIR"' EXIT

# Total chapter count (for the status line)
TOTAL=$(( ${#NARRATIVE_CHAPTERS[@]} + ${#SUBSYSTEM_CHAPTERS[@]} \
       + ${#APPENDIX_CHAPTERS[@]}  + ${#REFERENCE_CHAPTERS[@]} ))
echo "Concatenating $TOTAL arch-pack chapters across 4 tracks..."

# Initialise the combined file empty before the first track.  Each
# els_concat_chapters / els_insert_part_divider call only appends.
: > "$COMBINED"

# Track 1: narrative
els_concat_chapters "$COMBINED" "$ARCH_PACK_DIR" "${NARRATIVE_CHAPTERS[@]}"
# Track 2: subsystem reference (subsystems/README.md opens with "# Subsystem Reference")
els_concat_chapters "$COMBINED" "$ARCH_PACK_DIR" "${SUBSYSTEM_CHAPTERS[@]}"
# Track 3: appendices (each file opens with "# Appendix X — ...")
els_concat_chapters "$COMBINED" "$ARCH_PACK_DIR" "${APPENDIX_CHAPTERS[@]}"
# Track 4: reference data (reference/README.md opens with "# Reference Data")
els_concat_chapters "$COMBINED" "$ARCH_PACK_DIR" "${REFERENCE_CHAPTERS[@]}"

echo "Preprocessing markdown..."
python3 "$SCRIPT_DIR/preprocess.py" \
    --input="$COMBINED" \
    --output="$PROCESSED" \
    --mermaid-dir="$MERMAID_DIR" \
    --mermaid-rel-base="$SCRIPT_DIR"

echo "Stamping build date..."
els_stamp_date "$SCRIPT_DIR/metadata.yaml" "$STAMPED_METADATA"

echo "Generating Typst intermediate..."
els_run_pandoc "$PROCESSED" "$OUTPUT_TYP" "$STAMPED_METADATA"

echo "Post-processing Typst output..."
python3 "$SCRIPT_DIR/postprocess.py" "$OUTPUT_TYP" "$OUTPUT_TYP"
echo "  -> $OUTPUT_TYP"

if [[ "${1:-}" == "--pdf" ]]; then
    echo "Compiling PDF..."
    els_compile_pdf "$OUTPUT_TYP" "$OUTPUT_PDF"
    echo "  -> $OUTPUT_PDF"
    echo "  $(wc -c < "$OUTPUT_PDF" | xargs) bytes"
fi

echo "Done."

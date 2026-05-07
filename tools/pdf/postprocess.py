#!/usr/bin/env python3
"""Post-pandoc Typst transformations for the ELSPETH Architecture Pack.

Pandoc's typst writer emits some directives that are redundant or actively
counterproductive given the look-and-feel rules in our Typst template.
This script applies the small structural fixups that are easier expressed
as text edits than as Lua filters or template overrides.

Transformations:
    1. Strip per-cell alignment directives (template handles alignment globally)

Earlier versions of this pipeline (and the upstream wardline pipeline this
was ported from) also injected ``<section-X-Y>`` labels for clickable
``§N.N`` cross-references, plus appendix and Part labels.  The arch-pack
source uses ``§N`` only as in-heading numbering local to each chapter and
in cross-document body references like ``06-quality-assessment.md §1 E1``,
so injecting global labels would silently break references rather than
resolve them.  The simpler, honest behaviour is to render ``§N`` as plain
text — the styling shouldn't suggest a link the document can't honour.

Usage:
    python3 postprocess.py input.typ output.typ
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Strip per-cell alignment directives
# ─────────────────────────────────────────────────────────────
# Pandoc emits per-cell alignment on every table cell:
#     align: (left,),
# The template's ``set table(align: left)`` handles this globally,
# so these directives are redundant and add visual noise that makes
# manual review of the .typ intermediate harder.

_ALIGN_PATTERN = re.compile(r"^    align: \([^)]*\),\n", re.MULTILINE)


def strip_cell_alignments(content: str) -> str:
    """Remove pandoc's per-cell alignment directives."""
    result, count = _ALIGN_PATTERN.subn("", content)
    if count > 0:
        sys.stderr.write(f"  Stripped {count} per-cell alignment directives\n")
    return result


# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────
def postprocess(content: str) -> str:
    """Apply all post-pandoc transformations."""
    content = strip_cell_alignments(content)
    return content


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} input.typ output.typ\n")
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        sys.stderr.write(f"[error] input file not found: {input_path}\n")
        return 1

    content = input_path.read_text()
    result = postprocess(content)
    output_path.write_text(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

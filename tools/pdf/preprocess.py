#!/usr/bin/env python3
"""Markdown preprocessor for the ELSPETH Architecture Pack PDF pipeline.

Takes a concatenated Markdown file (built from the arch-pack source tree by
``build-arch-pack.sh``) and applies a small sequence of transforms before
handing the result to pandoc.  Each transform is a small composable function
so it can be unit-tested or bypassed.

Transforms applied (in order):

    1. ``strip_standalone_hrules``  — remove ``---`` lines used as visual
       separators in the source (the Typst template provides chapter-break
       structure already; standalone hrules confuse pandoc's table parser).
    2. ``rewrite_intra_pack_links`` — rewrite relative inter-chapter links
       (``[`02-architecture-overview.md`](02-architecture-overview.md)``)
       so they read sensibly in a single concatenated PDF.  We strip the
       link target and keep the inline label, so the PDF reads as plain
       text rather than a broken hyperlink.
    3. ``render_mermaid_blocks``    — render every ```mermaid``` fence to a
       300-DPI PNG via ``mmdc`` and replace the fence with a Typst
       ``#figure(kind: image)`` passthrough.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Standalone horizontal rules
# ─────────────────────────────────────────────────────────────
def strip_standalone_hrules(text: str) -> str:
    """Remove standalone ``---`` lines.

    The arch-pack uses ``---`` as a within-chapter visual separator
    (especially in 07-improvement-roadmap and 08-known-gaps).  In a
    paginated PDF the chapter and section structure carries the visual
    break already, and pandoc's GFM table parser is sensitive to bare
    ``---`` lines that happen to fall above a pipe-table.
    """
    return re.sub(r"(?m)^---\s*$\n?", "", text)


# ─────────────────────────────────────────────────────────────
# Intra-pack relative links
# ─────────────────────────────────────────────────────────────
# The arch-pack uses inline links between chapters and into reference
# subdirectories.  Examples:
#
#   [`02-architecture-overview.md`](02-architecture-overview.md)
#   [`05-cross-cutting-concerns.md`](05-cross-cutting-concerns.md)
#   [`../reference/l3-import-graph.json`](../reference/l3-import-graph.json)
#
# In a single concatenated PDF these targets don't resolve to anything
# meaningful — chapter files are gone, and the bare filename is more
# noise than navigation.  We strip the link and keep the label text so
# the prose still reads naturally.

_INTRAPACK_LINK = re.compile(
    r"\[(?P<label>[^\]]+)\]\((?P<target>(?:\.{1,2}/)?[A-Za-z0-9_./-]+\.(?:md|json|dot|mmd|txt))\)"
)


def rewrite_intra_pack_links(text: str) -> str:
    """Strip relative file-to-file links; keep their visible label."""
    return _INTRAPACK_LINK.sub(lambda m: m.group("label"), text)


# ─────────────────────────────────────────────────────────────
# Mermaid rendering
# ─────────────────────────────────────────────────────────────
# PDF-only sizing hints are placed on a dedicated HTML comment line
# immediately preceding the fence (a markdown-invisible directive):
#
#     <!-- pdf: size="height: 90%" alt="C4 L1 system context" -->
#     ```mermaid
#     ...
#     ```
#
# Supported attributes:
#   size    — a Typst image sizing expression (default ``width: 75%``)
#   orient  — ``vertical`` (default) pre-flips ``graph LR`` to ``graph TB``
#             before rendering; ``preserve`` keeps horizontal orientation.
#   alt     — alternative text for accessibility (PDF/UA compliance) and
#             figure caption.
_MERMAID_FENCE = re.compile(
    r"(?:^<!-- pdf:(?P<directive>[^\n]*)-->\s*\n)?"
    r"^```mermaid\s*\n(?P<body>.*?)^```",
    re.DOTALL | re.MULTILINE,
)
_ATTR_PAIR = re.compile(r'(\w+)="([^"]*)"')


@dataclass
class MermaidOptions:
    size: str = "width: 75%"
    orient: str = "vertical"
    alt: str = ""

    @classmethod
    def parse(cls, raw: str) -> "MermaidOptions":
        attrs = dict(_ATTR_PAIR.findall(raw or ""))
        return cls(
            size=attrs.get("size", "width: 75%"),
            orient=attrs.get("orient", "vertical"),
            alt=attrs.get("alt", ""),
        )

def render_mermaid_blocks(
    text: str,
    mermaid_dir: Path,
    relative_base: Path,
) -> str:
    """Render every ``mermaid`` fence to high-resolution PNG for Typst.

    Diagrams are rendered through ``mmdc`` (mermaid-cli) at 4× scale for
    print-quality output (~300 DPI).  While SVG would be ideal, Typst's
    SVG renderer doesn't fully support mermaid's ``foreignObject`` text.
    If rendering fails the fence is left in place so the build surfaces
    the problem rather than silently dropping a diagram.
    """
    mermaid_dir.mkdir(parents=True, exist_ok=True)
    counter = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        opts = MermaidOptions.parse(match.group("directive") or "")
        code = match.group("body")
        if opts.orient == "vertical":
            code = re.sub(r"(?m)^graph LR$", "graph TB", code)

        mmd_file = mermaid_dir / f"diagram-{counter}.mmd"
        png_file = mermaid_dir / f"diagram-{counter}.png"
        mmd_file.write_text(code)

        try:
            subprocess.run(
                [
                    "mmdc",
                    "-i", str(mmd_file),
                    "-o", str(png_file),
                    "-b", "white",
                    "-t", "neutral",
                    "-s", "4",  # 4× scale for ~300 DPI print quality
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            sys.stderr.write(
                f"  [warn] mermaid render failed for diagram-{counter}: {e}\n"
            )
            return match.group(0)

        if not png_file.exists():
            sys.stderr.write(f"  [warn] mermaid output missing for diagram-{counter}\n")
            return match.group(0)

        rel_path = os.path.relpath(png_file, relative_base)
        sys.stderr.write(f"  Rendered {rel_path} ({opts.size})\n")
        # Emit a raw Typst passthrough wrapped in #figure(kind: image)
        # so the diagram gets a "Figure N" caption and appears in the
        # List of Figures.  The alt text becomes the figure caption.
        alt_attr = f', alt: "{opts.alt}"' if opts.alt else ""
        caption = f", caption: [{opts.alt}]" if opts.alt else ""
        return (
            "```{=typst}\n"
            f'#figure(kind: image{caption})[#image("{rel_path}", {opts.size}{alt_attr})]\n'
            "```\n"
        )

    return _MERMAID_FENCE.sub(replace, text)


# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────
def run(text: str, mermaid_dir: Path, rel_base: Path) -> str:
    text = strip_standalone_hrules(text)
    text = rewrite_intra_pack_links(text)
    text = render_mermaid_blocks(text, mermaid_dir, rel_base)
    return text


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mermaid-dir",
        type=Path,
        required=True,
        help="Scratch directory for rendered mermaid PNGs.",
    )
    parser.add_argument(
        "--mermaid-rel-base",
        type=Path,
        required=True,
        help="Base path relative to which PNG references are emitted (the "
             "directory containing the final .typ file).",
    )
    args = parser.parse_args(argv)

    text = args.input.read_text()
    transformed = run(text, args.mermaid_dir, args.mermaid_rel_base)
    args.output.write_text(transformed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

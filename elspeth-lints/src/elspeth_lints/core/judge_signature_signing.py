"""Standalone wrapper for the operator-only judge-signature repair command."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Run ``elspeth-lints sign-judge-signatures`` as a console script."""
    from elspeth_lints.core.cli import main as cli_main

    args = sys.argv[1:] if argv is None else argv
    return cli_main(["sign-judge-signatures", *args])

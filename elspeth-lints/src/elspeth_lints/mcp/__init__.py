"""MCP server for the key-free ``elspeth-judge`` agent surface.

The agent stages an authority-free review *bundle* through this server; the
operator fires it with the key-bearing ``elspeth-lints sign-bundle`` / ``rekey``
CLI. Every tool here is structurally key-free ([O1]): it fails closed if
``ELSPETH_JUDGE_METADATA_HMAC_KEY`` is present in its environment, and no tool
ever mints a signature.

``main`` is exposed via a lazy import so importing this package never requires
the optional ``[mcp]`` extra until the server is actually launched.
"""


def main(argv: list[str] | None = None) -> None:
    """Run the ``elspeth-judge`` MCP server. Requires the ``[mcp]`` extra."""
    try:
        from elspeth_lints.mcp.server import main as _main
    except ModuleNotFoundError as exc:
        if "mcp" in str(exc):
            raise ImportError(
                "elspeth-judge MCP server requires the [mcp] extra. Install with: uv pip install -e '.[mcp]' from elspeth-lints/"
            ) from exc
        raise

    _main(argv)


__all__ = ["main"]

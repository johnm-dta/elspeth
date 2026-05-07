"""Unit tests for the enforce_tier_model.py ``dump-edges`` subcommand.

Δ7 contract — 10 required cases:

1. Smoke: scans a fixture tree with 3 known files, asserts node/edge counts.
2. Layer filtering: --include-layer L2 returns only L2↔L2 edges.
3. TYPE_CHECKING tagging: import inside ``if TYPE_CHECKING:`` block is tagged.
4. Conditional tagging: import inside ``try/except ImportError`` is tagged.
5. Re-export tagging: ``__init__.py`` doing ``from .x import y`` produces
   reexport=True on that edge.
6. Collapse: --collapse-to-subsystem aggregates file-level edges to
   subsystem-level with summed weights.
7. SCC detection: fixture with a known 3-node cycle reports it correctly.
8. Determinism: two runs with --no-timestamp produce byte-identical output.
9. CLI: invalid layer name rejected with non-zero exit.
10. Empty input: --root pointing at empty dir produces a valid empty graph
    (0 nodes, 0 edges) and exits 0.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from textwrap import dedent

import pytest
from scripts.cicd.enforce_tier_model import (
    render_dump_edges_dot,
    render_dump_edges_json,
    render_dump_edges_mermaid,
    scan_dump_edges,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_root() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def _write(path: Path, content: str) -> None:
    """Write ``content`` to ``path``, ensuring parent dirs exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip("\n"), encoding="utf-8")


def _make_minimal_l3_tree(root: Path) -> None:
    """Build a tiny three-file tree where ``plugins/`` and ``web/`` import each other.

    Layout (under ``root``, which is treated as ``src/elspeth`` equivalent):

        plugins/__init__.py        (empty package marker)
        plugins/transforms/llm/azure.py     (imports elspeth.web.composer.tools)
        web/composer/tools.py      (imports elspeth.plugins.transforms.llm.azure)
    """
    _write(root / "plugins" / "__init__.py", "")
    _write(root / "plugins" / "transforms" / "__init__.py", "")
    _write(root / "plugins" / "transforms" / "llm" / "__init__.py", "")
    _write(
        root / "plugins" / "transforms" / "llm" / "azure.py",
        """
        from elspeth.web.composer import tools  # noqa: F401
        """,
    )
    _write(root / "web" / "__init__.py", "")
    _write(root / "web" / "composer" / "__init__.py", "")
    _write(
        root / "web" / "composer" / "tools.py",
        """
        from elspeth.plugins.transforms.llm import azure  # noqa: F401
        """,
    )


# =============================================================================
# Case 1 — Smoke
# =============================================================================


def test_case01_smoke_scan(temp_root: Path) -> None:
    """3 known import-bearing files should produce ≥2 L3 nodes and ≥2 L3 edges."""
    _make_minimal_l3_tree(temp_root)
    nodes, edges, sccs = scan_dump_edges(
        root=temp_root,
        include_layers=frozenset({3}),
        collapse_to_subsystem=True,
    )

    node_ids = {n["id"] for n in nodes}
    # The two non-empty packages should appear; the empty plugins/__init__ may not
    # contribute outbound edges but is still a node by virtue of containing a .py file.
    assert "plugins/transforms/llm" in node_ids
    assert "web/composer" in node_ids

    edge_pairs = {(e["from"], e["to"]) for e in edges}
    assert ("plugins/transforms/llm", "web/composer") in edge_pairs
    assert ("web/composer", "plugins/transforms/llm") in edge_pairs

    # The 2-cycle is itself a non-trivial SCC.
    scc_sets = [frozenset(scc) for scc in sccs]
    assert frozenset({"plugins/transforms/llm", "web/composer"}) in scc_sets


# =============================================================================
# Case 2 — Layer filtering (--include-layer L2 returns only L2↔L2)
# =============================================================================


def test_case02_layer_filtering(temp_root: Path) -> None:
    """L2-only scan should ignore the L3↔L3 tree and produce zero edges."""
    _make_minimal_l3_tree(temp_root)
    # Add an L2 file and an L1 file — but no L2↔L2 edges, so the filtered graph is empty.
    _write(temp_root / "engine" / "__init__.py", "")
    _write(
        temp_root / "engine" / "orchestrator.py",
        """
        # Engine importing from contracts (downward, layer-conformant) — but contracts is L0,
        # not L2. With include-layer=L2, this edge must be filtered out.
        from elspeth.contracts import schema_contract  # noqa: F401
        """,
    )
    _write(temp_root / "contracts" / "__init__.py", "")
    _write(temp_root / "contracts" / "schema_contract.py", "FOO = 1\n")

    nodes_l2, edges_l2, _ = scan_dump_edges(
        root=temp_root,
        include_layers=frozenset({2}),
        collapse_to_subsystem=True,
    )
    # No L2 file imports another L2 file, so edges should be empty.
    assert edges_l2 == []
    # All node layers must be L2.
    assert all(n["layer"] == "L2/engine" for n in nodes_l2)


# =============================================================================
# Case 3 — TYPE_CHECKING tagging
# =============================================================================


def test_case03_type_checking_tagging(temp_root: Path) -> None:
    """Imports inside ``if TYPE_CHECKING:`` must be tagged type_checking_only=True."""
    _write(temp_root / "plugins" / "__init__.py", "")
    _write(temp_root / "plugins" / "alpha.py", "X = 1\n")
    _write(temp_root / "web" / "__init__.py", "")
    _write(
        temp_root / "web" / "consumer.py",
        """
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from elspeth.plugins import alpha  # noqa: F401
        """,
    )

    _, edges, _ = scan_dump_edges(
        root=temp_root,
        include_layers=frozenset({3}),
        collapse_to_subsystem=True,
    )
    # Find the web→plugins edge.
    found = [e for e in edges if e["from"] == "web" and e["to"] == "plugins"]
    assert len(found) == 1
    assert found[0]["type_checking_only"] is True


# =============================================================================
# Case 4 — Conditional tagging (import inside try/except ImportError)
# =============================================================================


def test_case04_conditional_tagging(temp_root: Path) -> None:
    """Imports inside try/except blocks must be tagged conditional=True."""
    _write(temp_root / "plugins" / "__init__.py", "")
    _write(temp_root / "plugins" / "optional.py", "X = 1\n")
    _write(temp_root / "web" / "__init__.py", "")
    _write(
        temp_root / "web" / "consumer.py",
        """
        try:
            from elspeth.plugins import optional  # noqa: F401
        except ImportError:
            optional = None  # type: ignore[assignment]
        """,
    )

    _, edges, _ = scan_dump_edges(
        root=temp_root,
        include_layers=frozenset({3}),
        collapse_to_subsystem=True,
    )
    found = [e for e in edges if e["from"] == "web" and e["to"] == "plugins"]
    assert len(found) == 1
    assert found[0]["conditional"] is True
    assert found[0]["type_checking_only"] is False


# =============================================================================
# Case 5 — Re-export tagging
# =============================================================================


def test_case05_reexport_tagging(temp_root: Path) -> None:
    """``__init__.py`` doing ``from .x import y`` should yield reexport=True."""
    # Two top-level subsystems so the edge isn't dropped by the collapse self-loop rule.
    _write(temp_root / "plugins" / "__init__.py", "")
    _write(temp_root / "plugins" / "submod.py", "X = 1\n")
    _write(temp_root / "web" / "_internal" / "submod.py", "Y = 2\n")
    _write(
        temp_root / "web" / "_internal" / "__init__.py",
        """
        # Re-export from a sibling package via a relative import. The Δ3 rule 9
        # heuristic flags this — the source is __init__.py and the import is relative
        # (level > 0), so the resulting edge is tagged reexport=True.
        from .submod import Y  # noqa: F401
        """,
    )
    _write(
        temp_root / "web" / "__init__.py",
        """
        from elspeth.plugins import submod  # noqa: F401
        """,
    )

    _, edges, _ = scan_dump_edges(
        root=temp_root,
        include_layers=frozenset({3}),
        collapse_to_subsystem=False,  # file-level so we can pinpoint the __init__.py edge
    )
    # Find the edge from web/_internal/__init__.py → web/_internal/submod.py.
    found = [e for e in edges if e["from"] == "web/_internal/__init__.py" and e["to"] == "web/_internal/submod.py"]
    assert len(found) == 1, f"expected exactly 1 reexport edge, got: {edges}"
    assert found[0]["reexport"] is True


# =============================================================================
# Case 6 — Collapse aggregation
# =============================================================================


def test_case06_collapse_aggregation(temp_root: Path) -> None:
    """Multiple file-level edges between two packages must aggregate to one
    subsystem-level edge with weight = sum.
    """
    _write(temp_root / "plugins" / "__init__.py", "")
    _write(temp_root / "plugins" / "alpha.py", "X = 1\n")
    _write(temp_root / "plugins" / "beta.py", "Y = 1\n")
    _write(temp_root / "web" / "__init__.py", "")
    _write(
        temp_root / "web" / "consumer_a.py",
        "from elspeth.plugins import alpha  # noqa: F401\n",
    )
    _write(
        temp_root / "web" / "consumer_b.py",
        "from elspeth.plugins import beta  # noqa: F401\n",
    )

    # Without collapse: two distinct file→file edges.
    _, edges_uncollapsed, _ = scan_dump_edges(
        root=temp_root,
        include_layers=frozenset({3}),
        collapse_to_subsystem=False,
    )
    web_to_plugin = [e for e in edges_uncollapsed if e["from"].startswith("web/") and e["to"].startswith("plugins/")]
    assert len(web_to_plugin) == 2

    # With collapse: one aggregated edge of weight 2.
    _, edges_collapsed, _ = scan_dump_edges(
        root=temp_root,
        include_layers=frozenset({3}),
        collapse_to_subsystem=True,
    )
    aggregated = [e for e in edges_collapsed if e["from"] == "web" and e["to"] == "plugins"]
    assert len(aggregated) == 1
    assert aggregated[0]["weight"] == 2


# =============================================================================
# Case 7 — SCC detection (3-node cycle)
# =============================================================================


def test_case07_scc_detection(temp_root: Path) -> None:
    """A → B → C → A must be reported as a single 3-node SCC."""
    _write(temp_root / "alpha" / "__init__.py", "")
    _write(temp_root / "beta" / "__init__.py", "")
    _write(temp_root / "gamma" / "__init__.py", "")
    _write(
        temp_root / "alpha" / "mod.py",
        "from elspeth.beta import mod  # noqa: F401\n",
    )
    _write(temp_root / "beta" / "mod.py", "from elspeth.gamma import mod  # noqa: F401\n")
    _write(temp_root / "gamma" / "mod.py", "from elspeth.alpha import mod  # noqa: F401\n")

    _, _, sccs = scan_dump_edges(
        root=temp_root,
        include_layers=frozenset({3}),
        collapse_to_subsystem=True,
    )
    cycle = {"alpha", "beta", "gamma"}
    assert any(set(scc) == cycle for scc in sccs), f"expected 3-node SCC {cycle}, got {sccs}"


# =============================================================================
# Case 8 — Determinism (--no-timestamp)
# =============================================================================


def test_case08_determinism(temp_root: Path) -> None:
    """Two scans with --no-timestamp must produce byte-identical JSON output."""
    _make_minimal_l3_tree(temp_root)

    nodes1, edges1, sccs1 = scan_dump_edges(root=temp_root, include_layers=frozenset({3}), collapse_to_subsystem=True)
    nodes2, edges2, sccs2 = scan_dump_edges(root=temp_root, include_layers=frozenset({3}), collapse_to_subsystem=True)

    rendered1 = render_dump_edges_json(
        root=temp_root,
        include_layers=frozenset({3}),
        collapse_to_subsystem=True,
        nodes=nodes1,
        edges=edges1,
        sccs=sccs1,
        use_stable_placeholder=True,
    )
    rendered2 = render_dump_edges_json(
        root=temp_root,
        include_layers=frozenset({3}),
        collapse_to_subsystem=True,
        nodes=nodes2,
        edges=edges2,
        sccs=sccs2,
        use_stable_placeholder=True,
    )
    assert rendered1 == rendered2

    # And the placeholder must actually appear (proving the elision happened).
    payload = json.loads(rendered1)
    assert payload["generated_at"] == "<stable>"
    assert payload["tool_version"] == "<stable>"


# =============================================================================
# Case 9 — CLI: invalid layer name rejected
# =============================================================================


def test_case09_cli_invalid_layer(temp_root: Path) -> None:
    """``--include-layer L9`` must be rejected by argparse with non-zero exit."""
    script = Path(__file__).resolve().parents[4] / "scripts" / "cicd" / "enforce_tier_model.py"
    out_path = temp_root / "out.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "dump-edges",
            "--root",
            str(temp_root),
            "--include-layer",
            "L9",
            "--output",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "L9" in result.stderr or "invalid choice" in result.stderr


# =============================================================================
# Case 10 — Empty input
# =============================================================================


def test_case10_empty_input(temp_root: Path) -> None:
    """An empty source tree must produce a valid empty graph and exit 0."""
    nodes, edges, sccs = scan_dump_edges(
        root=temp_root,
        include_layers=frozenset({3}),
        collapse_to_subsystem=True,
    )
    assert nodes == []
    assert edges == []
    assert sccs == []

    rendered = render_dump_edges_json(
        root=temp_root,
        include_layers=frozenset({3}),
        collapse_to_subsystem=True,
        nodes=nodes,
        edges=edges,
        sccs=sccs,
        use_stable_placeholder=True,
    )
    payload = json.loads(rendered)
    assert payload["stats"]["total_nodes"] == 0
    assert payload["stats"]["total_edges"] == 0
    assert payload["stats"]["scc_count"] == 0
    assert payload["nodes"] == []
    assert payload["edges"] == []
    assert payload["strongly_connected_components"] == []


# =============================================================================
# Bonus — output formatters round-trip / rendering smoke
# =============================================================================


def test_mermaid_renderer_smoke(temp_root: Path) -> None:
    """Mermaid output is non-empty, contains a flowchart header, and references nodes."""
    _make_minimal_l3_tree(temp_root)
    nodes, edges, _ = scan_dump_edges(root=temp_root, include_layers=frozenset({3}), collapse_to_subsystem=True)
    rendered = render_dump_edges_mermaid(nodes, edges)
    assert rendered.startswith("flowchart LR")
    # The cycle pair must both appear as nodes.
    assert "web_composer" in rendered
    assert "plugins_transforms_llm" in rendered


def test_dot_renderer_smoke(temp_root: Path) -> None:
    """DOT output starts with digraph and references node ids in quoted form."""
    _make_minimal_l3_tree(temp_root)
    nodes, edges, _ = scan_dump_edges(root=temp_root, include_layers=frozenset({3}), collapse_to_subsystem=True)
    rendered = render_dump_edges_dot(nodes, edges)
    assert rendered.startswith("digraph l3_imports {")
    assert '"web/composer"' in rendered
    assert '"plugins/transforms/llm"' in rendered
    assert rendered.rstrip().endswith("}")

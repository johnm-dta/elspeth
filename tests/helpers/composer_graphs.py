"""Surface-agnostic composition-graph isomorphism helper (Plan 05 Task 3).

This is the shared correctness core of the composer capability-parity matrix.
The three authoring surfaces (freeform, guided-full, guided-staged) each derive
a committed ``CompositionState`` through an independent production code path.
Parity means the *semantic* graphs agree even though surface-specific noise
(generated ids, connection names, temp paths, composition version, session /
profile metadata) legitimately differs. This module reduces a committed
``CompositionState`` (the primary surface) to a canonical, comparable structure
and provides :func:`assert_isomorphic`, which reports the first differing
*semantic* attribute.

Preserve / canonicalize split — design §8.1
(``2026-07-13-composer-guided-freeform-capability-parity-design.md:677``):

PRESERVE (must survive canonicalization; a difference here is a real regression):
  node and plugin kinds, normalized options, directed edge roles, route labels,
  gate conditions, coalesce policy + merge mode, aggregation trigger /
  output_mode, queue fan-in, field contracts / output business schemas, and
  every failure policy (``on_error`` / ``on_validation_failure`` /
  ``on_write_failure`` terminal-vs-routed distinction), plus the full wiring
  topology.

CANONICALIZE (normalized away; a difference here is surface noise, not a
regression):
  generated node ids, connection names, source keys, edge ids, map / list
  ordering, output temp paths (reduced to basename), composition version, and
  session / profile / naming metadata.

Connection names and node ids are canonicalized by a *topology-preserving
relabeling* — not erasure. Two graphs that differ only by a consistent renaming
of their wires are isomorphic; two graphs with the same names but different
wiring are not. Relabeling is computed by iterated colour refinement
(1-Weisfeiler-Leman) over the dataflow multigraph, so the canonical token a
wire receives is a function of its structural position, independent of its
name. Every *semantic* attribute is emitted verbatim into the canonical form,
so an attribute change (a swapped gate route, a flipped merge mode, a dropped
failure policy, a changed plugin) always changes the canonical form regardless
of the relabeling.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from elspeth.web.composer.state import CompositionState

__all__ = [
    "CanonicalGraph",
    "IsomorphismError",
    "assert_isomorphic",
    "canonical_graph",
    "public_pipeline_semantics",
]

# Sentinel terminal a routed failure/discard edge points at when the route is a
# literal ``discard`` (or absent) rather than a named connection. Keeping this a
# distinct atom preserves the terminal-vs-routed distinction that the design
# lists under PRESERVE (a failure policy that discards is not the same graph as
# one that routes to a real sink).
_DISCARD = "\x00discard"


class IsomorphismError(AssertionError):
    """Raised when two canonical graphs disagree on a preserved attribute."""


@dataclass(frozen=True)
class CanonicalGraph:
    """A canonical, comparable reduction of a committed ``CompositionState``.

    ``structure`` is a plain JSON-able dict with connection names / node ids
    relabelled to structural tokens and every list canonically sorted; equal
    ``structure`` dicts mean the two source graphs are isomorphic under the
    §8.1 split. ``fingerprint`` is a stable string digest of ``structure`` for
    cheap set/equality use.
    """

    structure: dict[str, Any]
    fingerprint: str


# --------------------------------------------------------------------------- #
# Option / value canonicalization                                             #
# --------------------------------------------------------------------------- #

_PATH_KEYS = frozenset({"path", "file"})


def _basename(value: str) -> str:
    """Reduce a filesystem-ish path to its final component.

    Source paths are rewritten under a per-test ``{data_dir}/blobs/`` prefix and
    output paths carry temp directories; only the leaf name is semantic.
    """
    normalized = value.replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


def _canon_value(value: Any, *, key: str | None = None) -> Any:
    """Recursively canonicalize an option value.

    ``path`` / ``file`` string values collapse to their basename; mappings sort
    their keys; sequences preserve order (option list order can be semantic).
    """
    if isinstance(value, Mapping):
        return {str(k): _canon_value(v, key=str(k)) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canon_value(item) for item in value]
    if isinstance(value, str) and key in _PATH_KEYS:
        return _basename(value)
    return value


def _effective_options(options: Any, plugin_kind: str | None, plugin_name: str | None) -> Any:
    """Drop a source/sink plugin's default-valued option keys so options compare by value.

    Different commit paths persist the *same* plugin config at different levels
    of explicitness: ``set_pipeline`` keeps the authored option keys, while the
    guided stage protocol persists the full pydantic ``model_dump`` (every
    default made explicit — e.g. csv ``delimiter`` / ``encoding`` / ``skip_rows``,
    json ``indent`` / ``headers``). Those are the identical effective
    configuration, so "normalized options" (a §8.1 PRESERVE attribute) must mean
    the *effective* options, not the authored surface form. Dropping every option
    key whose value equals its plugin config-model field default collapses that
    explicit-vs-default noise on both sides while keeping every non-default value
    (a real regression) fully visible. This reads field defaults only — it never
    validates — so structural (``on_validation_failure``) and aliased/nested
    (``schema``) keys, which have no matching plain field name, are left
    untouched, and options that carry a genuine non-default stay in the compared
    form. Falls back to the raw options when no model resolves.
    """
    from elspeth.contracts.freeze import deep_thaw

    raw = dict(deep_thaw(options)) if options is not None else {}
    if plugin_kind is None or plugin_name is None:
        return raw
    try:
        from pydantic_core import PydanticUndefined

        from elspeth.plugins.infrastructure.validation import get_sink_config_model, get_source_config_model

        if plugin_kind == "source":
            model = get_source_config_model(plugin_name)
        elif plugin_kind == "sink":
            model = get_sink_config_model(plugin_name)
        else:
            return raw
        if model is None:
            return raw
        fields = model.model_fields
        reduced: dict[str, Any] = {}
        for key, value in raw.items():
            field = fields.get(key)
            if field is not None:
                default = field.get_default(call_default_factory=True)
                if default is not PydanticUndefined and value == default:
                    continue  # explicit default == absent default: pure surface noise
            reduced[key] = value
        return reduced
    except Exception:
        return raw


def _canon_options(options: Any, *, plugin_kind: str | None = None, plugin_name: str | None = None) -> dict[str, Any]:
    canon = _canon_value(_effective_options(options, plugin_kind, plugin_name))
    if not isinstance(canon, dict):  # pragma: no cover - options are always mappings
        raise TypeError(f"options must canonicalize to a mapping, got {type(options)!r}")
    return canon


def _freeze(obj: Any) -> str:
    """Deterministic, hash-salt-independent string key for any JSON-able value."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


# --------------------------------------------------------------------------- #
# Atom / link model + colour refinement                                       #
# --------------------------------------------------------------------------- #


@dataclass
class _Model:
    base: dict[str, tuple[Any, ...]]  # atom key -> name-free semantic signature
    links: list[tuple[str, str, str]]  # (from_atom, role, to_atom)


def _conn(name: str) -> str:
    return f"C:{name}"


def _build_model(state: Mapping[str, Any]) -> _Model:
    """Build the dataflow multigraph atoms + role-typed links from a state dict."""
    base: dict[str, tuple[Any, ...]] = {_DISCARD: ("discard",)}
    links: list[tuple[str, str, str]] = []
    conns: set[str] = set()

    def conn_atom(name: str) -> str:
        conns.add(name)
        return _conn(name)

    def failure_target(value: Any) -> str:
        # A failure/validation/write policy that is ``discard`` or absent is a
        # terminal; anything else names a connection that must reach a sink.
        if value is None or value == "discard":
            return _DISCARD
        return conn_atom(str(value))

    # Sources -------------------------------------------------------------- #
    for key, source in state["sources"].items():
        atom = f"S:{key}"
        base[atom] = (
            "source",
            source["plugin"],
            _freeze(_canon_options(source.get("options"), plugin_kind="source", plugin_name=source["plugin"])),
        )
        links.append((atom, "source.on_success", conn_atom(str(source["on_success"]))))
        links.append((atom, "source.on_validation_failure", failure_target(source.get("on_validation_failure"))))

    # Nodes ---------------------------------------------------------------- #
    for node in state["nodes"]:
        atom = f"N:{node['id']}"
        node_type = node["node_type"]
        base[atom] = (
            "node",
            node_type,
            node.get("plugin"),
            _freeze(_canon_options(node.get("options"))),
            _freeze(node.get("condition")),
            _freeze(node.get("policy")),
            _freeze(node.get("merge")),
            _freeze(node.get("output_mode")),
            _freeze(node.get("trigger")),
            _freeze(node.get("expected_output_count")),
        )
        links.append((atom, "node.input", conn_atom(str(node["input"]))))
        if node.get("on_success") is not None:
            links.append((atom, "node.on_success", conn_atom(str(node["on_success"]))))
        if "on_error" in node and node_type in {"transform", "aggregation"}:
            links.append((atom, "node.on_error", failure_target(node.get("on_error"))))
        routes = node.get("routes")
        if isinstance(routes, Mapping):
            for route_key, target in routes.items():
                links.append((atom, f"node.route.{route_key}", conn_atom(str(target))))
        for forked in node.get("fork_to") or ():
            links.append((atom, "node.fork", conn_atom(str(forked))))
        branches = node.get("branches")
        if isinstance(branches, Mapping):
            for target in branches.values():
                links.append((atom, "node.branch", conn_atom(str(target))))

    # Outputs -------------------------------------------------------------- #
    for output in state["outputs"]:
        name = output["name"]
        atom = f"O:{name}"
        base[atom] = (
            "output",
            output["plugin"],
            _freeze(_canon_options(output.get("options"), plugin_kind="sink", plugin_name=output["plugin"])),
        )
        links.append((atom, "output.sink", conn_atom(str(name))))
        links.append((atom, "output.on_write_failure", failure_target(output.get("on_write_failure"))))

    # Explicit edges ------------------------------------------------------- #
    # ``from_node`` / ``to_node`` name a node id or the literal ``source`` (the
    # sole source). Directed edge roles are preserved; the label (a connection)
    # is linked structurally so a relabelled-but-equivalent edge stays equal.
    source_keys = list(state["sources"].keys())
    for edge in state["edges"]:
        from_ref = str(edge["from_node"])
        to_ref = str(edge["to_node"])
        from_atom = _edge_endpoint(from_ref, source_keys)
        to_atom = _edge_endpoint(to_ref, source_keys)
        links.append((from_atom, f"edge.{edge['edge_type']}", to_atom))
        if edge.get("label") is not None:
            links.append((from_atom, "edge.label", conn_atom(str(edge["label"]))))

    for name in conns:
        base.setdefault(_conn(name), ("conn",))
    return _Model(base=base, links=links)


def _edge_endpoint(ref: str, source_keys: Sequence[str]) -> str:
    if ref == "source" and "source" not in source_keys:
        # Legacy singular-source edge reference; bind to the one source.
        if len(source_keys) == 1:
            return f"S:{source_keys[0]}"
        return "S:source"
    if ref in source_keys:
        return f"S:{ref}"
    return f"N:{ref}"


def _rank(signatures: Mapping[str, Any]) -> dict[str, int]:
    """Assign each atom a small int colour by sorted distinct signature.

    Ranks are hash-salt independent (derived from a sorted order over
    deterministic string keys), so isomorphic graphs receive matching colours.
    """
    order = sorted({_freeze(sig) for sig in signatures.values()})
    rank_of = {frozen: index for index, frozen in enumerate(order)}
    return {atom: rank_of[_freeze(sig)] for atom, sig in signatures.items()}


def _refine(model: _Model) -> dict[str, int]:
    """Iterated 1-WL colour refinement to a stable partition."""
    out_links: dict[str, list[tuple[str, str]]] = {atom: [] for atom in model.base}
    in_links: dict[str, list[tuple[str, str]]] = {atom: [] for atom in model.base}
    for from_atom, role, to_atom in model.links:
        out_links[from_atom].append((role, to_atom))
        in_links[to_atom].append((role, from_atom))

    color = _rank(dict(model.base))
    for _ in range(len(model.base) + 2):
        signatures = {
            atom: (
                color[atom],
                tuple(sorted((role, color[to]) for role, to in out_links[atom])),
                tuple(sorted((role, color[frm]) for role, frm in in_links[atom])),
            )
            for atom in model.base
        }
        new_color = _rank(signatures)
        if _partition(new_color) == _partition(color):
            return new_color
        color = new_color
    return color


def _partition(color: Mapping[str, int]) -> frozenset[frozenset[str]]:
    groups: dict[int, set[str]] = {}
    for atom, value in color.items():
        groups.setdefault(value, set()).add(atom)
    return frozenset(frozenset(members) for members in groups.values())


# --------------------------------------------------------------------------- #
# Canonical emission                                                          #
# --------------------------------------------------------------------------- #


def _state_dict(state: CompositionState | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(state, CompositionState):
        return state.to_dict()
    if isinstance(state, Mapping):
        return dict(state)
    raise TypeError(f"expected CompositionState or its dict form, got {type(state)!r}")


def canonical_graph(state: CompositionState | Mapping[str, Any]) -> CanonicalGraph:
    """Reduce a committed ``CompositionState`` to its canonical semantic graph.

    Accepts a ``CompositionState`` or its ``to_dict()`` form so the helper is
    surface-agnostic. Composition version and pipeline metadata (name /
    description) are dropped as session/profile noise; every wire is relabelled
    to a structural token; every preserved attribute is emitted verbatim.
    """
    data = _state_dict(state)
    model = _build_model(data)
    color = _refine(model)

    def tok(name: str) -> str:
        return f"w{color[_conn(name)]}"

    def failure_tok(value: Any) -> str:
        if value is None or value == "discard":
            return "discard"
        return tok(str(value))

    def node_tok(ref: str) -> str:
        atom = _edge_endpoint(ref, list(data["sources"].keys()))
        return f"v{color[atom]}"

    sources = []
    for source in data["sources"].values():
        sources.append(
            {
                "plugin": source["plugin"],
                "options": _canon_options(source.get("options"), plugin_kind="source", plugin_name=source["plugin"]),
                "on_success": tok(str(source["on_success"])),
                "on_validation_failure": failure_tok(source.get("on_validation_failure")),
            }
        )

    nodes = []
    for node in data["nodes"]:
        entry: dict[str, Any] = {
            "node_type": node["node_type"],
            "plugin": node.get("plugin"),
            "options": _canon_options(node.get("options")),
            "input": tok(str(node["input"])),
        }
        if node.get("on_success") is not None:
            entry["on_success"] = tok(str(node["on_success"]))
        if "on_error" in node and node["node_type"] in {"transform", "aggregation"}:
            entry["on_error"] = failure_tok(node.get("on_error"))
        if node.get("condition") is not None:
            entry["condition"] = node["condition"]
        routes = node.get("routes")
        if isinstance(routes, Mapping):
            entry["routes"] = {str(k): tok(str(v)) for k, v in routes.items()}
        if node.get("fork_to"):
            entry["fork_to"] = sorted(tok(str(item)) for item in node["fork_to"])
        branches = node.get("branches")
        if isinstance(branches, Mapping):
            entry["branches"] = sorted(tok(str(v)) for v in branches.values())
        for optional in ("policy", "merge", "output_mode", "expected_output_count"):
            if node.get(optional) is not None:
                entry[optional] = node[optional]
        if node.get("trigger") is not None:
            entry["trigger"] = _canon_value(node["trigger"])
        nodes.append(entry)

    outputs = []
    for output in data["outputs"]:
        outputs.append(
            {
                "plugin": output["plugin"],
                "options": _canon_options(output.get("options"), plugin_kind="sink", plugin_name=output["plugin"]),
                "sink": tok(str(output["name"])),
                "on_write_failure": failure_tok(output.get("on_write_failure")),
            }
        )

    edges = []
    for edge in data["edges"]:
        edges.append(
            {
                "from": node_tok(str(edge["from_node"])),
                "to": node_tok(str(edge["to_node"])),
                "edge_type": edge["edge_type"],
                "label": tok(str(edge["label"])) if edge.get("label") is not None else None,
            }
        )

    structure = {
        "sources": sorted(sources, key=_freeze),
        "nodes": sorted(nodes, key=_freeze),
        "outputs": sorted(outputs, key=_freeze),
        "edges": sorted(edges, key=_freeze),
    }
    return CanonicalGraph(structure=structure, fingerprint=_freeze(structure))


# --------------------------------------------------------------------------- #
# Comparison + diff reporting                                                 #
# --------------------------------------------------------------------------- #


def _first_diff(a: Any, b: Any, path: str) -> str | None:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                return f"{path}.{key}: missing on left (right={b[key]!r})"
            if key not in b:
                return f"{path}.{key}: missing on right (left={a[key]!r})"
            diff = _first_diff(a[key], b[key], f"{path}.{key}")
            if diff is not None:
                return diff
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            diff = _first_diff(left, right, f"{path}[{index}]")
            if diff is not None:
                return diff
        return None
    if a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def assert_isomorphic(
    a: CompositionState | CanonicalGraph | Mapping[str, Any],
    b: CompositionState | CanonicalGraph | Mapping[str, Any],
    *,
    left: str = "left",
    right: str = "right",
) -> None:
    """Assert two committed graphs are semantically isomorphic under §8.1.

    Accepts raw ``CompositionState`` objects (canonicalized here) or already
    computed :class:`CanonicalGraph` values. On mismatch, raises
    :class:`IsomorphismError` naming the first differing semantic attribute.
    """
    graph_a = a if isinstance(a, CanonicalGraph) else canonical_graph(a)
    graph_b = b if isinstance(b, CanonicalGraph) else canonical_graph(b)
    if graph_a.fingerprint == graph_b.fingerprint:
        return
    diff = _first_diff(graph_a.structure, graph_b.structure, "graph")
    detail = diff if diff is not None else "canonical structures differ but no attribute path was isolated"
    raise IsomorphismError(f"composition graphs are not isomorphic ({left} vs {right}): {detail}")


# --------------------------------------------------------------------------- #
# Secondary check: public-YAML business semantics                             #
# --------------------------------------------------------------------------- #


def public_pipeline_semantics(state: CompositionState) -> dict[str, Any]:
    """Return the public compiled-pipeline dict with temp paths normalized.

    The public YAML compiler is the operator-facing export; comparing it across
    surfaces is a secondary confirmation that the two paths agree on the runtime
    pipeline shape (the primary check is :func:`assert_isomorphic` on the
    committed ``CompositionState``). Only ``path`` / ``file`` leaves are
    normalized to basenames — the compiler already strips web metadata.
    """
    from elspeth.web.composer.yaml_generator import generate_public_pipeline_dict

    return _canon_value(generate_public_pipeline_dict(state))

# Re-deriving the L3 Import Oracle

The L3 import-graph oracle in this pack
([`l3-import-graph.json`](l3-import-graph.json)) is a snapshot. To
verify it against the current tree, or to re-derive after codebase
changes, run:

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py dump-edges \
  --root src/elspeth \
  --format json \
  --output /tmp/l3-import-graph-fresh.json \
  --no-timestamp
```

`--no-timestamp` produces **byte-identical output across runs** given
the same source tree. So a `diff` against the snapshot reveals only
true import-graph drift, not noise from generation metadata.

---

## Verifying byte-equality against the snapshot

```bash
diff <(jq 'del(.metadata.generated_at, .metadata.tool_version)' \
        /tmp/l3-import-graph-fresh.json) \
     <(jq 'del(.metadata.generated_at, .metadata.tool_version)' \
        docs/arch-pack-2026-04-29-1500/reference/l3-import-graph.json)
```

A clean diff (exit code 0) means the import topology is unchanged from
the snapshot.

A non-empty diff means real drift. To characterise the drift, run:

```bash
# Stats overview
jq '.stats' /tmp/l3-import-graph-fresh.json

# Heaviest edges
jq -r '.edges[] | select(.weight >= 15) | "\(.from) → \(.to) (w=\(.weight))"' \
   /tmp/l3-import-graph-fresh.json | sort

# SCC topology
jq '.strongly_connected_components | map(length)' /tmp/l3-import-graph-fresh.json
jq '.strongly_connected_components[] | select(length >= 3)' /tmp/l3-import-graph-fresh.json
```

---

## Other output formats

The `dump-edges` subcommand supports three output formats:

| `--format` | Use case |
|------------|----------|
| `json` | Machine-readable graph + stats. The canonical artefact. |
| `mermaid` | Inline diagrams for documentation. See [`l3-import-graph.mmd`](l3-import-graph.mmd). |
| `dot` | Graphviz layout (`dot`, `circo`, `neato`, etc.). See [`l3-import-graph.dot`](l3-import-graph.dot). |

---

## Tier-model conformance check

Separate from the import oracle, the layer-conformance check verifies
that no module imports upward across the 4-layer boundary:

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check \
  --root src/elspeth \
  --allowlist config/cicd/enforce_tier_model
```

A clean run prints `No bug-hiding patterns detected. Check passed.`
This is the canonical evidence behind every "the layer model is
mechanically clean" claim in this pack.

---

## Drift status at this pack's HEAD

A re-derivation against this pack's HEAD (`5a5e05d7`) versus the
snapshot (taken at `47d3dd82`) shows minor numerical drift. Structural
claims hold; details in [`../08-known-gaps.md#6`](../08-known-gaps.md#6-the-head-drift-caveat).

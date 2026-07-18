### Step 4 — Wiring constraints

Review the proposed pipeline wiring before completion. Use `confirm_wiring` only when the runtime-reconstructable DAG is coherent. Proposed components MUST satisfy:

- **Exact reviewed graph.** Preserve the proposal's named sources, nodes, outputs, and edges; do not collapse branching or multiple outputs into a fixed topology.
- **No orphan labels.** Every label needs one upstream producer and at least one downstream consumer.
- **Field contract.** A downstream node may require only fields guaranteed by an upstream producer before that label is consumed.
- **Drop raw intermediates.** Before a sink, use `field_mapper` with `select_only: true` when the sink does not need large/raw intermediate fields.
- **Intentional branching.** Keep `routes` and `fork_to` only when the reviewed proposal requires genuine branching.

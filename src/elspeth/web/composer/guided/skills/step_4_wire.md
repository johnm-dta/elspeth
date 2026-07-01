### Step 4 — Wiring constraints

Review the proposed pipeline wiring before completion. Use `confirm_wiring` only when the runtime-reconstructable DAG is coherent. The runtime reconstructs the DAG from labels; proposed transforms MUST satisfy:

- **Single linear spine.** Source emits `on_success: "chain_in"`; the first transform reads `"chain_in"`; intermediate transforms use `chain_{k}`; the last transform emits `"main"`; sinks consume `"main"`.
- **No orphan labels.** Every label needs one upstream producer and at least one downstream consumer.
- **Field contract.** A downstream node may require only fields guaranteed by an upstream producer before that label is consumed.
- **Drop raw intermediates.** Before a sink, use `field_mapper` with `select_only: true` when the sink does not need large/raw intermediate fields.
- **No branching by default.** Avoid `routes` and `fork_to` unless the user asked for genuine branching.

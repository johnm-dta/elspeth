"""Narrative builder for the audit-readiness Explain view.

Generates deterministic prose describing what ELSPETH will record when
the composition runs. No LLM call; same composition + retention → same text.

Layer: L3 (application).
"""

from __future__ import annotations

from elspeth.web.composer.state import CompositionState, NodeSpec, OutputSpec


def build_narrative(state: CompositionState, *, retention_days: int) -> str:
    lines: list[str] = [
        "When you run this pipeline, ELSPETH will record:",
        "",
    ]

    if not state.sources:
        lines.append(
            "- No source configured yet — the composition is incomplete. Once you add a source, this view will describe what it records."
        )
    else:
        for source_name, source in state.sources.items():
            lines.append(_describe_source(source.plugin, source_name=source_name))

    for node in state.nodes:
        if node.node_type == "transform":
            lines.append(_describe_transform(node))

    for output in state.outputs:
        lines.append(_describe_output(output))

    lines.extend(
        [
            "",
            f"Retention: {retention_days} days by default. This applies to stored payloads; row-level hashes are retained indefinitely.",
            "",
            "Run metadata: when, who (you), and which plugin versions were in use at run time.",
            "",
            "This evidence is sufficient to answer questions about any output "
            "row of this pipeline, including which plugin produced it and "
            "from what input.",
        ]
    )
    return "\n".join(lines)


def _describe_source(plugin: str, *, source_name: str) -> str:
    prefix = "- Source data" if source_name == "source" else f"- Source data ({source_name})"
    if plugin == "csv":
        return f"{prefix} — each row from the CSV input. SHA-256 hash recorded for the source file and for each row."
    if plugin == "json":
        return f"{prefix} — each record from the JSON input. SHA-256 hash recorded for the source file and for each record."
    if plugin == "dataverse":
        return f"{prefix} — each record returned by the Dataverse query, with query parameters and result hashes recorded."
    return f"{prefix} — each row from the {plugin} source. Row-level hash recorded for every record."


def _describe_transform(node: NodeSpec) -> str:
    name = node.id
    # Callers filter to node_type == "transform"; per NodeSpec contract
    # (state.py docstring, line 113), `plugin` is None only for gates and
    # coalesces. If None ever appears here, that's a Tier-1 invariant breach.
    plugin = node.plugin
    if plugin is None:
        raise RuntimeError(f"transform node {name!r} has plugin=None — Tier-1 invariant breach")
    if plugin == "llm":
        return (
            f"- {name} (LLM transform) — for each row: the full prompt "
            f"(with your accepted definitions), the full response, the "
            f"model and version, and the timestamp. Recorded in the "
            f"audit database."
        )
    if plugin == "passthrough":
        return f"- {name} (passthrough) — copies the row unchanged. The audit trail records the hop; no new fields are written."
    if plugin == "web_scrape":
        return (
            f"- {name} (web scrape) — for each URL: HTTP status, "
            f"response time, and response body hash. Bytes are stored "
            f"under the payload retention policy."
        )
    if plugin == "rag_retrieval":
        return (
            f"- {name} (RAG retrieval) — for each query: the retrieval "
            f"request and top-k result hashes recorded. External call to "
            f"the configured vector store."
        )
    if plugin in ("azure_content_safety", "azure_prompt_shield"):
        return (
            f"- {name} ({plugin} — Azure safety) — for each row: the "
            f"safety analysis request and verdict recorded. External call "
            f"to Azure AI Services."
        )
    return f"- {name} ({plugin} transform) — input row hash, output row hash, and per-row outcome recorded."


def _describe_output(output: OutputSpec) -> str:
    # OutputSpec.plugin is a non-nullable str per state.py line 256.
    plugin = output.plugin
    if plugin in ("csv", "json"):
        return (
            f"- {output.name} ({plugin} file) — written to your session "
            f"storage. SHA-256-hashed; chain-of-custody recorded with "
            f"the run id and timestamp."
        )
    if plugin == "azure_blob":
        return (
            f"- {output.name} (Azure Blob) — uploaded to the configured "
            f"container. Content hash and remote path recorded; the "
            f"local emit is hashed before transit."
        )
    if plugin == "dataverse":
        return (
            f"- {output.name} (Dataverse sink — external boundary) — for "
            f"each record: the insert/upsert outcome and the remote write "
            f"request are recorded with a content hash. The local emit is "
            f"hashed before transit to the configured Dataverse instance, "
            f"so the call is recorded as an external operation."
        )
    return f"- {output.name} ({plugin} sink) — write outcome and output hash recorded for each row."

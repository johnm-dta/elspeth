# Semi-Autonomous Platform — Independent Architectural Critique

**Status:** Supplemental Analysis
**Date:** 2026-03-03
**Author:** Gemini CLI (Independent Analysis)
**Relates to:** `docs/architecture/semi-autonomous/design.md`, `design-review-synthesis.md`

---

## Executive Summary

While the 3-round design review synthesis (March 2, 2026) significantly strengthened the governance and auditability of the Semi-Autonomous Platform, several critical architectural flaws remain. These issues primarily concern **data lineage integrity**, **orchestration redundancy**, **user experience latency**, and **LLM generation stability**.

This document identifies six high-impact risks and proposes specific remediations to ensure the platform meets ELSPETH's core mandates of reliability and traceability.

---

## 1. Lineage Vulnerability: The "Refinement Diff" Approval Gap

### The Flaw
Feature F3 (Refinement Diff and Selective Re-Approval) allows unchanged nodes to carry their prior approval forward. This logic treats node approvals as isolated endorsements of a **configuration block** rather than an endorsement of a **data flow**.

In an SDA pipeline, a sink (Act) is approved to receive specific data produced by the upstream (Sense/Decide). If a user refines a middle-tier `llm_transform` to extract sensitive PII instead of sentiment, the downstream `database_sink` configuration remains identical. Under the current design, the system would carry forward the sink's approval, allowing PII to flow into a destination previously approved only for sentiment data.

### Remediation
- **Topological Hashing:** Approvals must bind to a **Topological Hash** of the upstream path.
- **Invalidation Rule:** Any mutation (parameter change, prompt edit, or topology shift) upstream of a gated node must invalidate all downstream approvals. An approval is a signature on a *contract of results*, not a static UI component.

---

## 2. Orchestration Redundancy: Temporal vs. ELSPETH Engine

### The Flaw
The design introduces Temporal for durable execution but treats the ELSPETH Engine as a black-box activity. This creates competing state machines:
1. **Competing Retries:** ELSPETH’s native `RetryManager` (tenacity-based) handles transient failures at the row level. Temporal’s activity retries operate at the pod/process level. If a pod is killed during a 5,000-row run, Temporal may blindly restart the entire pipeline.
2. **Idempotency Risk:** Unless ELSPETH's checkpointing is explicitly synchronized with Temporal's heartbeats, a restart could result in duplicate side-effects (double-writes) at the sink level.
3. **Competing Timeouts:** Temporal’s `start_to_close_timeout` could terminate an ELSPETH run that is intentionally waiting on a legitimate rate-limit backoff.

### Remediation
- **Checkpoint Handover:** Temporal must pass a `checkpoint_id` to the Worker Pod.
- **Activity Granularity:** Either ELSPETH must report its internal retry state to Temporal, or Temporal must be configured with zero retries for the `execute_pipeline` activity, deferring all recovery to ELSPETH’s native checkpointing system.

---

## 3. Performance Friction: Kubernetes Cold Starts vs. Interactive UI

### The Flaw
The architecture relies on Kubernetes Jobs/Deployments to spin up worker pods for every task, including 5-row "Previews."
- **Latency:** K8s pod scheduling and Python container startup (loading ELSPETH, LiteLLM, and DB drivers) typically take 10–30 seconds.
- **UX Impact:** In a "Refine" loop, a user asking to "add a date field" expects an immediate visual update. Waiting 30 seconds for a 5-row preview to render on the canvas will kill the "fluid reasoning" experience intended for the platform.

### Remediation
- **Warm Worker Pool:** Implement a pool of standby workers that stay initialized.
- **Socket-Based Tasking:** Standby workers should accept `PipelineArtifact` payloads over a low-latency socket (gRPC/Unix) to execute previews in <1 second.

---

## 4. Generation Fragility: Imperative Knobs vs. Declarative State

### The Intent
The LLM does not generate YAML; it uses **MCP knobs** (structured tools) to tweak a high-level configuration. Once the user signs off, a programmatic process transforms this config into executable YAML.

### The Flaw
Using "knobs" (e.g., `set_node_parameter(node_id, key, value)`) is an **imperative** pattern.
- **The State Synchronization Risk:** If the LLM calls 5 knobs in a row, it must maintain a perfect mental model of the resulting configuration. If one "tweak" fails validation, the LLM often struggles to reconcile the partial state. This leads to "hallucinated state" where the LLM believes a parameter is set because it called the tool, even if the tool returned an error or the state was overwritten.

### Remediation
- **Declarative Patching:** Even when using MCP tools, the primary "knob" should be `update_node_spec(node_id, partial_json)`. This allows the LLM to propose a **desired state** for a component rather than a sequence of imperative commands.
- **State-Reflecting Feedback:** Every knob call must return the *entire* updated state of the affected node, not just a success message. This forces the LLM's context to stay synchronized with the actual configuration.

---

## 5. Preview Safety: Unbounded and Destructive Sources

### The Flaw
The `HeadSourceWrapper` strategy assumes all sources can be safely and cheaply sampled by yielding the first N rows.
- **Destructive Sources:** Reading 5 rows from a Message Queue (SQS/RabbitMQ) or a Kafka stream is a destructive operation. A "Preview" would consume real production data before the pipeline is even approved.
- **Unbounded Costs:** Sampling 5 rows from a 10GB JSON file or an API without native pagination may still require the plugin to download/buffer the entire dataset, leading to "Preview" runs that take minutes or OOM the worker.

### Remediation
- **Source Capability Check:** The `SourceProtocol` must expose `supports_preview()`.
- **Static Sampling:** For destructive or expensive sources, the UI must require the user to upload a "Design-Time Sample" file rather than attempting an automatic live preview.

---

## 6. The Visibility Gap: Knob Mapping vs. Executable Contract

### The Intent
The user signs off on the "knob-based" configuration, which is then programmatically transformed into ELSPETH YAML.

### The Flaw
The programmatic transformation introduces an **Audit Translation Layer**.
- **Mapping Ambiguity:** If the logic that transforms "Knob A = Value X" into "YAML Path Y = Value Z" is complex or contains defaults/magic values, the user is signing a contract they don't fully see. ELSPETH's core mandate is that every decision is traceable. If the engine behaves a certain way because of a "magic default" added during the programmatic transformation, the audit trail is compromised.

### Remediation
- **Deterministic, Reversible Mapping:** The transformation from "Knobs" to "YAML" must be a **Tier 1 validated process**. It should be a simple, deterministic mapping with no "hidden" logic.
- **Spec Transparency:** The "User-Intent Config" (the state of the knobs) must be stored in the `PipelineArtifact` alongside the generated YAML.
- **Audit Verification:** Auditors must be able to view the specific "knob settings" the user approved. The approval signature must cryptographically cover the **entire configuration state**, which serves as the "source of truth" for the subsequent YAML generation.

---

## Summary of Remediation Priorities

| Priority | Risk | Mitigation |
|---|---|---|
| **Critical** | Data Lineage Bypass | Approvals bind to Upstream Topological Hash |
| **High** | Orchestration Conflict | Sync Temporal timeouts with ELSPETH RetryManager |
| **High** | Generation Instability | Shift from Imperative Tools to Declarative DAG Patching |
| **Medium** | Preview Latency | Warm Worker Pool for interactive previews |
| **Medium** | Destructive Previews | Augment SourceProtocol with `supports_preview()` |
| **Low** | Audit Visibility | Bind signatures to Raw YAML, not just UI widgets |

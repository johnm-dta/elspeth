# What ELSPETH Guarantees

**Version:** RC-3
**Date:** February 2026

This document defines the promises ELSPETH makes to its users. These are not aspirational features - they are contractual guarantees that the system must uphold.

---

## The Core Promise

**Every output can be traced to its source with complete audit trail.**

If ELSPETH produced an output, you can ask "why?" and get a complete answer:
- What source row it came from
- What transforms were applied
- What external calls were made
- What routing decisions occurred
- Why it ended up where it did

This is not optional. This is not best-effort. This is the reason ELSPETH exists.

---

## 1. AUDIT GUARANTEES

### 1.1 Complete Lineage

**Promise:** For any output token, `explain(run_id, token_id)` returns:

| Component | Guarantee |
|-----------|-----------|
| Source row | Original data as received, with hash |
| Transform chain | Every transform that touched this data |
| Input/output hashes | Cryptographic proof of data at each boundary |
| External calls | Full request/response for LLM/HTTP calls |
| Routing decisions | Why data went where it went |
| Terminal state | Final disposition (completed, routed, failed, etc.) |

### 1.2 No Silent Drops

**Promise:** Every row that enters the system has a recorded outcome.

| Possible Outcomes | Meaning |
|-------------------|---------|
| COMPLETED | Reached output sink successfully |
| ROUTED | Sent to named sink by gate |
| FORKED | Split into child tokens (parent token) |
| CONSUMED_IN_BATCH | Aggregated into batch |
| COALESCED | Merged in join operation |
| QUARANTINED | Failed validation, stored for investigation |
| FAILED | Processing failed, not recoverable |
| EXPANDED | Parent token for deaggregation (1→N expansion) |

**What this means:** You will never ask "what happened to row 42?" and get silence. The system recorded what happened.

### 1.3 Hash Integrity

**Promise:** Hashes are stable and verifiable.

- Same data → same hash (deterministic)
- Hash algorithm versioned (`sha256-rfc8785-v1`)
- NaN/Infinity strictly rejected (not silently converted)
- Payload store verifies hash on read

### 1.4 Payload Retention

**Promise:** Hashes survive payload deletion.

When retention policies purge old payloads:
- Metadata remains (who, what, when)
- Hashes remain (integrity verification)
- `explain()` reports "payload no longer available"
- Audit trail integrity preserved

---

## 2. EXECUTION GUARANTEES

### 2.1 DAG Execution Order

**Promise:** Transforms execute in topological order.

- Dependencies respected
- No transform sees data before its predecessors complete
- Parallel paths are independent (no cross-contamination)

### 2.2 Token Isolation

**Promise:** Forked tokens are independent.

When a row forks to parallel paths:
- Each branch gets its own copy of the data
- Mutations in one branch don't affect siblings
- Audit trail records each branch separately

### 2.3 Gate Routing

**Promise:** Gates route deterministically.

- Same row + same condition = same destination
- Routing reason recorded in audit trail
- Invalid destinations rejected at configuration time

### 2.4 Aggregation Triggers

**Promise:** Triggers fire as configured.

| Trigger | Behavior |
|---------|----------|
| Count | Fires when count threshold reached |
| Timeout | Fires when next row arrives after timeout period* |
| End-of-source | Always fires when source exhausted |

*Known limitation: Timeout requires next row arrival. True idle timeout not supported without heartbeat.

### 2.5 Retry Semantics

**Promise:** Retries are explicit and recorded.

- Each attempt is a separate audit record
- Transient errors retry with backoff
- Permanent errors fail immediately
- Max retries respected
- Final outcome clear

---

## 3. DATA GUARANTEES

### 3.1 Source Validation

**Promise:** Invalid source data doesn't crash the pipeline.

- Malformed rows quarantined
- Original data preserved in quarantine
- Valid rows continue processing
- Quarantine reason recorded

### 3.2 Schema Contracts

**Promise:** Plugins declare their data requirements.

- Sources declare guaranteed output fields
- Transforms declare required input fields
- DAG construction validates compatibility
- Mismatches rejected before execution

### 3.3 Field Normalization

**Promise:** Messy headers become valid identifiers.

- Unicode normalized (NFC)
- Non-identifier characters replaced
- Collisions detected and reported
- Original→normalized mapping recorded

---

## 4. EXTERNAL CALL GUARANTEES

### 4.1 Call Recording

**Promise:** External calls are fully recorded.

| Recorded | Details |
|----------|---------|
| Request | Full payload, hash, timestamp |
| Response | Full payload, hash, timestamp |
| Latency | Milliseconds |
| Status | HTTP status code or error type |
| Provider | Service identifier |

### 4.2 Rate Limiting

**Promise:** Rate limits are respected (when configured).

LLM plugins include built-in rate limiting:
- Configurable requests per second
- Automatic backoff on 429 responses
- No silent failures from rate exhaustion

---

## 5. RECOVERY GUARANTEES

### 5.1 Checkpoint Recovery

**Promise:** Interrupted runs can resume.

- Checkpoints created at processing boundaries
- `elspeth resume` continues from last checkpoint
- Already-processed rows not reprocessed
- Aggregation state restored

### 5.2 Idempotent Sinks

**Promise:** Sinks can be safely re-run.

- Idempotency keys provided to sinks
- Same key = same operation (for idempotent sinks)
- Non-idempotent sinks explicitly flagged

---

## 6. CONFIGURATION GUARANTEES

### 6.1 Validation Before Execution

**Promise:** Invalid configurations fail fast.

`elspeth validate` catches:
- Invalid plugin references
- Invalid sink references in routes
- Schema incompatibilities
- Missing required fields

### 6.2 Environment Variables

**Promise:** `${VAR}` syntax works.

- Variables expanded at load time
- Missing required variables fail with clear message
- Default values supported: `${VAR:-default}`

### 6.3 No Implicit Behavior

**Promise:** Explicit over implicit.

- `--execute` required to actually run
- Dry-run is the safe default
- No silent data modification

---

## 7. WHAT ELSPETH DOES NOT GUARANTEE

### 7.1 Performance

ELSPETH prioritizes correctness and auditability over throughput. It is not designed for:
- High-throughput streaming (use Kafka/Flink)
- Sub-millisecond latency (audit recording has overhead)
- Concurrent processing (single-threaded in RC-3)

### 7.2 Access Control

**ELSPETH is not multi-user.** It assumes single-user execution or a fully trusted network environment. There is no built-in authentication, authorization, or access control.

Specifically, RC-3 does not include:
- User authentication
- Role-based access control
- Data redaction profiles
- Network-level access restrictions

If ELSPETH is exposed on a network, the deployer is responsible for providing access control at the infrastructure level (e.g., VPN, firewall rules, reverse proxy authentication).

### 7.3 External System Behavior

ELSPETH records what external systems return, but cannot guarantee:
- LLM response quality or consistency
- External API availability
- Third-party rate limit behavior

### 7.4 True Idle Timeouts

Timeout triggers fire when the next row arrives, not during complete idle periods. If no rows arrive, buffered data waits for:
- A new row (triggering timeout check)
- Source completion (triggering end-of-source flush)

---

## 8. BREAKING THE CONTRACT

If ELSPETH fails to uphold these guarantees, that's a bug. Report it.

**Contract violations are P0 bugs.** They block release.

Examples of contract violations:
- Row entered system but no outcome recorded
- Hash changed for same input data
- Explain returned incomplete lineage
- Fork tokens shared mutable state
- Checkpoint resume reprocessed completed rows

---

## 9. VERSIONING

This contract is versioned with the software.

| Version | Date | Changes |
|---------|------|---------|
| RC-3 | Feb 2026 | Declarative DAG wiring, graceful shutdown, DROP-mode handling, gate plugin removal, telemetry hardening, test suite v2 migration |
| RC-2 | Feb 2026 | Initial contract, bug fixes, checkpoint compatibility |

Future versions may:
- Add new guarantees (backward compatible)
- Deprecate guarantees with migration path
- Never silently remove guarantees

---

## 10. THE ATTRIBUTABILITY TEST

The ultimate test of ELSPETH's contract:

```python
def test_attributability(run_id: str, token_id: str):
    """Given any output, prove complete lineage to source."""
    lineage = landscape.explain(run_id, token_id=token_id)

    # Source exists
    assert lineage.source_row is not None
    assert lineage.source_row.data_hash is not None

    # Processing recorded
    assert len(lineage.node_states) > 0
    for state in lineage.node_states:
        assert state.input_hash is not None
        if state.status == "completed":
            assert state.output_hash is not None

    # Terminal state recorded
    assert lineage.outcome is not None
    assert lineage.outcome in [
        "COMPLETED", "ROUTED", "FORKED",
        "CONSUMED_IN_BATCH", "COALESCED",
        "QUARANTINED", "FAILED", "EXPANDED"
    ]

    # Call linkage valid
    for call in lineage.calls:
        assert any(s.state_id == call.state_id for s in lineage.node_states)
```

If this test fails for any output that ELSPETH produced, the contract is broken.

---

*This is what ELSPETH promises. Nothing more, nothing less.*

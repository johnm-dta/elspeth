# What ELSPETH Guarantees

> **LAYERED ASSURANCE APPENDIX — RC-3 contract language with RC-5.2 additions.**
> §1 through §6 preserve the RC-3 guarantees as originally drafted (3 March 2026) because downstream engineering and audit references cite that wording. §7 has been amended to reflect that the **no-multi-user / no-access-control** disclaimer is no longer true as of RC-4 and RC-5. §11 through §14 document RC-5.2 guarantees for authentication, secret references, multi-user sessions, and composer authoring. Read this as a versioned assurance appendix, not as marketing copy.

**Versions:** RC-3 (§1–§10) + RC-5.2 additions (§11–§14)
**Original date:** 3 March 2026 (§1–§10)
**Refreshed:** 19 May 2026 (§7 amendment; §11–§14 additions)
**Audience:** Users, integrators, auditors, and assurance staff evaluating contractual claims
**Register:** Technical / contractual

## Document scope

This document defines the promises ELSPETH makes to its users and to anyone evaluating the audit trail. These are not aspirational features — they are assurance guarantees that the system must uphold. A violation of any clause here is a release-blocking bug.

This document is **not** a roadmap, a feature list, or a marketing surface. For the inventory of what has shipped, see [`elspeth-progress-rc1-to-rc5.md`](elspeth-progress-rc1-to-rc5.md). For the high-level capability and assurance posture, see [`executive-summary.md`](executive-summary.md).

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

**Promise:** For any output token, `explain(query, data_flow, run_id, token_id=...)` returns:

| Component | Guarantee |
|-----------|-----------|
| Source row | Original data as received, with hash |
| Transform chain | Every transform that touched this data |
| Input/output hashes | Cryptographic proof of data at each boundary |
| External calls | Full request/response for LLM/HTTP calls |
| Routing decisions | Why data went where it went |
| Terminal state | Final disposition (completed, routed, failed, etc.) |

`query` and `data_flow` are the `QueryRepository` and `DataFlowRepository` exposed by `RecorderFactory` (the construction point for the Landscape repositories adopted in the RC-3.3 repository decomposition). In practice a caller writes `from elspeth.core.landscape import explain` and passes `factory.query` and `factory.data_flow`.

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

### 7.2 Access Control — AMENDED in RC-5

> **§7.2 amendment (RC-5.2, 19 May 2026):** The original RC-3 wording of this section ("ELSPETH is not multi-user") is no longer true. Multi-user session support landed in RC-5 with three authentication providers. The guarantees for the new surfaces are documented in §11 (Authentication and identity) and §13 (Multi-user session) below.

What ELSPETH **does not** guarantee at the policy level:

- **Organisational policy enforcement.** The platform enforces the access rules you configure. It does not decide *which* identities are valid, *which* groups have which roles, or *which* data classifications are permitted in which sinks — those decisions belong to the deploying organisation's policy.
- **Network-level isolation.** If ELSPETH is exposed on a network, the deployer remains responsible for network-level controls (TLS termination, firewall rules, reverse proxy authentication) in addition to ELSPETH's application-level authentication.
- **Data redaction profiles.** Output redaction at the sink boundary is on the roadmap but is not contractually guaranteed in RC-5.2.

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
| RC-5.2 (§11–§14 additions) | May 2026 | §11 Authentication and identity guarantees; §12 Secret-reference handling; §13 Multi-user session; §14 Composer authoring. §7.2 amended — "ELSPETH is not multi-user" disclaimer no longer accurate; replaced with organisational-policy boundary statement. |
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
from elspeth.core.landscape import explain
from elspeth.core.landscape.factory import RecorderFactory

def test_attributability(factory: RecorderFactory, run_id: str, token_id: str):
    """Given any output, prove complete lineage to source."""
    lineage = explain(factory.query, factory.data_flow, run_id, token_id=token_id)

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

## 11. AUTHENTICATION AND IDENTITY (RC-5.2)

### 11.1 Provider Coverage

**Promise:** Three authentication providers are supported, each with its declared validation surface.

| Provider | Use case | Validation surface |
|----------|----------|--------------------|
| Local username/password | Development; air-gapped deployments | Per-deployment user database; bcrypt-hashed credentials |
| OpenID Connect | Federated identity against a generic OIDC IdP | Issuer URL validation; JWKS verification; ID-token signature, audience, expiry, nonce |
| Microsoft Entra ID | Enterprise federation against Azure AD / Entra | All OIDC checks plus tenant ID validation; optional group-claim assertion |

### 11.2 Principal Recording

**Promise:** Every authenticated action records the principal that performed it.

- Session creation records the authenticated principal in the audit trail
- Pipeline configuration changes record the authoring principal
- Pipeline-execution attempts record the requesting principal
- Preference mutations record the mutating principal

### 11.3 Authentication Failure Is Recorded

**Promise:** Failed authentication attempts produce an audit record, not a silent rejection.

- The audit trail distinguishes "principal not recognised" from "principal recognised but credentials rejected"
- Repeated failures against the same principal are recorded in sequence

### 11.4 Authentication Is Not Authorisation

**Boundary clarification.** ELSPETH authenticates principals. It does not enforce organisational authorisation policy. The deploying organisation is responsible for deciding which principals may perform which actions; ELSPETH's audit trail provides the evidence on which such policy can be enforced and reviewed.

---

## 12. SECRET-REFERENCE HANDLING (RC-5.2)

### 12.1 Secret Values Never Appear in Configuration

**Promise:** Credentials are referenced by name in pipeline configuration; the value is resolved at run time.

- Configuration files contain `${VAULT_NAME}` references, not literal secret values
- A pipeline configuration containing a literal credential is a contract violation (P0 bug)
- Audit trail records the *reference*, never the *value*

### 12.2 Resolution Is Audited

**Promise:** Every secret resolution writes a record to the `secret_resolutions` audit table.

| Recorded | Detail |
|----------|--------|
| Vault source | Azure Key Vault URL or environment-variable namespace |
| Secret name | The reference key used to look up the secret |
| Resolution timestamp | When the secret was retrieved |
| HMAC fingerprint | Keyed hash of the resolved value (NOT the value itself) |
| Latency | Resolution time in milliseconds |

The HMAC fingerprint allows an auditor to verify that the same value was used across runs without ever recording or transmitting the value itself.

### 12.3 Failed Resolution Fails the Run

**Promise:** A secret that cannot be resolved is a fatal error, not a silent default.

- Missing secret → run fails with clear error pointing at the unresolved reference
- Vault-access failure → run fails; audit record captures the vault-source and failure mode
- The run does not proceed with a fallback or empty value

---

## 13. MULTI-USER SESSION (RC-5.2)

### 13.1 Session Isolation

**Promise:** Sessions are isolated by authenticated principal.

- A session created by principal A is not visible to principal B
- Preference state is keyed on the principal ID
- The session database enforces the principal-key boundary at the schema level: `user_id` is `NOT NULL` and indexed on every session-scoped table. The principal identifier is treated as opaque (it originates from an external auth provider — Local, OIDC, or Entra) and is therefore not a foreign key into a local users table; the contract is integrity-by-not-null-plus-index, not by referential constraint to an internal users table.

### 13.2 Session State Is Persisted

**Promise:** In-progress authoring work survives a process restart.

- Composer turn state is committed to the session database before the response is acknowledged
- A process crash mid-turn does not lose the prior conversation history
- The next session for the same principal can resume the work-in-progress

### 13.3 Session Mutation Is Audited

**Promise:** Every mutation to session state records the principal and the operation.

This is the audit-primacy rule from §7 applied to the session subsystem: a mutation that fails its audit write fails the operation.

---

## 14. COMPOSER AUTHORING (RC-5.2)

### 14.1 Pipeline Generation Does Not Relax Audit

**Promise:** A pipeline generated by the composer is executed by the same engine and recorded by the same audit trail as a hand-written pipeline.

- The composer is a configuration generator; the engine is unchanged
- Generated pipelines pass through the same validation as hand-written pipelines
- No "auto-generated" code path bypasses any guarantee in §1–§13

### 14.2 Composer Transcript Is Preserved

**Promise:** The full chat transcript that produced a pipeline is preserved in the session database.

- An auditor reviewing a generated pipeline can read the conversation that produced it
- Tool-call results are recorded with their inputs and outputs
- Sensitive tool-call payloads are redacted per the redaction MANIFEST; the redaction is itself recorded so the auditor knows that redaction occurred and what category

### 14.3 Composer Failures Do Not Silently Modify Pipelines

**Promise:** A composer operation that fails leaves the pipeline state unchanged.

- A failed validation does not partially apply
- A failed save reports the failure and preserves the prior committed state
- Recovery from a failed save is operator-driven, not silent

---

## Closing

*This is what ELSPETH promises. Nothing more, nothing less.*

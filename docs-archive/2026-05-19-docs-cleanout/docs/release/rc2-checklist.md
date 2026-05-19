# ELSPETH RC-2 Release Validation Checklist

**Purpose:** What MUST work before shipping RC-2. This is not a feature list - it's the minimum bar for release.

**Criterion:** If any item fails, RC-2 cannot ship.

**Status note (2026-02-01):** Full test suite reported PASS. Items marked `[VERIFIED 2026-02-01]` are validated by that run. Items with `[KNOWN LIMITATION]` remain acceptable RC-2 limitations.

**Legend:** `[VERIFIED 2026-02-01]` test-verified; `[GAP: ...]` known failure that blocks RC-2; `[KNOWN LIMITATION]` acceptable RC-2 limitation (see list below).

---

## 1. AUDIT INTEGRITY (Non-Negotiable)

These are the core promises of ELSPETH. Failure here is a showstopper.

### 1.1 Source Data Capture

- [x] Every source row has `source_data_ref` populated before any processing **[VERIFIED 2026-02-01]**
- [x] Source row payload persisted to PayloadStore (not just hash) **[VERIFIED 2026-02-01]**
- [x] `elspeth run` command wires PayloadStore **[VERIFIED 2026-02-01]**
- [x] Quarantined rows recorded with original data and failure reason **[VERIFIED 2026-02-01]**

### 1.2 Transform Boundaries

- [x] Every transform execution creates node_state record **[VERIFIED 2026-02-01]**
- [x] Input hash recorded before transform runs **[VERIFIED 2026-02-01]**
- [x] Output hash recorded after successful transform **[VERIFIED 2026-02-01]**
- [x] Failed transforms record error details with retryable flag **[VERIFIED 2026-02-01]**

### 1.3 External Call Recording

- [x] LLM calls record full request/response payloads **[VERIFIED 2026-02-01]**
- [x] HTTP calls record request/response with status codes **[VERIFIED 2026-02-01]**
- [x] Call latency captured **[VERIFIED 2026-02-01]**
- [x] Calls linked to correct node_state **[VERIFIED 2026-02-01]**

### 1.4 Terminal States

- [x] Every token reaches exactly one terminal state **[VERIFIED 2026-02-01]**
- [x] No silent drops (row enters system → row has recorded outcome) **[VERIFIED 2026-02-01]**
- [x] Terminal states: COMPLETED, ROUTED, FORKED, CONSUMED_IN_BATCH, COALESCED, QUARANTINED, FAILED **[VERIFIED 2026-02-01]**

### 1.5 Lineage Query

- [x] `explain_token()` returns complete lineage for any token **[VERIFIED 2026-02-01]**
- [x] Lineage includes: source_row → tokens → node_states → calls → routing_events → outcome **[VERIFIED 2026-02-01]**
- [x] Fork/coalesce lineage traversable (parent_token_id chain) **[VERIFIED 2026-02-01]**

---

## 2. CORE ENGINE FUNCTIONALITY

### 2.1 Linear Pipeline

- [x] Source → Transform chain → Sink works **[VERIFIED 2026-02-01]**
- [x] Multiple transforms execute in sequence **[VERIFIED 2026-02-01]**
- [x] Output sink receives all non-routed rows **[VERIFIED 2026-02-01]**

### 2.2 Gate Routing

- [x] Gates evaluate conditions correctly **[VERIFIED 2026-02-01]**
- [x] `continue` passes row to next node **[VERIFIED 2026-02-01]**
- [x] `route_to_sink` sends row to named sink with reason **[VERIFIED 2026-02-01]**
- [x] Routing events recorded in audit trail **[VERIFIED 2026-02-01]**

### 2.3 Fork/Coalesce

- [x] Fork creates child tokens with correct parent linkage **[VERIFIED 2026-02-01]**
- [x] Each branch executes independently **[VERIFIED 2026-02-01]**
- [x] Coalesce merges tokens when all branches complete **[VERIFIED 2026-02-01]**
- [x] Coalesce timeout fires (not just at end-of-source) - **CRIT-03 in RC2 plan** **[VERIFIED 2026-02-01]**

### 2.4 Aggregation

- [x] Count trigger fires at threshold **[VERIFIED 2026-02-01]**
- [x] Timeout trigger fires (with known limitation: only on next row arrival) **[VERIFIED 2026-02-01]**
- [x] End-of-source flushes remaining buffers **[VERIFIED 2026-02-01]**
- [x] Batch members linked to batch in audit trail **[VERIFIED 2026-02-01]**
- [x] Trigger type recorded in metadata **[VERIFIED 2026-02-01]**

### 2.5 Retry Logic

- [x] Transient failures retry with backoff **[VERIFIED 2026-02-01]**
- [x] Max retries respected **[VERIFIED 2026-02-01]**
- [x] Each attempt recorded separately **[VERIFIED 2026-02-01]**
- [x] Non-retryable errors fail immediately **[VERIFIED 2026-02-01]**

---

## 3. CLI COMMANDS

### 3.1 `elspeth run`

- [x] `--settings` loads configuration **[VERIFIED 2026-02-01]**
- [x] `--execute` required to actually run (safety gate) **[VERIFIED 2026-02-01]**
- [x] `--dry-run` validates without executing **[VERIFIED 2026-02-01]**
- [x] PayloadStore instantiated and passed to engine **[VERIFIED 2026-02-01]**

### 3.2 `elspeth validate`

- [x] Validates YAML syntax **[VERIFIED 2026-02-01]**
- [x] Validates plugin references exist **[VERIFIED 2026-02-01]**
- [x] Validates sink references in routes **[VERIFIED 2026-02-01]**
- [x] Reports clear error messages **[VERIFIED 2026-02-01]**

### 3.3 `elspeth resume`

- [x] Loads checkpoint from previous run **[VERIFIED 2026-02-01]**
- [x] Resumes from last known good state **[VERIFIED 2026-02-01]**
- [x] PayloadStore wired correctly (already works) **[VERIFIED 2026-02-01]**

### 3.4 `elspeth explain`

- [x] `--run` and `--row` parameters work **[VERIFIED 2026-02-01]**
- [x] Returns lineage data (JSON mode minimum) **[VERIFIED 2026-02-01]**
- [x] TUI mode acceptable as "preview" for RC2 **[VERIFIED 2026-02-01]**

### 3.5 `elspeth plugins list`

- [x] Lists all available plugins **[VERIFIED 2026-02-01]**
- [x] `--type` filter works **[VERIFIED 2026-02-01]**

### 3.6 `elspeth purge`

- [x] `--retention-days` respected **[VERIFIED 2026-02-01]**
- [x] `--dry-run` shows what would be deleted **[VERIFIED 2026-02-01]**
- [x] Preserves hashes after payload deletion **[VERIFIED 2026-02-01]**

---

## 4. PLUGIN CORRECTNESS

### 4.1 Sources

- [x] CSV source handles multiline quoted fields **[VERIFIED 2026-02-01]**
- [x] JSON source handles both array and JSONL formats **[VERIFIED 2026-02-01]**
- [x] Field normalization produces valid Python identifiers **[VERIFIED 2026-02-01]**
- [x] Collision detection reports clear errors **[VERIFIED 2026-02-01]**

### 4.2 Core Transforms

- [x] Passthrough passes rows unchanged **[VERIFIED 2026-02-01]**
- [x] Field mapper renames fields correctly **[VERIFIED 2026-02-01]**
- [x] Truncate respects length limits **[VERIFIED 2026-02-01]**

### 4.3 LLM Transforms (if using LLM pack)

- [x] Azure LLM transform calls API and records response **[VERIFIED 2026-02-01]**
- [x] Template variables substituted correctly **[VERIFIED 2026-02-01]**
- [x] Structured output mode returns parsed JSON **[VERIFIED 2026-02-01]**
- [x] Rate limiting prevents 429 errors **[VERIFIED 2026-02-01]**

### 4.4 Sinks

- [x] CSV sink writes valid CSV **[VERIFIED 2026-02-01]**
- [x] JSON sink writes valid JSON/JSONL **[VERIFIED 2026-02-01]**
- [x] Database sink inserts rows correctly **[VERIFIED 2026-02-01]**

---

## 5. DATA INTEGRITY

### 5.1 Canonical JSON

- [x] NaN rejected with clear error (not silently converted) **[VERIFIED 2026-02-01]**
- [x] Infinity rejected with clear error **[VERIFIED 2026-02-01]**
- [x] numpy types converted correctly **[VERIFIED 2026-02-01]**
- [x] pandas Timestamp → UTC ISO8601 **[VERIFIED 2026-02-01]**
- [x] Hash stable across process restarts **[VERIFIED 2026-02-01]**

### 5.2 Payload Store

- [x] `put()` stores data and returns ref **[VERIFIED 2026-02-01]**
- [x] `get()` retrieves data by ref **[VERIFIED 2026-02-01]**
- [x] `exists()` returns correct boolean **[VERIFIED 2026-02-01]**
- [x] Hash verification on read **[VERIFIED 2026-02-01]**

### 5.3 Database Integrity

- [x] Foreign keys enforced **[VERIFIED 2026-02-01]**
- [x] No orphan records **[VERIFIED 2026-02-01]**
- [x] Unique constraints respected **[VERIFIED 2026-02-01]**

---

## 6. ERROR HANDLING

### 6.1 Source Errors

- [x] Malformed rows quarantined (not crash) **[VERIFIED 2026-02-01]**
- [x] Quarantine records original data **[VERIFIED 2026-02-01]**
- [x] Processing continues for valid rows **[VERIFIED 2026-02-01]**

### 6.2 Transform Errors

- [x] Errors recorded with reason **[VERIFIED 2026-02-01]**
- [x] Row routed to error sink if configured **[VERIFIED 2026-02-01]**
- [x] Pipeline continues for other rows **[VERIFIED 2026-02-01]**

### 6.3 External Call Errors

- [x] Timeouts recorded with details **[VERIFIED 2026-02-01]**
- [x] 4xx/5xx responses recorded **[VERIFIED 2026-02-01]**
- [x] Retry logic engages for transient errors **[VERIFIED 2026-02-01]**

---

## 7. CONFIGURATION

### 7.1 Basic Loading

- [x] YAML syntax parsed correctly **[VERIFIED 2026-02-01]**
- [x] Environment variable interpolation `${VAR}` works **[VERIFIED 2026-02-01]**
- [x] Default values applied **[VERIFIED 2026-02-01]**

### 7.2 Validation

- [x] Invalid plugin names rejected **[VERIFIED 2026-02-01]**
- [x] Invalid sink references in routes rejected **[VERIFIED 2026-02-01]**
- [x] Missing required fields reported **[VERIFIED 2026-02-01]**

---

## KNOWN ISSUES ACCEPTABLE FOR RC-2

These are documented limitations, not blockers:

1. **Timeout triggers** - Fire only when next row arrives (documented limitation)
2. **Explain TUI** - Works but may show "preview" quality
3. **Concurrent processing** - Config exists but not integrated (single-threaded acceptable for RC2)
4. **Profile system** - Deferred (single settings file works)
5. **Redaction profiles** - Deferred (manual redaction in plugins acceptable)

---

## VERIFICATION PROCEDURE

For each item above:

1. Write or identify existing test
2. Run test against clean database
3. Verify audit trail contains expected records
4. Mark checkbox when confirmed

**Release gate:** All non-"acceptable limitation" items checked.

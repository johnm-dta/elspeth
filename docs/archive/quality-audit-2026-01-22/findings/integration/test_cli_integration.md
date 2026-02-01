# Test Quality Review: test_cli_integration.py

## Summary

This integration test file validates CLI end-to-end workflows for pipeline execution, dry-run behavior, plugin listing, and source quarantine routing. Tests are mostly well-structured with proper fixtures and isolation, but suffer from weak assertions, incomplete audit trail verification, and minimal error scenario coverage.

## Poorly Constructed Tests

### Test: test_full_workflow_csv_to_json (line 60)
**Issue**: Weak assertions - only checks output file exists and count, doesn't verify audit trail
**Evidence**:
```python
# Step 3: Check output exists and is valid
output_file = tmp_path / "output.json"
assert output_file.exists()

data = json.loads(output_file.read_text())
assert len(data) == 3
assert data[0]["name"] == "alice"
```
**Fix**: This is an INTEGRATION test for an AUDITABILITY framework. Must verify:
- Landscape database was created at `tmp_path / 'landscape.db'`
- Run metadata was recorded (run_id, start_time, completion status)
- Source entries exist for all 3 rows
- Node state transitions recorded for each row (PENDING → COMPLETED)
- Artifact descriptor recorded with content hash matching output file
- All rows reached terminal state (COMPLETED)

**Example of proper verification**:
```python
# Verify audit trail completeness
landscape_db_path = tmp_path / "landscape.db"
assert landscape_db_path.exists(), "Audit trail database must exist"

from elspeth.core.landscape.database import LandscapeDB
db = LandscapeDB(f"sqlite:///{landscape_db_path}")

# Verify run metadata
runs = db.execute("SELECT * FROM runs").fetchall()
assert len(runs) == 1
run = runs[0]
assert run["status"] == "completed"

# Verify all rows recorded
source_entries = db.execute("SELECT * FROM source_entries WHERE run_id = ?", (run["run_id"],)).fetchall()
assert len(source_entries) == 3

# Verify terminal states
states = db.execute(
    "SELECT DISTINCT terminal_state FROM node_states WHERE run_id = ?",
    (run["run_id"],)
).fetchall()
assert all(s["terminal_state"] == "COMPLETED" for s in states)
```

**Priority**: P0 - This test completely fails to verify the core value proposition (auditability)

---

### Test: test_full_workflow_csv_to_json (line 60)
**Issue**: Only tests happy path - no error scenarios
**Evidence**: Test assumes CSV parses, pipeline runs without errors, output writes successfully
**Fix**: Add companion tests for:
- Invalid CSV format (malformed headers, inconsistent column counts)
- Sink write failures (directory doesn't exist, permissions error)
- Configuration errors (missing required fields)
- Plugin instantiation failures

**Priority**: P1 - Integration tests must cover failure modes

---

### Test: test_plugins_list_shows_all_types (line 82)
**Issue**: Assertions are fragile substring matches, doesn't verify structure
**Evidence**:
```python
# Sources
assert "csv" in result.stdout
assert "json" in result.stdout

# Sinks
assert "database" in result.stdout
```
**Fix**: Verify structured output format. If CLI returns text, verify format consistency:
```python
# Verify output structure
lines = result.stdout.strip().split("\n")
assert any("SOURCES" in line.upper() for line in lines), "Should show sources section"
assert any("SINKS" in line.upper() for line in lines), "Should show sinks section"

# Verify expected plugins appear (not just substring match)
assert any(line.strip().startswith("csv") for line in lines), "csv source should be listed"
```

If CLI returns JSON (which it should for machine-readable output), parse and verify:
```python
# Better: Add --json flag to plugins list and verify structured output
result = runner.invoke(app, ["plugins", "list", "--json"])
data = json.loads(result.stdout)
assert "sources" in data
assert "csv" in data["sources"]
assert data["sources"]["csv"]["version"]  # Verify version present
```

**Priority**: P2 - Tests pass but are brittle and uninformative

---

### Test: test_dry_run_does_not_create_output (line 96)
**Issue**: Incomplete - doesn't verify landscape DB is also not created
**Evidence**: Only checks output file, but dry-run should avoid ALL side effects
**Fix**: Verify no database created, no landscape entries, no audit trail:
```python
# Output should NOT be created
assert not output_file.exists()

# Landscape DB should ALSO not be created in dry-run mode
landscape_db_path = tmp_path / "landscape.db"
assert not landscape_db_path.exists(), "Dry-run must not create audit database"

# Verify console output mentions dry-run mode
assert "dry run" in result.stdout.lower()
assert "would execute" in result.stdout.lower()
```

**Priority**: P1 - Dry-run creating audit DB would pollute production systems

---

### Test: test_run_without_flags_exits_with_warning (line 109)
**Issue**: Doesn't verify the exact warning message or that no execution occurred
**Evidence**: Only checks exit code and presence of `--execute` in output
**Fix**: Verify specific behavior:
```python
# Should exit with code 1 (safety feature)
assert result.exit_code == 1

# Should show helpful error message
assert "Pipeline configuration valid" in result.output
assert "To execute, add --execute" in result.output

# Should NOT have run pipeline (no landscape DB created)
landscape_db_path = tmp_path / "landscape.db"
assert not landscape_db_path.exists(), "Pipeline must not execute without --execute flag"
```

**Priority**: P2 - Existing assertions catch the issue, but don't verify full behavior

---

### Test: test_invalid_rows_routed_to_quarantine_sink (line 182)
**Issue**: Doesn't verify audit trail records quarantine routing decision
**Evidence**: Only checks file contents, not landscape metadata
**Fix**: Verify quarantine routing is auditable:
```python
# Existing file checks are good, ADD audit trail verification:
from elspeth.core.landscape.database import LandscapeDB

landscape_db_path = tmp_path / "landscape.db"
assert landscape_db_path.exists()

db = LandscapeDB(f"sqlite:///{landscape_db_path}")

# Verify quarantined row has terminal state ROUTED
quarantine_states = db.execute(
    """
    SELECT ns.row_id, ns.terminal_state, ns.route_destination
    FROM node_states ns
    WHERE ns.terminal_state = 'ROUTED' AND ns.route_destination = 'quarantine'
    """
).fetchall()
assert len(quarantine_states) == 1, "One row should be routed to quarantine"

# Verify valid rows reached default sink
completed_states = db.execute(
    """
    SELECT ns.row_id, ns.terminal_state
    FROM node_states ns
    WHERE ns.terminal_state = 'COMPLETED'
    """
).fetchall()
assert len(completed_states) == 2, "Two valid rows should complete"

# Verify source entry recorded validation failure
quarantined_source = db.execute(
    """
    SELECT se.validation_status, se.validation_error
    FROM source_entries se
    WHERE se.row_id = ?
    """,
    (quarantine_states[0]["row_id"],)
).fetchone()
assert quarantined_source["validation_status"] == "FAILED"
assert "score" in quarantined_source["validation_error"].lower()
```

**Priority**: P0 - Quarantine routing is a high-stakes accountability feature (CLAUDE.md: "I don't know what happened" is never acceptable)

---

### Test: test_discard_does_not_write_to_sink (line 209)
**Issue**: Doesn't verify landscape still records discarded row metadata
**Evidence**: Only checks output file, doesn't verify audit trail
**Fix**: Even discarded rows must be traceable:
```python
# Only valid rows in output
output_file = tmp_path / "output.json"
data = json.loads(output_file.read_text())
assert len(data) == 2  # alice and carol only

# BUT audit trail must record ALL source rows including discarded ones
from elspeth.core.landscape.database import LandscapeDB

landscape_db_path = tmp_path / "landscape.db"
db = LandscapeDB(f"sqlite:///{landscape_db_path}")

# Verify 3 source entries (including discarded row)
source_entries = db.execute("SELECT * FROM source_entries").fetchall()
assert len(source_entries) == 3, "All source rows must be recorded, including discarded"

# Verify discarded row has proper status
discarded = [se for se in source_entries if se["validation_status"] == "FAILED"]
assert len(discarded) == 1, "One row should be marked as validation failure"
assert "bob" in str(discarded[0]["row_data"]), "Discarded row should be bob"
```

**Priority**: P0 - Audit trail completeness is non-negotiable (CLAUDE.md: "if it's not recorded, it didn't happen")

---

## Misclassified Tests

No misclassification issues. Tests are appropriately integration-level (CLI → engine → database → file output).

---

## Infrastructure Gaps

### Gap: No shared fixture for verifying audit trail completeness
**Issue**: Every test should verify landscape DB consistency, but no helper exists
**Evidence**: Tests only check output files, never verify audit trail
**Fix**: Add to `tests/integration/conftest.py`:
```python
from elspeth.core.landscape.database import LandscapeDB

@pytest.fixture
def verify_audit_trail():
    """Helper to verify audit trail completeness after pipeline run."""
    def _verify(tmp_path: Path, expected_row_count: int) -> LandscapeDB:
        landscape_db_path = tmp_path / "landscape.db"
        assert landscape_db_path.exists(), "Landscape DB must exist after run"

        db = LandscapeDB(f"sqlite:///{landscape_db_path}")

        # Verify run completed
        runs = db.execute("SELECT * FROM runs").fetchall()
        assert len(runs) == 1, "Should have exactly one run"
        assert runs[0]["status"] == "completed", "Run should be marked completed"

        # Verify all rows recorded
        source_entries = db.execute("SELECT * FROM source_entries").fetchall()
        assert len(source_entries) == expected_row_count, (
            f"Expected {expected_row_count} source entries, got {len(source_entries)}"
        )

        # Verify all rows reached terminal state
        non_terminal = db.execute(
            "SELECT * FROM node_states WHERE terminal_state IS NULL"
        ).fetchall()
        assert len(non_terminal) == 0, "All rows must reach terminal state"

        return db
    return _verify
```

Usage:
```python
def test_full_workflow_csv_to_json(pipeline_config, tmp_path, verify_audit_trail):
    # ... run pipeline ...

    # Verify audit trail
    db = verify_audit_trail(tmp_path, expected_row_count=3)

    # Additional audit queries for specific tests
    terminal_states = db.execute("SELECT terminal_state FROM node_states").fetchall()
    assert all(s["terminal_state"] == "COMPLETED" for s in terminal_states)
```

**Priority**: P0 - Critical infrastructure for testing auditability framework

---

### Gap: No fixture for creating CSV files with controlled schema violations
**Issue**: `csv_with_invalid_rows` is ad-hoc, need reusable pattern
**Evidence**: Line 129 creates CSV inline, but other tests may need similar setups
**Fix**: Add parametrized fixture:
```python
@pytest.fixture
def csv_with_schema_violations(tmp_path: Path, request):
    """Create CSV with controlled schema violations.

    Pass violations param as indirect fixture:
        @pytest.mark.parametrize("csv_with_schema_violations", [
            {"row": 2, "field": "score", "invalid_value": "bad", "expected_type": "int"}
        ], indirect=True)
    """
    violations = request.param
    csv_file = tmp_path / "data_with_violations.csv"

    # Generate CSV with controlled violations
    # Implementation omitted for brevity
    return csv_file, violations
```

**Priority**: P2 - Quality of life improvement, not blocking

---

### Gap: No test for CLI output format consistency (console vs JSON)
**Issue**: CLI supports `--format json` but no tests verify JSON output structure
**Evidence**: Line 149 in cli.py shows `output_format` parameter, never tested
**Fix**: Add test:
```python
def test_run_json_output_format(pipeline_config: Path, tmp_path: Path) -> None:
    """Verify --format json produces valid structured output."""
    from elspeth.cli import app

    result = runner.invoke(
        app,
        ["run", "-s", str(pipeline_config), "--execute", "--format", "json"]
    )
    assert result.exit_code == 0

    # Verify each line is valid JSON (streaming JSON events)
    lines = [line for line in result.stdout.split("\n") if line.strip()]
    for line in lines:
        event = json.loads(line)  # Should not raise
        assert "event" in event, "Each JSON event must have 'event' field"

    # Verify specific events present
    event_types = [json.loads(line)["event"] for line in lines]
    assert "run_started" in event_types
    assert "run_completed" in event_types
```

**Priority**: P1 - JSON output is critical for CI/CD integration

---

### Gap: No test for concurrent CLI invocations
**Issue**: CLI uses global plugin manager singleton (line 32 in cli.py), not tested for concurrency
**Evidence**: `_plugin_manager_cache` is module-level, could cause issues in parallel test runs
**Fix**: Add test:
```python
def test_concurrent_cli_runs_isolated(tmp_path: Path, sample_csv: Path) -> None:
    """Verify concurrent CLI invocations don't interfere via plugin manager cache."""
    import concurrent.futures
    from elspeth.cli import app

    # Create two separate configs with different landscape DBs
    config1 = create_config(tmp_path / "run1", sample_csv)
    config2 = create_config(tmp_path / "run2", sample_csv)

    def run_pipeline(config_path):
        return runner.invoke(app, ["run", "-s", str(config_path), "--execute"])

    # Run both pipelines concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(run_pipeline, config1)
        future2 = executor.submit(run_pipeline, config2)

        result1 = future1.result()
        result2 = future2.result()

    # Both should succeed
    assert result1.exit_code == 0
    assert result2.exit_code == 0

    # Verify separate audit trails
    db1 = LandscapeDB(f"sqlite:///{tmp_path / 'run1' / 'landscape.db'}")
    db2 = LandscapeDB(f"sqlite:///{tmp_path / 'run2' / 'landscape.db'}")

    runs1 = db1.execute("SELECT * FROM runs").fetchall()
    runs2 = db2.execute("SELECT * FROM runs").fetchall()

    assert runs1[0]["run_id"] != runs2[0]["run_id"], "Runs must have distinct IDs"
```

**Priority**: P1 - Plugin manager singleton is a potential production bug

---

### Gap: Missing fixture cleanup verification
**Issue**: Tests create temporary databases but don't verify cleanup
**Evidence**: `tmp_path` fixture auto-cleans, but tests don't verify connections closed
**Fix**: Add verification that database connections are properly closed:
```python
@pytest.fixture
def pipeline_config_with_cleanup_verification(tmp_path: Path, sample_csv: Path):
    """Pipeline config fixture that verifies DB connections closed after test."""
    config = create_config(tmp_path, sample_csv)
    yield config

    # Verify no lingering connections (SQLite-specific check)
    landscape_db_path = tmp_path / "landscape.db"
    if landscape_db_path.exists():
        # Should be able to open DB without "database is locked" error
        import sqlite3
        conn = sqlite3.connect(str(landscape_db_path))
        conn.execute("SELECT 1")  # Should succeed immediately
        conn.close()
```

**Priority**: P2 - Safety check, unlikely to fail but good hygiene

---

## Positive Observations

1. **Proper test isolation**: Each test uses `tmp_path` fixture, no shared state between tests
2. **Fixture composition**: Good use of fixture dependencies (`pipeline_config` depends on `sample_csv`)
3. **Clear test names**: Names describe behavior being tested (e.g., `test_dry_run_does_not_create_output`)
4. **Docstrings present**: Most tests have docstrings explaining intent
5. **Safety feature tested**: Line 109 test verifies `--execute` flag requirement (critical safety feature)
6. **Fixture documentation**: Line 28-32 explains why "default" sink is required
7. **Realistic test data**: Line 23 creates CSV with multiple rows covering edge cases (alice, bob, carol)
8. **Error path testing**: `TestSourceQuarantineRouting` class specifically tests quarantine routing (lines 121-247)

---

## Recommendations Summary

**Critical (P0) - Fix before RC-1 release:**
1. Add audit trail verification to all integration tests
2. Verify quarantine routing decisions recorded in landscape
3. Verify discarded rows still appear in audit trail
4. Create `verify_audit_trail` fixture helper

**Important (P1) - Fix soon:**
1. Add error scenario tests (malformed CSV, sink failures)
2. Test `--format json` output structure
3. Test concurrent CLI invocations (plugin manager singleton safety)
4. Verify dry-run doesn't create landscape DB

**Nice to have (P2):**
1. Improve `test_plugins_list` assertions (parse structured output)
2. Create reusable `csv_with_schema_violations` fixture
3. Add cleanup verification fixtures

---

## Overall Assessment

**Test Coverage**: 40% - Tests verify CLI mechanics and file I/O but completely ignore audit trail verification, which is the framework's core value proposition.

**Test Quality**: 50% - Well-structured and isolated, but assertions are weak and don't verify the behaviors that matter for an auditability framework.

**Blocking Issues**: Yes - No audit trail verification in ANY integration test. This is a critical gap for a framework whose entire purpose is "high-stakes accountability" and "every decision must be traceable."

**Recommended Action**: BLOCK RC-1 release until at least P0 items are addressed. An audit framework with no audit trail testing is fundamentally untested.

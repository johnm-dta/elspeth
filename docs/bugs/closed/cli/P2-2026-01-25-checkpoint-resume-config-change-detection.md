# P2: Checkpoint Resume Missing Pipeline Config Change Detection

**Date:** 2026-01-25
**Severity:** P2 (High)
**Status:** Open
**Component:** Checkpoint Resume
**Affects:** RC-1

## Problem

When resuming from a checkpoint, the system does not validate that the pipeline configuration matches the checkpointed run. Changes to pipeline structure (inserting transforms, reordering plugins) can cause silent incorrect behavior.

## Root Cause

Node IDs are now position-based with sequence numbers (`transform_passthrough_abc_0`, `transform_passthrough_abc_1`). This ensures uniqueness but creates fragility when config changes:

```yaml
# Original config (checkpointed)
row_plugins:
  - plugin: passthrough  # seq=0 → transform_passthrough_abc_0
  - plugin: passthrough  # seq=1 → transform_passthrough_abc_1

# Modified config (resume)
row_plugins:
  - plugin: field_mapper  # seq=0 → transform_field_mapper_xyz_0 (NEW)
  - plugin: passthrough   # seq=1 → transform_passthrough_abc_1 (SAME ID!)
  - plugin: passthrough   # seq=2 → transform_passthrough_abc_2 (NEW)
```

Checkpoint says "resume from `transform_passthrough_abc_1`" but that ID now refers to a **different transform** in the modified config.

## Impact

- Silent incorrect processing during resume
- Transforms may be skipped or duplicated
- Audit trail becomes inconsistent
- No error raised - behavior just wrong

## Current Validation

The system validates:
- ✅ Checkpoint timestamp (must be after deterministic IDs commit)
- ✅ Run status (must be "failed")
- ✅ Route destinations (must reference existing sinks)
- ❌ **Pipeline config hash** (NOT validated)

## Proposed Solutions

### Option A: Store Pipeline Config Hash in Run Record
```python
# In orchestrator.py when creating run
run_id = db.create_run(
    config_hash=hashlib.sha256(canonical_json(full_config).encode()).hexdigest(),
    ...
)

# In recovery.py when resuming
stored_hash = db.get_run(run_id).config_hash
current_hash = hashlib.sha256(canonical_json(full_config).encode()).hexdigest()
if stored_hash != current_hash:
    raise ConfigMismatchError(f"Pipeline config changed since checkpoint")
```

**Pros:** Simple, catches any config change
**Cons:** Byte-for-byte match required - even comment changes fail

### Option B: Store Pipeline Config Hash in Checkpoint
```python
# In CheckpointManager.create_checkpoint()
checkpoint = Checkpoint(
    pipeline_config_hash=hash(config),
    ...
)

# In RecoveryManager.get_resume_point()
if checkpoint.pipeline_config_hash != hash(current_config):
    raise ConfigMismatchError()
```

**Pros:** Checkpoint-specific validation
**Cons:** Requires checkpoint schema migration

### Option C: Add `--force-resume` Flag
```bash
# Strict mode (default)
elspeth resume <run_id> --execute  # Fails on config mismatch

# Force mode (expert users)
elspeth resume <run_id> --execute --force-resume  # Allows mismatch
```

**Pros:** Allows expert override for intentional changes
**Cons:** Users might default to --force

## Recommendation

**Implement Option A + Option C:**
1. Store config hash in Run record (always)
2. Validate hash on resume (strict by default)
3. Add `--force-resume` flag for intentional config changes
4. Log warning when --force-resume used

## Related

- Commit 04d5605: Introduced deterministic node IDs
- Commit [current]: Added sequence numbers to prevent collisions
- Architecture review: Identified this gap during systematic debugging

## Test Case

```python
def test_resume_rejects_changed_pipeline_config():
    # Run pipeline, checkpoint midway
    run_id = orchestrator.run(config=original_config)

    # Modify config
    modified_config = insert_transform_at_position_0(original_config)

    # Attempt resume - should FAIL
    with pytest.raises(ConfigMismatchError):
        orchestrator.resume(run_id=run_id, config=modified_config)
```

## Priority Justification

**P2 (High)** because:
- Affects production checkpoint resume (feature is implemented and usable)
- Silent incorrect behavior (no error raised)
- Data integrity impact (audit trail inconsistent)
- BUT: Checkpoint resume is new (RC-1), limited production usage

# Archived Scripts

This directory contains scripts that were used for one-time migrations or are no longer needed in active development.

## migrations/

One-time migration scripts that updated configuration or code during specific refactorings.

### 2026-01-fix-allowlist-post-ruff-format.py

**Purpose:** Updated `config/cicd/no_bug_hiding.yaml` allowlist with new line numbers after `ruff format` changed code formatting.

**Context:** When ruff formatter was applied, defensive programming exemptions that were keyed by file:line became stale. This script:
- Removed 40+ stale entries with old line numbers
- Added new entries with updated line numbers post-formatting

**One-time use:** This script contains hardcoded line numbers specific to a single point in time. It was run once and should not be used again.

**Preserved for:** Git history context - if someone wonders how/when the allowlist was bulk-updated.

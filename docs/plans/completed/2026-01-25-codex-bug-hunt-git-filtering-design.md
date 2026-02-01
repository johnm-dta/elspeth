# Design: Git-Based File Filtering for codex_bug_hunt.py

**Date:** 2026-01-25
**Status:** Approved

## Overview

Extend `scripts/codex_bug_hunt.py` with additional git-based file filtering options to target specific branches and commit ranges for bug analysis.

## Current State

The script currently supports:
- `--changed-since REF` - filters files changed since a git ref using `git diff --name-only REF`

## New Functionality

### 1. Branch Comparison (`--branch`)

**Purpose:** Analyze only files changed on current branch vs a base branch (e.g., feature branch vs main).

**Implementation:**
- Uses `git merge-base BASE HEAD` to find common ancestor
- Runs `git diff --name-only $(git merge-base BASE HEAD)..HEAD -- <path>`
- Equivalent to GitHub PR "Files changed" view

**Example:**
```bash
./scripts/codex_bug_hunt.py --branch main
```
On branch `fix/rc1-bug-burndown-session-4`, this analyzes only files modified since the branch diverged from `main`.

### 2. Commit Range (`--commit-range`)

**Purpose:** Analyze files changed between two specific commits.

**Implementation:**
- Validates range format (contains `..`)
- Runs `git diff --name-only START..END -- <path>`
- Works with commit hashes, tags, HEAD~N, etc.

**Example:**
```bash
./scripts/codex_bug_hunt.py --commit-range abc123..def456
```

## Design Decisions

### Mutual Exclusivity

These three options are mutually exclusive (only one allowed per run):
- `--changed-since`
- `--branch`
- `--commit-range`

**Rationale:** Combining them creates ambiguous semantics. Users should pick the most appropriate filter for their use case.

### Separate Flags vs Smart Detection

**Chosen:** Separate flags
**Alternative:** Smart `--changed-since` that detects `..` vs `...` syntax

**Rationale:** Separate flags have clearer semantics and don't require users to understand git's two-dot vs three-dot syntax.

## Code Structure

### New Helper Functions

```python
def _changed_files_on_branch(repo_root: Path, root_dir: Path, base_branch: str) -> list[Path]:
    """Get files changed on current branch vs base branch using merge-base."""
    # 1. Find merge base: git merge-base BASE HEAD
    # 2. Diff from merge base: git diff --name-only MERGE_BASE..HEAD -- <path>
    # 3. Filter and return paths (same logic as _changed_files_since)

def _changed_files_in_range(repo_root: Path, root_dir: Path, commit_range: str) -> list[Path]:
    """Get files changed in commit range (e.g., 'abc123..def456')."""
    # 1. Validate format (must contain '..')
    # 2. Run: git diff --name-only START..END -- <path>
    # 3. Filter and return paths (same logic as _changed_files_since)
```

### Modified Functions

**`_list_files`:**
- Add parameters: `branch: str | None`, `commit_range: str | None`
- Add mutual exclusivity validation
- Call appropriate helper based on which option is set
- Maintain existing intersection logic with `--paths-from`

**`main`:**
- Add `--branch` argument
- Add `--commit-range` argument
- Pass to `_list_files`

## Error Handling

- **Invalid git ref:** Propagate git error message to user
- **Invalid range format:** Raise `ValueError` with helpful message
- **Mutual exclusivity violation:** Raise `ValueError` explaining only one option allowed
- **Merge base not found:** Propagate git error (e.g., no common ancestor)

## Testing Scenarios

1. **Branch comparison:** `--branch main` on feature branch
2. **Commit range:** `--commit-range HEAD~5..HEAD~2`
3. **With existing filters:** `--branch main --paths-from subset.txt` (intersection)
4. **Error cases:**
   - `--branch nonexistent-branch`
   - `--commit-range invalid-format`
   - `--branch main --changed-since HEAD~5` (mutual exclusivity)

## Success Criteria

- Users can filter by branch divergence for focused PR reviews
- Users can filter by commit ranges for analyzing specific changes
- Error messages clearly explain git failures
- Behavior is consistent with existing `--changed-since` pattern

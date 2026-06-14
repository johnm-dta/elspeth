# Bolt's Performance Journal

This journal is used to capture critical learnings, unexpected performance behaviors, and important insights from optimizing the codebase.

## 2026-06-03 - Canonical JSON Normalization Fast-Path
**Learning:** Checking for Python's standard primitive types (`None`, `str`, `int`, `bool`) using `type(obj)` in a tuple is significantly faster than performing multiple `isinstance` checks against `np.floating`, `float`, and other NumPy/Pandas types first. In a recursive pipeline data serializer like `_normalize_for_canonical`, fast-pathing these primitives directly at the recursion entry point yields a ~52% performance improvement.
**Action:** Always place direct type checks for standard primitives (`str`, `int`, `bool`, `type(None)`) before complex class hierarchy `isinstance` checks when parsing or normalising recursive data structures.

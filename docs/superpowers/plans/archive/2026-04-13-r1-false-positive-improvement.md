# R1 False Positive Rule Improvement

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the R1 (dict.get) detection in `enforce_tier_model.py` to distinguish legitimate `.get()` method calls from defensive dict access, eliminating ~30 false positive allowlist entries.

**Architecture:** Add a consolidated `_is_likely_non_dict_get()` predicate method to `TierModelVisitor` that combines three heuristics: (1) decorator usage (`@router.get("/path")`), (2) URL-like first arguments for HTTP methods, (3) ChromaDB-specific keyword arguments. The heuristics are conservative — when in doubt, flag it (prefer false positives over false negatives).

**Tech Stack:** Python AST module, pytest for testing

**Filigree Issue:** elspeth-ba868c3896

**Review amendments incorporated:**
- Narrowed SDK keywords to `{"ids", "include", "where"}` (removed `limit`/`offset` — too generic, collides with ORM patterns)
- Consolidated three heuristics into single `_is_likely_non_dict_get()` predicate (cleaner than `pass`-chain)
- Added tests for async decorators, f-string URLs (intentional limitation), and `limit=` regression
- Improved Task 5 verification with fingerprint capture before/after

---

## File Structure

| File | Responsibility |
|------|----------------|
| `scripts/cicd/enforce_tier_model.py` | Add `_is_likely_non_dict_get()` consolidated predicate to `TierModelVisitor` |
| `tests/unit/scripts/cicd/test_enforce_tier_model.py` | Add test class `TestR1FalsePositiveFiltering` |
| `config/cicd/enforce_tier_model/web.yaml` | Remove false positive entries after verification |
| `config/cicd/enforce_tier_model/plugins.yaml` | Remove false positive entries after verification |

---

## Task 1: Add Tests for False Positive Filtering

**Files:**
- Modify: `tests/unit/scripts/cicd/test_enforce_tier_model.py`

- [ ] **Step 1: Write failing test for sync decorator context filtering**

Add a new test class after the existing `TestR1DictGet` class:

```python
class TestR1FalsePositiveFiltering:
    """Tests for R1 false positive filtering — non-dict .get() patterns."""

    def test_fastapi_router_decorator_not_flagged(self) -> None:
        """@router.get('/path') decorator should NOT be flagged as R1."""
        source = dedent("""
            from fastapi import APIRouter
            router = APIRouter()

            @router.get("/items")
            def list_items():
                return []
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 0, f"Decorator .get() should not be flagged: {r1_findings}"
```

- [ ] **Step 2: Write failing test for async decorator context filtering**

```python
    def test_async_router_decorator_not_flagged(self) -> None:
        """@router.get('/path') on async def should NOT be flagged as R1."""
        source = dedent("""
            from fastapi import APIRouter
            router = APIRouter()

            @router.get("/items")
            async def list_items():
                return []
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 0, f"Async decorator .get() should not be flagged: {r1_findings}"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/scripts/cicd/test_enforce_tier_model.py::TestR1FalsePositiveFiltering::test_fastapi_router_decorator_not_flagged tests/unit/scripts/cicd/test_enforce_tier_model.py::TestR1FalsePositiveFiltering::test_async_router_decorator_not_flagged -v`

Expected: FAIL — current implementation flags all `.get()` calls

- [ ] **Step 4: Write failing test for httpx client.get() filtering**

```python
    def test_httpx_client_get_not_flagged(self) -> None:
        """client.get('http://...') HTTP method should NOT be flagged as R1."""
        source = dedent("""
            import httpx

            def fetch_data():
                client = httpx.Client()
                response = client.get("https://api.example.com/data")
                return response.json()
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 0, f"HTTP .get() should not be flagged: {r1_findings}"
```

- [ ] **Step 5: Write failing test for URL path argument filtering**

```python
    def test_get_with_url_path_not_flagged(self) -> None:
        """.get('/path') with URL-like arg should NOT be flagged as R1."""
        source = dedent("""
            def make_request(client):
                return client.get("/api/v1/users")
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 0, f"URL path .get() should not be flagged: {r1_findings}"
```

- [ ] **Step 6: Write test documenting f-string URL limitation**

```python
    def test_fstring_url_still_flagged(self) -> None:
        """f-string URLs are NOT filtered — intentional limitation.

        We cannot statically analyze f-strings to determine if they produce
        URL-like values. These remain flagged and must be allowlisted if
        they are legitimate HTTP calls.
        """
        source = dedent("""
            def make_request(client, user_id):
                return client.get(f"/api/v1/users/{user_id}")
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 1, "f-string URLs should still be flagged (limitation)"
```

- [ ] **Step 7: Write failing test for SDK collection.get() filtering**

```python
    def test_collection_get_with_ids_kwarg_not_flagged(self) -> None:
        """collection.get(ids=[...]) SDK pattern should NOT be flagged as R1."""
        source = dedent("""
            def retrieve_documents(collection):
                result = collection.get(ids=["doc1", "doc2"])
                return result
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 0, f"SDK .get(ids=) should not be flagged: {r1_findings}"
```

- [ ] **Step 8: Write regression test for limit/offset kwargs**

```python
    def test_limit_kwarg_still_flagged(self) -> None:
        """Generic kwargs like limit= must NOT suppress R1.

        limit/offset are common ORM pagination kwargs (SQLAlchemy, Django).
        A call like session.get(..., limit=10) should still be flagged.
        Only ChromaDB-specific kwargs (ids, include, where) are exempted.
        """
        source = dedent("""
            def paginated_query(session, key):
                return session.get(key, limit=10)
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 1, "Generic limit= kwarg should NOT suppress R1"
```

- [ ] **Step 9: Write test to ensure real dict.get() is still flagged**

```python
    def test_real_dict_get_still_flagged(self) -> None:
        """Actual dict.get('key', default) must still be flagged."""
        source = dedent("""
            def process(data):
                value = data.get("key", "default")
                return value
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 1, "Real dict.get() must be flagged"
        assert r1_findings[0].line == 3

    def test_dict_get_without_default_still_flagged(self) -> None:
        """dict.get('key') without default must still be flagged."""
        source = dedent("""
            config = {"setting": True}
            value = config.get("setting")
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 1, "dict.get() without default must be flagged"
```

- [ ] **Step 10: Write test for edge case — ambiguous patterns still flagged**

```python
    def test_ambiguous_get_still_flagged(self) -> None:
        """Ambiguous .get() calls should still be flagged (conservative)."""
        source = dedent("""
            def process(obj):
                # Could be dict or something else — flag it
                return obj.get("field")
        """)
        findings = parse_and_visit(source)

        r1_findings = [f for f in findings if f.rule_id == "R1"]
        assert len(r1_findings) == 1, "Ambiguous .get() should be flagged"
```

- [ ] **Step 11: Run all new tests to confirm expected pass/fail pattern**

Run: `pytest tests/unit/scripts/cicd/test_enforce_tier_model.py::TestR1FalsePositiveFiltering -v`

Expected results:
- FAIL: `test_fastapi_router_decorator_not_flagged`
- FAIL: `test_async_router_decorator_not_flagged`
- FAIL: `test_httpx_client_get_not_flagged`
- FAIL: `test_get_with_url_path_not_flagged`
- PASS: `test_fstring_url_still_flagged` (documents limitation)
- FAIL: `test_collection_get_with_ids_kwarg_not_flagged`
- PASS: `test_limit_kwarg_still_flagged` (regression guard)
- PASS: `test_real_dict_get_still_flagged`
- PASS: `test_dict_get_without_default_still_flagged`
- PASS: `test_ambiguous_get_still_flagged`

- [ ] **Step 12: Commit test scaffolding**

```bash
git add tests/unit/scripts/cicd/test_enforce_tier_model.py
git commit -m "test: add R1 false positive filtering tests (red)

Tests for distinguishing non-dict .get() patterns:
- FastAPI decorator @router.get() (sync and async)
- httpx client.get() HTTP methods
- URL path arguments
- ChromaDB collection.get(ids=) patterns

Regression tests to ensure we don't over-filter:
- f-string URLs (intentional limitation, still flagged)
- Generic limit= kwargs (ORM patterns, still flagged)
- Real dict.get() and ambiguous patterns

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Implement Consolidated Predicate with Decorator Detection

**Files:**
- Modify: `scripts/cicd/enforce_tier_model.py:276-350` (add tracking for decorator context)
- Modify: `scripts/cicd/enforce_tier_model.py:380-390` (replace with consolidated predicate)

- [ ] **Step 1: Add decorator line tracking to the visitor**

In the `TierModelVisitor.__init__` method (around line 276), add a set to track decorator lines:

```python
def __init__(self, filename: str, source_lines: list[str]) -> None:
    self.filename = filename
    self.source_lines = source_lines
    self.findings: list[Finding] = []
    self.symbol_stack: list[str] = []
    self.type_checking_lines: set[int] = set()
    self._decorator_lines: set[int] = set()  # Track lines that are decorators
```

- [ ] **Step 2: Collect decorator lines in visit_FunctionDef and visit_AsyncFunctionDef**

Update the `visit_FunctionDef` method (around line 343) to collect decorator line numbers:

```python
def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
    # Collect decorator lines — .get() calls here are not dict access
    for decorator in node.decorator_list:
        self._decorator_lines.add(decorator.lineno)
    self.symbol_stack.append(node.name)
    self.generic_visit(node)
    self.symbol_stack.pop()
```

Also update `visit_AsyncFunctionDef` (around line 347) similarly:

```python
def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
    # Collect decorator lines — .get() calls here are not dict access
    for decorator in node.decorator_list:
        self._decorator_lines.add(decorator.lineno)
    self.symbol_stack.append(node.name)
    self.generic_visit(node)
    self.symbol_stack.pop()
```

- [ ] **Step 3: Add consolidated predicate method**

Add this method to `TierModelVisitor` class (before `visit_Call`):

```python
def _is_likely_non_dict_get(self, node: ast.Call) -> bool:
    """Return True if this .get() call is likely NOT a dict.get().

    Heuristics (conservative — only skip when confident):
    1. Decorator context: @router.get("/path") is a route decorator
    2. URL-like first arg: client.get("https://...") is an HTTP method
    3. ChromaDB keywords: collection.get(ids=[...]) is SDK retrieval

    Note: f-string URLs (client.get(f"/api/{id}")) are NOT filtered
    because we cannot statically determine their runtime value.
    These must be allowlisted if they are legitimate HTTP calls.
    """
    # Heuristic 1: Decorator context
    if node.lineno in self._decorator_lines:
        return True

    # Heuristic 2: URL-like first argument
    if node.args:
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            val = first_arg.value
            if val.startswith(("/", "http://", "https://")):
                return True

    # Heuristic 3: ChromaDB-specific keywords
    # IMPORTANT: Only include keywords that are unambiguous to ChromaDB/vector DBs.
    # Generic pagination keywords (limit, offset) are NOT included because they
    # collide with SQLAlchemy, Django ORM, and other common patterns.
    chromadb_keywords = {"ids", "include", "where"}
    call_keywords = {kw.arg for kw in node.keywords if kw.arg is not None}
    if call_keywords & chromadb_keywords:
        return True

    return False
```

- [ ] **Step 4: Update visit_Call to use the consolidated predicate**

Replace the R1 detection in `visit_Call` (line 382-390):

```python
# R1: dict.get() - Call(func=Attribute(attr="get"))
if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
    if not self._is_likely_non_dict_get(node):
        self._add_finding(
            "R1",
            node,
            f"Potential dict.get() usage: {self._get_code_snippet(node.lineno)}",
        )
```

- [ ] **Step 5: Run the decorator tests to verify they pass**

Run: `pytest tests/unit/scripts/cicd/test_enforce_tier_model.py::TestR1FalsePositiveFiltering::test_fastapi_router_decorator_not_flagged tests/unit/scripts/cicd/test_enforce_tier_model.py::TestR1FalsePositiveFiltering::test_async_router_decorator_not_flagged -v`

Expected: PASS

- [ ] **Step 6: Run the URL tests to verify they pass**

Run: `pytest tests/unit/scripts/cicd/test_enforce_tier_model.py::TestR1FalsePositiveFiltering::test_httpx_client_get_not_flagged tests/unit/scripts/cicd/test_enforce_tier_model.py::TestR1FalsePositiveFiltering::test_get_with_url_path_not_flagged -v`

Expected: PASS

- [ ] **Step 7: Run the SDK keyword test to verify it passes**

Run: `pytest tests/unit/scripts/cicd/test_enforce_tier_model.py::TestR1FalsePositiveFiltering::test_collection_get_with_ids_kwarg_not_flagged -v`

Expected: PASS

- [ ] **Step 8: Run all filtering tests to verify implementation**

Run: `pytest tests/unit/scripts/cicd/test_enforce_tier_model.py::TestR1FalsePositiveFiltering -v`

Expected: All 10 tests PASS

- [ ] **Step 9: Run full R1 test suite to ensure no regressions**

Run: `pytest tests/unit/scripts/cicd/test_enforce_tier_model.py::TestR1DictGet tests/unit/scripts/cicd/test_enforce_tier_model.py::TestR1FalsePositiveFiltering -v`

Expected: All tests PASS

- [ ] **Step 10: Commit consolidated predicate implementation**

```bash
git add scripts/cicd/enforce_tier_model.py
git commit -m "feat(tier-model): add _is_likely_non_dict_get() predicate for R1 filtering

Consolidated heuristics to reduce R1 false positives:
1. Decorator context — @router.get('/path') is a route decorator
2. URL-like first arg — client.get('https://...') is an HTTP method
3. ChromaDB keywords — collection.get(ids=[...]) is SDK retrieval

Intentional limitations:
- f-string URLs not filtered (cannot determine runtime value)
- Generic kwargs (limit, offset) not filtered (ORM collision risk)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Verify and Clean Up Allowlist Entries

**Files:**
- Modify: `config/cicd/enforce_tier_model/web.yaml`
- Modify: `config/cicd/enforce_tier_model/plugins.yaml`

- [ ] **Step 1: Capture baseline R1 fingerprints before cleanup**

Run this to capture the current state of R1-related allowlist entries:

```bash
grep -h ":R1:" config/cicd/enforce_tier_model/*.yaml | grep "^- key:" | sort > /tmp/r1_entries_before.txt
wc -l /tmp/r1_entries_before.txt
```

Note the count for comparison.

- [ ] **Step 2: Run enforcement without allowlist to see what the heuristics now filter**

```bash
python scripts/cicd/enforce_tier_model.py check --root src/elspeth 2>&1 | grep ":R1:" | sort > /tmp/r1_findings_after_heuristics.txt
wc -l /tmp/r1_findings_after_heuristics.txt
```

Compare to a baseline run (if available) to see which findings are now filtered.

- [ ] **Step 3: Identify removable entries in web.yaml**

Search for entries with "False positive" in the reason:

```bash
grep -n "False positive.*router.get\|False positive.*httpx" config/cicd/enforce_tier_model/web.yaml
```

These are candidates for removal.

- [ ] **Step 4: Remove FastAPI decorator false positive entries from web.yaml**

Remove entries matching this pattern (approximately 10-15 entries):
```yaml
- key: web/catalog/routes.py:R1:list_sources:fp=ccdc6bb55bdef578
  owner: web-catalog
  reason: False positive — @catalog_router.get() is FastAPI route decorator, not dict.get()
  safety: No dict access involved
  expires: null
```

- [ ] **Step 5: Remove httpx false positive entries from web.yaml**

Remove entries matching this pattern (approximately 2-5 entries):
```yaml
- key: web/auth/oidc.py:R1:JWKSTokenValidator:ensure_jwks:fp=5d9ec6c99446cd42
  owner: web-auth
  reason: False positive — httpx.AsyncClient.get() is an HTTP GET request, not dict.get()
  safety: HTTP errors caught by raise_for_status() and except block
  expires: null
```

- [ ] **Step 6: Remove ChromaDB/SDK false positive entries from plugins.yaml**

Remove entries matching this pattern (approximately 2-3 entries):
```yaml
- key: plugins/sinks/chroma_sink.py:R1:ChromaSink:write:fp=54913df13e849326
  owner: feature
  reason: Collection.get() is the ChromaDB SDK method for retrieving documents by ID, not dict.get()
  safety: Returns GetResult dict with "ids" key; subscript access on result["ids"] crashes if ChromaDB SDK contract breaks
  expires: null
```

- [ ] **Step 7: Run enforcement to verify no new failures**

Run: `python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`

Expected: EXIT 0 (no unallowlisted findings)

- [ ] **Step 8: Capture R1 fingerprints after cleanup and compare**

```bash
grep -h ":R1:" config/cicd/enforce_tier_model/*.yaml | grep "^- key:" | sort > /tmp/r1_entries_after.txt
wc -l /tmp/r1_entries_after.txt

# Show the diff
diff /tmp/r1_entries_before.txt /tmp/r1_entries_after.txt
```

Verify that approximately 20-30 entries were removed.

- [ ] **Step 9: Commit allowlist cleanup**

```bash
git add config/cicd/enforce_tier_model/web.yaml config/cicd/enforce_tier_model/plugins.yaml
git commit -m "chore(tier-model): remove R1 false positive allowlist entries

Removed allowlist entries that are now correctly filtered by heuristics:
- FastAPI @router.get() decorators (decorator line detection)
- httpx client.get() HTTP methods (URL-like argument detection)
- ChromaDB collection.get() SDK calls (ids= keyword detection)

Entries removed: $(wc -l < /tmp/r1_entries_before.txt) -> $(wc -l < /tmp/r1_entries_after.txt)

Remaining R1 entries are legitimate Tier 3 boundary dict.get() calls
that require individual review (OIDC claims, JSON schema parsing, etc.).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Final Verification and Documentation

**Files:**
- Modify: `scripts/cicd/enforce_tier_model.py` (update docstring if needed)

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/unit/scripts/cicd/test_enforce_tier_model.py -v`

Expected: All tests PASS

- [ ] **Step 2: Run enforcement on full codebase**

Run: `python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`

Expected: EXIT 0

- [ ] **Step 3: Verify final R1 allowlist entry count**

```bash
grep -c ":R1:" config/cicd/enforce_tier_model/*.yaml
```

Compare to the baseline captured in Task 3, Step 1.

- [ ] **Step 4: Update RULES docstring**

Update the R1 rule description at line 172-175 to mention the auto-filtering:

```python
"R1": {
    "name": "dict.get",
    "description": "dict.get() usage can hide missing key bugs (decorators, HTTP methods, and ChromaDB calls are auto-filtered)",
    "remediation": "Access dict keys directly (dict[key]) and fix the schema/contract if KeyError occurs",
},
```

- [ ] **Step 5: Commit final updates**

```bash
git add scripts/cicd/enforce_tier_model.py
git commit -m "docs(tier-model): update R1 rule description for auto-filtering

Mention that decorators, HTTP methods, and ChromaDB calls are auto-filtered.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Close the Filigree issue**

```bash
filigree close elspeth-ba868c3896 --reason="Implemented R1 false positive filtering — allowlist entries reduced, heuristics documented"
```

---

## Summary

This plan implements a consolidated `_is_likely_non_dict_get()` predicate with three heuristics:

1. **Decorator context** — tracks decorator lines and skips `.get()` calls on those lines
2. **URL-like arguments** — skips when first arg is a string literal starting with `/` or `http`
3. **ChromaDB keywords** — skips when call uses `ids=`, `include=`, or `where=` (narrowed from original set)

**Intentional limitations documented in tests:**
- f-string URLs (`client.get(f"/api/{id}")`) are NOT filtered — cannot determine runtime value
- Generic kwargs (`limit=`, `offset=`) are NOT filtered — collision risk with ORM patterns

The heuristics are conservative: ambiguous cases are still flagged, ensuring no false negatives for actual dict.get() defensive patterns.

**Review amendments incorporated:**
- Narrowed SDK keywords per architecture/systems/python reviewers
- Consolidated to single predicate per python reviewer
- Added regression tests per QA reviewer
- Added fingerprint capture verification per systems reviewer

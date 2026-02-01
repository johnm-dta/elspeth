# Bug Report: HTTP Client Has Silent JSON Parse Fallback

## Summary

- The HTTP client silently falls back to raw text when JSON parsing fails, swallowing the parse error. This hides malformed API responses and makes debugging external call failures difficult.

## Severity

- Severity: major
- Priority: P1 (RC-1 Blocker - diagnostics)

## Reporter

- Name or handle: Release Validation Analysis
- Date: 2026-01-29
- Related run/issue ID: QW-02, TD-008

## Environment

- Commit/branch: fix/P2-aggregation-metadata-hardcoded
- OS: Linux
- Python version: 3.12+
- Config profile / env vars: Any pipeline using HTTP transforms
- Data set or fixture: External API returning malformed JSON

## Agent Context (if relevant)

- Goal or task prompt: RC-1 release validation - identify blockers
- Model/version: Claude Opus 4.5
- Tooling and permissions: Read-only analysis
- Determinism details: N/A
- Notable tool calls or steps: Cross-referenced RC1-remediation.md, http.py

## Steps To Reproduce

1. Configure a pipeline with an HTTP transform
2. Point it at an endpoint that returns malformed JSON (or HTML error page)
3. Run the pipeline
4. Check audit trail for the external call

## Expected Behavior

- JSON parse failure should be recorded with:
  - Error type: `json_parse_failed`
  - Content-Type header value
  - Preview of the malformed response body
- TransformResult.error() returned so row can be routed appropriately

## Actual Behavior

- Parse error silently caught
- Raw text stored as response (may be HTML error page, etc.)
- No indication in audit trail that JSON parsing failed
- Downstream code may fail cryptically when expecting dict but getting string

## Evidence

- `src/elspeth/plugins/clients/http.py:164-169`:
  ```python
  # CURRENT (problematic):
  except Exception:
      response_body = response.text  # Silent fallback
  ```

## Impact

- User-facing impact: Difficult to debug external API failures
- Data integrity / security impact: Audit trail doesn't record parse failure
- Performance or cost impact: Time wasted debugging downstream failures

## Root Cause Hypothesis

- Defensive programming pattern added to "handle any response"
- Violates Three-Tier Trust Model: external data should be validated at boundary, not silently coerced

## Proposed Fix

- Code changes (modules/files):
  ```python
  # src/elspeth/plugins/clients/http.py:164-169

  # BEFORE (problematic):
  except Exception:
      response_body = response.text

  # AFTER (correct):
  except json.JSONDecodeError as e:
      logger.warning(
          "JSON parse failed",
          content_type=response.headers.get("content-type"),
          body_preview=response.text[:200],
          error=str(e),
      )
      return TransformResult.error({
          "reason": "json_parse_failed",
          "error": str(e),
          "content_type": response.headers.get("content-type"),
          "body_preview": response.text[:200],
      })
  ```

- Config or schema changes: None

- Tests to add/update:
  - `test_http_client_json_parse_error_not_silent`
  - Mock endpoint returning HTML error page
  - Verify TransformResult.error() returned with details

- Risks or migration steps:
  - Low risk for new pipelines
  - Existing pipelines that accidentally worked with malformed JSON will now fail explicitly
  - This is correct behavior - silent failures are worse than explicit failures

## Architectural Deviations

- Spec or doc reference: CLAUDE.md "Three-Tier Trust Model" - External data (Tier 3) should be validated at boundary
- Observed divergence: External data silently coerced instead of validated
- Reason (if known): Overly defensive programming

## Verification Criteria

- [x] JSON parse failure returns TransformResult.error()
- [x] Error includes content_type and body_preview
- [x] Audit trail records the parse failure
- [x] No silent fallback to raw text

## Resolution

**Already fixed - bug report inaccurate about current implementation.**

The bug report described a pattern that does not exist in the current codebase. The actual implementation has proper error handling at multiple layers:

**HTTP Client Layer (`plugins/clients/http.py:229-249`):**
- Catches `JSONDecodeError` specifically (not broad `Exception`)
- Logs warning with URL, status_code, body_preview, and error
- Records `{_json_parse_failed: True, _error: ..., _raw_text: ...}` in audit trail

**Transform Layer (openrouter.py:294-305, openrouter_multi_query.py:527-537):**
- Wraps `response.json()` in try/except
- Returns `TransformResult.error()` with `reason="invalid_json_response"`
- Includes `content_type` and `body_preview` in error details
- Row is routed appropriately (to error handling/quarantine)

**Test coverage:**
- `test_openrouter.py::test_invalid_json_response_emits_error` - PASSED
- Verifies error includes content_type="text/html" and body_preview

**Note:** The bug report described `except Exception: response_body = response.text` which does not exist in the codebase. The HTTP client catches `JSONDecodeError` specifically and records structured error data.

## Cross-References

- RC1-remediation.md: QW-02, TD-008
- CLAUDE.md: Three-Tier Trust Model, External Call Boundaries
- docs/release/rc1-checklist.md: Section 6.3

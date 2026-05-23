# CI/CD Overflow — `web/composer/service.py` Cleanup (2026-05-23)

Cluster-3 overflow: 4 lint events blocking lint-green state. Brief stated
"5 items (3 unallowed + 2 stale, symbols deleted)" but verification showed
**4 events** and the symbols were **not** deleted — the 2 "stale" entries
were AST-shift fingerprint rotations (per `feedback_ast_shift_fingerprint_rotation`),
not dead weight.

## Per-item verdict

| Fingerprint / Line | Rule | Verdict | Notes |
|---|---|---|---|
| `web.yaml` L1990 `fp=be827cf6910d415f` (`_call_llm` stale) | R2 | **RENEWED** → `fp=94c34bdaa32431a8` | Symbol intact at `service.py:3717`; same LiteLLM `BadRequestError.status_code` boundary; reason/safety/expires unchanged. |
| `web.yaml` L1999 `fp=d5ac8e63a7f0492b` (`_call_text_llm` stale) | R2 | **RENEWED** → `fp=abee4a6dabdcbb01` | Symbol intact at `service.py:3751`; same LiteLLM `BadRequestError.status_code` boundary; reason/safety/expires unchanged. |
| `service.py` L3622 `fp=e183f4ada88826c2` | R1 | **FIX-CODE** | Internal dict `self._schemas_loaded_by_session: dict[str, set[tuple[str, str]]]`; no trust boundary. Replaced `.get(session_id)` + `is None` check with `session_id not in dict` membership-check. |
| `service.py` L3641 `fp=813e1a7c3f425032` | R8 | **FIX-CODE** | Same internal dict. Replaced `setdefault(session_id, set())` with explicit `if session_id not in dict: dict[session_id] = set()` followed by direct index. |

Lines 3742 / 3774 (R2 `getattr(exc, "status_code", None)`) were already covered
by the now-renewed allowlist entries — they are the same construct the existing
reason text described.

## Source changes

`src/elspeth/web/composer/service.py`:

- `_loaded_plugin_schemas_for_session` (around L3620): swapped `.get()`-and-None-test
  for membership check + direct index.
- `_mark_plugin_schema_loaded` (around L3640): swapped `setdefault` for explicit
  presence check + direct assignment.

Both are pure refactors with identical semantics — the dict is internal Tier-1
state, no behaviour change.

## Allowlist (`config/cicd/enforce_tier_model/web.yaml`)

- 2 entries **modified in place** (fingerprint hash only; key path,
  owner, reason, safety, expires preserved verbatim).
- 0 entries added, 0 entries removed.
- Permanent-vs-bounded split: unchanged (both renewed entries retain
  `expires: '2026-08-23'` — bounded).

## Verification

- `elspeth_lints check --rules trust_tier.tier_model`: composer/service.py
  findings = **0**; no stale warnings against composer/service.py.
- `pytest tests/unit/web/composer/ tests/integration/web/`: **2393 passed**,
  4 unrelated `InsecureKeyLengthWarning` (test fixture key length).

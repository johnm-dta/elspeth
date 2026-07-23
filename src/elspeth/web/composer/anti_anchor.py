"""§7.7 anti-anchor hints: break LLM retry loops with no progress.

When the composer LLM retries the same tool with the same canonical arguments
three times in a row, it has stopped reading new evidence — a context-anchored
loop. The original §7.7 repair targeted that byte-identical pattern. Later
audit of the captured Tier 1 final cohort RED (`53bc3cf2-…`, see
`notes/composer-tier1.5-step-c-diagnosis-2026-05-06.md`) proved the actual
payloads were distinct: the model drifted across failed `set_pipeline` calls,
then surrendered to a text-only "I'm stuck" reply. The tracker therefore covers
both anchored identity and same-tool drift without progress.

The fix is small: after N=3 consecutive identical failures, inject a
`role="user"` message before the next LLM call telling the model to enumerate
which fields the validator named, what shape they expect, and to change at
least one of them. After firing, the tracker resets so the hint doesn't repeat
on the same anchor — a 4th identical failure won't re-trigger; only a NEW run
of 3 identical failures (after at least one success or different-payload
failure has reset the chain) will fire again.

The hint is `role="user"` rather than `role="tool"` because tool-role messages
require a `tool_call_id` matched to a prior assistant tool_call envelope; a
free-floating system-injected message has no such id. Wrapping in a
[ELSPETH-SYSTEM-HINT] prefix makes the system-injected origin visible in
chat_messages persistence (audit trail) and to operators inspecting transcripts.

Detection criteria (deliberately conservative):

- Failure means: any non-success outcome — JSON-decode failure, non-dict args,
  canonicalization rejection, MissingRequiredPaths, ToolArgumentError, or a
  successful dispatch whose `result.success == False` (e.g., set_pipeline
  whose validation rejects the resulting state).
- Identical means: same `(tool_name, arguments_hash)` pair where
  `arguments_hash` is the canonical SHA-256 from `DispatchAudit.arguments_hash`.
- Three-in-a-row means: the LAST 3 entries in the bounded failure deque are
  all the same `(tool_name, hash)` pair. Earlier non-matching entries do not
  poison the trigger; the deque length is bounded so stale entries age out.
- Drift without convergence means: the LAST 3 entries are failures for the same
  tool but at least two distinct argument hashes. The model is changing payloads
  but not making progress, so it needs a different repair strategy rather than
  more small variations.
- Any tool *success* clears the deque immediately. The system has, by
  definition, just made progress.
"""

from __future__ import annotations

from collections import deque

_RETRY_THRESHOLD = 3
_DEQUE_MAXLEN = 5

FailureKey = tuple[str, str]
FailureDeque = deque[FailureKey]


def should_inject_hint(failures: FailureDeque) -> bool:
    """Return True iff the last `_RETRY_THRESHOLD` failures are identical."""
    if len(failures) < _RETRY_THRESHOLD:
        return False
    last = list(failures)[-_RETRY_THRESHOLD:]
    return all(entry == last[0] for entry in last)


def should_inject_drift_hint(failures: FailureDeque) -> bool:
    """Return True iff the same tool has failed with drifting payloads."""
    if len(failures) < _RETRY_THRESHOLD:
        return False
    last = list(failures)[-_RETRY_THRESHOLD:]
    tool_names = {tool_name for tool_name, _arguments_hash in last}
    argument_hashes = {arguments_hash for _tool_name, arguments_hash in last}
    return len(tool_names) == 1 and len(argument_hashes) > 1


def build_anti_anchor_hint(failures: FailureDeque) -> str:
    """Compose the hint string for the current anchor.

    Always ends with a safe structural scaffold: identify validator-named
    fields, mismatch categories, and expected schema shapes without restating
    prior literal values. Generic-by-design — works for connection-naming AND
    options-validation failures (the diagnosed RED was the latter, not the
    former, so the hint cannot be connection-name-specific).
    """
    if len(failures) < _RETRY_THRESHOLD:
        # Caller mis-used the helper; defensive RAISE rather than silent default
        # since the hint message must always be meaningful.
        raise ValueError(f"build_anti_anchor_hint requires {_RETRY_THRESHOLD}+ identical failures; got {len(failures)}")
    tool_name = failures[-1][0]
    return (
        f"[ELSPETH-SYSTEM-HINT] Your last {_RETRY_THRESHOLD} calls to `{tool_name}` "
        "all failed with byte-identical arguments and the same error. "
        "Retrying with the same payload will keep failing. "
        "Before the next attempt, reason only about redacted structure. "
        "Do not repeat, quote, summarize, or otherwise reproduce any literal values from prior arguments or errors "
        "in assistant prose. Identify only: "
        "(a) which field names the validator named in the error, "
        "(b) each field's structural mismatch category (for example missing, type, shape, allowed-key, or cardinality), "
        "(c) what schema shape each field expects "
        "(re-read the relevant `get_plugin_schema` result if you have not already), "
        "then submit a tool call that changes at least one structural choice."
    )


def build_drift_hint(failures: FailureDeque) -> str:
    """Compose a hint for same-tool drift without convergence."""
    if not should_inject_drift_hint(failures):
        raise ValueError("build_drift_hint requires 3 same-tool failures with different arguments")
    tool_name = failures[-1][0]
    return (
        f"[ELSPETH-SYSTEM-HINT] Your last {_RETRY_THRESHOLD} calls to `{tool_name}` "
        "all failed while sending different arguments. This is drift without convergence: "
        "small payload variations are not escaping the validator failure. "
        "Before the next attempt, stop varying fields opportunistically and choose a new repair strategy: "
        "reason only about field names, types, shapes, allowed keys, and cardinality; "
        "do not repeat, quote, summarize, or otherwise reproduce literal values from the failed payloads in assistant prose; "
        "(a) compare the last failed payloads structurally against the validator-named fields, "
        "(b) re-read the relevant schema or assistance result if the expected shape is unclear, "
        "(c) rebuild the complete requested topology in one coherent payload, or "
        "(d) surface a named gap if the requested shape is unsupported by the available tools. "
        "When the goal is a one-node linear insertion, switch to `splice_transform` instead of "
        "varying full-replacement payloads."
    )


class AntiAnchorTracker:
    """Maintains the bounded failure deque across the compose loop.

    The instance is created once per `compose()` call and lives for the
    lifetime of that call. State is *not* persisted between sessions — the
    anchor pattern is per-conversation, and a new compose() call gets a fresh
    tracker so an unrelated earlier failure cannot poison a new convergence.
    """

    def __init__(self) -> None:
        self._failures: FailureDeque = deque(maxlen=_DEQUE_MAXLEN)

    def record_failure(self, tool_name: str, arguments_hash: str) -> None:
        """Append a failure. Bounded deque drops the oldest beyond `_DEQUE_MAXLEN`."""
        self._failures.append((tool_name, arguments_hash))

    def record_success(self) -> None:
        """Clear the deque. Any tool success means we have made progress."""
        self._failures.clear()

    def should_fire(self) -> bool:
        return should_inject_hint(self._failures) or should_inject_drift_hint(self._failures)

    def build_hint(self) -> str:
        if should_inject_hint(self._failures):
            return build_anti_anchor_hint(self._failures)
        if should_inject_drift_hint(self._failures):
            return build_drift_hint(self._failures)
        return build_anti_anchor_hint(self._failures)

    def consume_fire(self) -> None:
        """Reset after a hint is injected to prevent immediate re-fire.

        The hint addressed the model; we wait for it to act. A 4th failure does
        NOT immediately re-fire — only a fresh run of N failures after the
        reset does.
        """
        self._failures.clear()

"""§7.6 Option C empty-state recovery nudge: bookkeeping for one-shot fire.

After the §7.6 empty-state finalize-time passthrough shipped (commit
2e2c7bc8), the residual ~20% RED rate manifests as: model produces a
no-tool-call prose reply on a structurally-empty state — i.e., it tried
to assemble the pipeline, the validator kept rejecting the build, and
the model surrendered with prose instead of looping further.

§7.7 anti-anchor catches the *byte-identical retry* sub-case (model locks
onto a payload). Option C addresses the orthogonal *drift-then-surrender*
sub-case: model varies arguments every attempt, never converges, gives
up. The recovery move is to inject a single GENERIC "fall back to a
minimal shape" nudge before finalizing, then let the loop run one more
turn. If that turn also produces no-tool-call → fall through to the
existing empty-state passthrough. Bounded by single-fire-per-compose.

The tracker shape mirrors AntiAnchorTracker: per-compose-call instance,
never shared across requests. A new compose() call (next user message)
gets a fresh tracker and CAN nudge again — each user message deserves
its own recovery affordance. Single-fire WITHIN a compose() call is
load-bearing: re-firing on a second arrival would loop indefinitely
since identical nudge content is unlikely to produce a different
outcome.
"""

from __future__ import annotations

_RECOVERY_NUDGE_CONTENT = (
    "[ELSPETH-RECOVERY-NUDGE] Your previous attempts to assemble the "
    "pipeline did not produce a valid build. Fall back to a minimal "
    "shape: one source, one transform (or pass-through), one output "
    "sink. Re-read get_plugin_schema for any plugin whose options you "
    "are unsure about. Confirm that source.options and outputs[N].options "
    "blocks are present for every plugin, even ones that look "
    "option-less (csv, json, etc.). Once a minimal pipeline validates "
    "via preview_pipeline, you can evolve from there."
)


class EmptyStateRecoveryTracker:
    """Per-compose-call single-fire guard for the recovery nudge.

    The instance is created once per ``compose()`` call and lives for
    the lifetime of that call. State is *not* persisted between
    sessions — a new compose() call gets a fresh tracker so each user
    message (HTTP request) gets its own at-most-one recovery attempt.

    Single-fire is enforced by ``has_fired()`` / ``record_fire()``: a
    second arrival at the no-tool-call branch in the same compose call
    finds ``has_fired() is True`` and falls through to the existing
    empty-state passthrough.
    """

    def __init__(self) -> None:
        self._fired: bool = False

    def has_fired(self) -> bool:
        return self._fired

    def record_fire(self) -> None:
        self._fired = True

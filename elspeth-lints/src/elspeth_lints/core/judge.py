"""LLM-judge gate for new allowlist entries.

This is the synchronous, single-vendor judge path described in the
``cicd-judge-cli`` prototype plan ("Pillar A — Judge"). Agents call
``call_judge`` with a structured ``JudgeRequest`` (the proposed suppression
metadata plus the surrounding code excerpt) and receive a ``JudgeResponse``
recording the model's verdict, the model's rationale verbatim, and the
model identity / timestamp that produced it.

Design constraints (from the prototype plan, all load-bearing):

* The judge **reads** code + rationale and decides ``ACCEPTED`` /
  ``BLOCKED``. It does **not** propose code fixes. A ``BLOCKED`` verdict
  returns the failure to the agent; the agent figures out remediation.
* The judge is rule-agnostic — it consumes ``(file, rule_id, symbol,
  rationale, surrounding_code)`` and the input shape does not bake in
  ``tier_model``-specific assumptions. This is the abstraction that
  ports to ``wardline``.
* The judge's rationale is the new audit primitive. It is recorded
  verbatim and is independently re-readable months later when an auditor
  asks "why did we exempt this?". The YAML answers without re-running
  the model.
* The third verdict, ``OVERRIDDEN_BY_OPERATOR``, is **set by the CLI**
  rather than emitted by the model. It records that a human exercised
  authority to bypass the judge, with the model's own
  verdict-as-rationale preserved so the bypass is visible in audit.
* Malformed responses crash — per the project's offensive-programming
  policy, silently coercing a malformed judge response into a
  default-shaped one would destroy the audit primitive's integrity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from elspeth_lints.core.allowlist import JudgeVerdict

# Default model identifier. The prototype is single-vendor by design;
# wardline will make this configurable per-project.
DEFAULT_JUDGE_MODEL: str = "claude-opus-4-7"

# System prompt the judge sees. Worded for explicit role boundaries:
# the model is a reviewer, not a fixer; verdict structure is named;
# blocking is preferred to fabrication when the rationale is shallow.
# The decorator-suggestion path (``should_use_decorator``) is the
# structural counterpart to BLOCK-without-suggestion: when the finding
# is a textbook Tier-3 boundary, the judge nudges the agent toward the
# in-source ``@trust_boundary`` decorator instead of letting them add
# yet another per-line YAML entry. See ``notes/cicd-judge-cli-prototype-plan.md``
# ("How the two pillars interact").
_SYSTEM_PROMPT: str = """\
You are the cicd-judge, an automated reviewer of proposed exemptions to
a static-analysis trust-tier rule. An agent (another LLM, or a human) is
asking to add an allowlist entry that suppresses a specific finding at
a specific code location. They have supplied a written rationale.

Your role:
- Read the surrounding code and the agent's rationale.
- Decide whether the rationale honestly explains why the suppression is
  legitimate at this site. Common legitimate reasons: a Tier-3 trust
  boundary (external data, user input, file/API/DB response) where
  defensive coercion or shape-checking is required; an offensive
  programming guard where the test branch is reached only on a real
  invariant violation that must surface a meaningful error; a TOCTOU
  race window where a defensive read precedes an atomic write.
- BLOCK when the rationale is vague, generic, contradicts the visible
  code, or attempts to suppress a finding that should be fixed in code
  rather than allowlisted (e.g. a regular defensive .get() inside a
  function that already received validated typed data).
- You do NOT propose a code fix. Your only outputs are a verdict and
  the reasoning behind it. If the suppression is wrong, the agent is
  responsible for figuring out remediation — refactor, broaden a
  per-file rule, move the boundary into a decorator, or abandon the
  suppression.

The ``@trust_boundary`` structural alternative
----------------------------------------------

The project ships an in-source decorator at
``src/elspeth/contracts/trust_boundary.py`` that marks a function as a
Tier-3 external-data trust boundary and suppresses the project's two
highest-volume Tier-3 defensive-pattern rules (``R1`` — silent
``.get()`` on external data; ``R5`` — ``isinstance`` shape guard at
the boundary) only on names derived from the decorator's named
``source_param``. The decorator is the structural replacement for
per-line YAML allowlist entries for the function-scoped Tier-3
boundary pattern.

For each block-candidate finding, ask three questions:

1. Does the finding occur inside a function that takes external or
   untrusted data as a parameter (an LLM tool-call ``arguments`` dict,
   a parsed JSON payload from a webhook, a row read from an external
   CSV, etc.)?
2. Is the finding's rule_id one of ``R1`` or ``R5`` — the rules the
   decorator is allowed to suppress?
3. Is the finding's subject rooted at one of that function's
   parameters — i.e. is there a clear data-flow path from the external
   parameter into the suppressed expression (via subscript,
   ``.get(...)``, iteration, unpacking, attribute, or assignment)?

If the answer to ALL THREE is yes, the correct remediation is the
decorator, not an allowlist entry. Emit:

* ``verdict``: ``"BLOCKED"`` (the allowlist entry must not land);
* ``should_use_decorator``: the parameter name (e.g. ``"arguments"``)
  that the agent should pass as the decorator's ``source_param``;
* ``rationale``: explain that this finding is a structural Tier-3
  boundary case and the remediation is
  ``@trust_boundary(tier=3, source=<one-line description of the
  external source>, source_param=<the parameter name>,
  suppresses=(<the rule_id>,), invariant=<what the function
  guarantees on malformed input>)`` on the enclosing function,
  followed by deletion of any related allowlist entries.

If ANY of the three answers is no — the finding is not in a boundary
function, the rule is not in {R1, R5}, or the subject is not rooted at
an external parameter — emit ``should_use_decorator: null`` and decide
``ACCEPTED`` or ``BLOCKED`` on the rationale's merits as before.

One caveat about excerpt visibility: the surrounding code excerpt is
truncated to ±15 lines around the finding. If you cannot see the
function's decorators in the excerpt, do not assume the function is
undecorated — prefer the no-suggestion path (``should_use_decorator:
null``) in that case. A wrong "use the decorator" nudge on an already-
decorated function wastes the agent's time and erodes trust in the
gate.

Examples
--------

Example A — should suggest the decorator (BLOCKED + should_use_decorator):

  Finding: R1 on ``arguments.get("nodes")`` at line 47 of
  ``web/composer/tools/sessions.py``.

  Excerpt shows::

      def _execute_set_pipeline(self, arguments: dict[str, Any]) -> ToolResult:
          nodes = arguments.get("nodes")  # <-- R1 reported here
          if not isinstance(nodes, list):
              raise ToolArgumentError("nodes must be a list")
          ...

  Agent rationale: "arguments comes from the composer model's tool call,
  external data, need to defensively read."

  Verdict: ``BLOCKED``. Reason: all three conditions met (function
  takes external ``arguments``; R1 is in the suppressible set; the
  reported subject is rooted at ``arguments``). Emit
  ``should_use_decorator: "arguments"`` and recommend the decorator
  in the rationale.

Example B — regular ACCEPT inside an already-decorated function:

  Finding: R1 on ``arguments.get("path")`` at line 52 of a function
  that already carries ``@trust_boundary(tier=3, source=...,
  source_param='arguments', suppresses=('R1', 'R5'), ...)``. The
  excerpt shows the decorator above the function definition.

  Agent rationale: "arguments comes from the composer model's tool
  call; behaviour documented in the decorator's invariant; finding
  is one the decorator was intended to cover but the static analyzer
  reported it anyway."

  Verdict: ``ACCEPTED``. The decorator already exists and the
  suppression scope is appropriate; the agent is not trying to
  broaden coverage. Emit ``should_use_decorator: null``.

Example C — regular BLOCK (rationale shallow, no decorator help):

  Finding: R1 on ``self._cache.get(key)`` inside a method that does
  NOT take external data as a parameter — ``self._cache`` is a
  process-local dict populated by code we control.

  Agent rationale: "defensive, might be None."

  Verdict: ``BLOCKED``. The decorator would not help (no external
  parameter; ``self._cache`` is not Tier-3 data). Emit
  ``should_use_decorator: null``; the rationale describes a code-fix
  task (use direct attribute access; let it KeyError if the cache
  invariant is broken), not a legitimate suppression.

Output schema: you must return a JSON object with exactly these fields,
and nothing else (no markdown fences, no prose preamble):

{
  "verdict": "ACCEPTED" | "BLOCKED",
  "rationale": "<your reasoning, in 2-6 sentences, recorded verbatim>",
  "should_use_decorator": "<parameter_name>" | null
}

``should_use_decorator`` is valid ONLY paired with ``verdict``:
``BLOCKED``. An ``ACCEPTED`` verdict must always carry
``should_use_decorator: null``.
"""

# Per-request user-message template. Kept as plain text concatenation
# (not f-string at module level) so the substitutions happen per call.
_USER_PROMPT_TEMPLATE: str = """\
File: {file_path}
Rule: {rule_id}
Symbol: {symbol}
Fingerprint: {fingerprint}

Agent's rationale for the suppression:
---
{rationale}
---

Surrounding code (the finding is approximately at the middle of this
excerpt):
---
{surrounding_code}
---

Return your verdict JSON now.
"""


class JudgeConfigurationError(RuntimeError):
    """The judge cannot be called because its environment is incomplete.

    Distinct from API or response errors — this signals an
    operator-actionable misconfiguration (missing SDK install or missing
    API key). Surfaces remediation guidance in the message.
    """


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    """Input to the judge.

    All fields are scalars; ``frozen=True`` is sufficient for
    immutability (no container fields require ``deep_freeze``).
    """

    file_path: str
    rule_id: str
    symbol: str
    fingerprint: str
    rationale: str
    surrounding_code: str


@dataclass(frozen=True, slots=True)
class JudgeResponse:
    """Output from the judge.

    ``verdict`` is the model's verdict (always ``ACCEPTED`` or ``BLOCKED``
    when returned by ``call_judge``). The CLI may later transform the
    surrounding ``AllowlistEntry`` to record ``OVERRIDDEN_BY_OPERATOR``;
    this dataclass captures what the *model* said, not the final
    write-time decision.

    ``judge_rationale`` is recorded verbatim and is the new audit
    primitive. ``should_use_decorator`` carries the structured
    decorator-suggestion signal: when non-None, it names the parameter
    the agent should pass as ``source_param`` to
    ``@trust_boundary`` on the enclosing function, and the verdict is
    always ``BLOCKED`` (enforced at parse time in ``call_judge``). When
    ``None``, the verdict stands on its own merits — ``ACCEPTED`` means
    the suppression lands; ``BLOCKED`` means the agent figures out
    remediation without a structured nudge.
    """

    verdict: JudgeVerdict
    model_id: str
    judge_rationale: str
    recorded_at: datetime
    should_use_decorator: str | None


def call_judge(
    request: JudgeRequest,
    *,
    model_id: str = DEFAULT_JUDGE_MODEL,
) -> JudgeResponse:
    """Send a judge request to Anthropic and return the parsed verdict.

    Raises:
        JudgeConfigurationError: SDK not installed, or
            ``ANTHROPIC_API_KEY`` env var missing.
        RuntimeError: API call failed, or the model returned a malformed
            response (missing fields, unexpected verdict value, bad JSON).
            We crash rather than coerce per the project's
            offensive-programming policy — a malformed judge response is
            never an acceptable audit primitive.
    """
    # Lazy import: keeping ``anthropic`` out of module-level scope means
    # importing ``elspeth_lints.core.judge`` does not require the SDK to
    # be installed; only callers of ``call_judge`` do. This mirrors the
    # lazy-import pattern ``cli._run_rotate`` uses for the rotate module.
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise JudgeConfigurationError(
            "The anthropic SDK is not installed. The `justify` subcommand "
            "requires it. Install with:\n\n"
            "    uv pip install -e 'elspeth-lints/[judge]'\n\n"
            "(from the repo root), or add `anthropic` to your dev "
            "environment."
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise JudgeConfigurationError(
            "ANTHROPIC_API_KEY is not set. The `justify` subcommand calls "
            "the Anthropic API to gate allowlist writes. Set the key in "
            "your shell environment (`export ANTHROPIC_API_KEY=sk-ant-...`) "
            "and re-run."
        )

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        file_path=request.file_path,
        rule_id=request.rule_id,
        symbol=request.symbol,
        fingerprint=request.fingerprint,
        rationale=request.rationale,
        surrounding_code=request.surrounding_code,
    )

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model_id,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw_text = _extract_text_block(message)
    parsed = _parse_judge_payload(raw_text)
    verdict = _verdict_from_string(parsed["verdict"])
    rationale = _required_str_field(parsed, "rationale")
    should_use_decorator = _optional_str_field(parsed, "should_use_decorator")

    # Cross-field contract: should_use_decorator is the structured
    # "use @trust_boundary instead" nudge. It is meaningful ONLY when the
    # verdict is BLOCKED (the entry must not land; do the structural fix
    # instead). An ACCEPTED verdict paired with a non-null
    # should_use_decorator is incoherent — the model would be saying both
    # "this allowlist entry is fine" and "you should use the decorator
    # instead". Crash per the offensive-programming policy: silently
    # ignoring the suggestion would erode the audit primitive.
    if should_use_decorator is not None and verdict is not JudgeVerdict.BLOCKED:
        raise RuntimeError(
            f"judge emitted should_use_decorator={should_use_decorator!r} with "
            f"verdict={verdict.value}; should_use_decorator is only valid with "
            "BLOCKED (the decorator suggestion is a structured alternative to "
            "the proposed allowlist entry, which only applies when the entry "
            "is being rejected)."
        )

    return JudgeResponse(
        verdict=verdict,
        model_id=model_id,
        judge_rationale=rationale,
        recorded_at=datetime.now(UTC),
        should_use_decorator=should_use_decorator,
    )


def _extract_text_block(message: Any) -> str:
    """Pull the single text block out of a ``messages.create`` response.

    The judge's response is constrained to one JSON object; the SDK
    returns a list of content blocks. We refuse to merge multiple
    blocks (that would silently flatten a model that started ignoring
    the instructions) and we refuse to coerce a non-text block.
    """
    content = message.content
    if not isinstance(content, list) or len(content) != 1:
        raise RuntimeError(
            f"judge response must be a single content block; got {len(content) if isinstance(content, list) else type(content).__name__}"
        )
    block = content[0]
    block_type = block.type
    if block_type != "text":
        raise RuntimeError(f"judge response block type must be 'text'; got {block_type!r}")
    text = block.text
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("judge response text block is empty")
    return text


def _parse_judge_payload(raw_text: str) -> dict[str, Any]:
    """Parse the model's JSON output, crashing loudly on malformation.

    We strip whitespace but do NOT attempt to extract JSON from
    surrounding markdown fences — the system prompt forbids them and a
    response that ignores that constraint is itself a malformation
    signal worth surfacing.
    """
    stripped = raw_text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"judge returned non-JSON response; refusing to coerce. raw: {stripped!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"judge JSON must be an object; got {type(parsed).__name__}")
    for required in ("verdict", "rationale", "should_use_decorator"):
        if required not in parsed:
            raise RuntimeError(f"judge JSON missing required field {required!r}; got keys={sorted(parsed.keys())}")
    return parsed


def _verdict_from_string(value: Any) -> JudgeVerdict:
    """Map the model's ``verdict`` string to ``JudgeVerdict``.

    Only ``ACCEPTED`` and ``BLOCKED`` are valid model-emitted values.
    ``OVERRIDDEN_BY_OPERATOR`` is set by the CLI, not by the model;
    seeing it from the model is a contract violation worth crashing on.
    """
    if not isinstance(value, str):
        raise RuntimeError(f"judge verdict must be a string; got {type(value).__name__}")
    if value == JudgeVerdict.ACCEPTED.value:
        return JudgeVerdict.ACCEPTED
    if value == JudgeVerdict.BLOCKED.value:
        return JudgeVerdict.BLOCKED
    raise RuntimeError(
        f"judge verdict must be ACCEPTED or BLOCKED (the model does not emit "
        f"OVERRIDDEN_BY_OPERATOR); got {value!r}"
    )


def _required_str_field(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"judge field {key!r} must be a non-empty string; got {value!r}")
    return value


def _optional_str_field(payload: dict[str, Any], key: str) -> str | None:
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"judge field {key!r} must be a non-empty string or null; got {value!r}")
    return value

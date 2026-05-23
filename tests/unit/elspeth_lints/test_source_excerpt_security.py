"""Adversarial tests for the source-excerpt outbound-content boundary.

Closes:
  * elspeth-9bbb9df9a5 / C1-2(c) + C2-2 — justify path:
      - path-traversal in --file-path or absolute escape
      - secrets in the ±15-line excerpt
  * elspeth-ebb2b88753 / C3-4 — reaudit sweep path:
      - forged path-traversal allowlist key
      - secrets in the ±15-line excerpt amplified across N entries

The threat shape these tests guard against: ELSPETH's
cicd-judge-cli is the FIRST outbound-content channel in the project
that ships local source bytes to a third-party LLM. Any actor with
write access to the allowlist YAML (or the operator CLI) could
otherwise cause an arbitrary file under the OS to be read and
forwarded to OpenRouter, OR cause a legitimate file's inline secrets
to leak unredacted. The tests in this file are deliberately
adversarial: each test simulates the attack and asserts the boundary
holds (no read, no exfil, or scrubbed-not-literal).

Test-file organisation:
  1. Path containment — helper unit tests + CLI integration + reaudit
     per-entry isolation.
  2. Secrets scrubber — per-pattern coverage + bypass-resistance + the
     audit-record shape.
  3. Bypass-resistance — every code path that builds a JudgeRequest
     must funnel through extract_safe_excerpt.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from elspeth_lints.core.cli import main
from elspeth_lints.core.reaudit import ReauditDivergence, reaudit_entries
from elspeth_lints.core.source_excerpt import (
    RedactionRecord,
    SafeExcerpt,
    SourceExcerptPathOutsideRootError,
    resolve_safe_excerpt_path,
    scrub_secrets,
)

# ---------- shared fixtures ----------

_SYNTHETIC_SOURCE = '''\
"""Synthetic module used in source-excerpt security tests."""


class Widget:
    def lookup(self, payload: dict) -> str:
        # R1: dict.get on Tier-2 data — the kind of finding an agent
        # might want to suppress with judge approval.
        return payload.get("name", "anonymous")
'''


def _build_source_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Lay out a minimal source root with one finding-producing file.

    Mirrors the helpers in test_justify.py / test_reaudit.py so the
    security tests exercise the same harness shape end-to-end.
    """
    root = tmp_path / "src_root"
    (root / "plugins").mkdir(parents=True)
    target = root / "plugins" / "widget.py"
    target.write_text(_SYNTHETIC_SOURCE, encoding="utf-8")
    return root, target


def _build_allowlist_dir(tmp_path: Path) -> Path:
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir(parents=True)
    (allowlist_dir / "_defaults.yaml").write_text(
        "version: 1\ndefaults:\n  fail_on_stale: false\n  fail_on_expired: false\n",
        encoding="utf-8",
    )
    return allowlist_dir


def _mock_openrouter_completion(*, verdict: str = "ACCEPTED", rationale: str = "ok") -> MagicMock:
    """OpenAI-shape chat-completion mock for OpenRouter routing."""
    message = MagicMock()
    message.content = json.dumps({"verdict": verdict, "rationale": rationale, "should_use_decorator": None})
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    completion.model = "anthropic/claude-opus-4"
    completion.usage = MagicMock(prompt_tokens=4000, prompt_tokens_details=MagicMock(cached_tokens=0))
    return completion


@contextmanager
def _mock_judge_call(*, verdict: str = "ACCEPTED", rationale: str = "ok") -> Iterator[MagicMock]:
    """Patch ``openai.OpenAI`` so the judge call runs offline.

    Yields the patched OpenAI class so tests can inspect what prompt
    text would have been sent — this is the bypass-resistance check
    point: assert the scrubbed text reached the mock, not the literal
    secret.
    """
    fake_completion = _mock_openrouter_completion(verdict=verdict, rationale=rationale)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_completion
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=fake_client) as client_class,
    ):
        yield client_class


def _captured_user_prompt(client_class: MagicMock) -> str:
    """Extract the user-message text from a single recorded OpenAI call.

    The judge sends two messages: a static system block (the policy)
    and a per-call user block carrying the excerpt. The user block is
    the only place the source excerpt is interpolated, so any
    bypass-resistance assertion that the scrubber actually ran needs
    to read this text.
    """
    fake_client = client_class.return_value
    assert fake_client.chat.completions.create.call_count == 1
    call = fake_client.chat.completions.create.call_args
    messages = call.kwargs["messages"]
    user_message = next(m for m in messages if m["role"] == "user")
    return str(user_message["content"][0]["text"])


def _write_forged_allowlist_entry(allowlist_dir: Path, *, forged_path: str) -> Path:
    """Write an allowlist entry whose key encodes a path-traversal file_path.

    Mirrors the shape ``_write_widget_lookup_entry`` produces but the
    file_path segment is attacker-controlled. The C8-3 binding
    fields are omitted (judge_verdict=None) because the loader's
    C8-3 gate would fire before reaudit even runs the per-entry
    dispatch; this test exercises the SOURCE_EXCERPT_REJECTED path
    via the pre-judge entry surface (which include_pre_judge=True
    activates) so the security gate inside _reaudit_one_entry is the
    one that catches the attack.
    """
    key = f"{forged_path}:R1:_module_:fp=deadbeef"
    lines = [
        "allow_hits:",
        f"- key: {key}",
        "  owner: attacker",
        "  reason: |-",
        "    forged path-traversal entry for security test",
        "  safety: |-",
        "    Suppression gated by cicd-judge; see judge_rationale below.",
        "  expires: '2030-01-01'",
    ]
    target = allowlist_dir / "_forged.yaml"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# =====================================================================
# 1. Path containment — helper unit tests
# =====================================================================


def test_resolve_safe_excerpt_path_accepts_in_root(tmp_path: Path) -> None:
    root, target = _build_source_tree(tmp_path)
    resolved = resolve_safe_excerpt_path(root=root, target_file=target)
    assert resolved == target.resolve()


def test_resolve_safe_excerpt_path_rejects_absolute_escape(tmp_path: Path) -> None:
    """The classic ``--file-path /etc/passwd`` attack.

    The attacker passes an absolute path outside the project root.
    ``resolve_safe_excerpt_path`` MUST raise
    ``SourceExcerptPathOutsideRootError`` before any read, and the
    raised exception is a ``ValueError`` subclass (NOT RuntimeError)
    so reaudit's transport-failure net does not swallow it.
    """
    root, _target = _build_source_tree(tmp_path)
    # Use a path we know exists on Linux but isn't under our root.
    escape = Path("/etc/passwd") if Path("/etc/passwd").exists() else Path("/etc/hostname")
    if not escape.exists():
        pytest.skip("no stable absolute file exists outside tmp_path for this test environment")
    with pytest.raises(SourceExcerptPathOutsideRootError) as exc_info:
        resolve_safe_excerpt_path(root=root, target_file=escape)
    # Both resolved paths appear in the message — operators correlate
    # the attacker's intent against the allowlist diff.
    msg = str(exc_info.value)
    assert str(escape.resolve()) in msg
    assert str(root.resolve()) in msg
    # Subclass of ValueError, NOT RuntimeError: this is the load-bearing
    # separation that keeps T6b's RuntimeError net from swallowing the
    # security signal.
    assert isinstance(exc_info.value, ValueError)
    assert not isinstance(exc_info.value, RuntimeError)


def test_resolve_safe_excerpt_path_rejects_relative_traversal(tmp_path: Path) -> None:
    """The ``../../../etc/passwd`` attack via the allowlist key file_path.

    The attacker forges an entry key whose file_path segment is a
    relative traversal that resolves outside root. We compute enough
    ``..`` segments to definitely escape the tmp tree so the test is
    not flaky on deep tmp paths.
    """
    root, _target = _build_source_tree(tmp_path)
    traversal_target = Path("/etc/passwd") if Path("/etc/passwd").exists() else Path("/etc/hostname")
    if not traversal_target.exists():
        pytest.skip("no stable absolute file exists outside tmp_path")
    # Compute the actual depth of root so we definitely escape it. The
    # production attack uses a fixed ``../../../`` shape that works on a
    # source tree at depth 3-ish; in the test harness, tmp paths are
    # deeper, so the test computes the right depth dynamically.
    resolved_root = root.resolve()
    depth = len(resolved_root.parts) - 1  # parts[0] is the filesystem root '/'
    candidate = root.joinpath(*([".."] * depth), str(traversal_target).lstrip("/"))
    with pytest.raises(SourceExcerptPathOutsideRootError):
        resolve_safe_excerpt_path(root=root, target_file=candidate)


def test_resolve_safe_excerpt_path_raises_file_not_found_for_missing_in_root(tmp_path: Path) -> None:
    """Missing-file-in-root is a separate signal from path-traversal.

    ``Path.resolve(strict=True)`` raises FileNotFoundError when the
    target does not exist. The caller distinguishes this from the
    security branch by exception type — operational vs security.
    """
    root, _target = _build_source_tree(tmp_path)
    missing = root / "plugins" / "does_not_exist.py"
    with pytest.raises(FileNotFoundError):
        resolve_safe_excerpt_path(root=root, target_file=missing)


# =====================================================================
# 1b. Path containment — CLI integration (justify path)
# =====================================================================


def test_justify_rejects_absolute_path_escape(tmp_path: Path) -> None:
    """`elspeth-lints justify --file-path /etc/passwd ...` exits non-zero.

    Adversarial: an operator (or malicious config-driven invocation)
    supplies an absolute path outside the project root. The CLI must
    surface a security-violation error, exit non-zero, and never call
    OpenRouter.
    """
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    escape = Path("/etc/passwd") if Path("/etc/passwd").exists() else Path("/etc/hostname")
    if not escape.exists():
        pytest.skip("no stable absolute file outside tmp_path")

    # Mock the OpenAI client and assert it is NEVER called.
    with _mock_judge_call() as client_class:
        exit_code = main(
            [
                "justify",
                "--root",
                str(root),
                "--allowlist-dir",
                str(allowlist_dir),
                "--file-path",
                str(escape),
                "--symbol",
                "_module_",
                "--rationale",
                "exfil probe",
                "--owner",
                "attacker",
            ]
        )
    assert exit_code == 2
    # The OpenAI client must NEVER be instantiated, much less called.
    # If the security gate runs before _scan_single_file_findings and
    # before call_judge, the patched openai.OpenAI is untouched.
    assert client_class.call_count == 0


def test_justify_rejects_relative_traversal(tmp_path: Path) -> None:
    """`elspeth-lints justify --file-path ../../../etc/passwd ...` exits non-zero.

    Adversarial: the file_path segment is a relative traversal that
    resolves outside root after joining.
    """
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    with _mock_judge_call() as client_class:
        exit_code = main(
            [
                "justify",
                "--root",
                str(root),
                "--allowlist-dir",
                str(allowlist_dir),
                "--file-path",
                "../../../etc/passwd",
                "--symbol",
                "_module_",
                "--rationale",
                "exfil probe",
                "--owner",
                "attacker",
            ]
        )
    assert exit_code == 2
    assert client_class.call_count == 0


# =====================================================================
# 1c. Path containment — reaudit per-entry isolation
# =====================================================================


def test_reaudit_classifies_forged_path_as_source_excerpt_rejected(tmp_path: Path) -> None:
    """A forged allowlist key with an absolute-path file_path is per-entry quarantined.

    Adversarial: an attacker with write access to the allowlist YAML
    inserts an entry whose canonical key is
    ``/etc/passwd:R1:_module_:fp=...``. Python's ``Path / "/etc/passwd"``
    returns the absolute path verbatim — the most realistic exfil
    shape because it doesn't depend on the depth of the tmp tree.
    The reaudit sweep MUST classify this entry as
    SOURCE_EXCERPT_REJECTED, MUST NOT read /etc/passwd, MUST NOT
    exfiltrate to OpenRouter, and MUST continue processing other
    (legitimate) entries.
    """
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    # Pick a file that exists outside root so the security branch (not
    # the missing-file branch) fires. /etc/passwd is the canonical
    # exfil target on Linux; /etc/hostname is a fallback.
    forged = "/etc/passwd" if Path("/etc/passwd").exists() else "/etc/hostname"
    if not Path(forged).exists():
        pytest.skip("no stable absolute file exists outside tmp_path")
    _write_forged_allowlist_entry(allowlist_dir, forged_path=forged)

    with _mock_judge_call() as client_class:
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=True,  # pre-judge so the C8-3 load gate doesn't fire first
        )

    assert len(report.outcomes) == 1
    outcome = report.outcomes[0]
    assert outcome.divergence is ReauditDivergence.SOURCE_EXCERPT_REJECTED
    # No exfil: the judge was never even attempted.
    assert client_class.call_count == 0
    # The audit record carries the security-error message so the
    # operator can see WHICH path was rejected.
    assert outcome.fresh_rationale is not None
    assert "outside" in outcome.fresh_rationale.lower()


def test_reaudit_amplification_attack_makes_zero_openrouter_calls(tmp_path: Path) -> None:
    """100 forged entries in one sweep -> zero exfils.

    This is the amplification scenario in the bug filing: a single
    sweep could otherwise ship ±15 lines * N files to OpenRouter.
    With the path-containment gate, the sweep classifies every
    forged entry as either SOURCE_EXCERPT_REJECTED (the path
    resolves outside root to an existing file — exfil attempt) or
    ENTRY_OBSOLETE (the path resolves outside root but to a
    non-existent file — still no exfil, just a missing file). The
    SECURITY guarantee both branches uphold is identical: the judge
    is never called for a forged path. The test asserts the no-exfil
    invariant directly.
    """
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    lines = ["allow_hits:"]
    # Mix real-but-out-of-root paths (security branch) with non-
    # existent traversals (operational branch). Both share the
    # security invariant we care about: NO openrouter call.
    real_targets = [p for p in ("/etc/hostname", "/etc/passwd", "/etc/shells", "/etc/os-release") if Path(p).exists()]
    if not real_targets:
        pytest.skip("no stable absolute files exist outside tmp_path")
    for i in range(100):
        if i % 2 == 0 and real_targets:
            forged = real_targets[i % len(real_targets)]
        else:
            forged = f"../../../etc/nonexistent_host_{i}"
        key = f"{forged}:R1:_module_:fp=cafebabe{i:08d}"
        lines.extend(
            [
                f"- key: {key}",
                "  owner: attacker",
                "  reason: |-",
                f"    forged entry {i}",
                "  safety: |-",
                "    Suppression gated by cicd-judge; see judge_rationale below.",
                "  expires: '2030-01-01'",
            ]
        )
    (allowlist_dir / "_amplified.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with _mock_judge_call() as client_class:
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=True,
        )
    assert len(report.outcomes) == 100
    # Every outcome is either SOURCE_EXCERPT_REJECTED (real-file
    # exfil attempt blocked) or ENTRY_OBSOLETE (non-existent file).
    # No outcome is JUDGE_CALL_FAILED, STILL_AGREES, or any verdict-
    # bearing classification — the judge was never asked.
    quarantined = {ReauditDivergence.SOURCE_EXCERPT_REJECTED, ReauditDivergence.ENTRY_OBSOLETE}
    assert all(o.divergence in quarantined for o in report.outcomes)
    # At least one entry must have hit the SECURITY branch (the test
    # would be vacuous otherwise — i.e. confirming the
    # /etc/passwd-style attempt actually triggered SOURCE_EXCERPT_REJECTED,
    # not just the non-existent ones).
    assert any(o.divergence is ReauditDivergence.SOURCE_EXCERPT_REJECTED for o in report.outcomes)
    # And critically: zero OpenRouter calls — the security invariant.
    assert client_class.call_count == 0


# =====================================================================
# 2. Secrets scrubber — per-pattern coverage
# =====================================================================


@pytest.mark.parametrize(
    ("pattern_name", "planted_text", "expected_pattern_label"),
    [
        # AWS access key id (load-bearing prefix + 16 base32 chars).
        ("aws_access_key", "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", "aws_access_key"),  # secret-scan: allow-this-line
        # GitHub PAT (classic).
        ("github_pat_classic", "ghp_abcdefghijklmnopqrstuvwxyz0123456789", "github_pat_classic"),  # secret-scan: allow-this-line
        # GitHub PAT (fine-grained).
        (
            "github_pat_fine_grained",
            "github_pat_11ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMN_OPQ",
            "github_pat_fine_grained",
        ),
        # Slack token.
        ("slack_token", "xox" + "b-" + "1234567890-1234567890-abcdef", "slack_token"),
        # JWT (three base64url segments).
        ("jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.Pj8Vfg8RtQ_oABCdEFghIJklmnopQRsTUv", "jwt"),
        # GCP service-account marker.
        ("gcp_marker", '"type": "service_account"', "gcp_service_account_marker"),
        # SSH public-key shape.
        (
            "ssh_public_key",
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIabcdefghijklmnopqrstuvwxyz0123456789",
            "ssh_public_key",
        ),
        # PEM private-key header (the most load-bearing signal).
        ("pem_private_key_header", "-----BEGIN RSA PRIVATE KEY-----", "pem_private_key_header"),
        ("pem_private_key_footer", "-----END RSA PRIVATE KEY-----", "pem_private_key_footer"),
        # .env-style assignment.
        ("dotenv_secret", 'API_KEY="sk-abcdef1234567890"', "dotenv_secret_assignment"),
        # Labelled high-entropy value (proximity-gated catch-all).
        (
            "labelled_high_entropy",
            'config = {"secret": "abcdef1234567890abcdef1234567890XY"}',  # secret-scan: allow-this-line
            "labelled_high_entropy_value",
        ),
    ],
)
def test_scrub_secrets_redacts_pattern(pattern_name: str, planted_text: str, expected_pattern_label: str) -> None:
    """Each curated pattern fires and the matched literal vanishes."""
    excerpt_text = f"line before\n{planted_text}\nline after"
    result = scrub_secrets(excerpt_text)
    # The planted literal is gone.
    assert planted_text not in result.text, f"pattern {pattern_name} did not redact: text still contains literal"
    # The redaction token IS present.
    assert "[REDACTED-SECRET-" in result.text
    # The audit record names the matching pattern.
    pattern_names = {r.pattern_name for r in result.redactions}
    assert expected_pattern_label in pattern_names


def test_scrub_secrets_emits_no_redactions_for_clean_excerpt() -> None:
    """A SHA-256 hex constant is NOT flagged (proximity-gate works).

    This is the false-positive resistance check: a 32+ char hex
    string in a test fixture or rfc8785 vector must not trip the
    scrubber because the generic high-entropy pattern is gated to
    label-word proximity. Without that gate, every hashlib output in
    the codebase would be scrubbed and the report would lose signal.
    """
    excerpt_text = "fingerprint = '4f6b8a8e3c2d1a0b9c8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1908'"
    result = scrub_secrets(excerpt_text)
    assert result.redactions == ()
    assert result.text == excerpt_text


def test_scrub_secrets_empty_input() -> None:
    result = scrub_secrets("")
    assert isinstance(result, SafeExcerpt)
    assert result.text == ""
    assert result.redactions == ()


def test_scrub_secrets_redaction_token_carries_stable_hash() -> None:
    """The redaction token's hash is stable across re-runs.

    Operators rely on grep-equality of the redacted token to
    correlate repeat leakage of the same secret across multiple
    excerpts. The hash is the first 16 hex chars of SHA-256.
    """
    excerpt_one = "key = AKIAIOSFODNN7EXAMPLE"  # secret-scan: allow-this-line
    excerpt_two = "another file: AKIAIOSFODNN7EXAMPLE here"  # secret-scan: allow-this-line
    out_one = scrub_secrets(excerpt_one)
    out_two = scrub_secrets(excerpt_two)
    assert len(out_one.redactions) == 1
    assert len(out_two.redactions) == 1
    # The redacted_hash field is stable; the token text contains it.
    assert out_one.redactions[0].redacted_hash == out_two.redactions[0].redacted_hash
    assert out_one.redactions[0].redacted_hash in out_one.text
    assert out_one.redactions[0].redacted_hash in out_two.text


# =====================================================================
# 2b. Secrets scrubber — bypass-resistance (end-to-end via justify)
# =====================================================================


def test_justify_excerpt_with_aws_key_does_not_exfiltrate_literal(tmp_path: Path) -> None:
    """A planted AKIA key in source produces a scrubbed prompt.

    Adversarial: source file contains an inline AWS access key (the
    canonical "secret left in code" attack). The justify path MUST
    redact the literal before the OpenRouter call. We assert against
    the captured user-prompt that the literal AKIA key does NOT
    appear, the redaction token DOES appear, and the YAML entry's
    audit record names the pattern.
    """
    root = tmp_path / "src_root"
    (root / "plugins").mkdir(parents=True)
    target = root / "plugins" / "widget.py"
    target.write_text(
        '"""Synthetic module with planted AWS key."""\n\n\n'
        "class Widget:\n"
        "    def lookup(self, payload: dict) -> str:\n"
        '        AKIA_KEY = "AKIAIOSFODNN7EXAMPLE"  # planted secret\n'  # secret-scan: allow-this-line
        "        # R1 on next line:\n"
        '        return payload.get("name", AKIA_KEY)\n',
        encoding="utf-8",
    )
    allowlist_dir = _build_allowlist_dir(tmp_path)

    with _mock_judge_call(verdict="ACCEPTED", rationale="ok") as client_class:
        exit_code = main(
            [
                "justify",
                "--root",
                str(root),
                "--allowlist-dir",
                str(allowlist_dir),
                "--file-path",
                "plugins/widget.py",
                "--symbol",
                "Widget.lookup",
                "--rationale",
                "tier-3 boundary",
                "--owner",
                "test-agent",
            ]
        )
    assert exit_code == 0, "justify should succeed end-to-end (the scrubber doesn't fail-closed)"
    user_prompt = _captured_user_prompt(client_class)
    # The literal secret MUST NOT appear in the prompt.
    assert "AKIAIOSFODNN7EXAMPLE" not in user_prompt  # secret-scan: allow-this-line
    # The redaction token MUST appear.
    assert "[REDACTED-SECRET-" in user_prompt
    # The written YAML carries an audit record.
    written = (allowlist_dir / "plugins.yaml").read_text(encoding="utf-8")
    assert "judge_excerpt_redactions:" in written
    assert "pattern: aws_access_key" in written


def test_justify_excerpt_with_pem_block_does_not_exfiltrate_literal(tmp_path: Path) -> None:
    """A planted PEM private-key header is redacted before transit."""
    root = tmp_path / "src_root"
    (root / "plugins").mkdir(parents=True)
    target = root / "plugins" / "widget.py"
    target.write_text(
        '"""Module with planted PEM block."""\n\n\n'
        "class Widget:\n"
        "    def lookup(self, payload: dict) -> str:\n"
        "        # -----BEGIN RSA PRIVATE KEY-----\n"
        "        # MIIEpAIBAAKCAQEA...\n"
        "        # -----END RSA PRIVATE KEY-----\n"
        "        # R1 on next line:\n"
        '        return payload.get("name", "anonymous")\n',
        encoding="utf-8",
    )
    allowlist_dir = _build_allowlist_dir(tmp_path)
    with _mock_judge_call(verdict="ACCEPTED", rationale="ok") as client_class:
        exit_code = main(
            [
                "justify",
                "--root",
                str(root),
                "--allowlist-dir",
                str(allowlist_dir),
                "--file-path",
                "plugins/widget.py",
                "--symbol",
                "Widget.lookup",
                "--rationale",
                "tier-3 boundary",
                "--owner",
                "test-agent",
            ]
        )
    assert exit_code == 0
    user_prompt = _captured_user_prompt(client_class)
    assert "-----BEGIN RSA PRIVATE KEY-----" not in user_prompt
    assert "-----END RSA PRIVATE KEY-----" not in user_prompt
    assert "[REDACTED-SECRET-" in user_prompt
    written = (allowlist_dir / "plugins.yaml").read_text(encoding="utf-8")
    assert "pattern: pem_private_key_header" in written
    assert "pattern: pem_private_key_footer" in written


# =====================================================================
# 3. Bypass-resistance — every prompt-builder funnels through extract_safe_excerpt
# =====================================================================


def test_no_other_prompt_builder_constructs_judge_request_without_scrubber() -> None:
    """Static guard: every JudgeRequest constructor must be paired with extract_safe_excerpt.

    This is the structural bypass-resistance check the task plan
    calls for. The grep is intentionally narrow: any file in the
    elspeth_lints.core surface that builds a JudgeRequest must do so
    only after extract_safe_excerpt has produced the surrounding_code
    value. If a future contributor adds a third call site that bypasses
    the helper, this test fails.

    The check is dual: every JudgeRequest(...) construction has, in
    the same file, an extract_safe_excerpt call. The two production
    sites in this commit are cli._run_justify and reaudit._reaudit_one_entry.
    """
    core_dir = Path(__file__).resolve().parents[3] / "elspeth-lints" / "src" / "elspeth_lints" / "core"
    offending_files: list[str] = []
    for py_file in core_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if "JudgeRequest(" not in text:
            continue
        # The dataclass definition itself lives in judge.py; that file
        # builds no instances, only declares the class.
        if py_file.name == "judge.py":
            continue
        if "extract_safe_excerpt" not in text:
            offending_files.append(py_file.name)
    assert offending_files == [], (
        f"these files construct JudgeRequest without going through "
        f"extract_safe_excerpt: {offending_files}. Every code path "
        f"that builds a judge prompt MUST funnel the surrounding_code "
        f"through the scrubber — see source_excerpt.py for the contract."
    )


def test_reaudit_records_redactions_on_outcome(tmp_path: Path) -> None:
    """A reaudit sweep over a file containing a secret records redactions on the outcome.

    End-to-end assertion that the sidecar trail captures what the
    scrubber redacted, so post-hoc forensic inspection of the JSONL
    is sufficient evidence of "what was sanitised".
    """
    root = tmp_path / "src_root"
    (root / "plugins").mkdir(parents=True)
    target = root / "plugins" / "widget.py"
    target.write_text(
        '"""Module with planted secret for reaudit."""\n\n\n'
        "class Widget:\n"
        "    def lookup(self, payload: dict) -> str:\n"
        '        AKIA_KEY = "AKIAIOSFODNN7EXAMPLE"\n'  # secret-scan: allow-this-line
        '        return payload.get("name", AKIA_KEY)\n',
        encoding="utf-8",
    )
    allowlist_dir = _build_allowlist_dir(tmp_path)

    # Compute the live fingerprint + ast_path so the C8-3 binding
    # passes and reaudit reaches the per-entry dispatch.
    from elspeth_lints.rules.trust_tier.tier_model.rule import scan_file

    findings = [f for f in scan_file(target.resolve(), root) if f.rule_id == "R1"]
    assert len(findings) >= 1
    finding = findings[0]
    import hashlib as _hashlib

    file_fp = _hashlib.sha256(target.read_bytes()).hexdigest()
    yaml = (
        "allow_hits:\n"
        f"- key: {finding.canonical_key}\n"
        "  owner: test-owner\n"
        "  reason: |-\n"
        "    boundary\n"
        "  safety: |-\n"
        "    Suppression gated by cicd-judge; see judge_rationale below.\n"
        "  expires: '2030-01-01'\n"
        "  judge_verdict: ACCEPTED\n"
        "  judge_recorded_at: '2024-01-01T00:00:00+00:00'\n"
        "  judge_model: claude-opus-4-7\n"
        "  judge_rationale: |-\n"
        "    original judge said the boundary was genuine\n"
        f"  file_fingerprint: '{file_fp}'\n"
        f"  ast_path: '{finding.ast_path}'\n"
    )
    (allowlist_dir / "plugins.yaml").write_text(yaml, encoding="utf-8")

    with _mock_judge_call(verdict="ACCEPTED", rationale="still genuine"):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=False,
        )
    assert len(report.outcomes) == 1
    outcome = report.outcomes[0]
    # The outcome carries the redaction record.
    assert len(outcome.excerpt_redactions) >= 1
    pattern_names = {r.pattern_name for r in outcome.excerpt_redactions}
    assert "aws_access_key" in pattern_names
    # The code_snapshot stored on the outcome is the SCRUBBED text, not
    # the literal — defense-in-depth: even the audit dump never carries
    # the secret.
    assert "AKIAIOSFODNN7EXAMPLE" not in outcome.code_snapshot  # secret-scan: allow-this-line
    assert "[REDACTED-SECRET-" in outcome.code_snapshot


def test_redaction_record_is_immutable_dataclass() -> None:
    """RedactionRecord is frozen + slotted (deep_freeze contract for the
    in-memory audit primitive).
    """
    import dataclasses

    record = RedactionRecord(pattern_name="aws_access_key", byte_count=20, redacted_hash="deadbeefcafebabe")
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.pattern_name = "other"


def test_resolve_safe_excerpt_path_rejects_symlink_escape(tmp_path: Path) -> None:
    """A symlink inside root pointing OUTSIDE root is rejected.

    Adversarial: an attacker (or careless build) plants a symlink at
    ``src/elspeth/leaked.py -> /etc/passwd``. The symlink lives
    inside root, but ``Path.resolve(strict=True)`` follows it. The
    canonical absolute path is /etc/passwd, which lives outside
    root, so the containment check rejects it. Without symlink-
    aware resolution this would be the simplest exfil bypass.
    """
    root, _target = _build_source_tree(tmp_path)
    escape_target = Path("/etc/passwd") if Path("/etc/passwd").exists() else Path("/etc/hostname")
    if not escape_target.exists():
        pytest.skip("no stable absolute file exists outside tmp_path")
    symlink = root / "plugins" / "leaked.py"
    symlink.symlink_to(escape_target)
    with pytest.raises(SourceExcerptPathOutsideRootError):
        resolve_safe_excerpt_path(root=root, target_file=symlink)

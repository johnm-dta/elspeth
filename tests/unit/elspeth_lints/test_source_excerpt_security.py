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
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from elspeth_lints.core.allowlist import JudgeVerdict, compute_judge_metadata_signature
from elspeth_lints.core.cli import main
from elspeth_lints.core.judge import DEFAULT_JUDGE_MODEL, JUDGE_POLICY_HASH
from elspeth_lints.core.reaudit import ReauditDivergence, reaudit_entries
from elspeth_lints.core.source_excerpt import (
    RedactionRecord,
    SafeExcerpt,
    SourceExcerptPathOutsideRootError,
    extract_safe_excerpt,
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

_TEST_JUDGE_METADATA_HMAC_KEY = "test-judge-metadata-hmac-key-2026-05-24"


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
    message.content = json.dumps({"verdict": verdict, "rationale": rationale, "confidence": 0.91, "should_use_decorator": None})
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    completion.model = DEFAULT_JUDGE_MODEL
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
        patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "sk-or-test-key",
                "ELSPETH_JUDGE_METADATA_HMAC_KEY": _TEST_JUDGE_METADATA_HMAC_KEY,
            },
            clear=False,
        ),
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
    return "\n".join(str(block["text"]) for block in user_message["content"])


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
    """Static guard: every JudgeRequest constructor must be paired with a scrubber.

    This is the structural bypass-resistance check the task plan
    calls for. The scan walks the ENTIRE elspeth_lints package — not
    just ``core/`` — so a future contributor who adds a
    ``JudgeRequest(...)`` builder in ``rules/`` (or any sibling
    package) cannot slip past the gate. The pre-T8b version of this
    test only swept ``core/*.py``, missing exactly that class of
    bypass; T8b broadens the radius.

    The check is dual: every ``JudgeRequest(...)`` construction site
    has, in the SAME FILE, either an ``extract_safe_excerpt`` call
    (filesystem source) or a ``scrub_secrets`` call (already-inline,
    labelled quality-corpus excerpts). The same-file requirement is
    deliberate over a project-wide grep — a file that constructs the
    request must locally evidence that its ``surrounding_code`` value
    came from the scrubber.

    The dataclass definition itself (``judge.py``) is excluded — it
    declares the class but builds no instances. Test fixtures /
    helpers under ``tests/`` are excluded because they are not the
    production trust boundary (test code may legitimately construct
    a request to exercise downstream consumers).
    """
    package_root = Path(__file__).resolve().parents[3] / "elspeth-lints" / "src" / "elspeth_lints"
    assert package_root.is_dir(), f"elspeth_lints package not found at {package_root}"
    offending_files: list[str] = []
    for py_file in package_root.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if "JudgeRequest(" not in text:
            continue
        if py_file.name == "judge.py":
            continue
        if "extract_safe_excerpt" not in text and "scrub_secrets" not in text:
            # Report a stable, project-relative path so the failure
            # message points at the offending file regardless of
            # where the package tree lives.
            offending_files.append(str(py_file.relative_to(package_root)))
    assert offending_files == [], (
        f"these files construct JudgeRequest without going through "
        f"extract_safe_excerpt or scrub_secrets: {offending_files}. "
        f"Every code path that builds a judge prompt MUST funnel the "
        f"surrounding_code through the scrubber — see source_excerpt.py "
        f"for the contract."
    )


def test_bypass_resistance_grep_fires_when_injected_file_lacks_scrubber(tmp_path: Path) -> None:
    """The bypass-resistance grep CATCHES a planted offender.

    Defends against the test-is-vacuous failure mode: if the grep is
    incorrectly scoped or the helper string changes, the static guard
    could pass silently while production drifts. We construct a
    miniature package tree under ``tmp_path``, plant a file containing
    a ``JudgeRequest(...)`` call WITHOUT ``extract_safe_excerpt``, and
    assert the same grep logic flags it. This pins the grep's
    semantics to a deliberate failure case rather than just the
    happy-path production tree.
    """
    fake_pkg = tmp_path / "fake_elspeth_lints"
    (fake_pkg / "rules" / "bad").mkdir(parents=True)
    (fake_pkg / "core").mkdir()
    # Clean file (the production shape) — has the scrubber call.
    (fake_pkg / "core" / "good.py").write_text(
        "from elspeth_lints.core.judge import JudgeRequest\n"
        "from elspeth_lints.core.source_excerpt import extract_safe_excerpt\n"
        "def f():\n"
        "    excerpt = extract_safe_excerpt(root=..., target_file=..., line=1, context_lines=15)\n"
        "    return JudgeRequest(surrounding_code=excerpt.text)\n",
        encoding="utf-8",
    )
    # Offending file in a non-core package — what the broadened grep
    # is supposed to catch.
    (fake_pkg / "rules" / "bad" / "rogue.py").write_text(
        "from elspeth_lints.core.judge import JudgeRequest\ndef f():\n    return JudgeRequest(surrounding_code='raw unscrubbed bytes')\n",
        encoding="utf-8",
    )
    # Apply the same grep logic the bypass-resistance test uses.
    offending_files: list[str] = []
    for py_file in fake_pkg.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if "JudgeRequest(" not in text:
            continue
        if py_file.name == "judge.py":
            continue
        if "extract_safe_excerpt" not in text:
            offending_files.append(str(py_file.relative_to(fake_pkg)))
    assert offending_files == ["rules/bad/rogue.py"]


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
    signature = compute_judge_metadata_signature(
        key=finding.canonical_key,
        file_fingerprint=file_fp,
        ast_path=finding.ast_path,
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
        judge_model="claude-opus-4-7",
        judge_policy_hash=JUDGE_POLICY_HASH,
        judge_rationale="original judge said the boundary was genuine",
        hmac_key=_TEST_JUDGE_METADATA_HMAC_KEY.encode("utf-8"),
    )
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
        f"  judge_policy_hash: '{JUDGE_POLICY_HASH}'\n"
        "  judge_rationale: |-\n"
        "    original judge said the boundary was genuine\n"
        f"  file_fingerprint: '{file_fp}'\n"
        f"  ast_path: '{finding.ast_path}'\n"
        f"  judge_metadata_signature: '{signature}'\n"
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


# =====================================================================
# 2c. T8b expanded coverage — bare-prefix high-value secrets
# =====================================================================
#
# The T8 commit's pattern set required a label proximity gate for the
# generic high-entropy catch-all. That gate misses the most-common
# leaked-credential shapes: bare ``sk_live_…`` / ``sk-ant-…`` /
# ``AIza…`` strings appear in source unaccompanied by a label word
# (assigned to bare variables, embedded in URLs, pasted into
# comments). Each test below plants exactly the unlabelled shape T8
# would have shipped unredacted, asserts the scrubber catches it, and
# pins the pattern_name into the audit vocabulary.


@pytest.mark.parametrize(
    ("pattern_label", "planted_text"),
    [
        # Stripe — six prefix variants. Bodies are synthetic but match
        # the regex shape (24+ base62 for sk/pk/rk; 32+ for whsec).
        ("stripe_secret_key", "sk" + "_live_" + "4eC39HqLyjWDarjtT1zdp7dc"),
        ("stripe_test_secret_key", "sk" + "_test_" + "4eC39HqLyjWDarjtT1zdp7dc"),
        ("stripe_publishable_key", "pk" + "_live_" + "TYooMQauvdEDq54NiTphI7jx"),
        ("stripe_test_publishable_key", "pk" + "_test_" + "TYooMQauvdEDq54NiTphI7jx"),
        ("stripe_restricted_key", "rk" + "_live_" + "51HG8z0KvU9rT2bWqXm1nP4dF"),
        ("stripe_webhook_secret", "whsec" + "_" + "5WbX9NheWmkP3FvY1k2nC8oRtZ4vUxJqLmA0pYsBgEdJ"),
        # OpenAI — three shapes. Legacy 48-char, project-scoped, session.
        ("openai_api_key", "sk-" + "A" * 48),
        ("openai_project_key", "sk-proj-Wx7AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-AbCd"),  # secret-scan: allow-this-line
        ("openai_session_key", "sess-" + "Q" * 45),  # secret-scan: allow-this-line
        # Anthropic — bound BEFORE the generic openai_api_key pattern.
        ("anthropic_api_key", "sk-ant-api01-" + "B" * 80),  # secret-scan: allow-this-line
        # HuggingFace — bare hf_ prefix is uniquely diagnostic.
        ("huggingface_token", "hf_" + "C" * 36),  # secret-scan: allow-this-line
        # Google API key — exactly 35 char suffix; the FIXED length is
        # what disambiguates from other AI*-prefixed identifiers.
        ("google_api_key", "AIza" + "D" * 35),  # secret-scan: allow-this-line
        # GitLab PAT and OAuth secret.
        ("gitlab_pat", "glpat-xyzABC123_-456DEFghIjKlMnOpQ"),  # secret-scan: allow-this-line
        ("gitlab_oauth_secret", "gloas-applicationSecret_1234567890abcdef"),  # secret-scan: allow-this-line
        # Discord webhook URL — the URL IS the secret.
        (
            "discord_webhook_url",
            "https://discord.com/api/webhooks/123456789012345678/" + "E" * 60,  # secret-scan: allow-this-line
        ),
        # Slack webhook URL — distinct T... / B... segments + token tail.
        (
            "slack_webhook_url",
            "https://hooks.slack.com/services/" + "T01ABCDEFGH/B02ZYXWVUTSR/" + "F" * 24,
        ),
    ],
)
def test_scrub_secrets_redacts_bare_prefix_pattern(pattern_label: str, planted_text: str) -> None:
    """Each new bare-prefix pattern fires WITHOUT a label keyword nearby.

    Adversarial: the T8 pattern set required a label-proximity hit for
    the generic catch-all, so a bare ``KEY = "sk_live_…"`` assignment
    where the bare-variable name carries no label keyword slipped
    through unredacted. T8b adds explicit per-provider patterns so
    these high-value credentials are caught on the prefix alone.

    Each parameter plants the secret on a no-label line so a regression
    that re-introduces label-proximity gating would fail this test
    (the line carries only ``var =`` which the dotenv label set does
    not include).
    """
    # Plant the secret on a line whose only identifier is ``var``
    # (NOT in the dotenv label set: secret/password/passwd/token/
    # api_key/access_key/private_key/auth/bearer). If the new
    # specific pattern fires the literal vanishes; if only the
    # label-proximity-gated catch-all fires we'd miss this shape.
    excerpt_text = f"line before\nvar = {planted_text!r}\nline after"
    result = scrub_secrets(excerpt_text)
    assert planted_text not in result.text, (
        f"pattern {pattern_label!r} did not redact the literal secret. This is the bare-prefix shape T8b is meant to close."
    )
    assert "[REDACTED-SECRET-" in result.text
    pattern_names = {r.pattern_name for r in result.redactions}
    assert pattern_label in pattern_names, (
        f"expected pattern {pattern_label!r} to fire on {planted_text!r}; "
        f"got {sorted(pattern_names)}. A regression here means the audit "
        f"vocabulary lost a specific-pattern tag and operators can no "
        f"longer filter the scrubber report by provider."
    )


def test_scrub_secrets_authorization_bearer_token_only(tmp_path: Path) -> None:
    """``Authorization: Bearer <token>`` — the label survives, the token is redacted.

    The Bearer pattern uses ``redact_group=1`` to keep the structural
    shape of the header visible in the LLM prompt (operators reading
    the transcript see "an HTTP call carrying a Bearer token") while
    the credential itself is scrubbed. The byte_count and
    redacted_hash measure ONLY the token, not the surrounding label —
    confirming the group-bounded redaction path works as designed.
    """
    token = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH"
    # Use the canonical HTTP header form (label, colon, space, "Bearer ", token).
    # The regex requires a literal colon after ``Authorization``, mirroring
    # the RFC 7235 header syntax — both because that is the realistic
    # exfil shape AND because matching ``Authorization`` without a
    # colon would FP on every prose mention of the word.
    excerpt_text = f"# Example header\nAuthorization: Bearer {token}\n# next line"
    result = scrub_secrets(excerpt_text)
    # Token literal MUST vanish, the label MUST survive.
    assert token not in result.text
    assert "Authorization:" in result.text
    assert "Bearer" in result.text
    assert "[REDACTED-SECRET-" in result.text
    # Audit record names the pattern AND measures only the token's
    # bytes (44 chars * 1 byte each in UTF-8), not the label.
    matching = [r for r in result.redactions if r.pattern_name == "authorization_bearer"]
    assert len(matching) == 1
    assert matching[0].byte_count == len(token)


def test_scrub_secrets_azure_account_key_redacted(tmp_path: Path) -> None:
    """Azure storage ``AccountKey=<base64>`` — the credential is scrubbed.

    The connection string's other components (``DefaultEndpointsProtocol``,
    ``AccountName``, ``EndpointSuffix``) are not sensitive on their own —
    we redact only the ``AccountKey=<base64>`` segment so the prompt
    retains enough context for the judge to understand what shape of
    config it's looking at without seeing the credential.
    """
    account_key = "A" * 64 + "B" * 24 + "=="
    excerpt_text = f"conn = ('DefaultEndpointsProtocol=https;AccountName=stor1;AccountKey={account_key};EndpointSuffix=core.windows.net')"
    result = scrub_secrets(excerpt_text)
    assert account_key not in result.text
    assert "[REDACTED-SECRET-" in result.text
    # Surrounding context preserved.
    assert "DefaultEndpointsProtocol=https" in result.text
    assert "AccountName=stor1" in result.text
    assert "EndpointSuffix=core.windows.net" in result.text
    matching = [r for r in result.redactions if r.pattern_name == "azure_storage_account_key"]
    assert len(matching) == 1


def test_scrub_secrets_ssh_body_requires_path_hint(tmp_path: Path) -> None:
    """Bare 60+ char base64 lines redact ONLY when path_hint indicates a key file.

    Without the hint (the bare ``scrub_secrets`` diagnostic surface)
    the pattern is too permissive — every long base64 chunk in source
    (sourcemaps, embedded crypto material, minified JS) would
    false-positive. With the hint the file is structurally a key file
    and over-redaction is the desired behaviour.

    The asymmetry IS the security property: production callers go
    through ``extract_safe_excerpt`` which always supplies the hint;
    diagnostic callers passing ``scrub_secrets`` directly opt-out
    deliberately and accept the under-redaction.
    """
    pem_body = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDX"  # secret-scan: allow-this-line
    body_line = pem_body + "X" * (61 - len(pem_body))  # ensure >= 60 chars
    excerpt_text = f"line before\n{body_line}\nline after"
    # No hint -> pattern is skipped, literal survives.
    unguarded = scrub_secrets(excerpt_text)
    assert body_line in unguarded.text
    assert all(r.pattern_name != "ssh_private_key_body" for r in unguarded.redactions)
    # With a path hint matching a key-file extension -> pattern fires.
    guarded = scrub_secrets(excerpt_text, path_hint="/repo/secrets/id_rsa")
    assert body_line not in guarded.text
    assert any(r.pattern_name == "ssh_private_key_body" for r in guarded.redactions)
    # Other hint extensions also trigger.
    guarded_pem = scrub_secrets(excerpt_text, path_hint="/repo/keys/cert.pem")
    assert body_line not in guarded_pem.text


# =====================================================================
# 2d. T8b — file-fingerprint salting of the redacted-hash audit primitive
# =====================================================================


def test_extract_safe_excerpt_salts_redacted_hash_with_file_fingerprint(tmp_path: Path) -> None:
    """The SAME secret in two DIFFERENT files produces DIFFERENT redacted_hash.

    Adversarial: an internal actor with file-system access to the
    audit YAML / JSONL would otherwise be able to brute-force the
    16-hex SHA-256 prefix (64-bit search space). A low-entropy secret
    (``PASSWORD=hunter12``) is recoverable in seconds. Salting with
    the file's fingerprint scopes the brute-force cost per file: the
    same secret in file A and file B produces DIFFERENT redacted
    hashes, so an attacker must repeat the search per file rather
    than across the whole corpus.

    This is the load-bearing entropy property — it would silently
    regress if a future contributor removed the salt argument or
    changed ``scrub_secrets`` to ignore it.
    """
    root = tmp_path / "src_root"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    # Same secret, embedded in two files that differ only by one
    # extra comment line so the SHA-256 file fingerprints diverge.
    planted = "AKIAIOSFODNN7EXAMPLE"  # secret-scan: allow-this-line
    common = f'KEY = "{planted}"\n# R1 on next line\nprint(KEY)\n'
    (root / "a" / "mod.py").write_text(common, encoding="utf-8")
    (root / "b" / "mod.py").write_text("# extra context line\n" + common, encoding="utf-8")
    excerpt_a = extract_safe_excerpt(root=root, target_file=root / "a" / "mod.py", line=1, context_lines=15)
    excerpt_b = extract_safe_excerpt(root=root, target_file=root / "b" / "mod.py", line=2, context_lines=15)
    # Both excerpts must have caught the AKIA key.
    assert any(r.pattern_name == "aws_access_key" for r in excerpt_a.redactions)
    assert any(r.pattern_name == "aws_access_key" for r in excerpt_b.redactions)
    hash_a = next(r.redacted_hash for r in excerpt_a.redactions if r.pattern_name == "aws_access_key")
    hash_b = next(r.redacted_hash for r in excerpt_b.redactions if r.pattern_name == "aws_access_key")
    # Distinct salts -> distinct hashes for the SAME secret.
    assert hash_a != hash_b, (
        "redacted_hash collided across two file fingerprints; the salt "
        "is not being applied. This silently degrades the per-file "
        "brute-force scoping the salt is meant to provide."
    )
    # And the file_fingerprint on SafeExcerpt matches the actual
    # SHA-256 of the file bytes (single read source of truth).
    import hashlib as _hashlib

    assert excerpt_a.file_fingerprint == _hashlib.sha256((root / "a" / "mod.py").read_bytes()).hexdigest()
    assert excerpt_b.file_fingerprint == _hashlib.sha256((root / "b" / "mod.py").read_bytes()).hexdigest()


def test_scrub_secrets_unsalted_path_preserves_cross_file_grep_equality() -> None:
    """The diagnostic ``scrub_secrets`` surface (no salt) keeps cross-text hash equality.

    The salted regime is the production path (via
    ``extract_safe_excerpt``); the unsalted regime is a diagnostic
    surface that preserves the original grep-equality property —
    operators using ``scrub_secrets`` directly on a string can
    correlate identical secrets across two inputs.
    """
    planted = "AKIAIOSFODNN7EXAMPLE"  # secret-scan: allow-this-line
    out_one = scrub_secrets(f"a = {planted}")
    out_two = scrub_secrets(f"b = {planted}")
    assert out_one.redactions[0].redacted_hash == out_two.redactions[0].redacted_hash
    # And ``file_fingerprint`` on the diagnostic surface is empty —
    # the SafeExcerpt shape distinguishes salted from unsalted at the
    # type level.
    assert out_one.file_fingerprint == ""


# =====================================================================
# 2e. T8b — docstring fidelity: the dotenv FP claim is honest
# =====================================================================


def test_dotenv_secret_assignment_false_positives_on_prose_value() -> None:
    """The ``dotenv_secret_assignment`` pattern DOES false-positive on prose values.

    The T8 commit's comment claimed the scrubber would not FP on
    obvious-non-secret shapes; T8b's docstring revision is honest
    about the actual FP shape so future contributors know the
    trade-off. Pinning the FP in a test means a future "fix" that
    silently breaks this behaviour will fail loudly here, forcing a
    deliberate decision rather than an accidental drift.

    The trade-off (under-redaction is the security failure,
    over-redaction is the LLM-context failure) means we deliberately
    accept this FP rather than narrow the regex.
    """
    # A line that looks structurally like a secret assignment but is
    # actually a long prose description. The dotenv regex matches
    # ``label = value`` where value is 8+ non-quote/non-space chars.
    # The label ``secret`` is in the dotenv label set; the unquoted
    # RHS is 38 chars of non-whitespace, fully inside the value
    # alphabet. This is the realistic over-redaction shape: any
    # ``secret = <unquoted-long-identifier>`` line — including code
    # that uses ``secret`` as a variable name for a non-credential
    # value — gets scrubbed.
    excerpt_text = "secret = some_long_string_that_is_not_a_secret"
    result = scrub_secrets(excerpt_text)
    # The pattern fires (this is the FP shape we're pinning).
    pattern_names = {r.pattern_name for r in result.redactions}
    assert "dotenv_secret_assignment" in pattern_names, (
        "the dotenv FP shape changed; either the regex was tightened "
        "(update the docstring + this test) or the catch-all stopped "
        "firing on prose values (verify the security implications "
        "before celebrating)."
    )


# =====================================================================
# 3. T8c — production-path contract tests
# =====================================================================
#
# These tests exercise the FULL ``extract_safe_excerpt`` pipeline
# (path containment + raw-window read + scrubber + line-number-prefix
# renderer). The earlier T8b coverage tested ``scrub_secrets`` in
# isolation — that's where the SSH-body MAJOR slipped past: the
# render-then-scrub ordering meant the production text never reached
# the patterns in the shape the patterns expected. The unit test
# passed; the integration was broken.
#
# Every pattern in ``_SECRET_PATTERNS`` MUST survive the production
# pipeline. The parametrized contract below plants each pattern's
# literal into a freshly-written file under tmp_path, runs the file
# through ``extract_safe_excerpt`` with a path hint that activates
# any ``path_hint_required`` gating, and asserts the literal is absent
# from the rendered output AND that a redaction with the expected
# pattern_name is recorded.
#
# Add a new pattern to ``_SECRET_PATTERNS``? Add a corresponding row
# to ``_PATTERN_CONTRACT_CASES`` below. If you forget, the
# ``test_pattern_contract_coverage_is_exhaustive`` test fires and
# names the missing pattern.


def test_extract_safe_excerpt_redacts_ssh_private_key_body(tmp_path: Path) -> None:
    """T8c MAJOR: SSH body redaction works in the FULL production path.

    Pre-T8c this test failed silently: the render-then-scrub ordering
    put a line-number prefix in front of every body line, the
    ``ssh_private_key_body`` pattern's ``^...$`` anchor matched the
    prefix instead of the base64 body, and the literal slipped through
    unredacted. The unit test
    (``test_scrub_secrets_ssh_body_requires_path_hint``) continued to
    pass because it called ``scrub_secrets`` on un-prefixed text.

    Scrub-before-render closes the gap. This test pins the behaviour
    end-to-end so the regression cannot return without firing.
    """
    root = tmp_path / "src_root"
    root.mkdir()
    # PEM-shaped file with realistic 64-char base64 body lines (the
    # length the ``openssl`` writer produces).
    body_line_one = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDXAAAAAAAAAAAA"
    body_line_two = "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
    pem_path = root / "id_rsa"
    pem_path.write_text(
        f"-----BEGIN OPENSSH PRIVATE KEY-----\n{body_line_one}\n{body_line_two}\n-----END OPENSSH PRIVATE KEY-----\n",
        encoding="utf-8",
    )
    excerpt = extract_safe_excerpt(root=root, target_file=pem_path, line=2, context_lines=5)
    # Both body lines MUST be absent from the rendered prompt.
    assert body_line_one not in excerpt.text, (
        "ssh_private_key_body redaction regressed in the production path. "
        "If the unit test still passes, the cause is render/scrub ordering."
    )
    assert body_line_two not in excerpt.text
    # Header and footer also redacted (different patterns).
    body_pattern_names = {r.pattern_name for r in excerpt.redactions}
    assert "ssh_private_key_body" in body_pattern_names
    assert "pem_private_key_header" in body_pattern_names
    assert "pem_private_key_footer" in body_pattern_names
    # Line-number prefix preserved on the rendered output (the body
    # was scrubbed; the structural prefix remains).
    assert ">>" in excerpt.text or "  " in excerpt.text


# Pattern contract cases: one row per entry in ``_SECRET_PATTERNS``.
# ``surrounding_text`` is a small file body that embeds the literal in
# a no-label position; ``filename`` activates any ``path_hint_required``
# gating. The renderer adds a line-number prefix, so the literal must
# survive the prefix-then-scrub previously and now the scrub-then-prefix
# pipeline.
_PATTERN_CONTRACT_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "pem_private_key_header",
        "mod.py",
        "x = 1\n-----BEGIN RSA PRIVATE KEY-----\ny = 2\n",
    ),
    (
        "pem_private_key_footer",
        "mod.py",
        "x = 1\n-----END RSA PRIVATE KEY-----\ny = 2\n",
    ),
    (
        "aws_access_key",
        "mod.py",
        'x = 1\nvar = "AKIAIOSFODNN7EXAMPLE"\ny = 2\n',  # secret-scan: allow-this-line
    ),
    (
        "github_pat_classic",
        "mod.py",
        'x = 1\nvar = "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"\ny = 2\n',  # secret-scan: allow-this-line
    ),
    (
        "github_pat_fine_grained",
        "mod.py",
        'x = 1\nvar = "github_pat_' + "A" * 82 + '"\ny = 2\n',  # secret-scan: allow-this-line
    ),
    (
        "gitlab_pat",
        "mod.py",
        'x = 1\nvar = "glpat-xyzABC123_-456DEFghIjKlMnOpQ"\ny = 2\n',  # secret-scan: allow-this-line
    ),
    (
        "gitlab_oauth_secret",
        "mod.py",
        'x = 1\nvar = "gloas-applicationSecret_1234567890abcdef"\ny = 2\n',  # secret-scan: allow-this-line
    ),
    (
        "slack_token",
        "mod.py",
        'x = 1\nvar = "' + "xox" + "b-" + '1234567890-AbCdEfGhIjKlMnOp"\ny = 2\n',
    ),
    (
        "stripe_secret_key",
        "mod.py",
        'x = 1\nvar = "' + "sk" + "_live_" + '4eC39HqLyjWDarjtT1zdp7dc"\ny = 2\n',
    ),
    (
        "stripe_test_secret_key",
        "mod.py",
        'x = 1\nvar = "' + "sk" + "_test_" + '4eC39HqLyjWDarjtT1zdp7dc"\ny = 2\n',
    ),
    (
        "stripe_publishable_key",
        "mod.py",
        'x = 1\nvar = "' + "pk" + "_live_" + 'TYooMQauvdEDq54NiTphI7jx"\ny = 2\n',
    ),
    (
        "stripe_test_publishable_key",
        "mod.py",
        'x = 1\nvar = "' + "pk" + "_test_" + 'TYooMQauvdEDq54NiTphI7jx"\ny = 2\n',
    ),
    (
        "stripe_restricted_key",
        "mod.py",
        'x = 1\nvar = "' + "rk" + "_live_" + '51HG8z0KvU9rT2bWqXm1nP4dF"\ny = 2\n',
    ),
    (
        "stripe_webhook_secret",
        "mod.py",
        'x = 1\nvar = "' + "whsec" + "_" + '5WbX9NheWmkP3FvY1k2nC8oRtZ4vUxJqLmA0pYsBgEdJ"\ny = 2\n',
    ),
    (
        "anthropic_api_key",
        "mod.py",
        'x = 1\nvar = "sk-ant-api01-' + "B" * 80 + '"\ny = 2\n',  # secret-scan: allow-this-line
    ),
    (
        "openai_project_key",
        "mod.py",
        'x = 1\nvar = "sk-proj-Wx7AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-AbCd"\ny = 2\n',  # secret-scan: allow-this-line
    ),
    (
        "openai_session_key",
        "mod.py",
        'x = 1\nvar = "sess-' + "Q" * 45 + '"\ny = 2\n',  # secret-scan: allow-this-line
    ),
    (
        "openai_api_key",
        "mod.py",
        'x = 1\nvar = "sk-' + "A" * 48 + '"\ny = 2\n',  # secret-scan: allow-this-line
    ),
    (
        "huggingface_token",
        "mod.py",
        'x = 1\nvar = "hf_' + "C" * 36 + '"\ny = 2\n',  # secret-scan: allow-this-line
    ),
    (
        "google_api_key",
        "mod.py",
        'x = 1\nvar = "AIza' + "D" * 35 + '"\ny = 2\n',  # secret-scan: allow-this-line
    ),
    (
        "jwt",
        "mod.py",
        'x = 1\nvar = "eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0NTY3.SflKxwRJSMeKKF2QT4"\ny = 2\n',  # secret-scan: allow-this-line
    ),
    (
        "gcp_service_account_marker",
        "mod.py",
        'x = 1\ncfg = {"type": "service_account"}\ny = 2\n',
    ),
    (
        "ssh_public_key",
        "mod.py",
        "x = 1\nkey = 'ssh-rsa " + "A" * 80 + "'\ny = 2\n",  # secret-scan: allow-this-line
    ),
    (
        "discord_webhook_url",
        "mod.py",
        'x = 1\nurl = "https://discord.com/api/webhooks/123456789012345678/' + "E" * 60 + '"\ny = 2\n',  # secret-scan: allow-this-line
    ),
    (
        "slack_webhook_url",
        "mod.py",
        'x = 1\nurl = "https://hooks.slack.com/services/' + "T01ABCDEFGH/B02ZYXWVUTSR/" + "F" * 24 + '"\ny = 2\n',
    ),
    (
        "authorization_bearer",
        "mod.py",
        "x = 1\nheaders = 'Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH'\ny = 2\n",  # secret-scan: allow-this-line
    ),
    (
        "azure_storage_account_key",
        "mod.py",
        'x = 1\nconn = "AccountKey=' + "A" * 64 + "B" * 24 + '=="\ny = 2\n',  # secret-scan: allow-this-line
    ),
    (
        "dotenv_secret_assignment",
        "mod.py",
        'x = 1\nAPI_KEY = "sk-abcdef1234567890"\ny = 2\n',  # secret-scan: allow-this-line
    ),
    (
        "labelled_high_entropy_value",
        "mod.py",
        # Use a base64-shaped value paired with the ``token`` label.
        # Length 40 puts it above the 32-char floor; the alphabet is
        # the gated set [A-Za-z0-9+/=_-].
        "x = 1\nconfig_token: 'aGVsbG8td29ybGQtdGVzdC1iYXNlNjQtdmFsdWVoaXg='\ny = 2\n",  # secret-scan: allow-this-line
    ),
    (
        "ssh_private_key_body",
        # File NAME activates ``path_hint_required`` gating.
        "id_rsa",
        # 64-char base64-shaped body line per the openssl writer
        # convention. The pattern requires 60+ chars and re.MULTILINE
        # ``^...$`` matching against the raw body.
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDXAAAAAAAAAAAA\n",
    ),
)


@pytest.mark.parametrize(
    ("expected_pattern_name", "filename", "file_body"),
    _PATTERN_CONTRACT_CASES,
    ids=[case[0] for case in _PATTERN_CONTRACT_CASES],
)
def test_extract_safe_excerpt_contract_per_pattern(
    expected_pattern_name: str,
    filename: str,
    file_body: str,
    tmp_path: Path,
) -> None:
    """Contract: every pattern in _SECRET_PATTERNS MUST redact in the production path.

    This is the meta-test that prevents the T8c recurrence: a future
    pattern added with an anchor or shape that doesn't survive the
    ``extract_safe_excerpt`` pipeline (path resolve + raw read +
    scrubber + line-number prefix renderer) will fail HERE rather than
    silently shipping unredacted secrets to the LLM.

    The literal that the parametrised body embeds MUST be absent from
    the rendered ``SafeExcerpt.text`` AND a ``RedactionRecord`` with
    the expected pattern_name MUST appear in
    ``SafeExcerpt.redactions``.
    """
    root = tmp_path / "src_root"
    root.mkdir()
    target = root / filename
    target.write_text(file_body, encoding="utf-8")

    # Identify the planted literal: it's the body content that the
    # pattern is supposed to match. We can derive it generically by
    # finding the longest pattern-specific substring in the body. To
    # keep the test honest (and avoid re-running the regex to find
    # what we planted) we embed a known sentinel: for each case the
    # secret literal is the longest run of consecutive non-quote /
    # non-space chars in the body. The test below scans the body to
    # locate that segment, then asserts it's absent from the output.
    #
    # That's robust to the body shapes above (most embed the literal
    # in quotes) but for the PEM header/footer / ssh_private_key_body
    # / gcp_service_account_marker cases the literal IS the whole
    # body content of interest — we assert pattern_name presence
    # instead.

    excerpt = extract_safe_excerpt(
        root=root,
        target_file=target,
        line=2,
        context_lines=10,
    )

    pattern_names = {r.pattern_name for r in excerpt.redactions}
    assert expected_pattern_name in pattern_names, (
        f"pattern {expected_pattern_name!r} did NOT fire in the production "
        f"path. This is the T8c failure shape: scrub_secrets may pass in "
        f"isolation but extract_safe_excerpt's pipeline mutated the text "
        f"so the pattern no longer matches. Recorded redactions: "
        f"{sorted(pattern_names)}."
    )

    # Locate the literal in the body and assert it doesn't appear in
    # the rendered prompt. Different pattern shapes plant their
    # literal differently — we extract the substring that the
    # pattern is meant to scrub by re-running the regex against the
    # RAW body (the same shape ``extract_safe_excerpt`` sees before
    # rendering).
    from elspeth_lints.core.source_excerpt import _SECRET_PATTERNS

    pattern = next(p for p in _SECRET_PATTERNS if p.name == expected_pattern_name)
    match = pattern.regex.search(file_body)
    assert match is not None, (
        f"test setup error: pattern {expected_pattern_name!r} does not match its own contract body — fix the body for this row."
    )
    planted_literal = match.group(pattern.redact_group)
    assert planted_literal not in excerpt.text, (
        f"pattern {expected_pattern_name!r} fired (recorded as redaction) "
        f"but the literal {planted_literal!r} is still present in the "
        f"rendered prompt. The redaction record is honest about WHAT was "
        f"redacted but the replacement didn't actually substitute. This "
        f"is a Tier-1 audit-trail integrity failure."
    )


def test_pattern_contract_coverage_is_exhaustive() -> None:
    """Every pattern in _SECRET_PATTERNS MUST have a contract-test case.

    Adding a new pattern to ``_SECRET_PATTERNS`` without adding the
    corresponding row to ``_PATTERN_CONTRACT_CASES`` would silently
    leave that pattern uncovered by the production-path contract.
    This guard names the missing pattern so the contributor knows
    exactly which test row to add.
    """
    from elspeth_lints.core.source_excerpt import _SECRET_PATTERNS

    pattern_names_in_module = {p.name for p in _SECRET_PATTERNS}
    pattern_names_in_contract = {case[0] for case in _PATTERN_CONTRACT_CASES}
    missing_from_contract = pattern_names_in_module - pattern_names_in_contract
    extras_in_contract = pattern_names_in_contract - pattern_names_in_module
    assert not missing_from_contract, (
        f"_SECRET_PATTERNS gained {sorted(missing_from_contract)} since "
        f"the contract test was last updated. Add a row to "
        f"_PATTERN_CONTRACT_CASES so the production-path redaction is "
        f"pinned end-to-end."
    )
    assert not extras_in_contract, (
        f"_PATTERN_CONTRACT_CASES references {sorted(extras_in_contract)} which no longer exist in _SECRET_PATTERNS. Remove the stale rows."
    )


# =====================================================================
# 3a. T8c — exact-length pattern trailing anchor (\b -> (?!\w))
# =====================================================================
#
# Pre-T8c the trailing anchor on ``openai_api_key`` and
# ``google_api_key`` was ``\b``. ``\b`` matches a transition between
# a word char and a non-word char (or string boundary adjacent to a
# word char). The vulnerability shape is specific to
# ``google_api_key`` because its body alphabet ``[A-Za-z0-9_-]{35}``
# includes ``-`` (non-word) AND it has a fixed length so the regex
# engine cannot extend the match. When a real key ends in ``-`` and is
# followed by another non-word char (``,`` / ``"`` / ``)`` / EOS),
# ``\b`` finds no transition and the literal slips through. ``(?!\w)``
# asserts only "no word char follows" — strictly stricter than ``\b``
# in this case AND honest at end-of-string. The change to
# ``openai_api_key`` is consistency-only (its body alphabet has no
# non-word chars so the empirical behaviour is unchanged).


def test_scrub_secrets_google_key_with_non_word_terminator() -> None:
    """T8c MINOR: google_api_key body ending in ``-`` redacts before non-word terminator.

    Pre-T8c with the ``\\b`` trailing anchor: if the 35th body char
    is ``-`` (non-word, allowed by the alphabet) and the source
    follows it with another non-word char (``,`` / ``"`` / ``)``),
    the boundary check requires a word/non-word transition and fails
    (both chars are non-word). Literal slips through unredacted.

    With ``(?!\\w)``: the lookahead requires only that the next char
    not be a word char. ``,`` is not a word char, so the lookahead
    succeeds and the match takes.

    This test plants the precise failure shape so a regression to
    ``\\b`` cannot land silently.
    """
    # Body of exactly 35 chars from the gated alphabet, ending in ``-``.
    body = "D" * 34 + "-"
    planted = "AIza" + body
    assert len(body) == 35  # contract: exactly the regex's body length
    # Follow the literal with another non-word char so ``\b`` fails.
    excerpt_text = f'config["google_key"] = ({planted},)'
    result = scrub_secrets(excerpt_text)
    assert planted not in result.text, (
        "google_api_key did not redact a body ending in ``-`` followed by "
        "a non-word char. The trailing anchor must be ``(?!\\w)`` not "
        "``\\b``: ``\\b`` requires a word/non-word transition and finds "
        "none when both sides are non-word."
    )
    pattern_names = {r.pattern_name for r in result.redactions}
    assert "google_api_key" in pattern_names


def test_scrub_secrets_google_key_at_eof_with_dash_terminator() -> None:
    """T8c MINOR: google_api_key body ending in ``-`` redacts at end-of-string.

    The other vulnerability shape: a key whose body ends in ``-`` and
    sits at the very end of the excerpt. ``\\b`` requires a
    neighbouring word char on at least one side. With ``-`` (non-word)
    on one side and string-end on the other there is no word char to
    anchor against, so the boundary check fails. ``(?!\\w)`` is
    trivially satisfied at end-of-string.
    """
    body = "X" * 34 + "-"
    planted = "AIza" + body
    excerpt_text = f"key at eof: {planted}"
    result = scrub_secrets(excerpt_text)
    assert planted not in result.text, (
        "google_api_key did not redact a body ending in ``-`` at end-of-string. ``(?!\\w)`` must hold at EOS where ``\\b`` does not."
    )
    pattern_names = {r.pattern_name for r in result.redactions}
    assert "google_api_key" in pattern_names


# =====================================================================
# 6. Empty / out-of-bounds windows (T8d regression — closes
#    elspeth-9bbb9df9a5 MAJOR introduced by T8c, commit d88b737f2).
#
# The scrub-before-render refactor in T8c added an invariant comparing
# the post-scrub line count to ``end - start + 1``. When the requested
# window is empty (zero-byte file, or stale line number past EOF) the
# raw window is ``""``, ``"".split("\n") == [""]`` (length 1), but
# ``end - start + 1`` is 0 or negative — the invariant misfires with a
# misleading ``"scrubber altered newline count"`` ``RuntimeError``.
#
# The pre-refactor code at commit ``a80992528`` handled empty windows
# cleanly by short-circuiting the rendering loop. The fix restores that
# semantic via an early-return for ``start > end``, BEFORE the raw
# window is materialised but AFTER the file fingerprint is hashed (the
# fingerprint is still a meaningful audit fact for a zero-byte file).
# =====================================================================


_SHA256_EMPTY_BYTES = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_extract_safe_excerpt_empty_file_returns_empty_excerpt(tmp_path: Path) -> None:
    """T8d regression: zero-byte file does NOT raise the scrubber invariant.

    Pre-refactor (commit ``a80992528``): empty file → empty rendered
    excerpt, no crash. Post-T8c (commit ``d88b737f2``): empty file →
    ``RuntimeError("scrubber altered newline count")`` because
    ``start > end`` makes ``expected_lines`` non-positive while
    ``scrubbed_lines`` is always ``[""]``. The fix early-returns an
    empty ``SafeExcerpt`` carrying the SHA-256 of zero bytes (a stable,
    meaningful fingerprint that the C8-3 binding path can still record).
    """
    root = tmp_path / "src_root"
    root.mkdir()
    empty = root / "plugins"
    empty.mkdir()
    empty_init = empty / "__init__.py"
    empty_init.write_bytes(b"")

    excerpt = extract_safe_excerpt(
        root=root,
        target_file=empty_init,
        line=1,
        context_lines=15,
    )

    assert excerpt.text == ""
    assert excerpt.redactions == ()
    # Fingerprint of zero bytes is the well-known SHA-256 constant —
    # NOT the empty string. The audit trail can still bind the entry
    # to the file's content even though the content is empty.
    assert excerpt.file_fingerprint == _SHA256_EMPTY_BYTES


def test_extract_safe_excerpt_single_line_file_renders_single_marker(tmp_path: Path) -> None:
    """T8d regression: single-line file is NOT an empty window — the invariant must hold.

    A 1-line file has ``len(lines) == 1``, ``start = max(1, 1-15) = 1``,
    ``end = min(1, 1+15) = 1`` → window of exactly one line. This is
    the SHALLOW non-empty case the early-return must NOT swallow:
    rendering must produce the ``>>``-marker line and the invariant
    must hold (scrubbed body length == expected window length == 1).
    Guards against an overly aggressive early-return that captures the
    1-line case as if it were the empty case.
    """
    root = tmp_path / "src_root"
    root.mkdir()
    single = root / "single.py"
    single.write_text("x = 1\n", encoding="utf-8")

    excerpt = extract_safe_excerpt(
        root=root,
        target_file=single,
        line=1,
        context_lines=15,
    )

    # The line-number prefix uses ``>>`` for the marker line; the
    # 5-digit zero-padded line number is the existing render format.
    assert excerpt.text == ">>     1  x = 1"
    assert excerpt.redactions == ()
    # Fingerprint is the SHA-256 of the on-disk bytes including the
    # trailing newline — confirms the read path was exercised, not
    # short-circuited.
    import hashlib

    assert excerpt.file_fingerprint == hashlib.sha256(b"x = 1\n").hexdigest()


def test_extract_safe_excerpt_out_of_bounds_line_returns_empty_excerpt(tmp_path: Path) -> None:
    """T8d regression: line past EOF returns empty SafeExcerpt, not RuntimeError.

    A 5-line file with ``line=100`` triggers ``start = max(1, 85) =
    85``, ``end = min(5, 115) = 5`` → ``start > end``. Pre-fix this
    raised the misleading ``"pattern must have matched across a
    newline"`` ``RuntimeError`` even though no pattern ran — the
    invariant arithmetic was the bug. Post-fix we return an empty
    ``SafeExcerpt`` with the file's true fingerprint so the caller
    can decide whether to treat this as ENTRY_OBSOLETE, a no-op
    justify, or an operator-actionable refusal.
    """
    root = tmp_path / "src_root"
    root.mkdir()
    five_line = root / "five.py"
    body = "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n"
    five_line.write_text(body, encoding="utf-8")

    excerpt = extract_safe_excerpt(
        root=root,
        target_file=five_line,
        line=100,
        context_lines=15,
    )

    assert excerpt.text == ""
    assert excerpt.redactions == ()
    import hashlib

    assert excerpt.file_fingerprint == hashlib.sha256(body.encode("utf-8")).hexdigest()


def test_reaudit_stale_entry_past_eof_does_not_crash(tmp_path: Path) -> None:
    """T8d regression: a stale allowlist entry past current EOF is per-entry handled.

    Constructs a reaudit fixture where the allowlist entry points at
    a key whose finding has been deleted (or whose line is past
    current EOF). The reaudit dispatch must NOT raise the
    ``"scrubber altered newline count"`` ``RuntimeError`` —
    ``_find_matching_finding`` already returns ``None`` for a deleted
    finding so the sweep classifies the entry as
    ``ENTRY_OBSOLETE`` before ``extract_safe_excerpt`` is reached.
    The test asserts that classification rather than a crash, and
    confirms zero exfil calls.

    This is the realistic reaudit shape: a finding the scanner used
    to flag is no longer present (file edited, finding fixed, line
    moved past EOF) — the sweep must continue cleanly.
    """
    root, target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    # Allowlist entry binds to a symbol that doesn't exist in the
    # current source — the post-refactor file. Mimics "the operator
    # fixed the finding but forgot to prune the allowlist entry".
    forged_key = "plugins/widget.py:R1:Widget.gone_method:fp=deadbeef"
    (allowlist_dir / "_stale.yaml").write_text(
        "allow_hits:\n"
        f"- key: {forged_key}\n"
        "  owner: someone\n"
        "  reason: |-\n"
        "    stale entry left over after the underlying finding was fixed\n"
        "  safety: |-\n"
        "    Suppression gated by cicd-judge; see judge_rationale below.\n"
        "  expires: '2030-01-01'\n",
        encoding="utf-8",
    )
    # Replace the synthetic source with a version that no longer has
    # the R1 finding the entry purports to suppress. The simplest way:
    # remove the offending method body. This also shrinks the file so
    # any cached line number would be past EOF.
    target.write_text('"""Trimmed module."""\n', encoding="utf-8")

    with _mock_judge_call() as client_class:
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=True,
        )

    assert len(report.outcomes) == 1
    outcome = report.outcomes[0]
    # No RuntimeError leaked — the sweep classified the entry.
    assert outcome.divergence is ReauditDivergence.ENTRY_OBSOLETE
    # No exfil: judge was never called.
    assert client_class.call_count == 0


def test_justify_against_empty_file_refuses_cleanly(tmp_path: Path) -> None:
    """T8d regression: ``justify`` against a zero-byte file exits 2, no crash.

    Operator typo / autocomplete picks an empty ``__init__.py``.
    Pre-fix path: ``_scan_single_file_findings`` returns no findings
    so the CLI exits 2 BEFORE reaching ``extract_safe_excerpt`` —
    but if a future caller bypassed that guard the invariant would
    have fired. This test pins the operator-visible behaviour
    (exit 2, no judge call) AND ensures no ``RuntimeError`` from the
    excerpt pipeline can surface even when the scanner-stage guard
    is removed (covered by the direct ``extract_safe_excerpt`` tests
    above).
    """
    root = tmp_path / "src_root"
    (root / "plugins").mkdir(parents=True)
    empty_init = root / "plugins" / "__init__.py"
    empty_init.write_bytes(b"")
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
                "plugins/__init__.py",
                "--symbol",
                "_module_",
                "--rationale",
                "empty-file probe",
                "--owner",
                "operator",
            ]
        )

    # Exit 2 because there is nothing to justify; not a crash.
    assert exit_code == 2
    # Judge was never reached.
    assert client_class.call_count == 0

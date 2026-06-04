# Judge Signature Diagnosis

`diagnose-judge-signatures` is the read-only repair-discovery command for
cicd-judge signed allowlist metadata. It is packaged with `elspeth-lints` and is
safe for agents to run because it never writes YAML and does not require
`ELSPETH_JUDGE_METADATA_HMAC_KEY`.

## Install

Refresh the package command in the repo uv environment:

```bash
uv sync --frozen --all-extras
```

The root project declares `elspeth-lints` as a local editable uv source, so a
normal sync installs `elspeth-lints` plus the standalone
`diagnose-judge-signatures` and `sign-judge-signatures` commands.

## Run

```bash
diagnose-judge-signatures \
  --root src/elspeth \
  --allowlist-dir config/cicd/enforce_tier_model
```

Use `--format json` for machine-readable output.

Use `--env-file <path>` to load diagnosis-relevant keys from an operator-held
dotenv file before the report runs:

```bash
diagnose-judge-signatures \
  --root src/elspeth \
  --allowlist-dir config/cicd/enforce_tier_model \
  --env-file /path/to/operator.env
```

The command only imports `ELSPETH_JUDGE_METADATA_HMAC_KEY` and
`ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE` from that file. Existing
environment values win, unrelated keys are ignored, and secret values are not
printed.

## Verification Modes

`verification_mode: shape-only` is used when
`ELSPETH_JUDGE_METADATA_HMAC_KEY` is absent. The command validates metadata
shape, missing signatures, v1 `file_fingerprint` drift, v2
`scope_fingerprint`/`ast_path` binding drift, and missing live findings. It
prints repair commands with the key shown only as
`ELSPETH_JUDGE_METADATA_HMAC_KEY=<operator-held-key>`.

`verification_mode: authoritative` is used when the operator-held HMAC key is
present. The same checks run, and the command also recomputes and compares
`judge_metadata_signature`.

Exit code `0` means every inspected entry is clean for the active verification
mode. Exit code `1` means at least one entry needs operator action. Exit code
`2` means command configuration or allowlist parsing failed.

## Repair

The diagnosis command only reports. For diagnosed signed-entry drift, use the
operator-only repair command:

```bash
sign-judge-signatures \
  --root src/elspeth \
  --allowlist-dir config/cicd/enforce_tier_model \
  --env-file /path/to/operator.env \
  --owner "$USER"
```

For a cleanup that also needs first-time justifications for newly live findings,
pass a manifest:

```bash
sign-judge-signatures \
  --root src/elspeth \
  --allowlist-dir config/cicd/enforce_tier_model \
  --env-file /path/to/operator.env \
  --owner "$USER" \
  --manifest docs/elspeth-lints/judge-signature-signing-2026-06-04.yaml
```

`sign-judge-signatures` imports only signing-relevant keys from the dotenv file
(`ELSPETH_JUDGE_METADATA_HMAC_KEY`, signature verify mode, judge API keys, and
operator override token fields). Existing environment values win and secret
values are not printed. It removes diagnosed stale signed rows before rerunning
`justify` so the allowlist loader does not trip over the old invalid binding.
If one `justify` call is rejected, the command records the failure, restores
that entry's stale row when one was removed, and continues with the remaining
entries. The command exits nonzero at the end and prints a failure summary; a
rerun skips entries that were successfully signed.

Manual single-entry repairs still go through the lower-level signing commands:

```bash
env ELSPETH_JUDGE_METADATA_HMAC_KEY=<operator-held-key> \
  elspeth-lints justify ...

env ELSPETH_JUDGE_METADATA_HMAC_KEY=<operator-held-key> \
  elspeth-lints migrate-judge-scope ...
```

Only an operator-held environment should run those commands because they write
signed metadata.

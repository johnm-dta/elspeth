# Security Policy

## Supported Versions

ELSPETH is currently on the `0.5.3` / RC-5.3 release-candidate line. Security
fixes are prioritised against the current release-candidate branch and `main`.
Older release-candidate snapshots are retained for provenance, but are not
maintained as separately supported long-term release lines.

## Reporting a Vulnerability

Do not open a public GitHub issue containing exploit details, secrets, personal
data, proof-of-compromise material, or instructions that would help a third
party reproduce a vulnerability before maintainers have acknowledged a safe
disclosure path.

Preferred disclosure path:

1. Use GitHub private vulnerability reporting for this repository if it is
   enabled.
2. If private vulnerability reporting is not enabled or is temporarily
   unavailable, contact the repository maintainer through an existing trusted
   project channel and request a temporary private disclosure path.
3. If no private route is available, open a public issue that says only
   "Security disclosure path requested" and includes no exploit detail.

At RC-5.2, a dedicated public security mailbox has not yet been published in
this repository. Publishing or enabling a permanent private disclosure channel
remains a public-release readiness item.

## What To Include

When a private channel is available, include:

- affected version, commit, branch, or container digest;
- affected component or endpoint;
- impact summary;
- reproduction steps or proof of concept;
- whether secrets, personal data, audit records, or external systems were
  exposed;
- logs or audit identifiers, with sensitive values redacted.

## Handling Expectations

Security reports are triaged according to the highest credible impact:

- audit-integrity, secret-disclosure, authentication, authorisation,
  supply-chain, and remote-code-execution issues are release-blocking until
  assessed;
- suspected contract violations in audit, lineage, trust-tier handling,
  redaction, or secret references are treated as security-relevant even when
  they do not resemble a conventional CVE-class vulnerability;
- public disclosure timing is coordinated with the reporter where possible.

## Scope

In scope:

- source code in this repository;
- GitHub Actions workflows and release artifacts;
- the web Composer, authentication, sessions, execution, and audit surfaces;
- official container images and published release bundles;
- documentation that makes security or assurance claims.

Out of scope:

- third-party provider infrastructure such as Azure OpenAI, OpenRouter,
  Microsoft Entra, ChromaDB, Dataverse, and GitHub;
- denial-of-service testing against any live government or staging system
  without explicit written authorisation;
- social engineering or physical access attempts.

## Public Release Note

The project is MIT licensed and currently pre-production outside the scoped
interim ATO statement described in `docs/release/executive-summary.md`. The
security posture is documented candidly in `docs/release/assessment-mapping.md`
and `docs/release/guarantees.md`.

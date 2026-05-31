# Governance

## Project Status

ELSPETH is an open-source, MIT-licensed, high-assurance pipeline platform on the
RC-5.2 release-candidate line. The repository is maintained as a public
open-source project for evaluation and pilot-adoption planning. It does not
claim a completed whole-platform independent assurance assessment.

## Decision Authority

Until a broader maintainer group is appointed, release authority sits with the
repository maintainer. Public-release approval should be recorded in the
relevant release document provenance block, including:

- approving maintainer or release approver;
- date;
- release branch or tag;
- commit hash;
- evidence bundle or CI run links;
- any explicit exceptions or residual risks.

## Design And Assurance Decisions

Load-bearing design decisions should be recorded in:

- `docs/architecture/adr/` for architectural decisions;
- `docs/contracts/` and `docs/release/guarantees.md` for contractual behaviour;
- `docs/runbooks/` for operational procedures;
- the project issue tracker for active release blockers and follow-up work.

For ELSPETH, assurance claims are engineering commitments. If a documented
guarantee is violated, it should be treated as a bug, and in many cases as a
release-blocking bug.

## Release Governance

Before a public release, the maintainer should confirm:

- the version, README badge, changelog, and release documents agree;
- `SECURITY.md`, `SUPPORT.md`, `CODE_OF_CONDUCT.md`, and this file are present;
- CI and branch-protection evidence are current;
- supply-chain artifacts and verification commands are published or explicitly
  listed as not yet available;
- residual risks are stated plainly, especially maintainer continuity,
  independent assurance, manual migration, and third-party provider risks.

## Maintainer Continuity

The current executive summary identifies maintainer continuity as a material
residual risk. This governance file does not remove that risk. It makes the
release authority and decision record explicit so future maintainers and
adopting organisations can see where decisions are meant to land.

## Security Governance

Vulnerability handling follows `SECURITY.md`. Security-relevant changes include
authentication, authorisation, audit integrity, redaction, secret resolution,
CI/CD gates, dependency management, container artifacts, and any documentation
that makes assurance claims.

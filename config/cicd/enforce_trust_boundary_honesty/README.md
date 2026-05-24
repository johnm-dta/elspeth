Trust-boundary honesty-gate allowlist
=====================================

This directory is the narrow escape valve for the `trust_boundary.tests`,
`trust_boundary.scope`, and `trust_boundary.tier` CI gates.

Only exact `allow_hits` entries are permitted. Every entry must be written with
judge metadata, source-file fingerprint, AST binding, HMAC signature, and an
expiry. `per_file_rules` are rejected by the rule loader.

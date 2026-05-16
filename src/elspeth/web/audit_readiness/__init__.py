"""Audit-readiness package — composition-time presentation of audit signals.

Read-only: no audit-trail writes happen here. Composes existing checks
(validation, catalog, secrets, retention) into a single panel snapshot.

Layer: L3 (application).
"""

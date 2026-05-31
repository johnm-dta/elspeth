"""Pin: the Prometheus extras that ``src/elspeth/web/app.py`` imports at
module top must remain available on this deployment.

Phase 8 introduced unconditional imports of
``opentelemetry.exporter.prometheus.PrometheusMetricReader`` and
``prometheus_client`` (CONTENT_TYPE_LATEST, generate_latest) at
``src/elspeth/web/app.py:21-23``.  The merge body documents that the
``[all]`` extra has a resolver conflict with these packages and that
they live in the ``[webui]`` extra instead.  This test fails the suite
loudly if a future extras reshuffle drops the prometheus packages from
the deployment image, instead of letting the regression surface as a
boot-time ``ImportError`` during staging deploy.
"""

from __future__ import annotations


def test_prometheus_extras_are_importable() -> None:
    """Hard-pin: the imports app.py does at module top must succeed."""
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    assert PrometheusMetricReader is not None
    # Exposition is plain-text per Prometheus spec; the constant is the
    # canonical content-type string app.py returns on the /metrics route.
    assert CONTENT_TYPE_LATEST.startswith("text/plain"), (
        f"prometheus_client.CONTENT_TYPE_LATEST has unexpected prefix: {CONTENT_TYPE_LATEST!r}.  /metrics route relies on this string."
    )
    assert callable(generate_latest), "generate_latest must be callable"


def test_app_imports_prometheus_extras_at_module_top() -> None:
    """Pin the import path itself — a refactor moving these imports into
    a lazy-import wrapper would silently mask the deployment dependency.
    Use ``__dict__`` membership (not ``hasattr``) per CLAUDE.md's
    unconditional ban on ``hasattr``.
    """
    import elspeth.web.app as app_module

    # These names must be present in the module namespace; if a future
    # refactor moves them into ``create_app()`` body or behind a lazy
    # wrapper, this test forces the move to be deliberate (update this
    # pin to point at the new path).
    assert "generate_latest" in app_module.__dict__, (
        "prometheus_client.generate_latest not imported at module scope "
        "in elspeth.web.app — the /metrics handler relies on it being "
        "imported once at module load."
    )
    assert "CONTENT_TYPE_LATEST" in app_module.__dict__, (
        "prometheus_client.CONTENT_TYPE_LATEST not imported at module "
        "scope in elspeth.web.app — the /metrics handler relies on the "
        "module-level constant."
    )

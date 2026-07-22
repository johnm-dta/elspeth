"""Pin: the Prometheus extras that ``src/elspeth/web/app.py`` imports at
module top must remain available on this deployment.

Phase 8 introduced unconditional imports of
``opentelemetry.exporter.prometheus.PrometheusMetricReader`` and
``prometheus_client`` (CONTENT_TYPE_LATEST, generate_latest) at
``src/elspeth/web/app.py:21-23``.  The release Dockerfile installs the
``[all]`` extra by default, so both ``[webui]`` and ``[all]`` must carry
these imports.  These tests fail locally if a future extras reshuffle
drops the packages, instead of letting the regression surface as a
boot-time ``ImportError`` during deployment.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_ROOT = Path(__file__).resolve().parents[3]
_PROMETHEUS_DEPENDENCIES = {
    "opentelemetry-exporter-prometheus",
    "prometheus-client",
}


def test_prometheus_dependencies_are_in_webui_and_all_extras() -> None:
    """The default release extra must include every unconditional web import."""
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        optional = tomllib.load(handle)["project"]["optional-dependencies"]

    for extra in ("webui", "all"):
        dependency_names = {canonicalize_name(Requirement(dependency).name) for dependency in optional[extra]}
        assert dependency_names >= _PROMETHEUS_DEPENDENCIES, (
            f"[{extra}] is missing release-critical dependencies: {sorted(_PROMETHEUS_DEPENDENCIES - dependency_names)}"
        )


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

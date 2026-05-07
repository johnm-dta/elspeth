"""Query construction for RAG retrieval transform.

Three modes, all anchored on query_field:
1. Field only: use field value verbatim
2. Field + template: render Jinja2 template with {{ query }} and {{ row }}
3. Field + regex: extract search text via capture group
"""

from __future__ import annotations

import multiprocessing as mp
import re
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jinja2 import Template

from jinja2 import TemplateSyntaxError, UndefinedError
from jinja2.exceptions import SecurityError

from elspeth.contracts.errors import TransformErrorReason
from elspeth.plugins.infrastructure.templates import (
    TemplateError,
    create_sandboxed_environment,
)


def _regex_worker(
    pattern: re.Pattern[str],
    text: str,
) -> tuple[bool, str | None, str | None]:
    """Run regex search in a worker process and return picklable result.

    Returns a 3-tuple: (matched: bool, group0: str | None, group1: str | None).
    re.Match objects are not picklable, so we extract the data before returning.

    This is a module-level function (not a method or closure) to ensure it is
    picklable across process boundaries — required by ProcessPoolExecutor.
    """
    match = pattern.search(text)
    if match is None:
        return (False, None, None)
    g0 = match.group(0)
    g1 = match.group(1) if pattern.groups else None
    return (True, g0, g1)


@dataclass(frozen=True)
class QueryResult:
    """Result of query construction."""

    query: str | None = None
    error: TransformErrorReason | None = None


class QueryBuilder:
    """Constructs search queries from row data.

    Supports three modes:
    - Field only: query_field set, no template or pattern
    - Template: query_field + query_template (Jinja2)
    - Regex: query_field + query_pattern (re capture group)

    Regex mode uses a ProcessPoolExecutor (max 1 worker) for timeout enforcement.
    Python threads cannot be interrupted while running C extension code (re module),
    so process-level isolation is the only reliable timeout mechanism for ReDoS.
    The pool amortizes process creation cost across all rows in the run.
    """

    def __init__(
        self,
        query_field: str,
        *,
        query_template: str | None = None,
        query_pattern: str | None = None,
        regex_timeout: float = 5.0,
    ) -> None:
        self._query_field = query_field
        self._regex_timeout = regex_timeout
        self._compiled_template: Template | None = None
        self._compiled_pattern: re.Pattern[str] | None = None
        self._regex_pool: ProcessPoolExecutor | None = None

        if query_template is not None:
            env = create_sandboxed_environment()
            try:
                self._compiled_template = env.from_string(query_template)
            except TemplateSyntaxError as e:
                raise TemplateError(f"Invalid query template syntax: {e}") from e

        if query_pattern is not None:
            self._compiled_pattern = re.compile(query_pattern)
            # Lazily create the process pool on first use. Single worker
            # bounds concurrent process count while amortizing spawn cost.
            self._regex_pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))

    def build(self, row_data: dict[str, Any]) -> QueryResult:
        """Construct a search query from row data."""
        extracted = row_data[self._query_field]

        if extracted is None:
            return QueryResult(
                error=TransformErrorReason(
                    reason="invalid_input",
                    field=self._query_field,
                    cause="null_value",
                )
            )

        if self._compiled_template is not None:
            return self._build_template(extracted, row_data)
        elif self._compiled_pattern is not None:
            return self._build_regex(extracted)
        else:
            return self._build_field_only(extracted)

    def _build_field_only(self, extracted: Any) -> QueryResult:
        if not isinstance(extracted, str):
            raise TypeError(
                f"query_field '{self._query_field}' expected str, got {type(extracted).__name__} "
                f"— upstream plugin bug (Tier 2 data must not be coerced)"
            )
        return self._validate_non_empty(extracted)

    def _build_template(self, extracted: Any, row_data: dict[str, Any]) -> QueryResult:
        assert self._compiled_template is not None  # guaranteed by build() guard
        try:
            query = self._compiled_template.render(query=extracted, row=row_data)
        except (UndefinedError, SecurityError, OverflowError, ZeroDivisionError, ArithmeticError, TypeError, ValueError) as e:
            return QueryResult(
                error=TransformErrorReason(
                    reason="template_rendering_failed",
                    error=str(e),
                    field=self._query_field,
                )
            )
        return self._validate_non_empty(query)

    def _build_regex(self, extracted: Any) -> QueryResult:
        assert self._compiled_pattern is not None  # guaranteed by build() guard
        assert self._regex_pool is not None  # created when pattern is compiled
        if not isinstance(extracted, str):
            raise TypeError(
                f"query_field '{self._query_field}' expected str, got {type(extracted).__name__} "
                f"— upstream plugin bug (Tier 2 data must not be coerced)"
            )

        future = self._regex_pool.submit(_regex_worker, self._compiled_pattern, extracted)
        try:
            matched, group0, group1 = future.result(timeout=self._regex_timeout)
        except FuturesTimeoutError:
            future.cancel()
            # future.cancel() only prevents queued tasks from starting — it
            # cannot kill a worker already executing C extension code (re module).
            # shutdown(wait=False) does NOT terminate running workers; the stuck
            # process stays alive as an orphan. We must explicitly kill it.
            #
            # _processes is a private dict[int, Process] — there is no public
            # API to force-kill pool workers in CPython's ProcessPoolExecutor.
            stuck_processes = list(self._regex_pool._processes.values())
            self._regex_pool.shutdown(wait=False, cancel_futures=True)
            for proc in stuck_processes:
                if proc.is_alive():
                    proc.kill()
            self._regex_pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
            return QueryResult(
                error=TransformErrorReason(
                    reason="no_regex_match",
                    field=self._query_field,
                    cause="regex_timeout",
                )
            )
        except Exception as exc:
            # _regex_worker is system-owned code — a crash is a code bug, not a data issue.
            raise RuntimeError(
                f"Regex worker failed while evaluating pattern "
                f"{self._compiled_pattern.pattern!r} against field "
                f"'{self._query_field}': {exc}. This is a bug in the regex "
                f"worker or the Python regex engine — not a data issue."
            ) from exc

        if not matched:
            return QueryResult(
                error=TransformErrorReason(
                    reason="no_regex_match",
                    field=self._query_field,
                    pattern=self._compiled_pattern.pattern,
                )
            )

        # Use group1 if the pattern has capture groups (may be None if non-participating)
        captured = group1 if self._compiled_pattern.groups else group0
        if captured is None:
            return QueryResult(
                error=TransformErrorReason(
                    reason="no_regex_match",
                    field=self._query_field,
                    cause="capture_group_empty",
                )
            )

        return self._validate_non_empty(captured)

    def _validate_non_empty(self, query: str) -> QueryResult:
        if not query.strip():
            return QueryResult(
                error=TransformErrorReason(
                    reason="invalid_input",
                    field=self._query_field,
                    cause="empty_query",
                )
            )
        return QueryResult(query=query)

    def close(self) -> None:
        """Shut down the regex process pool if one was created."""
        if self._regex_pool is not None:
            self._regex_pool.shutdown(wait=False)
            self._regex_pool = None

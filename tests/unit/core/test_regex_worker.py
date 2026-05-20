"""Tests for regex worker helpers used by process-pool isolation."""

from __future__ import annotations

import multiprocessing as mp
import re
from concurrent.futures import ProcessPoolExecutor

from elspeth.core.regex_worker import run_regex_worker


def test_regex_worker_zero_group_match() -> None:
    pattern = re.compile(r"foo")

    assert run_regex_worker(pattern, "foobar") == (True, "foo", None)


def test_regex_worker_one_group_match() -> None:
    pattern = re.compile(r"(foo)bar")

    assert run_regex_worker(pattern, "foobar") == (True, "foobar", "foo")


def test_regex_worker_no_match() -> None:
    pattern = re.compile(r"foo")

    assert run_regex_worker(pattern, "qux") == (False, None, None)


def test_regex_worker_round_trip_via_process_pool() -> None:
    pattern = re.compile(r"(issue):\s*(.+)")

    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        future = pool.submit(run_regex_worker, pattern, "issue: payment failed")

    assert future.result(timeout=5) == (True, "issue: payment failed", "issue")

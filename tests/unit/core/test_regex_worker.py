"""Tests for regex worker helpers used by process-pool isolation."""

from __future__ import annotations

import multiprocessing as mp
import re
from concurrent.futures import ProcessPoolExecutor

from elspeth.core.regex_worker import run_regex_worker


def test_regex_worker_zero_group_match() -> None:
    pattern = re.compile(r"foo")

    result = run_regex_worker(pattern, "foobar")

    assert result.matched is True
    assert result.full_match == "foo"
    assert result.groups == ()


def test_regex_worker_one_group_match() -> None:
    pattern = re.compile(r"(foo)bar")

    result = run_regex_worker(pattern, "foobar")

    assert result.matched is True
    assert result.full_match == "foobar"
    assert result.groups == ("foo",)


def test_regex_worker_multiple_group_match_is_policy_neutral() -> None:
    pattern = re.compile(r"([A-Z]+)-(\d+)")

    result = run_regex_worker(pattern, "ABC-123")

    assert result.matched is True
    assert result.full_match == "ABC-123"
    assert result.groups == ("ABC", "123")


def test_regex_worker_no_match() -> None:
    pattern = re.compile(r"foo")

    result = run_regex_worker(pattern, "qux")

    assert result.matched is False
    assert result.full_match is None
    assert result.groups == ()


def test_regex_worker_round_trip_via_process_pool() -> None:
    pattern = re.compile(r"(issue):\s*(.+)")

    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        future = pool.submit(run_regex_worker, pattern, "issue: payment failed")

    result = future.result(timeout=5)

    assert result.matched is True
    assert result.full_match == "issue: payment failed"
    assert result.groups == ("issue", "payment failed")

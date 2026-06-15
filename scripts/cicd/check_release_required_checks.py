"""Verify release image publication is backed by required GitHub checks.

The build-push workflow uses this gate before publishing images for a commit.
It reads the active repository ruleset, extracts the required status-check
contexts and optional GitHub App integration bindings, and then verifies that
the image SHA has successful trusted evidence for every required context.

GitHub ruleset contexts are not always identical to the check-run names exposed
by the Checks API. In the current repository, the ruleset context ``CodeQL`` is
reported by the CodeQL workflow's ``Analyze Python`` job. Keep that mapping
explicit so a future ruleset or workflow rename fails closed instead of silently
weakening release proof.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

GITHUB_API_ROOT = "https://api.github.com"
DEFAULT_CONTEXT_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "CodeQL": ("Analyze Python",),
    }
)


@dataclass(frozen=True, slots=True)
class RequiredCheckSpec:
    """Ruleset-required status check plus optional GitHub App provenance."""

    context: str
    integration_id: int | None = None


@dataclass(frozen=True, slots=True)
class CheckRun:
    """Normalized GitHub check-run data used for release proof."""

    name: str
    status: str
    conclusion: str | None
    html_url: str | None
    app_id: int | None = None
    completed_at: str | None = None
    started_at: str | None = None


@dataclass(frozen=True, slots=True)
class CommitStatus:
    """Normalized legacy commit-status data used for release proof."""

    context: str
    state: str
    target_url: str | None
    updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class RequiredCheckResult:
    """Evaluation result for one ruleset-required context."""

    context: str
    matched_name: str | None
    state: str
    url: str | None


def extract_required_contexts(ruleset: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract required status-check contexts from a repository ruleset payload."""
    return tuple(spec.context for spec in extract_required_checks(ruleset))


def extract_required_checks(ruleset: Mapping[str, Any]) -> tuple[RequiredCheckSpec, ...]:
    """Extract required status-check contexts and app bindings from a ruleset payload."""
    required: list[RequiredCheckSpec] = []
    rules = ruleset.get("rules")
    if not isinstance(rules, list):
        raise ValueError("ruleset payload does not contain a rules list")

    for rule in rules:
        if not isinstance(rule, Mapping) or rule.get("type") != "required_status_checks":
            continue
        parameters = rule.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("required_status_checks rule is missing parameters")
        required_checks = parameters.get("required_status_checks")
        if not isinstance(required_checks, list):
            raise ValueError("required_status_checks rule is missing required_status_checks list")
        for item in required_checks:
            if not isinstance(item, Mapping):
                raise ValueError("required status check entry is not an object")
            context = item.get("context")
            if not isinstance(context, str) or not context:
                raise ValueError("required status check entry has invalid context")
            integration_id = item.get("integration_id")
            if integration_id is not None and type(integration_id) is not int:
                raise ValueError("required status check entry has invalid integration_id")
            required.append(RequiredCheckSpec(context=context, integration_id=integration_id))

    if not required:
        raise ValueError("ruleset does not define any required status-check contexts")
    return tuple(dict.fromkeys(required))


def evaluate_required_checks(
    *,
    required_contexts: Sequence[str | RequiredCheckSpec],
    check_runs: Sequence[CheckRun],
    statuses: Sequence[CommitStatus],
    context_aliases: Mapping[str, Sequence[str]] = DEFAULT_CONTEXT_ALIASES,
) -> tuple[RequiredCheckResult, ...]:
    """Evaluate whether every required context has successful commit evidence."""
    results: list[RequiredCheckResult] = []
    for required in required_contexts:
        spec = _coerce_required_check(required)
        context = spec.context
        accepted_check_names = (context, *tuple(context_aliases.get(context, ())))
        check_run = _latest_check_run(check_runs, accepted_check_names, integration_id=spec.integration_id)
        if check_run is not None:
            results.append(
                RequiredCheckResult(
                    context=context,
                    matched_name=check_run.name,
                    state=_check_run_state(check_run),
                    url=check_run.html_url,
                )
            )
            continue

        if spec.integration_id is None:
            status = _latest_status(statuses, context)
            if status is not None:
                results.append(
                    RequiredCheckResult(
                        context=context,
                        matched_name=status.context,
                        state=status.state,
                        url=status.target_url,
                    )
                )
                continue

        results.append(RequiredCheckResult(context=context, matched_name=None, state="missing", url=None))
    return tuple(results)


def all_required_checks_succeeded(results: Iterable[RequiredCheckResult]) -> bool:
    """Return true only when every evaluated context succeeded."""
    return all(result.state == "success" for result in results)


def fetch_ruleset_by_name(*, repo: str, ruleset_name: str, token: str) -> Mapping[str, Any]:
    """Fetch one active repository ruleset by name."""
    rulesets_payload = _github_api_json(f"/repos/{repo}/rulesets", token=token)
    if not isinstance(rulesets_payload, list):
        raise ValueError("GitHub rulesets response was not a list")

    matches = [
        item
        for item in rulesets_payload
        if isinstance(item, Mapping)
        and item.get("name") == ruleset_name
        and item.get("target") == "branch"
        and item.get("enforcement") == "active"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one active branch ruleset named {ruleset_name!r}, found {len(matches)}")

    ruleset_id = matches[0].get("id")
    if not isinstance(ruleset_id, int):
        raise ValueError(f"ruleset {ruleset_name!r} has invalid id {ruleset_id!r}")

    ruleset_payload = _github_api_json(f"/repos/{repo}/rulesets/{ruleset_id}", token=token)
    if not isinstance(ruleset_payload, Mapping):
        raise ValueError(f"GitHub ruleset {ruleset_id} response was not an object")
    return ruleset_payload


def fetch_check_runs(*, repo: str, sha: str, token: str) -> tuple[CheckRun, ...]:
    """Fetch all check runs for a commit SHA."""
    payloads = _github_api_json_pages(f"/repos/{repo}/commits/{sha}/check-runs", token=token, params={"per_page": "100"})
    runs: list[CheckRun] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise ValueError("GitHub check-runs page was not an object")
        raw_runs = payload.get("check_runs")
        if not isinstance(raw_runs, list):
            raise ValueError("GitHub check-runs response is missing check_runs list")
        for item in raw_runs:
            if not isinstance(item, Mapping):
                raise ValueError("GitHub check run entry was not an object")
            runs.append(
                CheckRun(
                    name=_required_str(item, "name"),
                    status=_required_str(item, "status"),
                    conclusion=_optional_str(item.get("conclusion")),
                    html_url=_optional_str(item.get("html_url")),
                    app_id=_optional_app_id(item.get("app")),
                    completed_at=_optional_str(item.get("completed_at")),
                    started_at=_optional_str(item.get("started_at")),
                )
            )
    return tuple(runs)


def fetch_commit_statuses(*, repo: str, sha: str, token: str) -> tuple[CommitStatus, ...]:
    """Fetch legacy commit statuses for a commit SHA."""
    payload = _github_api_json(f"/repos/{repo}/commits/{sha}/status", token=token)
    if not isinstance(payload, Mapping):
        raise ValueError("GitHub commit status response was not an object")
    statuses = payload.get("statuses")
    if not isinstance(statuses, list):
        raise ValueError("GitHub commit status response is missing statuses list")

    result: list[CommitStatus] = []
    for item in statuses:
        if not isinstance(item, Mapping):
            raise ValueError("GitHub commit status entry was not an object")
        result.append(
            CommitStatus(
                context=_required_str(item, "context"),
                state=_required_str(item, "state"),
                target_url=_optional_str(item.get("target_url")),
                updated_at=_optional_str(item.get("updated_at")),
            )
        )
    return tuple(result)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Repository in owner/name form.")
    parser.add_argument("--sha", required=True, help="Commit SHA represented by the release image.")
    parser.add_argument("--ruleset-name", default="main", help="Active branch ruleset name to mirror.")
    args = parser.parse_args(argv)

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GH_TOKEN or GITHUB_TOKEN is required to verify release checks.", file=sys.stderr)
        return 2

    ruleset = fetch_ruleset_by_name(repo=args.repo, ruleset_name=args.ruleset_name, token=token)
    required_checks = extract_required_checks(ruleset)
    check_runs = fetch_check_runs(repo=args.repo, sha=args.sha, token=token)
    statuses = fetch_commit_statuses(repo=args.repo, sha=args.sha, token=token)
    results = evaluate_required_checks(required_contexts=required_checks, check_runs=check_runs, statuses=statuses)

    print(f"Required contexts from active ruleset {args.ruleset_name!r}:")
    for result in results:
        matched = result.matched_name or "<none>"
        url = result.url or "no URL"
        print(f"- {result.context}: {result.state} via {matched} ({url})")

    if all_required_checks_succeeded(results):
        print(f"All ruleset-required checks succeeded for {args.sha}.")
        return 0

    sys.stdout.flush()
    print(f"Required checks are missing, pending, or failed for {args.sha}; refusing to publish image.", file=sys.stderr)
    return 1


def _check_run_state(check_run: CheckRun) -> str:
    if check_run.status == "completed" and check_run.conclusion == "success":
        return "success"
    return check_run.conclusion or check_run.status


def _coerce_required_check(required: str | RequiredCheckSpec) -> RequiredCheckSpec:
    if isinstance(required, RequiredCheckSpec):
        return required
    return RequiredCheckSpec(context=required)


def _latest_check_run(
    check_runs: Sequence[CheckRun],
    accepted_names: Sequence[str],
    *,
    integration_id: int | None,
) -> CheckRun | None:
    matches = [run for run in check_runs if run.name in accepted_names and (integration_id is None or run.app_id == integration_id)]
    if not matches:
        return None
    return max(matches, key=lambda run: run.completed_at or run.started_at or "")


def _latest_status(statuses: Sequence[CommitStatus], context: str) -> CommitStatus | None:
    matches = [status for status in statuses if status.context == context]
    if not matches:
        return None
    return max(matches, key=lambda status: status.updated_at or "")


def _github_api_json(path: str, *, token: str) -> Any:
    with urllib.request.urlopen(_github_request(path, token=token), timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _github_api_json_pages(path: str, *, token: str, params: Mapping[str, str]) -> tuple[Any, ...]:
    next_url: str | None = _github_url(path, params=params)
    pages: list[Any] = []
    while next_url is not None:
        request = urllib.request.Request(
            next_url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            pages.append(json.loads(response.read().decode("utf-8")))
            next_url = _next_link(response.headers.get("Link"))
    return tuple(pages)


def _github_request(path: str, *, token: str) -> urllib.request.Request:
    return urllib.request.Request(
        _github_url(path, params={}),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )


def _github_url(path: str, *, params: Mapping[str, str]) -> str:
    query = urllib.parse.urlencode(params)
    url = f"{GITHUB_API_ROOT}{path}"
    if query:
        return f"{url}?{query}"
    return url


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        url_part, _, rel_part = part.strip().partition(";")
        if 'rel="next"' not in rel_part:
            continue
        return url_part.strip()[1:-1]
    return None


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"GitHub payload field {key!r} is missing or not a string")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"GitHub payload optional string field had invalid value {value!r}")
    return value


def _optional_app_id(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"GitHub check run app field had invalid value {value!r}")
    app_id = value.get("id")
    if app_id is None:
        return None
    if type(app_id) is not int:
        raise ValueError(f"GitHub check run app.id had invalid value {app_id!r}")
    return app_id


if __name__ == "__main__":
    sys.exit(main())

# Judge Quality Corpus

`v1.jsonl` is the labelled discrimination corpus for the `cicd-judge`.
Each record is a strict JSON object that becomes one `JudgeRequest`; the
expected label scores the returned `verdict` and `should_use_decorator`
exactly.

Use this corpus for prompt or model changes:

```bash
PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli \
  check-judge-quality \
  --corpus config/cicd/judge-quality-corpus/v1.jsonl \
  --min-accuracy 0.90
```

Cadence:

- Run the real-LLM gate before and after any judge prompt, model, or
  policy-context edit.
- Add labelled cases when a review finds a new judge failure mode; keep
  the corpus between 10 and 30 cases so the trusted CI gate remains
  bounded.
- Re-baseline expected labels only when the underlying policy changes or
  an operator-reviewed prompt change intentionally moves the decision
  boundary. Do not lower the CI threshold as a workaround for prompt
  drift; threshold changes are policy changes and need explicit review.

# Statistical Batch Plugins

Runnable local examples for the statistical batch-aware aggregation plugins. The
inputs are small CSV fixtures that stand in for prompt-evaluation, model-QA, or
batch-observability data. No LLM or external service is required.

## Run

Run any example from the repository root:

```bash
elspeth run --settings examples/statistical_batch_plugins/settings_distribution_profile.yaml --execute
elspeth run --settings examples/statistical_batch_plugins/settings_experiment_compare.yaml --execute
elspeth run --settings examples/statistical_batch_plugins/settings_classifier_metrics.yaml --execute
elspeth run --settings examples/statistical_batch_plugins/settings_paired_preference.yaml --execute
elspeth run --settings examples/statistical_batch_plugins/settings_drift_compare.yaml --execute
elspeth run --settings examples/statistical_batch_plugins/settings_outlier_annotator.yaml --execute
```

Outputs are written under `examples/statistical_batch_plugins/output/` as JSONL.
Audit databases are written under `examples/statistical_batch_plugins/runs/`.

## Examples

| Settings file | Plugin | What it shows |
| --- | --- | --- |
| `settings_distribution_profile.yaml` | `batch_distribution_profile` | Latency distribution summaries grouped by prompt variant |
| `settings_experiment_compare.yaml` | `batch_experiment_compare` | Unpaired control-vs-treatment mean score comparison |
| `settings_classifier_metrics.yaml` | `batch_classifier_metrics` | Confusion matrix, F-scores, and binary metrics for classifier outputs |
| `settings_paired_preference.yaml` | `batch_paired_preference` | Paired control-vs-treatment preference scores by case |
| `settings_drift_compare.yaml` | `batch_drift_compare` | Baseline-vs-current numeric distribution drift |
| `settings_outlier_annotator.yaml` | `batch_outlier_annotator` | Row annotations for latency outliers inside a batch |

These examples use pre-scored data rather than making an LLM call. To combine
them with OpenRouter, run an upstream LLM evaluation pipeline that emits fields
like `prompt_variant`, `score`, `actual_label`, or `predicted_label`, then point
one of these aggregation settings at that output.

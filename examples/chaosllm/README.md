# ChaosLLM Response Data

This directory contains pre-recorded LLM response data (`responses.jsonl`) used by the ChaosLLM test server.

This is **not a runnable pipeline example**. For pipelines that use ChaosLLM, see:

- [`chaosllm_sentiment`](../chaosllm_sentiment/) — Sentiment analysis against a local ChaosLLM server
- [`chaosllm_endurance`](../chaosllm_endurance/) — Multi-query medical assessment with fault injection

## What is ChaosLLM?

ChaosLLM is ELSPETH's built-in LLM fault injection server. It serves an OpenAI-compatible API that can inject configurable failures (rate limits, timeouts, malformed JSON, content filtering) to test pipeline resilience without requiring real API keys.

```bash
# Start the server
chaosllm serve --port 8199 --preset=realistic

# Then run a pipeline that points to it
elspeth run --settings examples/chaosllm_sentiment/settings.yaml --execute
```

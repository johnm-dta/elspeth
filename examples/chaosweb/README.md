# ChaosWeb Scraping Pipeline Example

Demonstrates web scraping resilience testing using ChaosWeb fault injection.

## What This Shows

A pipeline fetches 10 pages from a local ChaosWeb server. ChaosWeb injects
realistic faults — rate limits, forbidden responses, timeouts, malformed HTML —
and the pipeline routes failures to a dedicated sink while successful scrapes
pass through a content-length gate.

```
source ─(urls)─> scraper ─┬─(scraped)─> [content_check] ─┬─ output
                           │                               └─ review
                           └─(on_error)─> scrape_failures
```

## Running

**Terminal 1** — Start the ChaosWeb server:

```bash
chaosweb serve --preset=realistic --port=8200
```

**Terminal 2** — Run the pipeline:

```bash
elspeth run --settings examples/chaosweb/settings.yaml --execute
```

## Output

Results appear in `output/`:
- `scraped_pages.csv` — Successfully fetched pages with content and fingerprint
- `review.csv` — Pages with suspiciously short content (< 50 chars)
- `scrape_failures.csv` — Failed fetches (429, 403, 404, timeout, etc.)

## Preset Variations

Try different ChaosWeb presets to test different failure profiles:

```bash
# Gentle (5% error rate) — most requests succeed
chaosweb serve --preset=gentle --port=8200

# Realistic (15-20% error rate) — mixed failures
chaosweb serve --preset=realistic --port=8200

# Stress (40%+ error rate) — heavy fault injection
chaosweb serve --preset=stress_scraping --port=8200

# Custom overrides
chaosweb serve --rate-limit-pct=30 --forbidden-pct=10 --port=8200
```

## Audit Trail

After running, explore the audit trail:

```bash
elspeth explain --run <run_id> --row 1
```

Or start the MCP analysis server:

```bash
elspeth-mcp --database sqlite:///examples/chaosweb/runs/audit.db
```

## Schema Notes

The `web_scrape` transform uses `schema.mode: fixed` with explicit `fields` declaring
the input fields this transform consumes (`id`, `url`, `label`). This is required so
the static `WebScrapeOutput` contract carries those pass-through fields through to the
downstream CSV sinks, which declare them as required input. Using `mode: observed` with
`guaranteed_fields` does not propagate pass-through fields the same way — the
gate→CSV-sink edge would fail Phase 2 type-validation. See the engine's contract
propagation rules for details.

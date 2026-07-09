# Blob Transform Examples

This folder shows the blob-ingress pattern without adding an end-to-end test that
depends on live network behaviour.

Two combinations are included:

- `settings_expand_csv_blobs.yaml` — offline row expansion. A helper script
  stores existing local example CSV files in the payload store, writes a manifest
  of `blob_ref` rows, then `blob_csv_expand` emits one row per CSV record while
  preserving the manifest fields.
- `settings_fetch_tutorial_html.yaml` — opt-in public fetch. It reads the same
  three hosted tutorial HTML URLs used by the first-run tutorial and stores them
  as payload-store blobs with fetch metadata. It deliberately stops at blob refs;
  HTML parsing stays with `web_scrape`.

## Offline CSV Expansion

Prepare the manifest and payload-store blobs:

```bash
python examples/blob_transforms/scripts/prepare_csv_blob_manifest.py
```

Run the expansion pipeline:

```bash
elspeth run --settings examples/blob_transforms/settings_expand_csv_blobs.yaml --execute
```

Output:

- `examples/blob_transforms/output/expanded_csv_rows.csv`

The output keeps `source_url`, `source_name`, `manifest_index`, and `blob_ref` on
every emitted CSV row, so rows from multiple source blobs remain disambiguated.

## Hosted Tutorial HTML Fetch

This example uses the public GitHub Pages tutorial files:

- `https://johnm-dta.github.io/elspeth/tutorial-site/project-1.html`
- `https://johnm-dta.github.io/elspeth/tutorial-site/project-2.html`
- `https://johnm-dta.github.io/elspeth/tutorial-site/project-3.html`

Run:

```bash
elspeth run --settings examples/blob_transforms/settings_fetch_tutorial_html.yaml --execute
```

Output:

- `examples/blob_transforms/output/tutorial_html_blobs.jsonl`

This configuration explicitly adds `text/html` to `allowed_content_types` because
`blob_fetch` is a generic blob fetcher, not an HTML scraper. The web-authored
path still blocks private-network allowlists; this example uses the default
`public_only` SSRF policy.

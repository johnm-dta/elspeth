# Web Scrape Transform

Fetch webpages from URLs, convert content to markdown/text, and generate fingerprints for change detection.

## Configuration

```yaml
transforms:
  - plugin: web_scrape
    options:
      url_field: url
      content_field: page_content
      fingerprint_field: page_fingerprint
      format: markdown  # markdown | text | raw
      text_separator: " "  # text format only
      fingerprint_mode: content  # content | full

      http:
        abuse_contact: compliance@example.com
        scraping_reason: "Compliance monitoring"
        timeout: 30

      strip_elements:
        - script
        - style
        - nav
```

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `{content_field}` | str | Extracted content |
| `{fingerprint_field}` | str | SHA-256 fingerprint |
| `fetch_status` | int | HTTP status code |
| `fetch_url_final` | str | Final URL after redirects |

## Text Extraction And Line Splitting

`format: text` defaults to `text_separator: " "`, which returns compact plain
text — a single logical line.

When a downstream `line_explode` transform consumes `web_scrape` output, the
composer's semantic validator emits a `semantic_contracts` violation with
`requirement_code: line_explode.source_field.line_framed_text`. Composer agents
should call

```text
get_plugin_assistance(
    plugin_name="line_explode",
    issue_code="line_explode.source_field.line_framed_text",
)
```

to retrieve the current structured fix guidance — including before/after
configuration examples — directly from the plugin. This document no longer
hardcodes the fix; the plugin owns it.

## Security

- SSRF prevention (blocks private IPs, loopback, cloud metadata)
- Scheme whitelist (http/https only)
- SSL certificate verification (always enabled)

## Installation

```bash
uv pip install -e ".[web]"
```

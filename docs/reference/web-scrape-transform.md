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

## Security

- SSRF prevention (blocks private IPs, loopback, cloud metadata)
- Scheme whitelist (http/https only)
- SSL certificate verification (always enabled)

## Installation

```bash
uv pip install -e ".[web]"
```

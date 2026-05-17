"""Content extraction utilities for web scraping.

Converts HTML to markdown, text, or raw format with configurable
element stripping.
"""

import html2text
from bs4 import BeautifulSoup


def extract_content(
    html: str,
    format: str,
    strip_elements: list[str] | None = None,
    text_separator: str = " ",
) -> str:
    """Extract content from HTML in specified format.

    This is a Tier 3 trust boundary: ``html`` is external data and
    third-party libraries (BeautifulSoup, html2text) may raise
    ``AttributeError`` or ``TypeError`` on pathological input.  These
    are caught here and re-raised as ``ValueError`` so callers only
    need to handle the documented exception contract.

    Args:
        html: Raw HTML content (Tier 3 — untrusted)
        format: Output format ("markdown", "text", "raw")
        strip_elements: HTML tags to remove before extraction
        text_separator: Separator used between DOM text nodes for text output

    Returns:
        Extracted content as string

    Raises:
        ValueError: If format is invalid, or if HTML parsing/extraction
            fails due to malformed external content.
    """
    if format == "raw":
        return html

    try:
        # Parse HTML and strip unwanted elements
        soup = BeautifulSoup(html, "html.parser")

        if strip_elements:
            for tag_name in strip_elements:
                for tag in soup.find_all(tag_name):
                    tag.decompose()

        # Extract based on format
        if format == "markdown":
            # Get cleaned HTML back from soup
            cleaned_html = str(soup)

            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.body_width = 0  # Don't wrap lines
            h.ignore_tables = False
            h.ignore_emphasis = False

            return h.handle(cleaned_html)

        elif format == "text":
            return soup.get_text(separator=text_separator, strip=True)

        else:
            raise ValueError(f"Unknown format: {format}")
    except ValueError:
        raise
    except (AttributeError, TypeError) as exc:
        raise ValueError(f"HTML extraction failed on malformed content: {exc}") from exc

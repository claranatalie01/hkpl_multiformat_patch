"""Provide deterministic HTML-text helpers shared by web acquisition.

Both the bulk crawler and the single-URL administrator path use these helpers
before handing saved HTML to the normal reader/chunker pipeline. Centralizing
normalization keeps page titles, content-length checks, and hashes consistent
across those intentionally separate acquisition entry points.
"""

import re


def normalize_html_text(value: str) -> str:
    """Collapse HTML-derived whitespace into a single trimmed text line."""

    return re.sub(r"\s+", " ", value or "").strip()

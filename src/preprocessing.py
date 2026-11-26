"""Scientific text normalization utilities."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

# Punctuation characters stripped from the outer boundaries of a mention while
# keeping technical suffixes like '+' (C++), '#' (C#), or '%' intact.
_OUTER_STRIP_CHARS = set(",.;:!?\"'`~|\\/@#$%^&*=<>()[]{}")
_SAFE_TRAILING_CHARS = {"+", "#"}
_MATCHING_OPEN = {")": "(", "]": "[", "}": "{"}


def _strip_unbalanced_or_outer_punct(text: str) -> str:
    """Strip outer punctuation while preserving technical symbols and balanced brackets."""
    left = 0
    right = len(text)

    # Strip matching outer bracket wrapper like "(BERT)" -> "BERT"
    while (
        right - left >= 2
        and text[right - 1] in _MATCHING_OPEN
        and text[left] == _MATCHING_OPEN[text[right - 1]]
    ):
        left += 1
        right -= 1

    while left < right and text[left] in _OUTER_STRIP_CHARS:
        left += 1

    while right > left and text[right - 1] in _OUTER_STRIP_CHARS:
        ch = text[right - 1]
        if ch in _SAFE_TRAILING_CHARS:
            break
        # Preserve closing bracket if a matching opening bracket exists inside the span
        if ch in _MATCHING_OPEN:
            opener = _MATCHING_OPEN[ch]
            segment = text[left:right]
            if segment.count(opener) >= segment.count(ch):
                break
        right -= 1

    return text[left:right]

_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }
)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_mention_text(
    text: Any,
    *,
    lowercase: bool = True,
    unicode_normalization: str = "NFKC",
    strip_outer_punctuation: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Normalize a scientific mention string without erasing technical tokens.

    Preserves internal hyphens, slashes, numbers, and technical suffixes (e.g.,
    ``bert-base``, ``c++``, ``f1``, ``yolov3``) while standardizing case,
    Unicode representation, typographic dashes/quotes, whitespace, and outer
    punctuation.

    Args:
        text: Input mention text (handles None or NaN safely by returning ``""``).
        lowercase: Whether to lowercase the text.
        unicode_normalization: Unicode normalization form (e.g., ``"NFKC"``).
        strip_outer_punctuation: Whether to strip leading/trailing punctuation.
        collapse_whitespace: Whether to collapse repeated whitespace to single spaces.

    Returns:
        Normalized string.
    """
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""

    value = str(text)
    if unicode_normalization:
        value = unicodedata.normalize(unicode_normalization, value)

    value = value.translate(_DASH_TRANSLATION)

    if collapse_whitespace:
        value = _WHITESPACE_RE.sub(" ", value).strip()
    else:
        value = value.strip()

    if strip_outer_punctuation and value:
        value = _strip_unbalanced_or_outer_punct(value)
        if collapse_whitespace:
            value = _WHITESPACE_RE.sub(" ", value).strip()
        else:
            value = value.strip()

    if lowercase:
        value = value.lower()

    return value

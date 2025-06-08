# utils/text_normalise.py
"""
Normalisation helpers used by both analysis and whitelist construction.

• `normalise_keep_marks`  – lower-cases text, keeps Unicode Letters
  (Lu, Ll, Lt, Lm, Lo) and Marks (Mn, Mc, Me); every other code-point
  becomes a space.  Apostrophes / hyphens are *not* preserved.
• `extract_words`  – splits the normalised text on spaces to yield
  individual tokens.
"""
from __future__ import annotations
import unicodedata
import re
from typing import List

__all__ = ["normalise_keep_marks", "extract_words"]

_SPACE_RE = re.compile(r"\s+")


def normalise_keep_marks(text: str) -> str:
    """
    Lower-case *text* and replace every non-letter / non-mark character
    with a single space, then collapse runs of whitespace.

    Returns the cleaned string (may be empty).
    """
    buffer: list[str] = []
    for ch in text:
        cat0 = unicodedata.category(ch)[0]          # first letter of category
        if cat0 in ("L", "M"):
            buffer.append(ch.lower())
        else:
            buffer.append(" ")
    return _SPACE_RE.sub(" ", "".join(buffer)).strip()


def extract_words(normalised_text: str) -> List[str]:
    """
    Split a string already processed by `normalise_keep_marks`
    into tokens.  Empty tokens are discarded.
    """
    if not normalised_text:
        return []
    return [token for token in normalised_text.split(" ") if token]

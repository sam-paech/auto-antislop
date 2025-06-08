# utils/text_normalise.py
"""
Unicode-aware normalisation helpers shared by analysis *and* whitelist
construction.  They keep every Letter (Lu, Ll, Lt, Lm, Lo) **and**
every Mark (Mn, Mc, Me) while replacing everything else with a space.
"""
from __future__ import annotations
import unicodedata, re

__all__ = ["normalise_keep_marks", "extract_words"]

_SPACES_RE = re.compile(r"\s+")


def normalise_keep_marks(text: str) -> str:
    buf: list[str] = []
    for ch in text:
        cat0 = unicodedata.category(ch)[0]  # first letter of “Lu”, “Mn”, …
        if cat0 in ("L", "M") or ch in ("'", "-"):
            buf.append(ch.lower())
        else:
            buf.append(" ")
    return _SPACES_RE.sub(" ", "".join(buf)).strip()


_WORD_RE = re.compile(r"[A-Za-z\p{Mn}\p{Mc}\p{Me}']+", re.UNICODE)


def extract_words(text: str) -> list[str]:
    """
    Return lower-cased tokens consisting of letters + marks + apostrophes.
    """
    # Python `re` cannot use \p{Mn} natively <3.12; the pattern above works
    # in 3.12+.  For 3.11 fallback to simple split.
    try:
        return [m.group(0) for m in _WORD_RE.finditer(text)]
    except re.error:      # <3.12
        return text.split()

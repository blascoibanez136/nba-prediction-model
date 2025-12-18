# src/utils/team_name_normalizer.py
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


_PUNCT_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizedName:
    raw: str
    normalized: str


def normalize_team_name(name: Optional[str]) -> NormalizedName:
    """
    Report-only normalization for *bucketing* strings.
    This does NOT enforce canonicalization in predictions/backtests.
    Deterministic and intentionally conservative.

    Steps:
      - unicode normalize (NFKD) + strip accents
      - lowercase
      - replace & with 'and'
      - remove punctuation
      - collapse whitespace
    """
    raw = "" if name is None else str(name)

    # Unicode normalize + strip diacritics for stability
    s = unicodedata.normalize("NFKD", raw)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = s.strip().lower()
    s = s.replace("&", " and ")

    # Remove punctuation, keep alnum + whitespace
    s = _PUNCT_RE.sub(" ", s)

    # Collapse whitespace
    s = _WS_RE.sub(" ", s).strip()

    return NormalizedName(raw=raw, normalized=s)

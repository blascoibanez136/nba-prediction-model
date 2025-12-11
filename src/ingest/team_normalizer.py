"""
Team name normalization utilities.

Different data sources (balldontlie, API-NBA, The Odds API, etc.) can use
slightly different strings for the same NBA team:

    - "LA Clippers" vs "Los Angeles Clippers"
    - "LA Lakers" vs "Los Angeles Lakers"
    - "NY Knicks" vs "New York Knicks", etc.

This module provides a single function:

    normalize_team_name(name: str) -> str

which maps a variety of aliases down to a canonical, lowercase form such as:

    "los angeles clippers"
    "new york knicks"

All merge_key construction and cross-source joins should go through this
function so that we never miss games due to naming mismatches.
"""

from __future__ import annotations

import re
from typing import Dict

# Canonical full franchise names (lowercase)
_CANONICAL_TEAMS = {
    "atlanta hawks",
    "boston celtics",
    "brooklyn nets",
    "charlotte hornets",
    "chicago bulls",
    "cleveland cavaliers",
    "dallas mavericks",
    "denver nuggets",
    "detroit pistons",
    "golden state warriors",
    "houston rockets",
    "indiana pacers",
    "los angeles clippers",
    "los angeles lakers",
    "memphis grizzlies",
    "miami heat",
    "milwaukee bucks",
    "minnesota timberwolves",
    "new orleans pelicans",
    "new york knicks",
    "oklahoma city thunder",
    "orlando magic",
    "philadelphia 76ers",
    "phoenix suns",
    "portland trail blazers",
    "sacramento kings",
    "san antonio spurs",
    "toronto raptors",
    "utah jazz",
    "washington wizards",
}

# Common aliases mapped to canonical names.
# NOTE: All keys and values should be lowercase and punctuation-free.
_ALIAS_MAP: Dict[str, str] = {
    # Clippers
    "la clippers": "los angeles clippers",
    "l a clippers": "los angeles clippers",
    "los angeles clippers": "los angeles clippers",
    "lac": "los angeles clippers",

    # Lakers
    "la lakers": "los angeles lakers",
    "l a lakers": "los angeles lakers",
    "los angeles lakers": "los angeles lakers",
    "lal": "los angeles lakers",

    # Knicks
    "ny knicks": "new york knicks",
    "n y knicks": "new york knicks",
    "new york knicks": "new york knicks",
    "nyk": "new york knicks",

    # Warriors
    "gs warriors": "golden state warriors",
    "g s warriors": "golden state warriors",
    "golden state warriors": "golden state warriors",
    "golden st warriors": "golden state warriors",
    "gsw": "golden state warriors",

    # Spurs
    "sa spurs": "san antonio spurs",
    "s a spurs": "san antonio spurs",
    "san antonio spurs": "san antonio spurs",
    "sas": "san antonio spurs",

    # Thunder
    "okc thunder": "oklahoma city thunder",
    "o k c thunder": "oklahoma city thunder",
    "oklahoma city thunder": "oklahoma city thunder",
    "okc": "oklahoma city thunder",

    # Pelicans
    "no pelicans": "new orleans pelicans",
    "n o pelicans": "new orleans pelicans",
    "new orleans pelicans": "new orleans pelicans",

    # 76ers
    "philadelphia sixers": "philadelphia 76ers",
    "philadelphia 76ers": "philadelphia 76ers",
    "philly 76ers": "philadelphia 76ers",
    "sixers": "philadelphia 76ers",

    # Simple ones where sources usually agree
    "atlanta hawks": "atlanta hawks",
    "boston celtics": "boston celtics",
    "brooklyn nets": "brooklyn nets",
    "charlotte hornets": "charlotte hornets",
    "chicago bulls": "chicago bulls",
    "cleveland cavaliers": "cleveland cavaliers",
    "dallas mavericks": "dallas mavericks",
    "denver nuggets": "denver nuggets",
    "detroit pistons": "detroit pistons",
    "houston rockets": "houston rockets",
    "indiana pacers": "indiana pacers",
    "memphis grizzlies": "memphis grizzlies",
    "miami heat": "miami heat",
    "milwaukee bucks": "milwaukee bucks",
    "minnesota timberwolves": "minnesota timberwolves",
    "orlando magic": "orlando magic",
    "phoenix suns": "phoenix suns",
    "portland trail blazers": "portland trail blazers",
    "sacramento kings": "sacramento kings",
    "toronto raptors": "toronto raptors",
    "utah jazz": "utah jazz",
    "washington wizards": "washington wizards",
}


def _clean(name: str) -> str:
    """
    Lowercase, remove punctuation, collapse whitespace.

    Turns things like "L.A. Clippers" into "la clippers".
    """
    s = str(name or "").strip().lower()
    # Remove punctuation except spaces
    s = re.sub(r"[^\w\s]", " ", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_team_name(name: str) -> str:
    """
    Normalize a raw team name string into a canonical, lowercase name.

    If the name is unknown, returns the cleaned version as-is so that
    behavior is still deterministic.
    """
    cleaned = _clean(name)
    if not cleaned:
        return ""

    # Alias map first
    if cleaned in _ALIAS_MAP:
        return _ALIAS_MAP[cleaned]

    # If it's already a canonical team, keep it
    if cleaned in _CANONICAL_TEAMS:
        return cleaned

    # Fallback: just return the cleaned form
    return cleaned

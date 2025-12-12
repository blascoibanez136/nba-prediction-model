from __future__ import annotations
import re

def canonical_team(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9\s]", "", s)   # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()  # normalize whitespace
    return s

def make_merge_key(home_team: str, away_team: str, game_date: str) -> str:
    return f"{canonical_team(home_team)}__{canonical_team(away_team)}__{str(game_date)[:10]}"

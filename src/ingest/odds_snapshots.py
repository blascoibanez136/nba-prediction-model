"""
Odds Movement & Dispersion utilities for NBA games (The Odds API, nested `bookmakers`).
- Saves time-stamped snapshots (open/mid/close)
- Flattens spreads per game/book/team
- Computes movement (open→close), velocity, and dispersion/consensus
- Produces a merge_key = norm(home) + "__" + norm(away) + "__" + game_date (UTC)
"""

from __future__ import annotations

import os
import json
import ast
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# existing odds fetcher using ODDS_API_KEY
from src.ingest.odds_ingest import get_nba_odds  # type: ignore

SNAPSHOT_DIR = os.path.join("data", "_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# -----------------------
# helpers
# -----------------------
def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M")

def _parse_snapshot_timestamp(path: str) -> datetime:
    # filename pattern: kind_YYYYMMDD_HHMM.csv
    base = os.path.basename(path)
    # e.g., "open_20251112_1830.csv"
    try:
        dt_part = base.split("_", 1)[1].split(".")[0]  # "20251112_1830"
        return datetime.strptime(dt_part, "%Y%m%d_%H%M")
    except Exception as e:
        # Fallback: now
        return datetime.utcnow()

def _date_utc_from_commence(ts: str) -> str:
    # The Odds API commence_time is ISO8601; support "Z" or explicit offset
    # Example: "2025-11-12T00:30:00Z" or "2025-11-12T00:30:00+00:00"
    ts = str(ts)
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d")

def _norm_team(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    aliases = {
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "ny knicks": "new york knicks",
        # add more aliases as you discover them
    }
    return aliases.get(s, s)

def _merge_key(home_team: str, away_team: str, game_date: str) -> str:
    return f"{_norm_team(home_team)}__{_norm_team(away_team)}__{game_date}"


# -----------------------
# snapshot I/O
# -----------------------
def save_snapshot(kind: str = "open") -> str:
    """
    Fetch current NBA odds and save to data/_snapshots/{kind}_YYYYMMDD_HHMM.csv
    (Raw The Odds API rows with nested 'bookmakers' JSON)
    """
    odds = get_nba_odds()
    df = pd.DataFrame(odds)
    ts = _timestamp()
    filename = f"{kind}_{ts}.csv"
    path = os.path.join(SNAPSHOT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"✅ saved {len(df)} rows to {path}")
    return path


# -----------------------
# flatten spreads
# -----------------------
def _to_list(obj):
    """Parse 'bookmakers' column reliably from CSV or live JSON."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str):
        # Try JSON first; fallback to safe literal_eval
        try:
            return json.loads(obj)
        except Exception:
            try:
                return ast.literal_eval(obj)
            except Exception:
                return []
    return []

def flatten_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """
    The Odds API returns a 'bookmakers' list per game.
    Turn it into one row per game + book + team with a spread (and price if present).
    Output columns: game_id, book, team, spread, price, home_team, away_team, commence_time
    """
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        game_id = row.get("id") or row.get("game_id")
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        commence_time = row.get("commence_time")
        books = _to_list(row.get("bookmakers", []))

        if not game_id or not home_team or not away_team:
            continue

        for bk in books:
            book_key = bk.get("key") or bk.get("title") or "unknown"
            for market in bk.get("markets", []):
                if (market.get("key") or "").lower() != "spreads":
                    continue
                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            "game_id": str(game_id),
                            "book": book_key,
                            "team": outcome.get("name"),
                            "spread": pd.to_numeric(outcome.get("point"), errors="coerce"),
                            "price": pd.to_numeric(outcome.get("price"), errors="coerce"),
                            "home_team": home_team,
                            "away_team": away_team,
                            "commence_time": commence_time,
                        }
                    )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["game_id"] = out["game_id"].astype(str)
    return out


# -----------------------
# movement & velocity
# -----------------------
def compute_movement(open_csv_path: str, close_csv_path: str) -> pd.DataFrame:
    """
    Load two snapshot CSVs, flatten both, align, and compute:
      - move_open_to_close
      - intraday_velocity (movement / hours)
    """
    open_df = pd.read_csv(open_csv_path)
    close_df = pd.read_csv(close_csv_path)

    open_flat = flatten_spreads(open_df)
    close_flat = flatten_spreads(close_df)

    merged = pd.merge(
        open_flat,
        close_flat,
        on=["game_id", "book", "team"],
        suffixes=("_open", "_close"),
    )

    merged["move_open_to_close"] = merged["spread_close"] - merged["spread_open"]

    try:
        t_open = _parse_snapshot_timestamp(open_csv_path)
        t_close = _parse_snapshot_timestamp(close_csv_path)
        hours = max((t_close - t_open).total_seconds() / 3600.0, 0.1)
    except Exception:
        hours = 1.0

    merged["intraday_velocity"] = merged["move_open_to_close"] / hours

    keep = [
        "game_id", "book", "team",
        "spread_open", "spread_close",
        "move_open_to_close", "intraday_velocity",
        "home_team_open", "away_team_open", "commence_time_open",
    ]
    # carry teams/commence from _open (they should match)
    for c in ["home_team", "away_team", "commence_time"]:
        merged[f"{c}_open"] = merged.get(f"{c}_open", merged.get(f"{c}_close"))

    return merged[[k for k in keep if k in merged.columns]]


# -----------------------
# dispersion & consensus
# -----------------------
def compute_dispersion(close_csv_path: str, allowed_books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Build dispersion and consensus close per game (HOME perspective),
    enriched with:
      - home_team, away_team
      - game_date (UTC from commence_time)
      - merge_key for cross-source joins
    Output columns: merge_key, book_dispersion, consensus_close, home_team, away_team, game_date
    """
    raw = pd.read_csv(close_csv_path)
    flat = flatten_spreads(raw)
    if flat.empty:
        return pd.DataFrame(columns=["merge_key","book_dispersion","consensus_close","home_team","away_team","game_date"])

    if allowed_books:
        flat = flat[flat["book"].isin(allowed_books)]

    # HOME perspective
    is_home = flat["team"].astype(str).str.lower() == flat["home_team"].astype(str).str.lower()
    flat["spread_home"] = np.where(is_home, flat["spread"], -flat["spread"])

    # UTC date for merge_key
    flat["game_date"] = flat["commence_time"].apply(_date_utc_from_commence)

    agg = (
        flat.groupby(["home_team","away_team","game_date"], as_index=False)
            .agg(book_dispersion=("spread_home","std"),
                 consensus_close=("spread_home","median"))
    )
    # If only one book → std is NaN. Treat as 0.0 dispersion.
    agg["book_dispersion"] = agg["book_dispersion"].fillna(0.0)

    agg["merge_key"] = agg.apply(lambda r: _merge_key(r["home_team"], r["away_team"], r["game_date"]), axis=1)
    return agg[["merge_key","book_dispersion","consensus_close","home_team","away_team","game_date"]]


# -----------------------
# CLI entry
# -----------------------
if __name__ == "__main__":
    kind = os.getenv("SNAPSHOT_KIND", "open")
    save_snapshot(kind)

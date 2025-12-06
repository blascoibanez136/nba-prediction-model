"""
Odds Movement & Dispersion utilities for NBA games (The Odds API, nested `bookmakers`).
"""

from __future__ import annotations

import os
import ast
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# we assume you already have this in your repo
# and that it uses ODDS_API_KEY from env
from src.ingest.odds_ingest import get_nba_odds  # type: ignore

SNAPSHOT_DIR = os.path.join("data", "_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M")


def save_snapshot(kind: str = "open") -> str:
    """
    Fetch current NBA odds and save to data/_snapshots/{kind}_YYYYMMDD_HHMM.csv
    """
    odds = get_nba_odds()
    df = pd.DataFrame(odds)
    ts = _timestamp()
    filename = f"{kind}_{ts}.csv"
    path = os.path.join(SNAPSHOT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"✅ saved {len(df)} rows to {path}")
    return path


def flatten_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """
    The Odds API returns a 'bookmakers' list per game.
    Turn it into one row per game + book + team with a spread.
    """
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        game_id = row.get("id")
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        books = row.get("bookmakers", [])

        # when loaded from CSV this is often a string
        if isinstance(books, str):
            try:
                books = ast.literal_eval(books)
            except Exception:
                books = []

        for bk in books:
            book_key = bk.get("key")
            for market in bk.get("markets", []):
                if market.get("key") == "spreads":
                    for outcome in market.get("outcomes", []):
                        rows.append(
                            {
                                "game_id": game_id,
                                "book": book_key,
                                "team": outcome.get("name"),
                                "spread": outcome.get("point"),
                                "price": outcome.get("price"),
                                "home_team": home_team,
                                "away_team": away_team,
                            }
                        )
    return pd.DataFrame(rows)


def _parse_snapshot_timestamp(path: str) -> datetime:
    # filename pattern: kind_YYYYMMDD_HHMM.csv
    base = os.path.basename(path)
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(f"Bad snapshot filename: {base}")
    dt_part = parts[1].split(".")[0]
    return datetime.strptime(dt_part, "%Y%m%d_%H%M")


def compute_movement(open_csv_path: str, close_csv_path: str) -> pd.DataFrame:
    """
    load two snapshot CSVs, flatten both, align, and compute:
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

    merged["move_open_to_close"] = (
        merged["spread_close"] - merged["spread_open"]
    )

    # time delta
    try:
        t_open = _parse_snapshot_timestamp(open_csv_path)
        t_close = _parse_snapshot_timestamp(close_csv_path)
        hours = max((t_close - t_open).total_seconds() / 3600.0, 0.1)
    except Exception:
        hours = 1.0

    merged["intraday_velocity"] = merged["move_open_to_close"] / hours

    return merged[
        [
            "game_id",
            "book",
            "team",
            "spread_open",
            "spread_close",
            "move_open_to_close",
            "intraday_velocity",
        ]
    ]


import pandas as pd
import numpy as np
from datetime import datetime, timezone

def _norm_team(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    aliases = {
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "ny knicks": "new york knicks",
        # add more aliases as needed
    }
    return aliases.get(s, s)

def _merge_key(home_team: str, away_team: str, game_date: str) -> str:
    return f"{_norm_team(home_team)}__{_norm_team(away_team)}__{game_date}"

def _date_utc_from_commence(ts: str) -> str:
    # The Odds API commence_time is ISO8601; we convert to UTC date (YYYY-MM-DD)
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d")

def compute_dispersion(close_csv_path: str, allowed_books: list[str] | None = None) -> pd.DataFrame:
    raw = pd.read_csv(close_csv_path)

    # Flatten 'bookmakers' (JSON) for spreads into tidy rows
    # Expected columns: id, commence_time, home_team, away_team, bookmakers
    rows = []
    for _, r in raw.iterrows():
        game_id = str(r.get("id") or r.get("game_id") or "")
        home = r.get("home_team")
        away = r.get("away_team")
        commence = r.get("commence_time")
        books = r.get("bookmakers")
        if pd.isna(books) or not home or not away or not commence:
            continue
        try:
            books = json.loads(books) if isinstance(books, str) else books
        except Exception:
            continue
        if not isinstance(books, list): 
            continue
        game_date = _date_utc_from_commence(str(commence))
        for bk in books:
            book_key = bk.get("key") or bk.get("title") or "unknown"
            if allowed_books and book_key not in allowed_books:
                continue
            for m in bk.get("markets", []):
                if (m.get("key") or "").lower() != "spreads":
                    continue
                for outc in m.get("outcomes", []):
                    team = outc.get("name")
                    point = outc.get("point")
                    # We keep spread values; we’ll compute dispersion across books
                    rows.append({
                        "game_id": game_id,
                        "book": book_key,
                        "home_team": home,
                        "away_team": away,
                        "game_date": game_date,
                        "team": team,
                        "spread": pd.to_numeric(point, errors="coerce"),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["merge_key","book_dispersion","consensus_close","home_team","away_team","game_date"])

    # Convert spreads to HOME perspective:
    # For home rows: spread_home = spread
    # For away rows: spread_home = -spread
    is_home = df["team"].astype(str).str.lower() == df["home_team"].astype(str).str.lower()
    df["spread_home"] = np.where(is_home, df["spread"], -df["spread"])

    # Compute dispersion and consensus close across books per game
    agg = (
        df.groupby(["home_team","away_team","game_date"], as_index=False)
          .agg(book_dispersion=("spread_home", "std"),
               consensus_close=("spread_home", "median"))
    )
    agg["merge_key"] = agg.apply(lambda r: _merge_key(r["home_team"], r["away_team"], r["game_date"]), axis=1)
    return agg[["merge_key","book_dispersion","consensus_close","home_team","away_team","game_date"]]


if __name__ == "__main__":
    kind = os.getenv("SNAPSHOT_KIND", "open")
    save_snapshot(kind)

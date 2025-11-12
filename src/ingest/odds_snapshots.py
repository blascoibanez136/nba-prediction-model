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


def compute_dispersion(
    close_csv_path: str, allowed_books: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    for one snapshot, compute:
    - book_dispersion = std of spreads across books
    - consensus_close = median spread across books
    """
    close_df = pd.read_csv(close_csv_path)
    flat = flatten_spreads(close_df)

    if allowed_books:
        flat = flat[flat["book"].isin(allowed_books)].reset_index(drop=True)

    disp = flat.groupby("game_id")["spread"].std().fillna(0).rename("book_dispersion")
    consensus = flat.groupby("game_id")["spread"].median().rename("consensus_close")

    out = pd.concat([disp, consensus], axis=1).reset_index()
    return out


if __name__ == "__main__":
    kind = os.getenv("SNAPSHOT_KIND", "open")
    save_snapshot(kind)

"""
Odds snapshot system for NBA Pro-Lite model.

This module:
    • Pulls raw odds from The Odds API via get_nba_odds()
    • Saves timestamped snapshots (open/mid/close)
    • Loads latest snapshots
    • Flattens nested bookmaker → market → outcome structures
    • Computes dispersion + consensus lines

Snapshots are ALWAYS saved relative to the repository root:
    <repo_root>/data/_snapshots/

This avoids incorrect relative paths when running in Colab,
GitHub Actions, Docker, or any external environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

from src.ingest.odds_ingest import get_nba_odds  # type: ignore


# -------------------------------------------------------------------
# 🔥 PATCH: Determine repo root dynamically for consistent save paths
# -------------------------------------------------------------------
# This file lives at: <repo>/src/ingest/odds_snapshots.py
# repo root = two directories above this file
REPO_ROOT = Path(__file__).resolve().parents[2]

# Final absolute snapshot location
OUT_DIR = REPO_ROOT / "data" / "_snapshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Snapshot creation
# -------------------------------------------------------------------
def save_snapshot(kind: str) -> Path:
    """
    Fetch current NBA odds and save a snapshot CSV under data/_snapshots.

    `kind` is a label like "open", "mid", or "close".
    """
    # Ensure the snapshot directory exists
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fn = SNAPSHOT_DIR / f"{kind}_{stamp}.csv"

    odds = get_nba_odds()
    df = pd.DataFrame(odds)

    df.to_csv(fn, index=False)
    print(f"✅ saved {len(df)} rows to {fn.relative_to(REPO_ROOT)}")
    return fn



# -------------------------------------------------------------------
# Load most recent snapshot
# -------------------------------------------------------------------
def load_latest_snapshot(kind: str) -> pd.DataFrame:
    """
    Load the newest snapshot for the given kind.

    Parameters
    ----------
    kind : str
        e.g., "open", "mid", "close"

    Returns
    -------
    pd.DataFrame
    """
    files = sorted(OUT_DIR.glob(f"{kind}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No snapshots found for '{kind}'.")
    return pd.read_csv(files[-1])


# -------------------------------------------------------------------
# Flatten spreads + totals from nested JSON
# -------------------------------------------------------------------
def flatten_spreads(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert nested bookmaker/market/outcome structures into rows like:

    game_id | book | market | team | price | point

    Returns
    -------
    pd.DataFrame
    """
    rows = []

    for _, row in raw_df.iterrows():
        game_id = row.get("id")
        home = row.get("home_team")
        away = row.get("away_team")
        start = row.get("commence_time")

        bookmakers = row.get("bookmakers", [])
        if not isinstance(bookmakers, list):
            continue

        for book in bookmakers:
            book_key = book.get("key")
            markets = book.get("markets", [])

            for m in markets:
                market_key = m.get("key")  # spreads, totals, h2h
                outcomes = m.get("outcomes", [])

                for o in outcomes:
                    rows.append({
                        "game_id": game_id,
                        "home_team": home,
                        "away_team": away,
                        "commence_time": start,
                        "book": book_key,
                        "market": market_key,
                        "team": o.get("name"),
                        "price": o.get("price"),
                        "point": o.get("point"),
                    })

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Compute dispersion + consensus
# -------------------------------------------------------------------
def compute_dispersion(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute:
        • spread dispersion
        • consensus close
        • merge keys for joining into prediction DataFrames

    Parameters
    ----------
    raw_df : pd.DataFrame
        Output of load_latest_snapshot()

    Returns
    -------
    pd.DataFrame
    """
    flat = flatten_spreads(raw_df)

    # Filter to spread market only
    spreads = flat[flat["market"] == "spreads"].copy()
    if spreads.empty:
        return pd.DataFrame(columns=[
            "merge_key", "consensus_close", "book_dispersion",
            "home_team", "away_team", "game_date"
        ])

    # Pivot: each book → point value
    piv = spreads.pivot_table(
        index=["game_id", "home_team", "away_team", "commence_time"],
        columns="book",
        values="point",
        aggfunc="mean"
    )

    piv["book_dispersion"] = piv.std(axis=1, skipna=True)
    piv["consensus_close"] = piv.mean(axis=1, skipna=True)

    piv = piv.reset_index()

    # Merge key for predictions: HomeTeam@AwayTeam_YYYYMMDD
    piv["game_date"] = piv["commence_time"].str[:10]
    piv["merge_key"] = (
        piv["home_team"] + "@" + piv["away_team"] + "_" + piv["game_date"]
    )

    return piv[[
        "merge_key", "consensus_close", "book_dispersion",
        "home_team", "away_team", "game_date"
    ]]

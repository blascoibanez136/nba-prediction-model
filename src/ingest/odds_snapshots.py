"""
Odds snapshot + dispersion utilities for NBA Pro-Lite.

FIXED VERSION:
- Snapshots stored as JSON instead of CSV (CSV corrupts nested data)
- JSON loaded cleanly before flattening
- Correct parsing of spreads from bookmaker->markets->outcomes
- merge_key unified with model: "<home>__<away>__YYYY-MM-DD"
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict

import pandas as pd

from src.ingest.odds_ingest import get_nba_odds


REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = REPO_ROOT / "data" / "_snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------

def _norm(name: str) -> str:
    return name.strip().lower() if isinstance(name, str) else ""


# ---------------------------------------------------------
# SNAPSHOT I/O
# ---------------------------------------------------------

def save_snapshot(kind: str) -> Path:
    """
    Save raw odds response directly as JSON.
    CSV destroys nested bookmaker structures.
    """
    odds = get_nba_odds()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fn = SNAPSHOT_DIR / f"{kind}_{ts}.json"

    with open(fn, "w") as f:
        json.dump(odds, f)

    print(f"[snapshot] Saved {len(odds)} games -> {fn.relative_to(REPO_ROOT)}")
    return fn


def load_latest_snapshot(kind: str) -> List[dict]:
    files = sorted(SNAPSHOT_DIR.glob(f"{kind}_*.json"))
    if not files:
        raise FileNotFoundError(f"No snapshots for kind={kind}")

    with open(files[-1]) as f:
        return json.load(f)


# ---------------------------------------------------------
# FLATTENING
# ---------------------------------------------------------

def flatten_spreads(data: List[dict]) -> pd.DataFrame:
    """
    Flatten Odds API structure:
    game -> bookmakers -> markets -> outcomes

    Extract only SPREAD markets.
    """
    rows = []

    for g in data:
        game_id = g.get("id")
        home = g.get("home_team")
        away = g.get("away_team")
        commence = g.get("commence_time")

        for book in g.get("bookmakers", []):
            book_key = book.get("key")

            for m in book.get("markets", []):
                if m.get("key") not in ("spreads", "spread", "spreads_alt"):
                    continue

                for o in m.get("outcomes", []):
                    rows.append(
                        {
                            "merge_key": f"{_norm(home)}__{_norm(away)}__{commence[:10]}",
                            "home_team": home,
                            "away_team": away,
                            "game_date": commence[:10],
                            "book": book_key,
                            "team": o.get("name"),
                            "price": o.get("price"),
                            "point": o.get("point"),
                        }
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# DISPERSION + CONSENSUS
# ---------------------------------------------------------

def compute_dispersion(data: List[dict]) -> pd.DataFrame:
    df = flatten_spreads(data)

    if df.empty:
        return pd.DataFrame(
            columns=["merge_key", "consensus_close", "book_dispersion",
                     "home_team", "away_team", "game_date"]
        )

    # pivot: rows = game, columns = book, values = spread point
    piv = df.pivot_table(
        index=["merge_key", "home_team", "away_team", "game_date"],
        columns="book",
        values="point",
        aggfunc="mean",
    )

    piv["consensus_close"] = piv.mean(axis=1, skipna=True)
    piv["book_dispersion"] = piv.std(axis=1, skipna=True)
    piv = piv.reset_index()

    return piv


# ---------------------------------------------------------
# MOVEMENT
# ---------------------------------------------------------

def compute_movement(open_data: List[dict], close_data: List[dict]) -> pd.DataFrame:
    df_open = flatten_spreads(open_data)
    df_close = flatten_spreads(close_data)

    if df_open.empty or df_close.empty:
        return pd.DataFrame(columns=[
            "merge_key", "open_consensus", "close_consensus", "line_move"
        ])

    piv_open = df_open.pivot_table(
        index=["merge_key"], columns="book", values="point", aggfunc="mean"
    )
    piv_close = df_close.pivot_table(
        index=["merge_key"], columns="book", values="point", aggfunc="mean"
    )

    piv_open["open_consensus"] = piv_open.mean(axis=1)
    piv_close["close_consensus"] = piv_close.mean(axis=1)

    merged = piv_open[["open_consensus"]].merge(
        piv_close[["close_consensus"]],
        left_index=True, right_index=True, how="inner"
    )

    merged["line_move"] = merged["close_consensus"] - merged["open_consensus"]
    merged = merged.reset_index()

    return merged[["merge_key", "open_consensus", "close_consensus", "line_move"]]


# ---------------------------------------------------------
# CLI FOR GITHUB ACTIONS
# ---------------------------------------------------------

if __name__ == "__main__":
    kind = os.getenv("SNAPSHOT_KIND")
    if not kind:
        print("No SNAPSHOT_KIND set")
        raise SystemExit

    save_snapshot(kind)

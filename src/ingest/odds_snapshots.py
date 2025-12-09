"""
Odds snapshot and dispersion utilities for NBA Pro-Lite.

- save_snapshot(kind): fetch current odds and write
  <repo_root>/data/_snapshots/<kind>_<timestamp>.csv
- load_latest_snapshot(kind): load the most recent snapshot of that kind.
- flatten_spreads(df): turn nested bookmaker/market/outcome data into tidy rows.
- compute_dispersion(df): compute consensus close + dispersion per game.
- compute_movement(open_path, close_path): compute line movement between open and close.

IMPORTANT:
- merge_key is standardized to match the prediction pipeline:
    "<home_team_lower>__<away_team_lower>__<YYYY-MM-DD>"
  so we can join odds features onto predictions cleanly.
"""

from __future__ import annotations

import os
import json
import ast
from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd

from src.ingest.odds_ingest import get_nba_odds  # type: ignore

# Repository root (…/nba-prediction-model)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Snapshot directory; alias SNAPSHOT_DIR for backward compatibility
OUT_DIR = REPO_ROOT / "data" / "_snapshots"
SNAPSHOT_DIR = OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_team_name(name: Any) -> str:
    """
    Normalize team names so they match prediction merge_key format.

    Assumes the prediction side uses:
        "<home_team_lower>__<away_team_lower>__<YYYY-MM-DD>"
    e.g. "orlando magic__miami heat__2025-12-09"
    """
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def _parse_bookmakers_field(val: Any) -> list[dict[str, Any]]:
    """
    Convert the CSV 'bookmakers' field back into a Python list of dicts.

    - If it's already a list, return it.
    - If it's a JSON string or repr of a list/dict, parse it.
    - Otherwise return an empty list.
    """
    if isinstance(val, list):
        return val

    if isinstance(val, str) and val.strip():
        s = val.strip()
        # Try JSON first
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            # Fall back to Python literal eval (what pandas may use)
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    return [parsed]
            except Exception:
                return []

    return []


# ---------------------------------------------------------------------------
# Snapshot I/O
# ---------------------------------------------------------------------------

def save_snapshot(kind: str) -> Path:
    """Fetch current odds and save a snapshot CSV under data/_snapshots."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fn = OUT_DIR / f"{kind}_{stamp}.csv"

    odds = get_nba_odds()
    df = pd.DataFrame(odds)
    df.to_csv(fn, index=False)

    print(f"✅ saved {len(df)} rows to {fn.relative_to(REPO_ROOT)}")
    return fn


def load_latest_snapshot(kind: str) -> pd.DataFrame:
    """Load the newest snapshot for the given kind (open/mid/close)."""
    files = sorted(OUT_DIR.glob(f"{kind}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No snapshots found for '{kind}'.")
    return pd.read_csv(files[-1])


# ---------------------------------------------------------------------------
# Odds flattening + feature builders
# ---------------------------------------------------------------------------

def flatten_spreads(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten nested bookmaker/market/outcome structures into tidy rows."""
    rows: list[dict[str, Any]] = []

    for _, row in raw_df.iterrows():
        game_id = row.get("id")
        home = row.get("home_team")
        away = row.get("away_team")
        start = row.get("commence_time")

        raw_books = row.get("bookmakers", [])
        books = _parse_bookmakers_field(raw_books)
        if not books:
            continue

        for book in books:
            book_key = book.get("key")
            markets = book.get("markets", []) or []

            for m in markets:
                mkey = m.get("key")
                outcomes = m.get("outcomes", []) or []

                for o in outcomes:
                    rows.append(
                        {
                            "game_id": game_id,
                            "home_team": home,
                            "away_team": away,
                            "commence_time": start,
                            "book": book_key,
                            "market": mkey,
                            "team": o.get("name"),
                            "price": o.get("price"),
                            "point": o.get("point"),
                        }
                    )

    return pd.DataFrame(rows)


def _add_merge_key_from_flat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the standardized merge_key to a flattened odds dataframe.

    Uses normalized team names (lowercase, stripped) and YYYY-MM-DD date.
    """
    if df.empty:
        df["merge_key"] = []
        return df

    df = df.copy()
    df["home_team_norm"] = df["home_team"].apply(_normalize_team_name)
    df["away_team_norm"] = df["away_team"].apply(_normalize_team_name)
    df["game_date"] = df["commence_time"].astype(str).str[:10]

    df["merge_key"] = (
        df["home_team_norm"]
        + "__"
        + df["away_team_norm"]
        + "__"
        + df["game_date"]
    )
    return df


def compute_dispersion(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute consensus close and book dispersion for spread markets.

    Returns columns:
      merge_key, consensus_close, book_dispersion,
      home_team, away_team, game_date
    """
    flat = flatten_spreads(raw_df)
    flat = _add_merge_key_from_flat(flat)

    # If we couldn't recover markets, bail out gracefully
    if flat.empty or "market" not in flat.columns:
        return pd.DataFrame(
            columns=[
                "merge_key",
                "consensus_close",
                "book_dispersion",
                "home_team",
                "away_team",
                "game_date",
            ]
        )

    spreads = flat[flat["market"] == "spreads"].copy()
    if spreads.empty:
        return pd.DataFrame(
            columns=[
                "merge_key",
                "consensus_close",
                "book_dispersion",
                "home_team",
                "away_team",
                "game_date",
            ]
        )

    piv = spreads.pivot_table(
        index=["merge_key", "home_team", "away_team", "game_date"],
        columns="book",
        values="point",
        aggfunc="mean",
    )

    piv["book_dispersion"] = piv.std(axis=1, skipna=True)
    piv["consensus_close"] = piv.mean(axis=1, skipna=True)
    piv = piv.reset_index()

    return piv[
        [
            "merge_key",
            "consensus_close",
            "book_dispersion",
            "home_team",
            "away_team",
            "game_date",
        ]
    ]


def _consensus_spread(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the average spread across books for each game."""
    flat = flatten_spreads(raw_df)
    flat = _add_merge_key_from_flat(flat)

    # If 'market' column never materializes, just return empty
    if flat.empty or "market" not in flat.columns:
        return pd.DataFrame(columns=["merge_key", "consensus_spread"])

    spreads = flat[flat["market"] == "spreads"].copy()
    if spreads.empty:
        return pd.DataFrame(columns=["merge_key", "consensus_spread"])

    # Optional: filter obviously bogus points once you've inspected raw data.
    # spreads = spreads[spreads["point"].notna()]

    piv = spreads.pivot_table(
        index=["merge_key"],
        columns="book",
        values="point",
        aggfunc="mean",
    )

    piv["consensus_spread"] = piv.mean(axis=1, skipna=True)
    piv = piv.reset_index()

    return piv[["merge_key", "consensus_spread"]]


def compute_movement(open_path: str | Path, close_path: str | Path) -> pd.DataFrame:
    """
    Compute line movement (close – open consensus) for each game.

    Returns:
      merge_key, open_consensus, close_consensus, line_move
    """
    open_raw = pd.read_csv(open_path)
    close_raw = pd.read_csv(close_path)

    open_cons = _consensus_spread(open_raw).rename(
        columns={"consensus_spread": "open_consensus"}
    )
    close_cons = _consensus_spread(close_raw).rename(
        columns={"consensus_spread": "close_consensus"}
    )

    if open_cons.empty or close_cons.empty:
        # Nothing to merge; return empty movement frame
        return pd.DataFrame(
            columns=["merge_key", "open_consensus", "close_consensus", "line_move"]
        )

    df = open_cons.merge(close_cons, on="merge_key", how="inner")
    df["line_move"] = df["close_consensus"] - df["open_consensus"]

    return df[["merge_key", "open_consensus", "close_consensus", "line_move"]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple CLI for GitHub Actions:
    #   SNAPSHOT_KIND=open|mid|close
    kind = os.getenv("SNAPSHOT_KIND")
    if kind:
        try:
            save_snapshot(kind)
        except Exception as e:
            print(f"[odds_snapshots] Failed to save snapshot for kind '{kind}': {e}")
    else:
        print(
            "[odds_snapshots] No SNAPSHOT_KIND provided. "
            "Set SNAPSHOT_KIND to 'open', 'mid' or 'close' to save a snapshot."
        )

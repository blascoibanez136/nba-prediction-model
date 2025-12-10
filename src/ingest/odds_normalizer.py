"""
odds_normalizer.py

Utilities to normalize The Odds API JSON snapshots into a flat, tabular CSV
that downstream code (market_ensemble, edge_picker, backtests) can use.

Input:  A single JSON snapshot from The Odds API, e.g.:
        data/_snapshots/open_20251209_225352.json

Structure (typical The Odds API v4-ish shape):

[
  {
    "id": "game-id",
    "commence_time": "2025-12-09T23:30:00Z",
    "home_team": "Orlando Magic",
    "away_team": "Miami Heat",
    "bookmakers": [
      {
        "key": "draftkings",
        "title": "DraftKings",
        "last_update": "2025-12-09T22:53:10Z",
        "markets": [
          {
            "key": "h2h",
            "outcomes": [
              {"name": "Orlando Magic", "price": -120},
              {"name": "Miami Heat", "price": 100}
            ]
          },
          {
            "key": "spreads",
            "outcomes": [
              {"name": "Orlando Magic", "point": -1.5, "price": -110},
              {"name": "Miami Heat", "point": 1.5, "price": -110}
            ]
          },
          {
            "key": "totals",
            "outcomes": [
              {"name": "Over", "point": 235.0, "price": -110},
              {"name": "Under", "point": 235.0, "price": -110}
            ]
          }
        ]
      },
      ...
    ]
  },
  ...
]

Output: CSV with one row per (game, bookmaker), e.g.:

snapshot_type,game_id,game_date,commence_time,home_team,away_team,book,book_title,last_update,
ml_home,ml_away,
spread_home_point,spread_home_price,spread_away_point,spread_away_price,
total_point,total_over_price,total_under_price

Usage (CLI):

    python -m src.ingest.odds_normalizer \
        --input data/_snapshots/open_20251209_225352.json \
        --output data/_snapshots/open_20251209_225352.csv \
        --snapshot-type open

Downstream, odds_snapshots.py can call normalize_snapshot(...) directly
after saving a JSON snapshot.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NormalizedOddsRow:
    snapshot_type: str
    game_id: str
    game_date: str  # YYYY-MM-DD
    commence_time: str  # ISO string
    home_team: str
    away_team: str
    book: str
    book_title: str
    last_update: str  # raw from API

    ml_home: Optional[float] = None
    ml_away: Optional[float] = None

    spread_home_point: Optional[float] = None
    spread_home_price: Optional[float] = None
    spread_away_point: Optional[float] = None
    spread_away_price: Optional[float] = None

    total_point: Optional[float] = None
    total_over_price: Optional[float] = None
    total_under_price: Optional[float] = None


def _ensure_game_list(obj: Any) -> List[Dict[str, Any]]:
    """
    The Odds API typically returns a list of games, but some wrappers
    might wrap that in a dict under 'data' or similar. This helper
    normalizes to a list of game dicts.
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # Common pattern: {"data": [...], ...}
        for key in ("data", "games", "results"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
    raise ValueError("Unsupported snapshot JSON structure for games.")


def _parse_game_date(commence_time: str) -> str:
    """
    Convert an ISO commence_time (UTC) to YYYY-MM-DD.

    Example: "2025-12-09T23:30:00Z" -> "2025-12-09"
    """
    try:
        # Handle trailing Z or offset
        if commence_time.endswith("Z"):
            dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(commence_time)
        return dt.date().isoformat()
    except Exception:  # pragma: no cover - defensive
        # Fallback: first 10 chars if it's already YYYY-MM-DD...
        return commence_time[:10]


def _extract_markets(bookmaker: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert a bookmaker's list of markets into a dict keyed by market key.

    Returns something like:
    {
        "h2h": { ... },
        "spreads": { ... },
        "totals": { ... },
    }
    """
    markets = bookmaker.get("markets") or []
    out: Dict[str, Dict[str, Any]] = {}
    for m in markets:
        key = m.get("key")
        if not key:
            continue
        out[key] = m
    return out


def _extract_prices_for_game_book(
    game: Dict[str, Any],
    bookmaker: Dict[str, Any],
    snapshot_type: str,
) -> NormalizedOddsRow:
    """
    For a single game + bookmaker, extract ML, spread, total info into a
    NormalizedOddsRow. Missing markets are left as None.
    """
    game_id = str(game.get("id", ""))
    commence_time = game.get("commence_time", "")
    home_team = game.get("home_team", "")
    away_team = game.get("away_team", "")
    game_date = _parse_game_date(commence_time)

    book_key = bookmaker.get("key", "")
    book_title = bookmaker.get("title", book_key)
    last_update = bookmaker.get("last_update", "")

    row = NormalizedOddsRow(
        snapshot_type=snapshot_type,
        game_id=game_id,
        game_date=game_date,
        commence_time=commence_time,
        home_team=home_team,
        away_team=away_team,
        book=book_key,
        book_title=book_title,
        last_update=last_update,
    )

    markets = _extract_markets(bookmaker)

    # --- Moneyline (h2h) ---
    h2h = markets.get("h2h")
    if h2h:
        for outcome in h2h.get("outcomes", []):
            name = outcome.get("name")
            price = outcome.get("price")
            if name == home_team:
                row.ml_home = float(price)
            elif name == away_team:
                row.ml_away = float(price)

    # --- Spreads ---
    spreads = markets.get("spreads")
    if spreads:
        for outcome in spreads.get("outcomes", []):
            name = outcome.get("name")
            price = outcome.get("price")
            point = outcome.get("point")
            if name == home_team:
                row.spread_home_point = float(point)
                row.spread_home_price = float(price)
            elif name == away_team:
                row.spread_away_point = float(point)
                row.spread_away_price = float(price)

    # --- Totals ---
    totals = markets.get("totals")
    if totals:
        for outcome in totals.get("outcomes", []):
            name = outcome.get("name")
            price = outcome.get("price")
            point = outcome.get("point")
            # We assume Over/Under share the same point
            if name == "Over":
                row.total_point = float(point)
                row.total_over_price = float(price)
            elif name == "Under":
                # If total_point hasn't been set by Over, set from Under
                if row.total_point is None:
                    row.total_point = float(point)
                row.total_under_price = float(price)

    return row


def normalize_snapshot(
    json_path: Path,
    csv_path: Path,
    snapshot_type: str,
) -> pd.DataFrame:
    """
    Normalize a single JSON snapshot into a flat CSV.

    Parameters
    ----------
    json_path : Path
        Path to The Odds API JSON snapshot.
    csv_path : Path
        Where to write the CSV.
    snapshot_type : str
        One of {"open", "mid", "close"} or any label you want to tag the snapshot.

    Returns
    -------
    pd.DataFrame
        The normalized dataframe (also written to csv_path).
    """
    json_path = Path(json_path)
    csv_path = Path(csv_path)

    logger.info("Normalizing odds snapshot %s -> %s", json_path, csv_path)

    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    games = _ensure_game_list(raw)
    rows: List[NormalizedOddsRow] = []

    for game in games:
        bookmakers = game.get("bookmakers") or []
        for bookmaker in bookmakers:
            try:
                row = _extract_prices_for_game_book(game, bookmaker, snapshot_type)
                rows.append(row)
            except Exception as e:
                logger.warning(
                    "Failed to parse bookmaker %s for game %s: %s",
                    bookmaker.get("key"),
                    game.get("id"),
                    e,
                )

    if not rows:
        logger.warning("No odds rows found in snapshot %s", json_path)
        df = pd.DataFrame(columns=[f.name for f in NormalizedOddsRow.__dataclass_fields__.values()])
    else:
        df = pd.DataFrame([asdict(r) for r in rows])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    logger.info("Wrote normalized odds CSV with %d rows to %s", len(df), csv_path)
    return df


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize The Odds API JSON snapshots to CSV.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSON snapshot (e.g. data/_snapshots/open_20251209_225352.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV (e.g. data/_snapshots/open_20251209_225352.csv)",
    )
    parser.add_argument(
        "--snapshot-type",
        type=str,
        default="close",
        help="Snapshot label: e.g. open, mid, close (default: close).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )
    args = _parse_args(argv)
    normalize_snapshot(Path(args.input), Path(args.output), args.snapshot_type)


if __name__ == "__main__":  # pragma: no cover
    main()

"""
odds_normalizer.py

Flatten Odds API JSON snapshots into a normalized CSV suitable for:

- market_ensemble.apply_market_ensemble
- edge_picker.flatten_spreads_from_snapshot

Each CSV row = one (game, bookmaker) with columns:

    snapshot_type          open | mid | close
    game_id                Odds API game id
    commence_time          ISO8601 string
    game_date              YYYY-MM-DD (UTC, from commence_time)
    home_team
    away_team
    book                   bookmaker key
    book_title
    last_update            bookmaker's last_update

    ml_home                American odds for home moneyline (if present)
    ml_away                American odds for away moneyline

    spread_home_point      home spread (home perspective)
    spread_home_price
    spread_away_point      away spread
    spread_away_price

    total_point            total line
    total_over_price
    total_under_price
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = REPO_ROOT / "data" / "_snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class NormalizedRow:
    snapshot_type: str
    game_id: str
    commence_time: str
    game_date: str
    home_team: str
    away_team: str
    book: str
    book_title: str
    last_update: str
    ml_home: float
    ml_away: float
    spread_home_point: float
    spread_home_price: float
    spread_away_point: float
    spread_away_price: float
    total_point: float
    total_over_price: float
    total_under_price: float


def _norm_team(name: Any) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip()


def _date_from_commence(commence: Optional[str]) -> str:
    if not commence:
        return ""
    ts = str(commence)
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts).astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        # fallback: just take first 10 chars if it looks like a date
        return ts[:10]


def normalize_odds_list(
    odds: List[Dict[str, Any]],
    snapshot_type: str,
) -> pd.DataFrame:
    """
    Flatten a list of Odds API game objects into normalized rows.
    """
    rows: List[Dict[str, Any]] = []

    stype = snapshot_type.lower().strip()

    for g in odds:
        game_id = str(g.get("id", ""))
        home_team = _norm_team(g.get("home_team"))
        away_team = _norm_team(g.get("away_team"))
        commence_time = g.get("commence_time") or ""
        game_date = _date_from_commence(commence_time)

        bookmakers = g.get("bookmakers") or []
        for book in bookmakers:
            book_key = str(book.get("key", ""))
            book_title = str(book.get("title", ""))
            last_update = str(book.get("last_update", ""))

            ml_home = np.nan
            ml_away = np.nan
            spread_home_point = np.nan
            spread_home_price = np.nan
            spread_away_point = np.nan
            spread_away_price = np.nan
            total_point = np.nan
            total_over_price = np.nan
            total_under_price = np.nan

            for m in book.get("markets", []):
                mkey = (m.get("key") or "").lower()

                # 1) Moneyline
                if mkey in ("h2h", "h2h_lay", "moneyline"):
                    for o in m.get("outcomes", []):
                        name = _norm_team(o.get("name"))
                        price = o.get("price")
                        if name == home_team:
                            ml_home = price
                        elif name == away_team:
                            ml_away = price

                # 2) Spreads
                elif mkey in ("spreads", "spread", "spreads_alt"):
                    for o in m.get("outcomes", []):
                        name = _norm_team(o.get("name"))
                        price = o.get("price")
                        point = o.get("point")
                        if name == home_team:
                            spread_home_point = point
                            spread_home_price = price
                        elif name == away_team:
                            spread_away_point = point
                            spread_away_price = price

                # 3) Totals
                elif mkey in ("totals", "total"):
                    for o in m.get("outcomes", []):
                        name = str(o.get("name") or "").lower()
                        price = o.get("price")
                        point = o.get("point")
                        # Some books use "Over" / "Under"; some "Over 210.5"
                        if name.startswith("over"):
                            total_point = point
                            total_over_price = price
                        elif name.startswith("under"):
                            # assume same total line; don't overwrite point if already set
                            if np.isnan(total_point):
                                total_point = point
                            total_under_price = price

            rows.append(
                {
                    "snapshot_type": stype,
                    "game_id": game_id,
                    "commence_time": commence_time,
                    "game_date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "book": book_key,
                    "book_title": book_title,
                    "last_update": last_update,
                    "ml_home": ml_home,
                    "ml_away": ml_away,
                    "spread_home_point": spread_home_point,
                    "spread_home_price": spread_home_price,
                    "spread_away_point": spread_away_point,
                    "spread_away_price": spread_away_price,
                    "total_point": total_point,
                    "total_over_price": total_over_price,
                    "total_under_price": total_under_price,
                }
            )

    df = pd.DataFrame(rows)
    # enforce column order
    cols = [
        "snapshot_type",
        "game_id",
        "commence_time",
        "game_date",
        "home_team",
        "away_team",
        "book",
        "book_title",
        "last_update",
        "ml_home",
        "ml_away",
        "spread_home_point",
        "spread_home_price",
        "spread_away_point",
        "spread_away_price",
        "total_point",
        "total_over_price",
        "total_under_price",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def normalize_odds_json_file(
    json_path: Path,
    snapshot_type: Optional[str] = None,
) -> Path:
    """
    Read a snapshot JSON file and write a normalized CSV next to it.
    If snapshot_type is None, infer from the filename prefix (open_|mid_|close_).
    """
    json_path = Path(json_path)
    if snapshot_type is None:
        stem = json_path.stem  # e.g. "open_20251210_011905"
        snapshot_type = stem.split("_", 1)[0].lower()

    with json_path.open() as f:
        odds = json.load(f)

    if not isinstance(odds, list):
        raise RuntimeError(
            f"Expected list in odds JSON, got {type(odds)} from {json_path}"
        )

    df = normalize_odds_list(odds, snapshot_type=snapshot_type)
    csv_path = json_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(
        f"[odds_normalizer] Normalized {len(df)} rows -> "
        f"{csv_path.relative_to(REPO_ROOT)}"
    )
    return csv_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Normalize Odds API JSON snapshot into per-book CSV."
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to snapshot JSON (e.g. data/_snapshots/close_20251210_011905.json)",
    )
    parser.add_argument(
        "--snapshot-type",
        type=str,
        default=None,
        help="Snapshot kind: open | mid | close. If omitted, inferred from filename.",
    )
    args = parser.parse_args()

    normalize_odds_json_file(Path(args.json_path), snapshot_type=args.snapshot_type)


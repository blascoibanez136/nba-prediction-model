"""
Odds snapshot + dispersion utilities for NBA Pro-Lite.

- Snapshots are stored as JSON (raw Odds API response)
- AND immediately normalized to CSV using odds_normalizer.py
- JSON is used for dispersion/movement; CSV is used by:
    - market_ensemble.apply_market_ensemble
    - edge_picker.flatten_spreads_from_snapshot
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from src.ingest.odds_ingest import get_nba_odds
from src.ingest.odds_normalizer import normalize_odds_list


REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = REPO_ROOT / "data" / "_snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------

def _norm(name: str) -> str:
    return name.strip().lower() if isinstance(name, str) else ""


def _ensure_games_list(obj: Union[str, Path, List[dict]]) -> List[dict]:
    """
    Accept:
      - list[dict]           (already parsed Odds API payload)
      - str / Path path      (JSON file path)
      - str JSON             (raw JSON string)
      - pandas.DataFrame     (treated as 'no games' for compute_*)

    Return list[dict] or [] if we can't sensibly parse odds from the input.
    """
    # Already a list of dicts
    if isinstance(obj, list):
        if obj and not isinstance(obj[0], dict):
            raise TypeError(
                f"_ensure_games_list expected list[dict], got list[{type(obj[0])}]"
            )
        return obj

    # If someone passes a DataFrame (e.g. a CSV they've already read),
    # we don't try to reverse-engineer it back into Odds API JSON.
    # For dispersion/movement, it's safe to treat this as "no games".
    if isinstance(obj, pd.DataFrame):
        print(
            "[odds_snapshots] Warning: _ensure_games_list received a DataFrame; "
            "treating as empty odds payload for compute_*."
        )
        return []

    # Path-like or path string
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        if p.exists():
            with p.open() as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # This happens if the path is actually a CSV or empty file.
                    # Rather than crashing the whole workflow, treat it as
                    # "no games" and let dispersion/movement return empty frames.
                    print(
                        f"[odds_snapshots] Warning: {p} is not valid JSON; "
                        f"treating as empty odds payload for compute_*."
                    )
                    return []
        else:
            # Assume it's a raw JSON string
            data = json.loads(str(obj))

        if not isinstance(data, list):
            raise TypeError(
                f"_ensure_games_list expected list from JSON, got {type(data)}"
            )
        return data

    raise TypeError(f"_ensure_games_list cannot handle type {type(obj)}")


# ---------------------------------------------------------
# SNAPSHOT I/O
# ---------------------------------------------------------

def save_snapshot(kind: str) -> Dict[str, Path]:
    """
    Fetch current NBA odds from The Odds API and save both:
      - JSON snapshot (raw odds)
      - normalized CSV (per-book markets; for ensemble + edge_picker)

    Returns dict with:
        {
          "json": Path(...json),
          "csv": Path(...csv),
        }
    """
    kind = kind.lower().strip()
    if kind not in {"open", "mid", "close"}:
        raise ValueError(f"Invalid snapshot kind: {kind} (expected open|mid|close)")

    odds = get_nba_odds()  # list[dict]
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = SNAPSHOT_DIR / f"{kind}_{ts}.json"
    csv_path = SNAPSHOT_DIR / f"{kind}_{ts}.csv"

    # 1) Save raw JSON
    with json_path.open("w") as f:
        json.dump(odds, f)
    print(
        f"[snapshot] Saved {len(odds)} games -> "
        f"{json_path.relative_to(REPO_ROOT)}"
    )

    # 2) Normalize to per-book CSV
    df = normalize_odds_list(odds, snapshot_type=kind)
    df.to_csv(csv_path, index=False)
    print(
        f"[snapshot] Normalized {len(df)} rows -> "
        f"{csv_path.relative_to(REPO_ROOT)}"
    )

    return {"json": json_path, "csv": csv_path}


def load_latest_snapshot(kind: str) -> List[dict]:
    """
    Load the most recent snapshot JSON for the given kind (open|mid|close).
    Returns the raw list[dict] from The Odds API.
    """
    kind = kind.lower().strip()
    files = sorted(SNAPSHOT_DIR.glob(f"{kind}_*.json"))
    if not files:
        raise FileNotFoundError(f"No snapshots for kind={kind} in {SNAPSHOT_DIR}")

    with files[-1].open() as f:
        return json.load(f)


# ---------------------------------------------------------
# FLATTENING FOR DISPERSION/MOVEMENT (JSON-based)
# ---------------------------------------------------------

def flatten_spreads(data: List[dict]) -> pd.DataFrame:
    """
    Flatten Odds API structure:
      game -> bookmakers -> markets -> outcomes

    Extract only SPREAD markets into a simple table used for
    dispersion + movement calculations.

    Output columns:
        merge_key, home_team, away_team, game_date,
        book, team, price, point
    """
    rows: List[Dict[str, Any]] = []

    for g in data:
        home = g.get("home_team")
        away = g.get("away_team")
        commence = g.get("commence_time") or ""
        game_date = commence[:10] if commence else ""

        merge_key = f"{_norm(home)}__{_norm(away)}__{game_date}"

        for book in g.get("bookmakers", []):
            book_key = book.get("key")

            for m in book.get("markets", []):
                mkey = (m.get("key") or "").lower()
                if mkey not in ("spreads", "spread", "spreads_alt"):
                    continue

                for o in m.get("outcomes", []):
                    rows.append(
                        {
                            "merge_key": merge_key,
                            "home_team": home,
                            "away_team": away,
                            "game_date": game_date,
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

def compute_dispersion(data: Union[str, Path, List[dict]]) -> pd.DataFrame:
    """
    Compute consensus spread and dispersion (std dev across books)
    from a raw odds JSON payload.

    `data` can be:
      - list[dict]      (already parsed)
      - path / str      (JSON file path or raw JSON string)

    Returns:
        columns = [
          "merge_key", "consensus_close", "book_dispersion",
          "home_team", "away_team", "game_date",
        ]
    """
    games = _ensure_games_list(data)
    df = flatten_spreads(games)

    if df.empty:
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

    piv = df.pivot_table(
        index=["merge_key", "home_team", "away_team", "game_date"],
        columns="book",
        values="point",
        aggfunc="mean",
    )

    piv["consensus_close"] = piv.mean(axis=1, skipna=True)
    piv["book_dispersion"] = piv.std(axis=1, skipna=True)
    piv = piv.reset_index()

    cols = [
        "merge_key",
        "consensus_close",
        "book_dispersion",
        "home_team",
        "away_team",
        "game_date",
    ]
    return piv[cols]


# ---------------------------------------------------------
# MOVEMENT
# ---------------------------------------------------------

def compute_movement(
    open_data: Union[str, Path, List[dict]],
    close_data: Union[str, Path, List[dict]],
) -> pd.DataFrame:
    """
    Compare consensus spread at open vs close and compute line_move.

    open_data / close_data can be:
      - list[dict]      (already parsed)
      - path / str      (JSON file path or raw JSON string)

    Returns:
        columns = [
          "merge_key", "open_consensus", "close_consensus", "line_move"
        ]
    """
    open_games = _ensure_games_list(open_data)
    close_games = _ensure_games_list(close_data)

    df_open = flatten_spreads(open_games)
    df_close = flatten_spreads(close_games)

    if df_open.empty or df_close.empty:
        return pd.DataFrame(
            columns=["merge_key", "open_consensus", "close_consensus", "line_move"]
        )

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
        left_index=True,
        right_index=True,
        how="inner",
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
        print("No SNAPSHOT_KIND set (expected open|mid|close).")
        raise SystemExit(1)

    save_snapshot(kind)

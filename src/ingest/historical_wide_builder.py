"""
historical_wide_builder.py

Fallback builder for historical odds snapshots.

The Odds API's historical endpoint exposes games from past seasons, but the
structure and completeness of those payloads can vary widely compared to the
current live endpoint. In particular, older seasons may omit certain
markets (e.g. spreads or totals), use different keys, or exclude
bookmakers entirely. The normal ``normalize_odds_list`` function will
therefore produce an empty DataFrame when it cannot find the expected
fields. To avoid terminating ingestion runs in these cases we provide a
best‑effort wide builder that extracts whatever information is available
from the raw payload.

Each output row corresponds to a single (game, bookmaker) pair and
includes the same set of columns defined in ``normalize_odds_list``:

    snapshot_type
    game_id
    commence_time
    game_date
    merge_key
    home_team
    away_team
    book
    book_title
    last_update
    ml_home
    ml_away
    spread_home_point
    spread_home_price
    spread_away_point
    spread_away_price
    total_point
    total_over_price
    total_under_price

Missing values are filled with ``NaN`` to preserve column order. This
builder intentionally does **not** attempt to infer missing markets or
impute values; it simply records what exists. The resulting DataFrame
can be passed directly into downstream logic such as the market
ensemble's wide aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from src.ingest.odds_normalizer import _norm_team, _norm_key, _date_from_commence


def build_wide_snapshot_from_raw(
    games: List[Dict[str, Any]],
    snapshot_type: str = "close",
) -> pd.DataFrame:
    """Construct a per‑book snapshot DataFrame from raw historical odds.

    Parameters
    ----------
    games : list of dict
        A list of Odds API game objects. Each item should conform to the
        structure returned by the ``/v4/historical/sports/.../odds`` endpoint
        (i.e. containing ``home_team``, ``away_team``, ``commence_time`` and
        possibly a ``bookmakers`` list). If ``bookmakers`` or other
        fields are missing the corresponding output columns will be NaN.
    snapshot_type : str, optional
        A string describing the snapshot type (e.g. ``"open"``, ``"mid"``,
        ``"close"``). Defaults to ``"close"``. The value is lower‑cased
        and stripped before assignment.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per (game, bookmaker) containing the same
        columns as produced by ``normalize_odds_list``. If ``games`` is
        empty or no bookmakers/markets are present, the returned DataFrame
        will also be empty (but with correct columns).
    """
    stype = snapshot_type.lower().strip() if snapshot_type else ""
    rows: List[Dict[str, Any]] = []

    for g in games or []:
        game_id = str(g.get("id", ""))
        home_team = _norm_team(g.get("home_team"))
        away_team = _norm_team(g.get("away_team"))
        commence_time = g.get("commence_time") or ""
        game_date = _date_from_commence(commence_time)
        merge_key = f"{_norm_key(home_team)}__{_norm_key(away_team)}__{game_date}"

        bookmakers = g.get("bookmakers") or []
        # If bookmakers is empty, still emit a single row with NaNs so that
        # downstream merges can occur. This ensures merge_key presence.
        if not bookmakers:
            rows.append(
                {
                    "snapshot_type": stype,
                    "game_id": game_id,
                    "commence_time": commence_time,
                    "game_date": game_date,
                    "merge_key": merge_key,
                    "home_team": home_team,
                    "away_team": away_team,
                    "book": "",
                    "book_title": "",
                    "last_update": "",
                    "ml_home": np.nan,
                    "ml_away": np.nan,
                    "spread_home_point": np.nan,
                    "spread_home_price": np.nan,
                    "spread_away_point": np.nan,
                    "spread_away_price": np.nan,
                    "total_point": np.nan,
                    "total_over_price": np.nan,
                    "total_under_price": np.nan,
                }
            )
            continue

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

            for m in book.get("markets", []) or []:
                mkey = (m.get("key") or "").lower()
                # Moneyline markets may appear under various keys historically
                if mkey in ("h2h", "h2h_lay", "moneyline"):
                    for o in m.get("outcomes", []) or []:
                        name = _norm_team(o.get("name"))
                        price = o.get("price")
                        if name == home_team:
                            ml_home = price
                        elif name == away_team:
                            ml_away = price
                # Spread markets
                elif mkey in ("spreads", "spread", "spreads_alt"):
                    for o in m.get("outcomes", []) or []:
                        name = _norm_team(o.get("name"))
                        price = o.get("price")
                        point = o.get("point")
                        if name == home_team:
                            spread_home_point = point
                            spread_home_price = price
                        elif name == away_team:
                            spread_away_point = point
                            spread_away_price = price
                # Totals markets. Historically totals may be under "totals" or "total".
                elif mkey in ("totals", "total"):
                    for o in m.get("outcomes", []) or []:
                        name_raw = (o.get("name") or "").lower()
                        price = o.get("price")
                        point = o.get("point")
                        # We intentionally do not require "over"/"under" names; if
                        # only a single outcome exists we'll assign the point.
                        if name_raw.startswith("over"):
                            total_point = point
                            total_over_price = price
                        elif name_raw.startswith("under"):
                            # If this is the first totals seen, assign point
                            if pd.isna(total_point):
                                total_point = point
                            total_under_price = price
                        else:
                            # Unknown or unlabeled total; assign the point if not already set
                            if pd.isna(total_point):
                                total_point = point
                # Otherwise ignore unknown markets

            rows.append(
                {
                    "snapshot_type": stype,
                    "game_id": game_id,
                    "commence_time": commence_time,
                    "game_date": game_date,
                    "merge_key": merge_key,
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

    # Build DataFrame and enforce column order to match normalize_odds_list
    df = pd.DataFrame(rows)
    cols = [
        "snapshot_type",
        "game_id",
        "commence_time",
        "game_date",
        "merge_key",
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
    # Ensure missing columns exist and preserve order
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]

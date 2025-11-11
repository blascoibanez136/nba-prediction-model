*** Begin Patch
*** Add File: src/ingest/odds_snapshots.py
+"""
+Odds Movement & Dispersion utilities for NBA games.
+
+This module provides a small ingest layer around the
+``The Odds API`` odds feed used by the project.  The Odds API
+returns a nested ``bookmakers`` field for each game where odds
+from multiple sports books are stored.  To compute useful
+features like line movement and market dispersion, we first
+flatten that nested data into a tabular structure (one row per
+game, bookmaker, and team).  A series of helper functions then
+persist snapshots and derive rolling metrics for model training.
+
+Functions
+---------
+save_snapshot(kind: str = "open")
+    Fetch the latest odds via ``get_nba_odds`` and write them to
+    a timestamped CSV under ``data/_snapshots``.  The ``kind``
+    parameter (e.g. ``open``, ``mid``, ``close``) is included in
+    the filename.  The function returns the full path to the
+    saved file.
+
+flatten_spreads(df: pandas.DataFrame) -> pandas.DataFrame
+    Flatten the nested ``bookmakers`` structure returned by
+    The Odds API into a tidy DataFrame.  Each output row
+    contains a ``game_id``, ``book``, ``team``, ``spread``, and
+    ``price`` along with the home and away team names.
+
+compute_movement(open_csv_path: str, close_csv_path: str) -> pandas.DataFrame
+    Given two flattened snapshot CSV paths (usually an early
+    snapshot and a late snapshot), compute the difference in
+    spreads for each game/book/team pair.  Also compute the
+    velocity of the line movement by dividing the change by the
+    hours between the snapshot timestamps.  The hours delta is
+    inferred from the filenames (``<kind>_YYYYMMDD_HHMM.csv``).
+
+compute_dispersion(close_csv_path: str, allowed_books: list[str] | None = None) -> pandas.DataFrame
+    For a single flattened snapshot, compute the standard
+    deviation of the closing spreads across bookmakers for each
+    game as ``book_dispersion`` and the median spread across
+    books as ``consensus_close``.  If ``allowed_books`` is
+    provided, only those books are considered when computing
+    dispersion and consensus.
+
+Command Line
+------------
+Running this module as a script will save a snapshot of the
+specified ``kind``.  The kind is read from the ``SNAPSHOT_KIND``
+environment variable or defaults to ``open`` if unset.
+
+Example
+-------
+>>> from src.ingest.odds_snapshots import save_snapshot
+>>> path = save_snapshot("open")
+>>> path  # doctest: +SKIP
+'data/_snapshots/open_20251111_1805.csv'
+"""
+
+from __future__ import annotations
+
+import os
+import ast
+from datetime import datetime
+from typing import Any, Dict, List, Optional
+
+import pandas as pd
+
+# Attempt to import the existing odds ingest.  This module is
+# expected to provide a ``get_nba_odds`` function that returns
+# raw odds data from The Odds API.
+try:
+    from src.ingest.odds_ingest import get_nba_odds  # type: ignore
+except Exception as exc:  # pragma: no cover - fallback
+    raise ImportError(
+        "Could not import src.ingest.odds_ingest.get_nba_odds. "
+        "Make sure that module exists or implement get_nba_odds()"
+    ) from exc
+
+
+# Directory where snapshot CSVs are written.  This directory is
+# relative to the project root and is created if it does not exist.
+SNAPSHOT_DIR = os.path.join("data", "_snapshots")
+os.makedirs(SNAPSHOT_DIR, exist_ok=True)
+
+
+def _timestamp() -> str:
+    """Return a UTC timestamp in YYYYMMDD_HHMM format."""
+    return datetime.utcnow().strftime("%Y%m%d_%H%M")
+
+
+def save_snapshot(kind: str = "open") -> str:
+    """
+    Fetch current NBA odds and persist to a timestamped CSV.
+
+    Parameters
+    ----------
+    kind : str, default "open"
+        A label describing the snapshot.  Common values are
+        "open", "mid", and "close".  The label is used in the
+        filename but otherwise has no impact on the data.
+
+    Returns
+    -------
+    str
+        The filesystem path to the written CSV.
+    """
+    odds = get_nba_odds()
+    df = pd.DataFrame(odds)
+    ts = _timestamp()
+    filename = f"{kind}_{ts}.csv"
+    path = os.path.join(SNAPSHOT_DIR, filename)
+    df.to_csv(path, index=False)
+    return path
+
+
+def flatten_spreads(df: pd.DataFrame) -> pd.DataFrame:
+    """
+    Flatten nested bookmaker odds into a tidy DataFrame.
+
+    The Odds API returns a list of bookmakers for each game.  Each
+    bookmaker contains markets (e.g. "spreads").  Markets
+    contain outcomes for each team.  This function iterates
+    through all games and bookmakers, emitting one row per
+    game/book/team entry with the associated spread and price.
+
+    Parameters
+    ----------
+    df : pandas.DataFrame
+        The raw odds DataFrame as returned by ``get_nba_odds``.
+
+    Returns
+    -------
+    pandas.DataFrame
+        A DataFrame with columns: ``game_id``, ``book``, ``team``,
+        ``spread``, ``price``, ``home_team``, and ``away_team``.
+    """
+    records: List[Dict[str, Any]] = []
+    for _, row in df.iterrows():
+        game_id: str = row.get("id")
+        home_team: str = row.get("home_team")
+        away_team: str = row.get("away_team")
+        # The bookmakers field may be a list or a serialized JSON string
+        bookmakers_data = row.get("bookmakers", [])
+        if isinstance(bookmakers_data, str):
+            try:
+                bookmakers_data = ast.literal_eval(bookmakers_data)
+            except Exception:
+                bookmakers_data = []
+        for bk in bookmakers_data:
+            book_key = bk.get("key")
+            markets = bk.get("markets", [])
+            for market in markets:
+                if market.get("key") == "spreads":
+                    outcomes = market.get("outcomes", [])
+                    for outcome in outcomes:
+                        records.append(
+                            {
+                                "game_id": game_id,
+                                "book": book_key,
+                                "team": outcome.get("name"),
+                                "spread": outcome.get("point"),
+                                "price": outcome.get("price"),
+                                "home_team": home_team,
+                                "away_team": away_team,
+                            }
+                        )
+    return pd.DataFrame.from_records(records)
+
+
+def _parse_snapshot_timestamp(path: str) -> datetime:
+    """
+    Parse a snapshot filename to a datetime.
+
+    Filenames are expected to follow the pattern
+    ``{kind}_YYYYMMDD_HHMM.csv``.  Only the date/time portion is
+    parsed.  If the filename does not conform, a runtime error is
+    raised.
+    """
+    base = os.path.basename(path)
+    # split on underscore and dot
+    parts = base.split("_")
+    if len(parts) < 2:
+        raise ValueError(f"Invalid snapshot filename: {path}")
+    # drop the kind and extension
+    dt_part = parts[1].split(".")[0]
+    return datetime.strptime(dt_part, "%Y%m%d_%H%M")
+
+
+def compute_movement(open_csv_path: str, close_csv_path: str) -> pd.DataFrame:
+    """
+    Compute line movement and velocity from two snapshot CSVs.
+
+    This function loads two snapshot CSVs (presumably an "open"
+    and a "close" snapshot), flattens them via
+    ``flatten_spreads``, merges on ``game_id``, ``book``, and
+    ``team`` to align spreads, and then computes the raw
+    difference and a time-normalized velocity.
+
+    Parameters
+    ----------
+    open_csv_path : str
+        Path to the earlier snapshot CSV (e.g. open).
+    close_csv_path : str
+        Path to the later snapshot CSV (e.g. close).
+
+    Returns
+    -------
+    pandas.DataFrame
+        A DataFrame with columns: ``game_id``, ``book``, ``team``,
+        ``spread_open``, ``spread_close``, ``move_open_to_close``,
+        and ``intraday_velocity``.
+    """
+    open_df = pd.read_csv(open_csv_path)
+    close_df = pd.read_csv(close_csv_path)
+    open_flat = flatten_spreads(open_df)
+    close_flat = flatten_spreads(close_df)
+    merged = pd.merge(
+        open_flat,
+        close_flat,
+        on=["game_id", "book", "team"],
+        suffixes=("_open", "_close"),
+    )
+    merged["move_open_to_close"] = merged["spread_close"] - merged["spread_open"]
+    # compute hours delta from filename timestamps
+    try:
+        t_open = _parse_snapshot_timestamp(open_csv_path)
+        t_close = _parse_snapshot_timestamp(close_csv_path)
+        hours_delta = max((t_close - t_open).total_seconds() / 3600.0, 0.1)
+    except Exception:
+        # fallback to 1 hour if parsing fails
+        hours_delta = 1.0
+    merged["intraday_velocity"] = merged["move_open_to_close"] / hours_delta
+    return merged[[
+        "game_id",
+        "book",
+        "team",
+        "spread_open",
+        "spread_close",
+        "move_open_to_close",
+        "intraday_velocity",
+    ]]
+
+
+def compute_dispersion(
+    close_csv_path: str, allowed_books: Optional[List[str]] = None
+) -> pd.DataFrame:
+    """
+    Compute market dispersion and consensus close for a snapshot.
+
+    Parameters
+    ----------
+    close_csv_path : str
+        Path to a snapshot CSV (usually the close snapshot).
+    allowed_books : list[str] | None, optional
+        If provided, only these bookmaker keys are considered when
+        computing statistics.
+
+    Returns
+    -------
+    pandas.DataFrame
+        A DataFrame with columns: ``game_id``, ``book_dispersion``,
+        and ``consensus_close``.
+    """
+    close_df = pd.read_csv(close_csv_path)
+    flat = flatten_spreads(close_df)
+    if allowed_books is not None:
+        flat = flat[flat["book"].isin(allowed_books)].reset_index(drop=True)
+    # compute standard deviation of spreads across books
+    disp = (
+        flat.groupby("game_id")["spread"].std().fillna(0).rename("book_dispersion")
+    )
+    # compute median (consensus) spread across books
+    consensus = (
+        flat.groupby("game_id")["spread"].median().rename("consensus_close")
+    )
+    result = pd.concat([disp, consensus], axis=1).reset_index()
+    return result
+
+
+if __name__ == "__main__":
+    # Simple CLI to save a snapshot when executed directly.
+    # The snapshot kind is read from the SNAPSHOT_KIND environment
+    # variable or defaults to "open".  Usage:
+    #   SNAPSHOT_KIND=mid python src/ingest/odds_snapshots.py
+    kind = os.getenv("SNAPSHOT_KIND", "open")
+    path = save_snapshot(kind)
+    print(f"Saved {kind} snapshot to {path}")
+
*** End Patch
*** End Patch

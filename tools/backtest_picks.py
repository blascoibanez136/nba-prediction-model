"""
Backtest picks (supports two modes):

Mode A (embedded odds):
- picks contain an odds column (american or decimal)

Mode B (snapshot odds):
- picks have team+side; odds are resolved from market snapshots by date + merge_key

Hard rules:
- Never silently succeed on empty data
- Always write audit JSON on failure in CLI
- Keep merge_key contract intact
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Merge key (LOCKED CONTRACT)
# ---------------------------------------------------------------------

def _merge_key(home_team: str, away_team: str, game_date: str) -> str:
    return f"{str(home_team).strip().lower()}__{str(away_team).strip().lower()}__{str(game_date).strip()}"


# ---------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------

def _american_to_decimal(american: object) -> float:
    if american is None:
        return np.nan
    try:
        o = float(american)
    except Exception:
        return np.nan
    if np.isnan(o) or o == 0:
        return np.nan
    if o > 0:
        return 1.0 + (o / 100.0)
    return 1.0 + (100.0 / abs(o))


def _coerce_decimal_odds(x: object) -> float:
    if x is None:
        return np.nan
    try:
        v = float(x)
    except Exception:
        return np.nan
    if np.isnan(v):
        return np.nan

    # If it looks like an American line, convert
    if abs(v) >= 100 and abs(v) <= 5000:
        return _american_to_decimal(v)

    # Otherwise assume it's decimal odds
    if v <= 1.0:
        return np.nan
    return v


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: object) -> None:
    _safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Snapshot discovery (CLOSE preferred, OPEN fallback)
# NOTE: This does NOT change ingestion rules; it preserves Commit-3 behavior.
# ---------------------------------------------------------------------

def _find_snapshot_file(snapshot_dir: Path, game_date: str) -> Optional[Path]:
    """
    Find a snapshot file for a given date.
    Supports:
    - close_YYYYMMDD.csv
    - YYYYMMDD.csv
    - YYYY-MM-DD.csv
    """
    ymd = str(game_date).replace("-", "")
    candidates = [
        snapshot_dir / f"close_{ymd}.csv",
        snapshot_dir / f"{ymd}.csv",
        snapshot_dir / f"{game_date}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _discover_team_columns(snapshot_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = [c.lower() for c in snapshot_df.columns]
    # Common variants
    home_candidates = ["home_team", "home", "team_home", "home_name", "home_team_name"]
    away_candidates = ["away_team", "away", "team_away", "away_name", "away_team_name"]
    home_col = None
    away_col = None
    for cand in home_candidates:
        if cand in cols:
            home_col = snapshot_df.columns[cols.index(cand)]
            break
    for cand in away_candidates:
        if cand in cols:
            away_col = snapshot_df.columns[cols.index(cand)]
            break
    return home_col, away_col


def _discover_ml_columns(snapshot_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = [c.lower() for c in snapshot_df.columns]
    # Common variants for moneyline
    home_ml_candidates = ["ml_home", "home_ml", "moneyline_home", "home_moneyline", "home_ml_close", "ml_home_close"]
    away_ml_candidates = ["ml_away", "away_ml", "moneyline_away", "away_moneyline", "away_ml_close", "ml_away_close"]
    home_ml = None
    away_ml = None
    for cand in home_ml_candidates:
        if cand in cols:
            home_ml = snapshot_df.columns[cols.index(cand)]
            break
    for cand in away_ml_candidates:
        if cand in cols:
            away_ml = snapshot_df.columns[cols.index(cand)]
            break
    return home_ml, away_ml


# ---------------------------------------------------------------------
# Picks normalization + mode detection
# ---------------------------------------------------------------------

def _normalize_picks(picks_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = picks_df.copy()
    audit: Dict = {"rows_in": int(len(df)), "inferred": {}, "mode": None}

    if "game_date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "game_date"})
        audit["inferred"]["game_date"] = "renamed from date"

    need = {"game_date", "home_team", "away_team"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"[picks_backtest] Picks missing required columns: {sorted(miss)}")

    if "merge_key" not in df.columns:
        df["merge_key"] = [
            _merge_key(h, a, d) for h, a, d in zip(df["home_team"], df["away_team"], df["game_date"])
        ]
        audit["inferred"]["merge_key"] = "constructed from home/away/game_date"

    # bet_side inference
    if "bet_side" not in df.columns:
        for c in ["pick_side", "side", "selection"]:
            if c in df.columns:
                df = df.rename(columns={c: "bet_side"})
                audit["inferred"]["bet_side"] = f"renamed from {c}"
                break
    if "bet_side" not in df.columns:
        raise RuntimeError(or("[picks_backtest] Cannot infer bet_side (need bet_side/pick_side/side)"))

    df["bet_side"] = df["bet_side"].astype(str).str.upper()
    df = df[df["bet_side"].isin(["HOME", "AWAY"])].copy()

    # Commit-4 (additive): support embedded execution odds when present.
    # We distinguish:
    # - bet_odds_decimal: execution / pick-time odds (if embedded)
    # - close_odds_decimal: closing odds from snapshots (attached later)
    # The existing 'odds_decimal' column remains the odds used for PnL (backward compatible).

    embedded_cols = [c for c in ["bet_ml", "odds_decimal", "odds", "american_odds", "price", "line"] if c in df.columns]
    if embedded_cols:
        src = embedded_cols[0]
        df["bet_odds_decimal"] = df[src].apply(_coerce_decimal_odds)
        # Backward compatible: use execution odds for PnL if we have them
        df["odds_decimal"] = df["bet_odds_decimal"]
        audit["mode"] = "embedded_odds"
        audit["inferred"]["bet_odds_decimal"] = f"from {src}"
        audit["inferred"]["odds_decimal"] = "set to bet_odds_decimal (execution odds)"
    else:
        df["bet_odds_decimal"] = np.nan
        df["odds_decimal"] = np.nan
        audit["mode"] = "snapshot_odds"

    # Always create close_odds_decimal (filled during snapshot attach when available)
    if "close_odds_decimal" not in df.columns:
        df["close_odds_decimal"] = np.nan

    audit["rows_out"] = int(len(df))
    return df, audit


# ---------------------------------------------------------------------
# Snapshot odds attach (FIXED: duplicate mk safe + scalar safe)
# ---------------------------------------------------------------------

def _first_non_null_scalar(val: Union[pd.Series, pd.DataFrame, object]) -> object:
    """
    Make 'val' scalar-safe:
    - If Series: return first non-null
    - If object: return as-is
    """
    if isinstance(val, pd.Series):
        for x in val.values.tolist():
            if pd.notna(x):
                return x
        return np.nan
    return val


def _get_cell(row: Union[pd.Series, pd.DataFrame], col: Optional[str]) -> object:
    """
    Return a scalar value for column 'col' from a snapshot row.
    If snapshot has duplicate mk, row will be a DataFrame: choose first non-null in that column.
    """
    if col is None:
        return np.nan
    if isinstance(row, pd.DataFrame):
        if col not in row.columns:
            return np.nan
        return _first_non_null_scalar(row[col])
    # Series
    try:
        return row.get(col, np.nan)  # type: ignore
    except Exception:
        return np.nan


def _attach_odds_from_snapshots(
    picks_df: pd.DataFrame,
    snapshot_dir: Optional[Path],
) -> Tuple[pd.DataFrame, Dict]:
    df = picks_df.copy()
    audit: Dict = {
        "snapshot_dir": str(snapshot_dir) if snapshot_dir else None,
        "dates_seen": int(df["game_date"].nunique()),
        "snapshots_found": 0,
        "rows_with_snapshot_match": 0,
        "odds_filled": 0,
        "odds_missing_after": 0,
        "team_cols": None,
        "ml_cols": None,
        "example_snapshot_columns": None,
        "dup_mk_rows_seen": 0,
        "note": None,
    }

    if snapshot_dir is None:
        audit["note"] = "no snapshot_dir provided; leaving odds unresolved"
        audit["odds_missing_after"] = int(df["odds_decimal"].isna().sum())
        return df, audit

    # Commit-4: Do NOT skip snapshot attach just because embedded odds exist.
    # We still want close_odds_decimal for CLV diagnostics.
    embedded_present = int(df["odds_decimal"].notna().sum()) > 0
    if embedded_present:
        audit["note"] = "embedded odds detected; attaching close odds for CLV"

    filled = 0
    matched = 0
    found = 0
    dup_mk_rows = 0

    # Load snapshot per date
    for game_date, grp in df.groupby("game_date"):
        snap_path = _find_snapshot_file(snapshot_dir, str(game_date))
        if snap_path is None:
            continue

        found += 1
        snap = _read_csv(snap_path)
        if audit["example_snapshot_columns"] is None:
            audit["example_snapshot_columns"] = list(snap.columns)

        home_col, away_col = _discover_team_columns(snap)
        home_ml_col, away_ml_col = _discover_ml_columns(snap)
        audit["team_cols"] = (home_col, away_col)
        audit["ml_cols"] = (home_ml_col, away_ml_col)

        if home_col is None or away_col is None or home_ml_col is None or away_ml_col is None:
            continue

        snap = snap.copy()
        snap["merge_key"] = [
            _merge_key(h, a, str(game_date)) for h, a in zip(snap[home_col], snap[away_col])
        ]
        snap_idx = snap.set_index("merge_key", drop=False)

        for i in grp.index.tolist():
            mk = df.at[i, "merge_key"]
            if mk not in snap_idx.index:
                continue

            row = snap_idx.loc[mk]
            if isinstance(row, pd.DataFrame):
                dup_mk_rows += 1

            matched += 1

            # Choose ML based on bet_side
            if df.at[i, "bet_side"] == "HOME":
                am = _get_cell(row, home_ml_col)
            else:
                am = _get_cell(row, away_ml_col)

            dec = _american_to_decimal(am)
            if pd.notna(dec):
                # Always fill close_odds_decimal (for CLV diagnostics)
                df.at[i, "close_odds_decimal"] = float(dec)
                # Backward compatible: if pnl odds not set (no embedded), use close as odds_decimal
                if pd.isna(df.at[i, "odds_decimal"]):
                    df.at[i, "odds_decimal"] = float(dec)
                filled += 1

    audit["snapshots_found"] = int(found)
    audit["rows_with_snapshot_match"] = int(matched)
    audit["odds_filled"] = int(filled)
    audit["odds_missing_after"] = int(df["odds_decimal"].isna().sum())
    audit["close_odds_missing_after"] = int(df["close_odds_decimal"].isna().sum())
    audit["dup_mk_rows_seen"] = int(dup_mk_rows)
    return df, audit


# ---------------------------------------------------------------------
# History join
# ---------------------------------------------------------------------

def _load_history(history_csv: Path) -> pd.DataFrame:
    hist = _read_csv(history_csv)
    need = {"merge_key", "home_win"}
    miss = need - set(hist.columns)
    if miss:
        raise RuntimeError(f"[picks_backtest] History missing required columns: {sorted(miss)}")
    hist["home_win"] = hist["home_win"].astype(int)
    return hist


def _join_history(picks_df: pd.DataFrame, history_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = picks_df.copy()
    joined = df.merge(history_df[["merge_key", "home_win"]], on="merge_key", how="left", validate="m:1")
    coverage = float(joined["home_win"].notna().mean()) if len(joined) else 0.0
    audit = {
        "rows_in": int(len(df)),
        "rows_out": int(len(joined)),
        "history_match_rate": coverage,
        "unmatched_rows": int(joined["home_win"].isna().sum()),
    }
    return joined, audit


# ---------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------

def backtest_picks(
    picks_csvs: List[Path],
    history_csv: Path,
    snapshot_dir: Optional[Path],
    out_dir: Path,
    stake: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    if not picks_csvs:
        raise RuntimeError("[picks_backtest] No picks files provided")

    _safe_mkdir(out_dir)

    # Load + concat
    parts = []
    for p in picks_csvs:
        if not p.exists():
            continue
        df = _read_csv(p)
        if len(df) == 0:
            continue
        parts.append(df)

    if not parts:
        raise RuntimeError("[picks_backtest] No picks rows found in provided files")

    picks = pd.concat(parts, ignore_index=True)

    # Normalize picks + infer mode
    picks, norm_audit = _normalize_picks(picks)

    # Attach odds
    picks, odds_audit = _attach_odds_from_snapshots(picks, snapshot_dir)

    # Load history + join
    hist = _load_history(history_csv)
    joined, join_audit = _join_history(picks, hist)

    # Compute result and pnl
    joined["home_win"] = joined["home_win"].fillna(np.nan)
    joined["result"] = joined.apply(
        lambda r: r["home_win"] if r["bet_side"] == "HOME" else 1 - r["home_win"],
        axis=1,
    )

    joined["stake"] = float(stake)
    joined["pnl"] = np.where(
        joined["odds_decimal"].notna(),
        np.where(joined["result"] == 1, (joined["odds_decimal"] - 1.0) * joined["stake"], -joined["stake"]),
        np.nan,
    )


    # -----------------------------------------------------------------
    # Commit-4: Edge + CLV columns (additive, audit-first)
    #
    # Definitions:
    # - bet_odds_decimal: execution odds if embedded, else odds_decimal
    # - close_odds_decimal: closing odds from snapshots if available, else odds_decimal
    # - edge_bet: p_model (chosen side) - implied_prob(bet_odds_decimal)
    # - clv_implied: implied_prob(bet) - implied_prob(close)
    # -----------------------------------------------------------------

    if "bet_odds_decimal" not in joined.columns:
        joined["bet_odds_decimal"] = np.nan

    joined["bet_odds_decimal"] = np.where(
        joined["bet_odds_decimal"].notna(),
        joined["bet_odds_decimal"].astype(float),
        joined["odds_decimal"].astype(float),
    )

    if "close_odds_decimal" not in joined.columns:
        joined["close_odds_decimal"] = np.nan

    joined["close_odds_decimal"] = np.where(
        joined["close_odds_decimal"].notna(),
        joined["close_odds_decimal"].astype(float),
        joined["odds_decimal"].astype(float),
    )

    joined["p_implied_bet"] = np.where(
        joined["bet_odds_decimal"].notna(), 1.0 / joined["bet_odds_decimal"], np.nan
    )
    joined["p_implied_close"] = np.where(
        joined["close_odds_decimal"].notna(), 1.0 / joined["close_odds_decimal"], np.nan
    )

    # p_model for the chosen side
    joined["p_model"] = np.where(
        joined["bet_side"] == "HOME",
        joined["home_win_prob"].astype(float),
        1.0 - joined["home_win_prob"].astype(float),
    )

    joined["edge_bet"] = joined["p_model"] - joined["p_implied_bet"]

    # CLV in probability space (primary)
    joined["clv_implied"] = joined["p_implied_bet"] - joined["p_implied_close"]
    joined["clv_positive"] = joined["clv_implied"].apply(lambda x: bool(x > 0) if pd.notna(x) else False)

    resolved = joined[joined["pnl"].notna()].copy()

    # Summary
    summary = pd.DataFrame(
        [
            {
                "bets": int(len(resolved)),
                "wins": int((resolved["pnl"] > 0).sum()),
                "win_rate": float((resolved["pnl"] > 0).mean()) if len(resolved) else np.nan,
                "units": float(resolved["pnl"].sum()) if len(resolved) else 0.0,
                "roi": float(resolved["pnl"].sum() / resolved["stake"].sum()) if len(resolved) else np.nan,
            }
        ]
    )

    audit = {
        "normalize": norm_audit,
        "odds_attach": odds_audit,
        "history_join": join_audit,
        "resolved_bets": int(len(resolved)),
        "unresolved_missing_odds": int(joined["pnl"].isna().sum()),
    }


    # Commit-4: CLV coverage stats (additive)
    clv_available_rate = float(joined["clv_implied"].notna().mean()) if "clv_implied" in joined.columns else 0.0
    if "clv_positive" in joined.columns and joined["clv_implied"].notna().any():
        clv_positive_rate = float(joined.loc[joined["clv_implied"].notna(), "clv_positive"].mean())
    else:
        clv_positive_rate = None

    audit["clv"] = {
        "clv_available_rate": clv_available_rate,
        "clv_positive_rate": clv_positive_rate,
        "embedded_execution_odds_present": bool(joined["bet_odds_decimal"].notna().any()),
    }

    # Write outputs (existing contract)
    backtest_csv = out_dir / "picks_backtest.csv"
    summary_csv = out_dir / "picks_backtest_summary.csv"
    audit_json = out_dir / "picks_backtest_audit.json"

    resolved.to_csv(backtest_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    _write_json(audit_json, audit)

    return resolved, summary, audit


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--picks_dir", type=str, required=True, help="Directory containing picks_YYYY-MM-DD.csv files")
    p.add_argument("--history_csv", type=str, required=True, help="Path to history CSV containing merge_key + home_win")
    p.add_argument("--snapshot_dir", type=str, default=None, help="Directory containing snapshot CSVs (optional)")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p.add_argument("--stake", type=float, default=1.0, help="Flat stake per bet (units)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    picks_dir = Path(args.picks_dir)
    history_csv = Path(args.history_csv)
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else None
    out_dir = Path(args.out_dir)

    try:
        picks_csvs = sorted(picks_dir.glob("picks_*.csv"))
        backtest_picks(
            picks_csvs=picks_csvs,
            history_csv=history_csv,
            snapshot_dir=snapshot_dir,
            out_dir=out_dir,
            stake=float(args.stake),
        )
        return 0
    except Exception as e:
        # Always write an audit JSON on failure
        _safe_mkdir(out_dir)
        _write_json(out_dir / "picks_backtest_audit.json", {"error": str(e)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())

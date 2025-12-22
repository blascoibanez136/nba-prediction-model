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
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

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

def _american_to_decimal(odds: object) -> float:
    if odds is None:
        return np.nan
    try:
        o = float(odds)
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
# Snapshot discovery (CLOSE preferred, OPEN supported)
# ---------------------------------------------------------------------

def _find_snapshot_file(snapshot_dir: Path, game_date: str, snapshot_type: str) -> Optional[Path]:
    """
    We support:
      close_YYYYMMDD.csv
      open_YYYYMMDD.csv

    For backward compatibility, close also tries legacy candidates (ymd.csv or date.csv).
    """
    ymd = str(game_date).replace("-", "")

    if snapshot_type not in {"open", "close"}:
        raise ValueError(f"Unsupported snapshot_type for snapshot discovery: {snapshot_type}")

    if snapshot_type == "open":
        candidates = [
            snapshot_dir / f"open_{ymd}.csv",
        ]
    else:
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
        audit["inferred"]["merge_key"] = "built from teams+date (normalized)"

    if "bet_side" not in df.columns:
        if "pick_side" in df.columns:
            df["bet_side"] = df["pick_side"].astype(str).str.upper()
            audit["inferred"]["bet_side"] = "from pick_side"
        elif "side" in df.columns:
            df["bet_side"] = df["side"].astype(str).str.upper()
            audit["inferred"]["bet_side"] = "from side"
        else:
            raise RuntimeError("[picks_backtest] Cannot infer bet_side (need bet_side/pick_side/side)")

    df["bet_side"] = df["bet_side"].astype(str).str.upper()
    df = df[df["bet_side"].isin(["HOME", "AWAY"])].copy()

    # Embedded execution odds when present
    embedded_cols = [c for c in ["bet_ml", "odds_decimal", "odds", "american_odds", "price", "line"] if c in df.columns]
    if embedded_cols:
        src = embedded_cols[0]
        df["bet_odds_decimal"] = df[src].apply(_coerce_decimal_odds)
        df["odds_decimal"] = df["bet_odds_decimal"]
        audit["mode"] = "embedded_odds"
        audit["inferred"]["bet_odds_decimal"] = f"from {src}"
        audit["inferred"]["odds_decimal"] = "set to bet_odds_decimal (execution odds)"
    else:
        df["bet_odds_decimal"] = np.nan
        df["odds_decimal"] = np.nan
        audit["mode"] = "snapshot_odds"

    # Ensure standard columns exist (additive)
    if "close_odds_decimal" not in df.columns:
        df["close_odds_decimal"] = np.nan
    if "open_odds_decimal" not in df.columns:
        df["open_odds_decimal"] = np.nan

    audit["rows_out"] = int(len(df))
    return df, audit


# ---------------------------------------------------------------------
# Snapshot odds attach (duplicate mk safe + scalar safe)
# ---------------------------------------------------------------------

def _first_non_null_scalar(val: Union[pd.Series, pd.DataFrame, object]) -> object:
    if isinstance(val, pd.Series):
        for x in val.values.tolist():
            if pd.notna(x):
                return x
        return np.nan
    return val


def _get_cell(row: Union[pd.Series, pd.DataFrame], col: Optional[str]) -> object:
    if col is None:
        return np.nan
    if isinstance(row, pd.DataFrame):
        if col not in row.columns:
            return np.nan
        return _first_non_null_scalar(row[col])
    try:
        return row.get(col, np.nan)  # type: ignore
    except Exception:
        return np.nan


def _attach_one_snapshot_type(
    df: pd.DataFrame,
    snapshot_dir: Path,
    snapshot_type: str,
    *,
    out_col: str,
    audit: Dict,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Attach moneyline odds for snapshot_type in {"open","close"} into df[out_col] as decimal odds.
    Uses merge_key+game_date and bet_side to select home/away ML.
    """
    filled = 0
    matched_rows = 0
    found_snaps = 0
    dup_mk_rows = 0
    team_cols_used = None
    ml_cols_used = None
    example_cols = None

    for game_date, grp in df.groupby("game_date"):
        snap_path = _find_snapshot_file(snapshot_dir, str(game_date), snapshot_type=snapshot_type)
        if snap_path is None:
            continue

        found_snaps += 1
        snap = pd.read_csv(snap_path)

        if example_cols is None:
            example_cols = list(snap.columns)

        home_col, away_col = _discover_team_columns(snap)
        home_ml_col, away_ml_col = _discover_ml_columns(snap)
        team_cols_used = (home_col, away_col)
        ml_cols_used = (home_ml_col, away_ml_col)

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

            matched_rows += 1

            if df.at[i, "bet_side"] == "HOME":
                am = _get_cell(row, home_ml_col)
            else:
                am = _get_cell(row, away_ml_col)

            dec = _american_to_decimal(am)
            if pd.notna(dec):
                df.at[i, out_col] = float(dec)
                filled += 1

    audit_key = f"{snapshot_type}_attach"
    audit[audit_key] = {
        "snapshots_found": int(found_snaps),
        "rows_with_snapshot_match": int(matched_rows),
        "odds_filled": int(filled),
        "odds_missing_after": int(df[out_col].isna().sum()),
        "team_cols": team_cols_used,
        "ml_cols": ml_cols_used,
        "example_snapshot_columns": example_cols,
        "dup_mk_rows_seen": int(dup_mk_rows),
        "snapshot_type": snapshot_type,
    }

    return df, audit


def _attach_odds_from_snapshots(
    picks_df: pd.DataFrame,
    snapshot_dir: Optional[Path],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Attach CLOSE odds (existing behavior) and optionally OPEN odds (additive).

    - close_odds_decimal is filled from CLOSE snapshots when possible
    - odds_decimal is filled from close when in snapshot_odds mode
    - if embedded odds exist, we keep them as execution odds and still attach close/open for CLV
    - open_odds_decimal is filled from OPEN snapshots when possible (diagnostics / CLV)
    """
    df = picks_df.copy()
    audit: Dict = {
        "snapshot_dir": str(snapshot_dir) if snapshot_dir else None,
        "dates_seen": int(df["game_date"].nunique()),
        "note": None,
    }

    if snapshot_dir is None:
        audit["note"] = "snapshot_dir not provided; cannot attach odds"
        audit["odds_missing_after"] = int(df["odds_decimal"].isna().sum())
        audit["close_odds_missing_after"] = int(df["close_odds_decimal"].isna().sum()) if "close_odds_decimal" in df.columns else None
        audit["open_odds_missing_after"] = int(df["open_odds_decimal"].isna().sum()) if "open_odds_decimal" in df.columns else None
        return df, audit

    embedded_present = int(df["odds_decimal"].notna().sum()) > 0
    if embedded_present:
        audit["note"] = "embedded odds detected; attaching close+open odds for CLV diagnostics"

    if "close_odds_decimal" not in df.columns:
        df["close_odds_decimal"] = np.nan
    if "open_odds_decimal" not in df.columns:
        df["open_odds_decimal"] = np.nan

    # CLOSE
    df, audit = _attach_one_snapshot_type(
        df=df,
        snapshot_dir=snapshot_dir,
        snapshot_type="close",
        out_col="close_odds_decimal",
        audit=audit,
    )

    # If snapshot mode, execution odds come from CLOSE
    if not embedded_present:
        df["odds_decimal"] = np.where(
            df["odds_decimal"].isna() & df["close_odds_decimal"].notna(),
            df["close_odds_decimal"].astype(float),
            df["odds_decimal"],
        )

    # OPEN (additive)
    df, audit = _attach_one_snapshot_type(
        df=df,
        snapshot_dir=snapshot_dir,
        snapshot_type="open",
        out_col="open_odds_decimal",
        audit=audit,
    )

    audit["odds_decimal_missing_after"] = int(df["odds_decimal"].isna().sum())
    audit["close_odds_missing_after"] = int(df["close_odds_decimal"].isna().sum())
    audit["open_odds_missing_after"] = int(df["open_odds_decimal"].isna().sum())
    return df, audit


# ---------------------------------------------------------------------
# History join helpers (build merge_key if missing)
# ---------------------------------------------------------------------

def _ensure_history_merge_key(history_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    If history has no merge_key column, build deterministically from common column names.
    """
    df = history_df.copy()
    audit: Dict = {"merge_key_built": False, "used_cols": None}

    if "merge_key" in df.columns:
        return df, audit

    cols = {c.lower(): c for c in df.columns}

    date_col = cols.get("game_date") or cols.get("date") or cols.get("game_day")
    home_col = cols.get("home_team") or cols.get("home") or cols.get("home_team_name")
    away_col = cols.get("away_team") or cols.get("away") or cols.get("away_team_name") or cols.get("visitor_team")

    if date_col is None or home_col is None or away_col is None:
        raise RuntimeError(
            "[picks_backtest] History CSV missing merge_key and could not infer "
            "date/home/away columns. Please add merge_key or rename columns to include "
            "game_date, home_team, away_team."
        )

    df["merge_key"] = [
        _merge_key(h, a, d) for h, a, d in zip(df[home_col], df[away_col], df[date_col])
    ]
    audit["merge_key_built"] = True
    audit["used_cols"] = {"game_date": date_col, "home_team": home_col, "away_team": away_col}
    return df, audit


def _find_score_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}
    home_col = cols.get("home_score") or cols.get("home_pts") or cols.get("home_points") or cols.get("pts_home")
    away_col = cols.get("away_score") or cols.get("away_pts") or cols.get("away_points") or cols.get("pts_away")
    if home_col is None or away_col is None:
        raise RuntimeError("[picks_backtest] Could not find home/away score columns in history")
    return home_col, away_col


def _join_history(picks_df: pd.DataFrame, history_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Join hygiene: merge_key is the only shared contract.

    CRITICAL FIX:
    - Drop overlapping columns from history before merge to avoid pandas suffix collisions
      (the exact error you hit: duplicate columns like home_team_x, game_date_x, etc.).
    """
    hist, mk_audit = _ensure_history_merge_key(history_df)

    overlapping = set(picks_df.columns).intersection(set(hist.columns))
    overlapping.discard("merge_key")  # merge_key is the only join contract

    hist_safe = hist.drop(columns=list(overlapping), errors="ignore")

    joined = picks_df.merge(hist_safe, on="merge_key", how="left", validate="m:1")

    audit = {
        "rows_in": int(len(picks_df)),
        "rows_out": int(len(joined)),
        "history_merge_key": mk_audit,
        "overlapping_columns_dropped_from_history": sorted(overlapping),
        "history_match_rate": float(joined["merge_key"].notna().mean()) if len(joined) else 0.0,
    }
    return joined, audit


# ---------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------

def backtest_picks(
    picks_df: pd.DataFrame,
    history_df: pd.DataFrame,
    snapshot_dir: Optional[Path],
    out_dir: Path,
    stake: float = 1.0,
) -> Tuple[pd.DataFrame, Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if picks_df is None or picks_df.empty:
        raise RuntimeError("[picks_backtest] picks_df is empty")

    picks_df, norm_audit = _normalize_picks(picks_df)
    picks_df, odds_audit = _attach_odds_from_snapshots(picks_df, snapshot_dir)

    joined, join_audit = _join_history(picks_df, history_df)
    if joined is None or joined.empty:
        raise RuntimeError("[picks_backtest] All picks dropped after history join")

    home_score_col, away_score_col = _find_score_cols(joined)
    joined["home_win"] = (joined[home_score_col] > joined[away_score_col]).astype(int)
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

    # Execution odds
    if "bet_odds_decimal" not in joined.columns:
        joined["bet_odds_decimal"] = np.nan
    joined["bet_odds_decimal"] = np.where(
        joined["bet_odds_decimal"].notna(),
        joined["bet_odds_decimal"].astype(float),
        joined["odds_decimal"].astype(float),
    )

    # Close odds
    if "close_odds_decimal" not in joined.columns:
        joined["close_odds_decimal"] = np.nan
    joined["close_odds_decimal"] = np.where(
        joined["close_odds_decimal"].notna(),
        joined["close_odds_decimal"].astype(float),
        joined["odds_decimal"].astype(float),
    )

    # Open odds (additive, diagnostics only if missing)
    if "open_odds_decimal" not in joined.columns:
        joined["open_odds_decimal"] = np.nan

    # Implied probabilities
    joined["p_implied_bet"] = np.where(
        joined["bet_odds_decimal"].notna(), 1.0 / joined["bet_odds_decimal"], np.nan
    )
    joined["p_implied_close"] = np.where(
        joined["close_odds_decimal"].notna(), 1.0 / joined["close_odds_decimal"], np.nan
    )
    joined["p_implied_open"] = np.where(
        joined["open_odds_decimal"].notna(), 1.0 / joined["open_odds_decimal"], np.nan
    )

    # Model prob -> pick-side prob (if available)
    if "home_win_prob" in joined.columns:
        joined["p_model"] = np.where(
            joined["bet_side"] == "HOME",
            joined["home_win_prob"].astype(float),
            1.0 - joined["home_win_prob"].astype(float),
        )
        joined["edge_bet"] = joined["p_model"] - joined["p_implied_bet"]
    else:
        joined["p_model"] = np.nan
        joined["edge_bet"] = np.nan

    # CLV: execution vs close
    joined["clv_implied"] = joined["p_implied_bet"] - joined["p_implied_close"]
    joined["clv_positive"] = joined["clv_implied"].apply(lambda x: bool(x > 0) if pd.notna(x) else False)

    # CLV: OPEN → CLOSE (THIS IS THE NEW WIRING YOU WANT)
    # Positive means market moved toward your side by close.
    joined["clv_implied_open_to_close"] = joined["p_implied_open"] - joined["p_implied_close"]
    joined["clv_open_positive"] = joined["clv_implied_open_to_close"].apply(lambda x: bool(x > 0) if pd.notna(x) else False)

    resolved = joined[joined["pnl"].notna()].copy()

    audit = {
        "normalize": norm_audit,
        "odds_attach": odds_audit,
        "history_join": join_audit,
        "resolved_bets": int(len(resolved)),
        "unresolved_missing_odds": int(joined["pnl"].isna().sum()),
    }

    embedded_execution = bool(norm_audit.get("mode") == "embedded_odds")

    clv_available_rate = float(joined["clv_implied"].notna().mean()) if "clv_implied" in joined.columns else 0.0
    if "clv_positive" in joined.columns and joined["clv_implied"].notna().any():
        clv_positive_rate = float(joined.loc[joined["clv_implied"].notna(), "clv_positive"].mean())
    else:
        clv_positive_rate = None

    clv_open_available_rate = float(joined["clv_implied_open_to_close"].notna().mean())
    if joined["clv_implied_open_to_close"].notna().any():
        clv_open_positive_rate = float(joined.loc[joined["clv_implied_open_to_close"].notna(), "clv_open_positive"].mean())
    else:
        clv_open_positive_rate = None

    audit["clv"] = {
        "clv_available_rate": clv_available_rate,
        "clv_positive_rate": clv_positive_rate,
        "clv_open_available_rate": clv_open_available_rate,
        "clv_open_positive_rate": clv_open_positive_rate,
        "embedded_execution_odds_present": embedded_execution,
        "definitions": {
            "clv_implied": "p_implied_bet - p_implied_close",
            "clv_implied_open_to_close": "p_implied_open - p_implied_close",
        },
    }

    # Outputs
    resolved.to_csv(out_dir / "picks_backtest.csv", index=False)

    total_pnl = float(resolved["pnl"].sum()) if not resolved.empty else 0.0
    total_staked = float(resolved["stake"].sum()) if not resolved.empty else 0.0
    roi = float(total_pnl / total_staked) if total_staked > 0 else np.nan

    summary = pd.DataFrame(
        [{
            "bets": int(len(resolved)),
            "wins": int((resolved["pnl"] > 0).sum()) if not resolved.empty else 0,
            "win_rate": float((resolved["pnl"] > 0).mean()) if not resolved.empty else np.nan,
            "total_pnl": total_pnl,
            "total_staked": total_staked,
            "roi": roi,
            "clv_open_available_rate": clv_open_available_rate,
            "clv_open_positive_rate": clv_open_positive_rate,
        }]
    )
    summary.to_csv(out_dir / "picks_backtest_summary.csv", index=False)

    with open(out_dir / "picks_backtest_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True, default=str)

    if resolved.empty:
        raise RuntimeError("[picks_backtest] No resolved bets. (All missing odds?)")

    return resolved, audit


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest picks (embedded odds OR snapshot odds).")
    ap.add_argument("--picks-dir", required=True)
    ap.add_argument("--pattern", default="picks_*.csv")
    ap.add_argument("--history", required=True)
    ap.add_argument("--snapshot-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stake", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "picks_backtest_audit.json"

    try:
        picks_dir = Path(args.picks_dir)
        files = sorted(picks_dir.glob(args.pattern))
        if not files:
            raise RuntimeError(f"[picks_backtest] No picks files matched: {picks_dir}/{args.pattern}")

        dfs = []
        for f in files:
            df = pd.read_csv(f)
            if df is None or df.empty:
                continue
            df = df.dropna(axis=1, how="all")
            df["_source_file"] = f.name
            dfs.append(df)

        if not dfs:
            raise RuntimeError("[picks_backtest] All picks files were empty")

        picks = pd.concat(dfs, ignore_index=True)
        history = pd.read_csv(args.history)

        snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else None
        backtest_picks(
            picks_df=picks,
            history_df=history,
            snapshot_dir=snapshot_dir,
            out_dir=out_dir,
            stake=float(args.stake),
        )
        return 0

    except Exception as e:
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump({"error": str(e)}, f, indent=2, sort_keys=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

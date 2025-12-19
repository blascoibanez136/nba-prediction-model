from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


# ---------- Helpers ----------

def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _american_to_implied_prob(odds: float) -> float:
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return np.nan


def _payout_from_american(odds: float, stake: float = 1.0) -> float:
    # returns profit (not including stake) for a 1-unit stake
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return np.nan
    if odds > 0:
        return stake * (odds / 100.0)
    if odds < 0:
        return stake * (100.0 / (-odds))
    return np.nan


def _norm_team(s: object) -> str:
    # ultra-light normalization for joining; your canonicalization already improved upstream
    if s is None:
        return ""
    return str(s).strip().lower()


def _merge_key(home: str, away: str, game_date: str) -> str:
    return f"{_norm_team(home)}__{_norm_team(away)}__{str(game_date).strip()}"


# ---------- Snapshot loader (audit-only) ----------

def _load_snapshot_for_date(snapshot_dir: Path, game_date: str) -> Optional[pd.DataFrame]:
    """
    Supports your observed naming convention:
      close_YYYYMMDD.csv
    Normalizes to game_date = YYYY-MM-DD externally.
    """
    if not snapshot_dir.exists():
        return None

    ymd = str(game_date).replace("-", "")
    candidates = [
        snapshot_dir / f"close_{ymd}.csv",
        snapshot_dir / f"{ymd}.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df["_snapshot_file"] = p.name
            return df
    return None


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------- Main logic ----------

@dataclass(frozen=True)
class PicksBacktestConfig:
    stake: float = 1.0  # flat unit stake for evaluation
    # Market column candidates (adjustable but safe defaults)
    home_ml_cols: Tuple[str, ...] = ("home_moneyline", "home_ml", "ml_home", "moneyline_home", "home_odds")
    close_home_ml_cols: Tuple[str, ...] = ("home_moneyline_close", "close_home_ml", "home_ml_close", "home_close_ml")
    open_home_ml_cols: Tuple[str, ...] = ("home_moneyline_open", "open_home_ml", "home_ml_open", "home_open_ml")


def backtest_picks(
    picks_df: pd.DataFrame,
    history_df: pd.DataFrame,
    *,
    snapshot_dir: Optional[Path] = None,
    config: PicksBacktestConfig = PicksBacktestConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    if picks_df is None or picks_df.empty:
        raise RuntimeError("[picks_backtest] picks_df is empty")

    # Normalize required columns
    df = picks_df.copy()
    if "game_date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "game_date"})
    required = {"game_date", "home_team", "away_team"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"[picks_backtest] picks_df missing required columns: {sorted(missing)}")

    # Only ML HOME supported for now (safe, explicit)
    if "pick_type" in df.columns:
        df = df[df["pick_type"].astype(str).str.upper() == "ML"].copy()
    if "pick_side" in df.columns:
        df = df[df["pick_side"].astype(str).str.upper() == "HOME"].copy()

    if df.empty:
        audit = {"n_picks_in": int(len(picks_df)), "n_picks_used": 0, "note": "No ML/HOME picks after filtering."}
        return df, pd.DataFrame(), audit

    # Build merge keys for picks and history
    df["merge_key"] = [
        _merge_key(h, a, d) for h, a, d in zip(df["home_team"].tolist(), df["away_team"].tolist(), df["game_date"].tolist())
    ]

    h = history_df.copy()
    # Expect history has at least home_team/away_team/date + scores
    # Try to align column names
    if "game_date" not in h.columns and "date" in h.columns:
        h = h.rename(columns={"date": "game_date"})
    if "home_score" not in h.columns:
        # common alt names
        for alt in ["home_points", "pts_home", "score_home"]:
            if alt in h.columns:
                h = h.rename(columns={alt: "home_score"})
                break
    if "away_score" not in h.columns:
        for alt in ["away_points", "pts_away", "score_away"]:
            if alt in h.columns:
                h = h.rename(columns={alt: "away_score"})
                break

    required_hist = {"game_date", "home_team", "away_team", "home_score", "away_score"}
    missing_hist = required_hist - set(h.columns)
    if missing_hist:
        raise RuntimeError(f"[picks_backtest] history missing required columns: {sorted(missing_hist)}")

    h["merge_key"] = [
        _merge_key(hh, aa, dd) for hh, aa, dd in zip(h["home_team"].tolist(), h["away_team"].tolist(), h["game_date"].tolist())
    ]

    joined = df.merge(
        h[["merge_key", "home_score", "away_score"]],
        on="merge_key",
        how="left",
        suffixes=("", "_hist"),
        validate="m:1",
    )

    # Join coverage
    n_in = int(len(df))
    n_joined = int(joined["home_score"].notna().sum())
    n_missing = n_in - n_joined

    # Compute result
    joined["home_win"] = (joined["home_score"] > joined["away_score"]).astype("float64")
    joined.loc[joined["home_score"].isna() | joined["away_score"].isna(), "home_win"] = np.nan

    # Determine odds to use for ROI
    odds_col = _find_col(joined, list(config.home_ml_cols))
    joined["bet_odds_home"] = _to_float(joined[odds_col]) if odds_col else np.nan

    joined["profit_if_win"] = joined["bet_odds_home"].apply(lambda o: _payout_from_american(o, stake=config.stake))
    joined["stake"] = float(config.stake)

    # Flat-stake ROI for ML:
    # win -> +profit_if_win, loss -> -stake
    joined["profit"] = np.where(
        joined["home_win"] == 1.0,
        joined["profit_if_win"],
        np.where(joined["home_win"] == 0.0, -joined["stake"], np.nan),
    )

    # CLV (optional): load snapshot per date and try to find entry/close odds
    joined["entry_home_ml"] = np.nan
    joined["close_home_ml"] = np.nan
    joined["entry_implied_prob"] = np.nan
    joined["close_implied_prob"] = np.nan
    joined["clv_prob"] = np.nan
    joined["_snapshot_file"] = ""

    if snapshot_dir is not None:
        # Group by date to avoid reloading per row
        for d, idx in joined.groupby("game_date").groups.items():
            snap = _load_snapshot_for_date(snapshot_dir, str(d))
            if snap is None or snap.empty:
                continue

            # identify team columns inside snapshot
            home_col = _find_col(snap, ["home_team", "home", "team_home"])
            away_col = _find_col(snap, ["away_team", "away", "team_away"])
            date_col = _find_col(snap, ["game_date", "date"])
            if date_col and date_col != "game_date":
                snap = snap.rename(columns={date_col: "game_date"})

            if not home_col or not away_col:
                continue

            # odds columns
            close_col = _find_col(snap, list(config.close_home_ml_cols) + list(config.home_ml_cols))
            open_col = _find_col(snap, list(config.open_home_ml_cols))

            snap["_mk"] = [
                _merge_key(hh, aa, d) for hh, aa in zip(snap[home_col].tolist(), snap[away_col].tolist())
            ]

            snap_map = snap.set_index("_mk")

            for i in idx:
                mk = joined.at[i, "merge_key"]
                if mk not in snap_map.index:
                    continue
                row = snap_map.loc[mk]

                joined.at[i, "_snapshot_file"] = str(row.get("_snapshot_file", ""))

                if open_col and open_col in row.index:
                    joined.at[i, "entry_home_ml"] = float(row[open_col]) if pd.notna(row[open_col]) else np.nan
                # fallback: if no open, treat close as entry reference (clv delta unavailable)
                if close_col and close_col in row.index:
                    joined.at[i, "close_home_ml"] = float(row[close_col]) if pd.notna(row[close_col]) else np.nan
                    if pd.isna(joined.at[i, "entry_home_ml"]):
                        joined.at[i, "entry_home_ml"] = joined.at[i, "close_home_ml"]

                ep = _american_to_implied_prob(joined.at[i, "entry_home_ml"])
                cp = _american_to_implied_prob(joined.at[i, "close_home_ml"])
                joined.at[i, "entry_implied_prob"] = ep
                joined.at[i, "close_implied_prob"] = cp
                if pd.notna(ep) and pd.notna(cp):
                    joined.at[i, "clv_prob"] = (cp - ep)

    # Summary stats (only on resolved picks)
    resolved = joined[joined["profit"].notna()].copy()
    n_res = int(len(resolved))
    total_profit = float(resolved["profit"].sum()) if n_res else 0.0
    total_staked = float(resolved["stake"].sum()) if n_res else 0.0
    roi = (total_profit / total_staked) if total_staked > 0 else np.nan
    win_rate = float((resolved["home_win"] == 1.0).mean()) if n_res else np.nan

    # Drawdown / streaks (flat stake profit series)
    resolved = resolved.sort_values(["game_date", "home_team", "away_team"])
    resolved["cum_profit"] = resolved["profit"].cumsum()
    resolved["cum_max"] = resolved["cum_profit"].cummax()
    resolved["drawdown"] = resolved["cum_profit"] - resolved["cum_max"]
    max_dd = float(resolved["drawdown"].min()) if n_res else np.nan

    # losing streak length
    losses = (resolved["profit"] < 0).astype(int).tolist()
    max_ls = 0
    cur = 0
    for x in losses:
        if x == 1:
            cur += 1
            max_ls = max(max_ls, cur)
        else:
            cur = 0

    # CLV summary
    clv_used = resolved[resolved["clv_prob"].notna()]
    clv_mean = float(clv_used["clv_prob"].mean()) if len(clv_used) else np.nan
    clv_n = int(len(clv_used))

    summary = pd.DataFrame([{
        "n_picks_in": int(len(picks_df)),
        "n_picks_used": n_in,
        "n_joined_results": n_joined,
        "n_missing_results": n_missing,
        "n_resolved": n_res,
        "total_profit": total_profit,
        "total_staked": total_staked,
        "roi": roi,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "max_losing_streak": max_ls,
        "clv_n": clv_n,
        "clv_mean_prob": clv_mean,
        "odds_col_used": odds_col or "",
        "snapshot_dir_used": str(snapshot_dir) if snapshot_dir is not None else "",
    }])

    audit = {
        "n_picks_in": int(len(picks_df)),
        "n_picks_used": n_in,
        "n_joined_results": n_joined,
        "n_missing_results": n_missing,
        "odds_col_used": odds_col,
        "snapshot_dir_used": bool(snapshot_dir is not None),
        "notes": [
            "Pick-conditioned backtest is additive; does not modify prediction/backtest core.",
            "ROI computed with flat unit stakes using available bet odds (if present).",
            "CLV computed when snapshot entry/close odds are available; otherwise omitted per-row.",
        ],
    }

    return joined, summary, audit


def main() -> int:
    ap = argparse.ArgumentParser(description="Pick-conditioned backtest + CLV (additive, Commit-4).")
    ap.add_argument("--picks-dir", default="outputs")
    ap.add_argument("--pattern", default="picks_*.csv", help="picks_*.csv or picks_edge_*.csv")
    ap.add_argument("--history", required=True)
    ap.add_argument("--snapshot-dir", default="", help="Optional snapshot dir for CLV")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--stake", type=float, default=1.0)
    args = ap.parse_args()

    picks_dir = Path(args.picks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audits_dir = out_dir / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)

    hist = pd.read_csv(args.history)

    snap_dir = Path(args.snapshot_dir) if args.snapshot_dir else None
    config = PicksBacktestConfig(stake=float(args.stake))

    # Load picks files
    files = sorted(picks_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No picks files matched: {picks_dir}/{args.pattern}")

    all_joined = []
    for f in files:
        df = pd.read_csv(f)
        joined, _, _ = backtest_picks(df, hist, snapshot_dir=snap_dir, config=config)
        if not joined.empty:
            joined["_picks_file"] = f.name
            all_joined.append(joined)

    if not all_joined:
        raise SystemExit("No joined picks produced.")

    joined_all = pd.concat(all_joined, ignore_index=True)

    # Recompute summary over all picks
    summary_all = pd.DataFrame()
    audit_all = {}

    joined_all, summary_all, audit_all = backtest_picks(joined_all, hist, snapshot_dir=snap_dir, config=config)

    out_csv = out_dir / "picks_backtest.csv"
    out_csv_summary = out_dir / "picks_backtest_summary.csv"
    out_audit = audits_dir / "picks_backtest_audit.json"

    joined_all.to_csv(out_csv, index=False)
    summary_all.to_csv(out_csv_summary, index=False)
    out_audit.write_text(json.dumps(audit_all, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[picks_backtest] wrote {out_csv} ({len(joined_all)} rows)")
    print(f"[picks_backtest] wrote {out_csv_summary}")
    print(f"[picks_backtest] wrote {out_audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

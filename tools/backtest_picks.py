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
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return np.nan
    if odds > 0:
        return stake * (odds / 100.0)
    if odds < 0:
        return stake * (100.0 / (-odds))
    return np.nan


def _norm_team(s: object) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def _merge_key(home: str, away: str, game_date: str) -> str:
    return f"{_norm_team(home)}__{_norm_team(away)}__{str(game_date).strip()}"


# ---------- Snapshot loader ----------

def _load_snapshot_for_date(snapshot_dir: Path, game_date: str) -> Optional[pd.DataFrame]:
    if not snapshot_dir.exists():
        return None
    ymd = str(game_date).replace("-", "")
    for p in [snapshot_dir / f"close_{ymd}.csv", snapshot_dir / f"{ymd}.csv"]:
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


# ---------- Core backtest ----------

@dataclass(frozen=True)
class PicksBacktestConfig:
    stake: float = 1.0
    home_ml_cols: Tuple[str, ...] = ("home_moneyline", "home_ml", "ml_home", "moneyline_home", "home_odds")
    close_home_ml_cols: Tuple[str, ...] = ("home_moneyline_close", "close_home_ml", "home_ml_close", "home_close_ml")
    open_home_ml_cols: Tuple[str, ...] = ("home_moneyline_open", "open_home_ml", "home_ml_open", "home_open_ml")


def backtest_picks(
    picks_df: pd.DataFrame,
    history_df: pd.DataFrame,
    *,
    snapshot_dir: Optional[Path],
    config: PicksBacktestConfig,
) -> Tuple[pd.DataFrame, Dict]:
    if picks_df is None or picks_df.empty:
        return pd.DataFrame(), {"note": "empty picks_df"}

    df = picks_df.copy()
    if "game_date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "game_date"})

    df = df[(df.get("pick_type", "ML") == "ML") & (df.get("pick_side", "HOME") == "HOME")].copy()
    if df.empty:
        return pd.DataFrame(), {"note": "no ML/HOME picks"}

    df["merge_key"] = [
        _merge_key(h, a, d)
        for h, a, d in zip(df["home_team"], df["away_team"], df["game_date"])
    ]

    hist = history_df.copy()
    if "game_date" not in hist.columns and "date" in hist.columns:
        hist = hist.rename(columns={"date": "game_date"})

    hist["merge_key"] = [
        _merge_key(h, a, d)
        for h, a, d in zip(hist["home_team"], hist["away_team"], hist["game_date"])
    ]

    joined = df.merge(
        hist[["merge_key", "home_score", "away_score"]],
        on="merge_key",
        how="left",
        validate="m:1",
    )

    joined["home_win"] = (joined["home_score"] > joined["away_score"]).astype(float)

    odds_col = _find_col(joined, list(config.home_ml_cols))
    joined["bet_odds_home"] = _to_float(joined[odds_col]) if odds_col else np.nan
    joined["profit_if_win"] = joined["bet_odds_home"].apply(
        lambda o: _payout_from_american(o, config.stake)
    )
    joined["stake"] = config.stake
    joined["profit"] = np.where(
        joined["home_win"] == 1.0,
        joined["profit_if_win"],
        np.where(joined["home_win"] == 0.0, -config.stake, np.nan),
    )

    joined["clv_prob"] = np.nan
    if snapshot_dir:
        for d, idx in joined.groupby("game_date").groups.items():
            snap = _load_snapshot_for_date(snapshot_dir, d)
            if snap is None:
                continue
            home_col = _find_col(snap, ["home_team"])
            away_col = _find_col(snap, ["away_team"])
            if not home_col or not away_col:
                continue
            snap["_mk"] = [
                _merge_key(h, a, d)
                for h, a in zip(snap[home_col], snap[away_col])
            ]
            snap = snap.set_index("_mk")
            close_col = _find_col(snap, list(config.close_home_ml_cols))
            open_col = _find_col(snap, list(config.open_home_ml_cols))
            for i in idx:
                mk = joined.at[i, "merge_key"]
                if mk not in snap.index:
                    continue
                row = snap.loc[mk]
                entry = row.get(open_col, row.get(close_col))
                close = row.get(close_col)
                ep = _american_to_implied_prob(entry)
                cp = _american_to_implied_prob(close)
                if pd.notna(ep) and pd.notna(cp):
                    joined.at[i, "clv_prob"] = cp - ep

    return joined, {"n_rows": len(joined)}


# ---------- Main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description="Pick-conditioned backtest + CLV")
    ap.add_argument("--picks-dir", default="outputs")
    ap.add_argument("--pattern", default="picks_*.csv")
    ap.add_argument("--history", required=True)
    ap.add_argument("--snapshot-dir", default="")
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
    config = PicksBacktestConfig(stake=args.stake)

    files = sorted(picks_dir.glob(args.pattern))
    if not files:
        raise SystemExit("No pick files found")

    joined_all = []
    for f in files:
        df = pd.read_csv(f)
        joined, _ = backtest_picks(df, hist, snapshot_dir=snap_dir, config=config)
        if not joined.empty:
            joined["_picks_file"] = f.name
            joined_all.append(joined)

    if not joined_all:
        raise SystemExit("No resolved picks")

    joined_all = pd.concat(joined_all, ignore_index=True)

    resolved = joined_all[joined_all["profit"].notna()].copy()
    resolved["cum_profit"] = resolved["profit"].cumsum()
    resolved["cum_max"] = resolved["cum_profit"].cummax()
    resolved["drawdown"] = resolved["cum_profit"] - resolved["cum_max"]

    summary = pd.DataFrame([{
        "n_picks": len(joined_all),
        "n_resolved": len(resolved),
        "total_profit": resolved["profit"].sum(),
        "total_staked": resolved["stake"].sum(),
        "roi": resolved["profit"].sum() / resolved["stake"].sum(),
        "win_rate": (resolved["profit"] > 0).mean(),
        "max_drawdown": resolved["drawdown"].min(),
        "max_losing_streak": (
            resolved["profit"].lt(0)
            .astype(int)
            .groupby((resolved["profit"] >= 0).cumsum())
            .sum()
            .max()
        ),
        "clv_mean_prob": resolved["clv_prob"].mean(),
        "clv_n": resolved["clv_prob"].notna().sum(),
    }])

    joined_all.to_csv(out_dir / "picks_backtest.csv", index=False)
    summary.to_csv(out_dir / "picks_backtest_summary.csv", index=False)
    (audits_dir / "picks_backtest_audit.json").write_text(
        json.dumps({"note": "Aggregate pick-conditioned backtest"}, indent=2)
    )

    print("[picks_backtest] completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

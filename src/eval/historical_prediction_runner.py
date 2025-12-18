from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Single source of truth for feature building + model inference.
from src.model.predict import predict_games  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY = REPO_ROOT / "data" / "history" / "games_2019_2024.csv"
DEFAULT_SNAPSHOTS_DIR = REPO_ROOT / "data" / "_snapshots"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs"
DEFAULT_AUDIT_PATH = DEFAULT_OUT_DIR / "historical_prediction_runner_audit.json"


@dataclass
class Audit:
    history_path: str
    snapshots_dir: str
    out_dir: str
    start: str
    end: str
    apply_market: bool
    overwrite: bool
    games_total_in_history: int
    games_in_window: int
    days_processed: int
    days_skipped_existing: int
    snapshot_files_found: int
    snapshot_files_missing: int
    prediction_files_written: int
    prediction_rows_written: int
    notes: list[str]


def _ensure_datetime_col(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        col = "date"
    elif "game_date" in df.columns:
        col = "game_date"
    elif "gameDate" in df.columns:
        col = "gameDate"
    else:
        raise RuntimeError("History CSV must include a date column (date/game_date).")
    out = df.copy()
    out[col] = pd.to_datetime(out[col]).dt.date
    out = out.rename(columns={col: "game_date"})
    return out


def _ensure_team_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "home_team" not in out.columns:
        for alt in ["homeTeam", "HOME_TEAM", "home"]:
            if alt in out.columns:
                out = out.rename(columns={alt: "home_team"})
                break
    if "away_team" not in out.columns:
        for alt in ["visitor_team", "awayTeam", "VISITOR_TEAM", "away", "visitor"]:
            if alt in out.columns:
                out = out.rename(columns={alt: "away_team"})
                break
    if "home_team" not in out.columns or "away_team" not in out.columns:
        raise RuntimeError("History CSV must include home_team and away_team (or visitor_team).")
    return out


def _american_to_prob(odds: float) -> Optional[float]:
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return 100.0 / (o + 100.0)


def _snapshot_path_for_day(snapshots_dir: Path, day) -> Path:
    ymd = pd.to_datetime(day).strftime("%Y%m%d")
    return snapshots_dir / f"close_{ymd}.csv"


def _try_apply_market_from_snapshot(df_pred: pd.DataFrame, snapshot_csv: Path) -> Tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    snap = pd.read_csv(snapshot_csv)

    if "home_team" not in snap.columns:
        for alt in ["home", "homeTeam", "HOME_TEAM"]:
            if alt in snap.columns:
                snap = snap.rename(columns={alt: "home_team"})
                break
    if "away_team" not in snap.columns:
        for alt in ["away", "awayTeam", "visitor_team", "VISITOR_TEAM"]:
            if alt in snap.columns:
                snap = snap.rename(columns={alt: "away_team"})
                break

    if "home_team" not in snap.columns or "away_team" not in snap.columns:
        notes.append("snapshot_missing_team_cols")
        return df_pred, notes

    # Moneylines -> implied home win probability
    home_ml_col = None
    for c in ["home_moneyline", "home_ml", "home_ml_close", "home_ml_line", "home_odds"]:
        if c in snap.columns:
            home_ml_col = c
            break
    if home_ml_col is not None:
        snap["_home_win_prob_market"] = snap[home_ml_col].apply(_american_to_prob)
    else:
        notes.append("snapshot_missing_home_moneyline_col")

    # Spread / total
    spread_col = None
    for c in ["spread", "spread_line", "home_spread", "close_spread"]:
        if c in snap.columns:
            spread_col = c
            break
    total_col = None
    for c in ["total", "total_line", "close_total"]:
        if c in snap.columns:
            total_col = c
            break

    keep_cols = ["home_team", "away_team"]
    if "_home_win_prob_market" in snap.columns:
        keep_cols.append("_home_win_prob_market")
    if spread_col is not None:
        snap = snap.rename(columns={spread_col: "_fair_spread_market"})
        keep_cols.append("_fair_spread_market")
    else:
        notes.append("snapshot_missing_spread_col")
    if total_col is not None:
        snap = snap.rename(columns={total_col: "_fair_total_market"})
        keep_cols.append("_fair_total_market")
    else:
        notes.append("snapshot_missing_total_col")

    snap_keep = snap[keep_cols].copy()
    out = df_pred.merge(snap_keep, on=["home_team", "away_team"], how="left")

    if "_home_win_prob_market" in out.columns:
        out = out.rename(columns={"_home_win_prob_market": "home_win_prob_market"})
    if "_fair_spread_market" in out.columns:
        out = out.rename(columns={"_fair_spread_market": "fair_spread_market"})
    if "_fair_total_market" in out.columns:
        out = out.rename(columns={"_fair_total_market": "fair_total_market"})

    return out, notes


def _write_csv(path: Path, df: pd.DataFrame, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return True


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate daily historical predictions from a games history CSV.")
    parser.add_argument("--history", type=str, default=str(DEFAULT_HISTORY), help="Path to games history CSV.")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD (inclusive).")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD (inclusive).")
    parser.add_argument("--apply-market", action="store_true", help="Overlay market columns from CLOSE snapshots (best-effort).")
    parser.add_argument("--snapshots-dir", type=str, default=str(DEFAULT_SNAPSHOTS_DIR), help="Directory containing close_YYYYMMDD.csv files.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Directory for outputs/predictions_YYYY-MM-DD.csv files.")
    parser.add_argument("--audit-path", type=str, default=str(DEFAULT_AUDIT_PATH), help="Where to write an audit JSON.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions_*.csv outputs.")
    args = parser.parse_args(argv)

    history_path = Path(args.history)
    snapshots_dir = Path(args.snapshots_dir)
    out_dir = Path(args.out_dir)
    audit_path = Path(args.audit_path)

    df_hist = pd.read_csv(history_path)
    df_hist = _ensure_datetime_col(df_hist)
    df_hist = _ensure_team_cols(df_hist)

    start_d = pd.to_datetime(args.start).date()
    end_d = pd.to_datetime(args.end).date()
    df_window = df_hist[(df_hist["game_date"] >= start_d) & (df_hist["game_date"] <= end_d)].copy()

    notes: list[str] = []
    if df_window.empty:
        raise RuntimeError("No games found in the requested date window.")

    days = sorted(df_window["game_date"].unique().tolist())

    snapshot_found = 0
    snapshot_missing = 0
    files_written = 0
    rows_written = 0
    days_skipped_existing = 0

    for day in days:
        df_day = df_window[df_window["game_date"] == day].copy()

        # predict_games handles loading the models from repo_root/models
        df_pred = predict_games(df_day)

        if args.apply_market:
            snap_path = _snapshot_path_for_day(snapshots_dir, day)
            if snap_path.exists():
                snapshot_found += 1
                df_pred, n = _try_apply_market_from_snapshot(df_pred, snap_path)
                notes.extend([f"{snap_path.name}:{x}" for x in n])
            else:
                snapshot_missing += 1
                print(f"[historical][WARNING] No CLOSE odds snapshot found for {day} in {snapshots_dir} — skipping market overlay")

        out_path = out_dir / f"predictions_{pd.to_datetime(day).strftime('%Y-%m-%d')}.csv"
        wrote = _write_csv(out_path, df_pred, overwrite=args.overwrite)
        if wrote:
            files_written += 1
            rows_written += int(df_pred.shape[0])
            print(f"[historical] Wrote {out_path} ({df_pred.shape[0]} rows)")
        else:
            days_skipped_existing += 1
            print(f"[historical] Skipped existing {out_path} (use --overwrite to replace)")

    audit = Audit(
        history_path=str(history_path),
        snapshots_dir=str(snapshots_dir),
        out_dir=str(out_dir),
        start=str(start_d),
        end=str(end_d),
        apply_market=bool(args.apply_market),
        overwrite=bool(args.overwrite),
        games_total_in_history=int(df_hist.shape[0]),
        games_in_window=int(df_window.shape[0]),
        days_processed=len(days),
        days_skipped_existing=int(days_skipped_existing),
        snapshot_files_found=int(snapshot_found),
        snapshot_files_missing=int(snapshot_missing),
        prediction_files_written=int(files_written),
        prediction_rows_written=int(rows_written),
        notes=notes[:5000],
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(asdict(audit), indent=2))
    print(f"[historical] wrote {audit_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY = REPO_ROOT / "data" / "history" / "games_2019_2024.csv"
DEFAULT_PRED_DIR = REPO_ROOT / "outputs"
DEFAULT_TEAM_INDEX = REPO_ROOT / "models" / "team_index.json"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs"
DEFAULT_AUDIT_PATH = DEFAULT_OUT_DIR / "backtest_join_audit.json"
DEFAULT_METRICS_PATH = DEFAULT_OUT_DIR / "backtest_metrics.json"


ALIASES = {
    "LA Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
    "NY Knicks": "New York Knicks",
    "GS Warriors": "Golden State Warriors",
    "G.S. Warriors": "Golden State Warriors",
    "Nets": "Brooklyn Nets",
    "Knicks": "New York Knicks",
    "Spurs": "San Antonio Spurs",
    "Mavs": "Dallas Mavericks",
    "Wolves": "Minnesota Timberwolves",
    "Blazers": "Portland Trail Blazers",
    "Sixers": "Philadelphia 76ers",
    "Suns": "Phoenix Suns",
    "Bucks": "Milwaukee Bucks",
    "Cavs": "Cleveland Cavaliers",
    "Pels": "New Orleans Pelicans",
}


@dataclass
class JoinAudit:
    pred_dir: str
    history_path: str
    start: str
    end: str
    pred_files: int
    pred_rows: int
    history_rows: int
    joined_rows: int
    missing_pred_rows: int
    missing_result_rows: int
    duplicate_keys_pred: int
    duplicate_keys_hist: int
    notes: list[str]


@dataclass
class Metrics:
    n_games: int
    win_prob_brier: Optional[float]
    win_prob_logloss: Optional[float]
    spread_mae: Optional[float]
    total_mae: Optional[float]
    ats_accuracy: Optional[float]
    ou_accuracy: Optional[float]


def _read_team_index(path: Path) -> Dict[str, int]:
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return {}
    if isinstance(obj, dict) and all(isinstance(v, (int, float)) for v in obj.values()):
        return {str(k): int(v) for k, v in obj.items()}
    return {}


def _norm_team(name: str) -> str:
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    s = str(name).strip()
    s = " ".join(s.split())
    return ALIASES.get(s, s)


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        col = "date"
    elif "game_date" in df.columns:
        col = "game_date"
    elif "gameDate" in df.columns:
        col = "gameDate"
    else:
        raise RuntimeError("Missing date column in input.")
    out = df.copy()
    out[col] = pd.to_datetime(out[col]).dt.date
    out = out.rename(columns={col: "game_date"})
    return out


def _ensure_teams(df: pd.DataFrame, home_col: str, away_col: str) -> pd.DataFrame:
    out = df.copy()
    if home_col not in out.columns:
        raise RuntimeError(f"Missing {home_col} column.")
    if away_col not in out.columns:
        raise RuntimeError(f"Missing {away_col} column.")
    out["home_team"] = out[home_col].apply(_norm_team)
    out["away_team"] = out[away_col].apply(_norm_team)
    return out


def _load_predictions(pred_dir: Path, start_d, end_d) -> pd.DataFrame:
    files = sorted(pred_dir.glob("predictions_*.csv"))
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df = _ensure_date(df)
        df = df[(df["game_date"] >= start_d) & (df["game_date"] <= end_d)]
        if df.empty:
            continue
        df = _ensure_teams(df, "home_team", "away_team")
        rows.append(df)
    if not rows:
        raise RuntimeError(f"No prediction files matched the date window in {pred_dir}.")
    return pd.concat(rows, ignore_index=True)


def _load_history(history_path: Path, start_d, end_d) -> pd.DataFrame:
    df = pd.read_csv(history_path)
    df = _ensure_date(df)

    if "home_team" not in df.columns:
        for alt in ["homeTeam", "HOME_TEAM", "home"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "home_team"})
                break
    if "away_team" not in df.columns:
        for alt in ["visitor_team", "awayTeam", "VISITOR_TEAM", "away", "visitor"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "away_team"})
                break
    df = _ensure_teams(df, "home_team", "away_team")

    if "home_score" not in df.columns:
        for alt in ["home_points", "home_pts", "HOME_SCORE", "homeScore"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "home_score"})
                break
    if "away_score" not in df.columns:
        for alt in ["visitor_score", "away_points", "away_pts", "AWAY_SCORE", "awayScore", "visitor_points"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "away_score"})
                break
    if "home_score" not in df.columns or "away_score" not in df.columns:
        raise RuntimeError("History CSV must include home_score and away_score columns (or common variants).")

    return df[(df["game_date"] >= start_d) & (df["game_date"] <= end_d)].copy()


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean((y - p) ** 2))


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Join daily prediction files to results and compute simple backtest metrics.")
    p.add_argument("--pred-dir", type=str, default=str(DEFAULT_PRED_DIR), help="Directory containing predictions_*.csv.")
    p.add_argument("--history", type=str, default=str(DEFAULT_HISTORY), help="Games history CSV (must include scores).")
    p.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD (inclusive).")
    p.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD (inclusive).")
    p.add_argument("--team-index", type=str, default=str(DEFAULT_TEAM_INDEX), help="team_index.json used for unseen-team checks.")
    p.add_argument("--prob-col", type=str, default="home_win_prob", help="Column in predictions for home win probability.")
    p.add_argument("--spread-col", type=str, default="fair_spread", help="Column in predictions for predicted/fair spread (home - away).")
    p.add_argument("--total-col", type=str, default="fair_total", help="Column in predictions for predicted/fair total.")
    p.add_argument("--audit-path", type=str, default=str(DEFAULT_AUDIT_PATH), help="Path to write join audit JSON.")
    p.add_argument("--metrics-path", type=str, default=str(DEFAULT_METRICS_PATH), help="Path to write metrics JSON.")
    args = p.parse_args(argv)

    pred_dir = Path(args.pred_dir)
    history_path = Path(args.history)
    audit_path = Path(args.audit_path)
    metrics_path = Path(args.metrics_path)

    start_d = pd.to_datetime(args.start).date()
    end_d = pd.to_datetime(args.end).date()

    team_index = _read_team_index(Path(args.team_index))
    notes: list[str] = []
    if not team_index:
        notes.append("team_index_missing_or_unreadable")

    df_pred = _load_predictions(pred_dir, start_d, end_d)
    df_hist = _load_history(history_path, start_d, end_d)

    if team_index:
        pred_teams = set(df_pred["home_team"]).union(set(df_pred["away_team"]))
        unseen = sorted([t for t in pred_teams if t and t not in team_index])
        if unseen:
            notes.append(f"unseen_teams_in_predictions:{len(unseen)}")
            notes.append("unseen_teams_sample:" + ",".join(unseen[:10]))

    key_cols = ["game_date", "home_team", "away_team"]
    dup_pred = int(df_pred.duplicated(subset=key_cols).sum())
    dup_hist = int(df_hist.duplicated(subset=key_cols).sum())

    df_pred_u = df_pred.drop_duplicates(subset=key_cols, keep="last")
    df_hist_u = df_hist.drop_duplicates(subset=key_cols, keep="last")

    joined = df_pred_u.merge(df_hist_u, on=key_cols, how="outer", indicator=True)
    missing_pred = int((joined["_merge"] == "right_only").sum())
    missing_res = int((joined["_merge"] == "left_only").sum())
    joined_ok = joined[joined["_merge"] == "both"].copy()

    audit = JoinAudit(
        pred_dir=str(pred_dir),
        history_path=str(history_path),
        start=str(start_d),
        end=str(end_d),
        pred_files=len(list(pred_dir.glob("predictions_*.csv"))),
        pred_rows=int(df_pred.shape[0]),
        history_rows=int(df_hist.shape[0]),
        joined_rows=int(joined_ok.shape[0]),
        missing_pred_rows=missing_pred,
        missing_result_rows=missing_res,
        duplicate_keys_pred=dup_pred,
        duplicate_keys_hist=dup_hist,
        notes=notes,
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(asdict(audit), indent=2))
    print(f"[backtest] wrote {audit_path}")

    n_games = int(joined_ok.shape[0])

    win_prob_brier = None
    win_prob_logloss = None
    if args.prob_col in joined_ok.columns:
        p_home = pd.to_numeric(joined_ok[args.prob_col], errors="coerce").to_numpy()
        y_home = (
            pd.to_numeric(joined_ok["home_score"], errors="coerce")
            > pd.to_numeric(joined_ok["away_score"], errors="coerce")
        ).astype(int).to_numpy()
        mask = np.isfinite(p_home) & np.isfinite(y_home)
        if mask.any():
            win_prob_brier = _brier(y_home[mask], p_home[mask])
            win_prob_logloss = _logloss(y_home[mask], p_home[mask])

    spread_mae = None
    if args.spread_col in joined_ok.columns:
        pred_spread = pd.to_numeric(joined_ok[args.spread_col], errors="coerce").to_numpy()
        actual_spread = (
            pd.to_numeric(joined_ok["home_score"], errors="coerce")
            - pd.to_numeric(joined_ok["away_score"], errors="coerce")
        ).to_numpy()
        mask = np.isfinite(pred_spread) & np.isfinite(actual_spread)
        if mask.any():
            spread_mae = _mae(actual_spread[mask], pred_spread[mask])

    total_mae = None
    if args.total_col in joined_ok.columns:
        pred_total = pd.to_numeric(joined_ok[args.total_col], errors="coerce").to_numpy()
        actual_total = (
            pd.to_numeric(joined_ok["home_score"], errors="coerce")
            + pd.to_numeric(joined_ok["away_score"], errors="coerce")
        ).to_numpy()
        mask = np.isfinite(pred_total) & np.isfinite(actual_total)
        if mask.any():
            total_mae = _mae(actual_total[mask], pred_total[mask])

    ats_accuracy = None
    if "spread" in joined_ok.columns and args.spread_col in joined_ok.columns:
        line = pd.to_numeric(joined_ok["spread"], errors="coerce").to_numpy()
        pred = pd.to_numeric(joined_ok[args.spread_col], errors="coerce").to_numpy()
        actual = (
            pd.to_numeric(joined_ok["home_score"], errors="coerce")
            - pd.to_numeric(joined_ok["away_score"], errors="coerce")
        ).to_numpy()
        mask = np.isfinite(line) & np.isfinite(pred) & np.isfinite(actual)
        if mask.any():
            bet_home = pred[mask] < line[mask]
            home_covers = actual[mask] > line[mask]
            ats_accuracy = float(np.mean(bet_home == home_covers))

    ou_accuracy = None
    if "total" in joined_ok.columns and args.total_col in joined_ok.columns:
        line = pd.to_numeric(joined_ok["total"], errors="coerce").to_numpy()
        pred = pd.to_numeric(joined_ok[args.total_col], errors="coerce").to_numpy()
        actual = (
            pd.to_numeric(joined_ok["home_score"], errors="coerce")
            + pd.to_numeric(joined_ok["away_score"], errors="coerce")
        ).to_numpy()
        mask = np.isfinite(line) & np.isfinite(pred) & np.isfinite(actual)
        if mask.any():
            bet_over = pred[mask] > line[mask]
            over_hits = actual[mask] > line[mask]
            ou_accuracy = float(np.mean(bet_over == over_hits))

    metrics = Metrics(
        n_games=n_games,
        win_prob_brier=win_prob_brier,
        win_prob_logloss=win_prob_logloss,
        spread_mae=spread_mae,
        total_mae=total_mae,
        ats_accuracy=ats_accuracy,
        ou_accuracy=ou_accuracy,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(asdict(metrics), indent=2))
    print(f"[backtest] wrote {metrics_path}")


if __name__ == "__main__":
    main()

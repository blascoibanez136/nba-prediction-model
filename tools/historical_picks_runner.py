"""
Historical Picks Runner (Commit-4 / Option A)
--------------------------------------------

Purpose:
- Generate pick files across a historical date range from existing daily prediction CSVs.
- Additively annotate pick rows with execution price (bet_ml) resolved from close snapshots.

Rules:
- Deterministic
- No silent drops (all skips logged)
- Does not modify predictions
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRED_RE = re.compile(r"^predictions_(\d{4}-\d{2}-\d{2})\.csv$")


# -----------------------------
# Config / Contracts
# -----------------------------

@dataclass(frozen=True)
class PicksRunnerConfig:
    prob_floor: float = 0.62
    max_picks_per_day: int = 3
    min_games_for_picks: int = 2
    require_calibration_keep: bool = False
    max_abs_gap: float = 0.08
    date_col: str = "game_date"
    prob_col: str = "home_win_prob"


# -----------------------------
# Commit-3 import (preferred)
# -----------------------------

def _try_import_commit3_picks():
    candidates = [
        ("src.picks.conservative_picks", "generate_conservative_picks"),
        ("src.picks.picks", "generate_conservative_picks"),
        ("src.picks.conservative", "generate_conservative_picks"),
    ]
    for mod, fn in candidates:
        try:
            m = __import__(mod, fromlist=[fn])
            return getattr(m, fn)
        except Exception:
            continue
    return None


# -----------------------------
# Minimal fallback picks (only if import fails)
# -----------------------------

def _fallback_generate_conservative_picks(
    preds_df: pd.DataFrame,
    *,
    config: PicksRunnerConfig,
) -> Tuple[pd.DataFrame, Dict]:
    df = preds_df.copy()

    if config.date_col not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": config.date_col})
    if config.date_col not in df.columns:
        df[config.date_col] = ""

    required = {"home_team", "away_team", config.prob_col}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"[historical_picks] Missing required columns in preds: {sorted(missing)}")

    n_games = int(len(df))
    if n_games < config.min_games_for_picks:
        audit = {
            "n_games": n_games,
            "n_candidates": 0,
            "n_picks": 0,
            "policy": {
                "prob_floor": config.prob_floor,
                "max_picks_per_day": config.max_picks_per_day,
                "min_games_for_picks": config.min_games_for_picks,
            },
            "reason": "min_games_for_picks not met",
        }
        return df.head(0), audit

    df["home_win_prob"] = pd.to_numeric(df[config.prob_col], errors="coerce").clip(0.0, 1.0)
    candidates = df[df["home_win_prob"] >= float(config.prob_floor)].copy()
    candidates = candidates.sort_values(["home_win_prob", "home_team", "away_team"], ascending=[False, True, True])

    picks = candidates.head(int(config.max_picks_per_day)).copy()
    if picks.empty:
        audit = {
            "n_games": n_games,
            "n_candidates": int(len(candidates)),
            "n_picks": 0,
            "policy": {
                "prob_floor": config.prob_floor,
                "max_picks_per_day": config.max_picks_per_day,
                "min_games_for_picks": config.min_games_for_picks,
            },
            "reason": "no candidates met prob_floor",
        }
        return picks, audit

    picks["pick_type"] = "ML"
    picks["pick_side"] = "HOME"
    picks["confidence"] = picks["home_win_prob"]
    picks["reason"] = picks["home_win_prob"].apply(lambda p: f"prob>={config.prob_floor:.2f};prob={float(p):.3f}")

    out_cols = [config.date_col, "home_team", "away_team", "pick_type", "pick_side", "confidence", "reason"]
    out_cols = [c for c in out_cols if c in picks.columns]

    audit = {
        "n_games": n_games,
        "n_candidates": int(len(candidates)),
        "n_picks": int(len(picks)),
        "policy": {
            "prob_floor": config.prob_floor,
            "max_picks_per_day": config.max_picks_per_day,
            "min_games_for_picks": config.min_games_for_picks,
        },
        "calibration_present": False,
        "note": "Fallback conservative picks used (Commit-3 module import not found).",
    }
    return picks[out_cols].copy(), audit


# -----------------------------
# Odds annotation helpers (additive)
# -----------------------------

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


def _merge_key(home_team: str, away_team: str, game_date: str) -> str:
    return f"{str(home_team).strip().lower()}__{str(away_team).strip().lower()}__{str(game_date).strip()}"


def _find_close_snapshot(snapshot_dir: Path, game_date: str) -> Optional[Path]:
    ymd = str(game_date).replace("-", "")
    p = snapshot_dir / f"close_{ymd}.csv"
    return p if p.exists() else None


def _discover_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _pick_book_row(snap: pd.DataFrame, preferred_books: list[str]) -> Tuple[pd.DataFrame, Dict]:
    audit: Dict = {"book_col": None, "selected_book": None, "rule": None}

    book_col = _discover_col(snap, ["book", "book_name", "sportsbook", "sportsbook_name"])
    audit["book_col"] = book_col

    if book_col and book_col in snap.columns:
        s = snap.copy()
        s[book_col] = s[book_col].astype(str)

        for b in preferred_books:
            m = s[s[book_col].str.lower() == b.lower()]
            if not m.empty:
                m = m.sort_values(by=[book_col], kind="mergesort").head(1)
                audit["selected_book"] = b
                audit["rule"] = "preferred_book"
                return m, audit

        s2 = s.sort_values(by=[book_col], kind="mergesort").head(1)
        audit["selected_book"] = str(s2.iloc[0][book_col])
        audit["rule"] = "lexicographic_book"
        return s2, audit

    s = snap.copy()
    sort_cols = list(s.columns)
    for c in sort_cols:
        s[c] = s[c].astype(str)
    s = s.sort_values(by=sort_cols, kind="mergesort").head(1)
    audit["rule"] = "stable_sort_all_cols"
    return s, audit


def _attach_bet_ml_from_close_snapshot(
    picks_df: pd.DataFrame,
    *,
    snapshot_dir: Path,
    preferred_books: list[str],
) -> Tuple[pd.DataFrame, Dict]:
    df = picks_df.copy()
    audit: Dict = {
        "snapshot_dir": str(snapshot_dir),
        "snapshot_file": None,
        "attached_rows": 0,
        "missing_snapshot": False,
        "missing_cols": [],
        "book_selection": None,
    }
    if df.empty:
        return df, audit

    if "game_date" not in df.columns:
        audit["missing_cols"].append("game_date")
        return df, audit

    game_date = str(df["game_date"].iloc[0])
    snap_path = _find_close_snapshot(snapshot_dir, game_date)
    if snap_path is None:
        audit["missing_snapshot"] = True
        return df, audit

    audit["snapshot_file"] = snap_path.name
    snap = pd.read_csv(snap_path)

    home_team_col = _discover_col(snap, ["home_team", "home", "team_home", "home_name", "home_team_name"])
    away_team_col = _discover_col(snap, ["away_team", "away", "team_away", "away_name", "away_team_name"])
    home_ml_col = _discover_col(snap, ["ml_home", "home_ml", "moneyline_home", "home_moneyline", "home_ml_close", "ml_home_close"])
    away_ml_col = _discover_col(snap, ["ml_away", "away_ml", "moneyline_away", "away_moneyline", "away_ml_close", "ml_away_close"])

    missing = []
    for name, col in [("home_team", home_team_col), ("away_team", away_team_col), ("ml_home", home_ml_col), ("ml_away", away_ml_col)]:
        if col is None:
            missing.append(name)
    if missing:
        audit["missing_cols"] = missing
        return df, audit

    snap = snap.copy()
    snap["merge_key"] = [_merge_key(h, a, game_date) for h, a in zip(snap[home_team_col], snap[away_team_col])]

    bet_mls = []
    bet_books = []
    bet_rules = []
    bet_odds_dec = []
    bet_snapfile = []

    for _, r in df.iterrows():
        mk = _merge_key(r["home_team"], r["away_team"], game_date)
        rows = snap[snap["merge_key"] == mk]
        if rows.empty:
            bet_mls.append(np.nan)
            bet_books.append(None)
            bet_rules.append("no_match")
            bet_odds_dec.append(np.nan)
            bet_snapfile.append(snap_path.name)
            continue

        chosen, book_audit = _pick_book_row(rows, preferred_books)
        audit["book_selection"] = book_audit

        side = str(r.get("pick_side", "HOME")).upper()
        if side == "AWAY":
            am = chosen.iloc[0][away_ml_col]
        else:
            am = chosen.iloc[0][home_ml_col]

        bet_mls.append(am)
        bet_books.append(book_audit.get("selected_book"))
        bet_rules.append(book_audit.get("rule"))
        bet_odds_dec.append(_american_to_decimal(am))
        bet_snapfile.append(snap_path.name)

    df["bet_ml"] = bet_mls
    df["bet_odds_decimal"] = bet_odds_dec
    df["bet_book"] = bet_books
    df["bet_book_rule"] = bet_rules
    df["bet_snapshot_file"] = bet_snapfile

    audit["attached_rows"] = int(df["bet_ml"].notna().sum())
    return df, audit


# -----------------------------
# File discovery
# -----------------------------

def _list_prediction_files(pred_dir: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for p in sorted(pred_dir.glob("predictions_*.csv")):
        m = PRED_RE.match(p.name)
        if not m:
            continue
        out.append((m.group(1), p))
    return out


# -----------------------------
# Main runner
# -----------------------------

def run_historical_picks(
    *,
    pred_dir: Path,
    out_dir: Path,
    snapshot_dir: Path,
    preferred_books: list[str],
    start: Optional[str],
    end: Optional[str],
    overwrite: bool,
    config: PicksRunnerConfig,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    audits_dir = out_dir / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)

    files = _list_prediction_files(pred_dir)
    if not files:
        raise RuntimeError(f"[historical_picks] No prediction files found in {pred_dir} matching predictions_YYYY-MM-DD.csv")

    if start:
        files = [(d, p) for (d, p) in files if d >= start]
    if end:
        files = [(d, p) for (d, p) in files if d <= end]

    if not files:
        raise RuntimeError("[historical_picks] No prediction files remain after date filtering.")

    commit3_fn = _try_import_commit3_picks()

    n_days = 0
    n_written = 0
    n_skipped_exists = 0
    n_skipped_empty = 0
    n_errors = 0
    errors: List[Dict] = []
    n_odds_attached = 0
    n_missing_snapshot = 0

    for game_date, pred_path in files:
        n_days += 1
        picks_path = out_dir / f"picks_{game_date}.csv"
        audit_path = audits_dir / f"picks_{game_date}_audit.json"

        if picks_path.exists() and not overwrite:
            n_skipped_exists += 1
            continue

        try:
            preds = pd.read_csv(pred_path)
            if preds.empty:
                n_skipped_empty += 1
                audit_path.write_text(
                    json.dumps(
                        {"game_date": game_date, "status": "skipped_empty_predictions", "pred_file": pred_path.name},
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                continue

            if config.date_col not in preds.columns:
                preds[config.date_col] = game_date

            if commit3_fn is not None:
                try:
                    picks, audit = commit3_fn(
                        preds,
                        prob_floor=config.prob_floor,
                        max_picks_per_day=config.max_picks_per_day,
                        min_games_for_picks=config.min_games_for_picks,
                        require_calibration_keep=config.require_calibration_keep,
                        max_abs_gap=config.max_abs_gap,
                    )
                except TypeError:
                    picks, audit = commit3_fn(preds)
            else:
                picks, audit = _fallback_generate_conservative_picks(preds, config=config)

            # Ensure game_date on picks
            if not picks.empty and "game_date" not in picks.columns:
                picks["game_date"] = game_date

            # NEW: annotate picks with bet_ml from close snapshot (additive)
            odds_audit = {}
            if not picks.empty:
                picks, odds_audit = _attach_bet_ml_from_close_snapshot(
                    picks,
                    snapshot_dir=snapshot_dir,
                    preferred_books=preferred_books,
                )
                n_odds_attached += int(odds_audit.get("attached_rows", 0))
                if odds_audit.get("missing_snapshot"):
                    n_missing_snapshot += 1

            picks.to_csv(picks_path, index=False)

            audit_out = {
                "game_date": game_date,
                "pred_file": pred_path.name,
                "picks_file": picks_path.name,
                **(audit if isinstance(audit, dict) else {"audit": str(audit)}),
                "odds_annotation": odds_audit,
            }
            audit_path.write_text(json.dumps(audit_out, indent=2, sort_keys=True), encoding="utf-8")
            n_written += 1

            logger.info("[historical_picks] Wrote %s (%d rows)", picks_path.as_posix(), len(picks))

        except Exception as e:
            n_errors += 1
            err = {"game_date": game_date, "pred_file": pred_path.name, "error": repr(e)}
            errors.append(err)
            logger.exception("[historical_picks] error for %s: %s", game_date, e)
            audit_path.write_text(json.dumps({"game_date": game_date, "status": "error", **err}, indent=2, sort_keys=True), encoding="utf-8")

    runner_audit = {
        "pred_dir": str(pred_dir),
        "out_dir": str(out_dir),
        "snapshot_dir": str(snapshot_dir),
        "preferred_books": preferred_books,
        "start": start,
        "end": end,
        "overwrite": overwrite,
        "n_days_considered": n_days,
        "n_days_written": n_written,
        "n_skipped_exists": n_skipped_exists,
        "n_skipped_empty_predictions": n_skipped_empty,
        "n_errors": n_errors,
        "errors": errors[:50],
        "odds_annotation": {
            "total_rows_with_bet_ml_attached": n_odds_attached,
            "days_missing_snapshot": n_missing_snapshot,
        },
        "policy": {
            "prob_floor": config.prob_floor,
            "max_picks_per_day": config.max_picks_per_day,
            "min_games_for_picks": config.min_games_for_picks,
            "require_calibration_keep": config.require_calibration_keep,
            "max_abs_gap": config.max_abs_gap,
        },
        "note": "Additive historical picks generation + bet_ml annotation; does not modify prediction artifacts.",
    }

    (out_dir / "historical_picks_runner_audit.json").write_text(
        json.dumps(runner_audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return runner_audit


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser(description="Generate picks_YYYY-MM-DD.csv across a date range from predictions_YYYY-MM-DD.csv.")
    ap.add_argument("--pred-dir", default="outputs", help="Directory containing predictions_YYYY-MM-DD.csv files")
    ap.add_argument("--out-dir", default="outputs", help="Directory to write picks_YYYY-MM-DD.csv files")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD inclusive")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD inclusive")
    ap.add_argument("--overwrite", action="store_true", default=False)

    # NEW: snapshots
    ap.add_argument("--snapshot-dir", default="data/_snapshots", help="Directory containing close_YYYYMMDD.csv")
    ap.add_argument(
        "--preferred-books",
        default="DraftKings,FanDuel,BetMGM,Caesars",
        help="Comma-separated list used for deterministic book selection if multiple rows exist",
    )

    # Conservative policy knobs
    ap.add_argument("--prob-floor", type=float, default=0.62)
    ap.add_argument("--max-picks-per-day", type=int, default=3)
    ap.add_argument("--min-games-for-picks", type=int, default=2)
    ap.add_argument("--require-calibration-keep", action="store_true", default=False)
    ap.add_argument("--max-abs-gap", type=float, default=0.08)

    args = ap.parse_args()

    preferred_books = [s.strip() for s in str(args.preferred_books).split(",") if s.strip()]

    audit = run_historical_picks(
        pred_dir=Path(args.pred_dir),
        out_dir=Path(args.out_dir),
        snapshot_dir=Path(args.snapshot_dir),
        preferred_books=preferred_books,
        start=args.start,
        end=args.end,
        overwrite=bool(args.overwrite),
        config=PicksRunnerConfig(
            prob_floor=float(args.prob_floor),
            max_picks_per_day=int(args.max_picks_per_day),
            min_games_for_picks=int(args.min_games_for_picks),
            require_calibration_keep=bool(args.require_calibration_keep),
            max_abs_gap=float(args.max_abs_gap),
        ),
    )
    print("[historical_picks] wrote outputs/historical_picks_runner_audit.json")
    print(f"[historical_picks] days_written={audit['n_days_written']} errors={audit['n_errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

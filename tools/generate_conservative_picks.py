from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.picks.conservative_picks import PicksPolicy, generate_conservative_picks, load_calibration_table


# -----------------------------
# Odds helpers (local, additive)
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
    """
    Deterministic selection:
      1) if a book column exists and preferred book present: choose first preferred
      2) else choose lexicographically smallest book
      3) else choose first row after stable sort by all columns (stringified)
    Returns a 1-row DataFrame and an audit dict.
    """
    audit: Dict = {"book_col": None, "selected_book": None, "rule": None}

    book_col = _discover_col(snap, ["book", "book_name", "sportsbook", "sportsbook_name"])
    audit["book_col"] = book_col

    if book_col and book_col in snap.columns:
        s = snap.copy()
        s[book_col] = s[book_col].astype(str)

        # Preferred list
        for b in preferred_books:
            m = s[s[book_col].str.lower() == b.lower()]
            if not m.empty:
                m = m.sort_values(by=[book_col], kind="mergesort").head(1)
                audit["selected_book"] = b
                audit["rule"] = "preferred_book"
                return m, audit

        # Lexicographically smallest book
        s2 = s.sort_values(by=[book_col], kind="mergesort").head(1)
        audit["selected_book"] = str(s2.iloc[0][book_col])
        audit["rule"] = "lexicographic_book"
        return s2, audit

    # No book column: stable sort by all columns as strings
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
    """
    Additive annotation step:
      - reads close snapshot for the picks date
      - resolves bet_ml based on pick_side (HOME/AWAY)
      - records bet_book + snapshot filename + rule
    """
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

    # Ensure we have game_date
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

    # Discover home/away team and ML cols
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

    # For each pick, resolve rows for mk then choose book deterministically
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
        audit["book_selection"] = book_audit  # representative (same policy for all)

        # Determine side column (you currently output pick_side=HOME; keep generic)
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate conservative picks from prediction CSV (audit-only).")
    ap.add_argument("--pred", required=True, help="Path to predictions_YYYY-MM-DD.csv")
    ap.add_argument("--calibration", default="outputs/backtest_calibration.csv", help="Path to backtest_calibration.csv")
    ap.add_argument("--out-dir", default="outputs", help="Output dir for picks file")

    # Conservative policy knobs
    ap.add_argument("--prob-floor", type=float, default=0.62)
    ap.add_argument("--max-abs-gap", type=float, default=0.08)
    ap.add_argument("--require-cal-keep", action="store_true", default=False)
    ap.add_argument("--max-picks", type=int, default=3)
    ap.add_argument("--min-games", type=int, default=2)

    # NEW (additive): snapshot annotation
    ap.add_argument("--snapshot-dir", default="data/_snapshots", help="Directory containing close_YYYYMMDD.csv")
    ap.add_argument(
        "--preferred-books",
        default="DraftKings,FanDuel,BetMGM,Caesars",
        help="Comma-separated list used for deterministic book selection if multiple rows exist",
    )

    args = ap.parse_args()

    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise SystemExit(f"Missing pred file: {pred_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audits_dir = out_dir / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(pred_path)

    calibration_df = None
    cal_path = Path(args.calibration)
    if cal_path.exists():
        calibration_df = load_calibration_table(cal_path)

    policy = PicksPolicy(
        prob_floor=float(args.prob_floor),
        max_abs_gap=float(args.max_abs_gap),
        require_calibration_keep=bool(args.require_cal_keep),
        max_picks_per_day=int(args.max_picks),
        min_games_for_picks=int(args.min_games),
        n_buckets=10,
    )

    picks_df, audit = generate_conservative_picks(preds, calibration_df=calibration_df, policy=policy)

    # Output name: picks_YYYY-MM-DD.csv derived from predictions filename
    stem = pred_path.stem.replace("predictions_", "")
    picks_path = out_dir / f"picks_{stem}.csv"

    # NEW: annotate picks with bet_ml from close snapshot (additive)
    snap_dir = Path(args.snapshot_dir)
    preferred_books = [s.strip() for s in str(args.preferred_books).split(",") if s.strip()]
    odds_audit = {}
    if not picks_df.empty:
        picks_df, odds_audit = _attach_bet_ml_from_close_snapshot(
            picks_df,
            snapshot_dir=snap_dir,
            preferred_books=preferred_books,
        )

    picks_df.to_csv(picks_path, index=False)

    audit_out = dict(audit) if isinstance(audit, dict) else {"audit": str(audit)}
    audit_out["odds_annotation"] = odds_audit
    audit_path = audits_dir / f"picks_{stem}_audit.json"
    audit_path.write_text(json.dumps(audit_out, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[picks] wrote {picks_path} ({len(picks_df)} rows)")
    print(f"[picks] wrote {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

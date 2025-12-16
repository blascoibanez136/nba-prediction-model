"""
Moneyline ROI analysis with probability, EV, and calibrated EV modes
(American-only, contract-first, bet-gated).

ML Selector v3 (PRODUCTION):
- Default ml_side=away_only
- EV-based trimming to enforce max_bet_rate (no more fail-loud bet-rate crash)
- Optional diagnostic override: --disable-bet-cap (keeps everything for inspection)

Modes:
  prob   : select by probability edge vs de-vig market
  ev     : select by EV using raw model probability
  ev_cal : select by EV using market-relative calibrated probability (delta-calibrator)

Critical invariants preserved:
- American-only moneyline contract (abs(odds) >= 100, no decimals)
- Zero/NaN/malformed odds removed upstream (belt+ suspenders here)
- Coverage reporting + warning (<95%) + hard fail (<80%)
- Profit sanity guard
- No change to daily prediction outputs (this is ROI/selection only)

Notes:
- This script is intentionally "selection-only" and safe to run on backtest_per_game.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.model.calibration import load_calibrator, apply_calibrator
from src.model.market_relative_calibration import load_delta_calibrator, apply_delta_calibrator
from src.utils.odds_math import (
    clean_american_ml,
    devig_home_prob,
    expected_value_units,
    win_profit_per_unit_american,
)

ML_ROI_VERSION = "ml_roi_v9_ev_trimmed_away_default_coverage_guards_2025-12-16"
REQUIRED_ODDS_COLS = ["ml_home_consensus", "ml_away_consensus"]


# -----------------------------
# Helpers
# -----------------------------
PROB_EPS = 1e-6


def _clip_prob(x: Any) -> Optional[float]:
    try:
        p = float(x)
    except Exception:
        return None
    if math.isnan(p) or math.isinf(p):
        return None
    if not (0.0 < p < 1.0):
        return None
    return max(PROB_EPS, min(1.0 - PROB_EPS, p))


def pick_model_prob_col(df: pd.DataFrame) -> str:
    candidates = ["home_win_prob_model_raw", "home_win_prob_model", "home_win_prob"]
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"[ml_roi] No model prob column found. Tried: {candidates}")


def ensure_home_win_actual(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "home_win_actual" in out.columns:
        out["home_win_actual"] = pd.to_numeric(out["home_win_actual"], errors="coerce")
        return out
    if "home_score" in out.columns and "away_score" in out.columns:
        hs = pd.to_numeric(out["home_score"], errors="coerce")
        aw = pd.to_numeric(out["away_score"], errors="coerce")
        out["home_win_actual"] = (hs > aw).astype(float)
        return out
    raise RuntimeError("[ml_roi] Missing home_win_actual and cannot infer from scores.")


def _get_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["game_date", "date", "gamedate"]:
        if c in df.columns:
            return c
    return None


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def summarize(bets: pd.DataFrame) -> Dict[str, Any]:
    if bets is None or bets.empty:
        return {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": None, "win_rate": None}
    stake = float(pd.to_numeric(bets["stake"], errors="coerce").fillna(0).sum())
    profit = float(pd.to_numeric(bets["profit"], errors="coerce").fillna(0).sum())
    roi = (profit / stake) if stake > 0 else None
    win_rate = float((bets["result"].astype(str).str.lower() == "win").mean())
    return {"bets": int(len(bets)), "stake": stake, "profit": profit, "roi": roi, "win_rate": win_rate}


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class MLROIConfig:
    per_game_path: str
    mode: str = "prob"              # prob | ev | ev_cal
    ml_side: str = "away_only"      # both | away_only | home_only

    prob_edge_threshold: float = 0.04
    ev_threshold: float = 0.01

    calibrator_path: Optional[str] = None          # diagnostics only
    delta_calibrator_path: Optional[str] = None    # required for ev_cal

    # eval window (optional)
    eval_start: Optional[str] = None
    eval_end: Optional[str] = None

    out_dir: str = "outputs"

    # guards
    max_profit_abs: float = 10.0
    max_bet_rate: float = 0.35

    # coverage guardrails
    coverage_warn: float = 0.95
    coverage_fail: float = 0.80

    # diagnostic override
    disable_bet_cap: bool = False

    # payout sanity gate
    enforce_max_profit_per_unit_gate: bool = True


# -----------------------------
# Core: build + settle bets
# -----------------------------
def build_bets(per_game: pd.DataFrame, cfg: MLROIConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if per_game is None or per_game.empty:
        raise RuntimeError("[ml_roi] per_game is empty")

    missing = [c for c in REQUIRED_ODDS_COLS if c not in per_game.columns]
    if missing:
        raise RuntimeError(f"[ml_roi] Missing required odds columns: {missing}")

    df = ensure_home_win_actual(per_game)

    # Optional eval window filter
    eval_info: Dict[str, Any] = {"enabled": False}
    date_col = _get_date_col(df)
    if (cfg.eval_start or cfg.eval_end):
        if not date_col:
            raise RuntimeError("[ml_roi] eval-start/eval-end provided but no date column found")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).copy()

        es = pd.to_datetime(cfg.eval_start, errors="coerce") if cfg.eval_start else None
        ee = pd.to_datetime(cfg.eval_end, errors="coerce") if cfg.eval_end else None
        if cfg.eval_start and pd.isna(es):
            raise RuntimeError("[ml_roi] invalid eval-start")
        if cfg.eval_end and pd.isna(ee):
            raise RuntimeError("[ml_roi] invalid eval-end")

        if es is not None:
            df = df[df[date_col] >= es].copy()
        if ee is not None:
            df = df[df[date_col] <= ee].copy()

        eval_info = {
            "enabled": True,
            "date_col": date_col,
            "start": str(es.date()) if es is not None else None,
            "end": str(ee.date()) if ee is not None else None,
            "rows": int(len(df)),
        }

    if df.empty:
        raise RuntimeError("[ml_roi] No rows after eval window filtering")

    # Sanitize odds (belt + suspenders)
    df["ml_home_consensus"] = pd.to_numeric(df["ml_home_consensus"], errors="coerce").apply(clean_american_ml)
    df["ml_away_consensus"] = pd.to_numeric(df["ml_away_consensus"], errors="coerce").apply(clean_american_ml)

    # Market prob (devig)
    dev = df.apply(
        lambda r: devig_home_prob(r["ml_home_consensus"], r["ml_away_consensus"]),
        axis=1,
        result_type="expand",
    )
    df["market_prob_home"] = dev[0]
    df["market_prob_method"] = dev[1]

    # Model probability
    model_col = pick_model_prob_col(df)
    df[model_col] = pd.to_numeric(df[model_col], errors="coerce")
    df["model_prob_home_raw"] = df[model_col].apply(_clip_prob)
    df["model_prob_away_raw"] = df["model_prob_home_raw"].apply(lambda x: (1.0 - x) if x is not None else None)

    # Optional absolute calibrator for diagnostics only
    if cfg.calibrator_path:
        cal = load_calibrator(cfg.calibrator_path)
        diag_target = "home_win_prob" if "home_win_prob" in df.columns else model_col
        df[diag_target] = pd.to_numeric(df[diag_target], errors="coerce")
        df[f"{diag_target}_calibrated"] = apply_calibrator(df[diag_target], cal)
        print(f"[ml_roi] Applied calibrator from {cfg.calibrator_path} to '{diag_target}' (diagnostics only).")

    # Coverage (ML odds + market prob availability)
    total_games = int(df["merge_key"].nunique()) if "merge_key" in df.columns else int(len(df))
    usable_mask = (
        df["ml_home_consensus"].notna()
        & df["ml_away_consensus"].notna()
        & df["market_prob_home"].notna()
        & df["model_prob_home_raw"].notna()
    )
    usable_games = int(df.loc[usable_mask, "merge_key"].nunique()) if "merge_key" in df.columns else int(usable_mask.sum())
    coverage = usable_games / max(total_games, 1)
    if coverage < cfg.coverage_warn:
        print(f"[ml_roi] WARNING: ML odds coverage {coverage:.3f} (<{cfg.coverage_warn})")
    if coverage < cfg.coverage_fail:
        raise RuntimeError(f"[ml_roi] Coverage too low: {coverage:.3f} (<{cfg.coverage_fail}).")

    # Payout sanity gate (pre-bet)
    if cfg.enforce_max_profit_per_unit_gate:
        home_ppu = df["ml_home_consensus"].apply(win_profit_per_unit_american)
        away_ppu = df["ml_away_consensus"].apply(win_profit_per_unit_american)
        ppu_gate = (
            (home_ppu.isna() | (home_ppu <= cfg.max_profit_abs))
            & (away_ppu.isna() | (away_ppu <= cfg.max_profit_abs))
        )
    else:
        ppu_gate = pd.Series([True] * len(df), index=df.index)

    df["eligible"] = usable_mask & ppu_gate

    # Edges (prob mode)
    df["home_edge_prob"] = pd.to_numeric(df["model_prob_home_raw"], errors="coerce") - pd.to_numeric(df["market_prob_home"], errors="coerce")
    df["away_edge_prob"] = pd.to_numeric(df["model_prob_away_raw"], errors="coerce") - (1.0 - pd.to_numeric(df["market_prob_home"], errors="coerce"))

    # EV raw
    df["home_ev_raw"] = df.apply(lambda r: expected_value_units(r["model_prob_home_raw"], r["ml_home_consensus"]), axis=1)
    df["away_ev_raw"] = df.apply(lambda r: expected_value_units(r["model_prob_away_raw"], r["ml_away_consensus"]), axis=1)

    # EV calibrated (market-relative)
    mode = str(cfg.mode).strip().lower()
    if mode == "ev_cal":
        if not cfg.delta_calibrator_path:
            raise RuntimeError("[ml_roi] mode=ev_cal requires --delta-calibrator")
        cal_obj = load_delta_calibrator(cfg.delta_calibrator_path)
        df["delta_home"] = pd.to_numeric(df["model_prob_home_raw"], errors="coerce") - pd.to_numeric(df["market_prob_home"], errors="coerce")

        df["model_prob_home_cal"] = df.apply(
            lambda r: apply_delta_calibrator(
                delta=(float(r["delta_home"]) if pd.notna(r["delta_home"]) else None),
                ml_home_odds=(float(r["ml_home_consensus"]) if pd.notna(r["ml_home_consensus"]) else None),
                calibrator=cal_obj,
            ),
            axis=1,
        )
        df["model_prob_home_cal"] = df["model_prob_home_cal"].apply(_clip_prob)
        df["model_prob_away_cal"] = df["model_prob_home_cal"].apply(lambda x: (1.0 - x) if x is not None else None)

        df["home_ev_cal"] = df.apply(lambda r: expected_value_units(r["model_prob_home_cal"], r["ml_home_consensus"]), axis=1)
        df["away_ev_cal"] = df.apply(lambda r: expected_value_units(r["model_prob_away_cal"], r["ml_away_consensus"]), axis=1)

    ml_side = str(cfg.ml_side).strip().lower()
    if ml_side not in ("both", "away_only", "home_only"):
        raise RuntimeError("[ml_roi] Invalid --ml-side. Use: both | away_only | home_only")

    if mode not in ("prob", "ev", "ev_cal"):
        raise RuntimeError("[ml_roi] Invalid --mode. Use: prob | ev | ev_cal")

    # Choose score columns
    if mode == "prob":
        home_score_col, away_score_col, threshold, metric_type = "home_edge_prob", "away_edge_prob", cfg.prob_edge_threshold, "prob_edge"
    elif mode == "ev":
        home_score_col, away_score_col, threshold, metric_type = "home_ev_raw", "away_ev_raw", cfg.ev_threshold, "ev_units"
    else:
        home_score_col, away_score_col, threshold, metric_type = "home_ev_cal", "away_ev_cal", cfg.ev_threshold, "ev_units_cal"

    def choose_side(r) -> Tuple[bool, Optional[str], Optional[float], Optional[str]]:
        if not bool(r["eligible"]):
            return False, None, None, None

        hs = r.get(home_score_col, None)
        as_ = r.get(away_score_col, None)

        # Side policy
        if ml_side == "away_only":
            hs = float("nan")
        elif ml_side == "home_only":
            as_ = float("nan")

        if pd.isna(hs) and pd.isna(as_):
            return False, None, None, None

        # Pick best side above threshold
        if pd.notna(hs) and float(hs) >= threshold and (pd.isna(as_) or float(hs) >= float(as_)):
            return True, "home", float(hs), metric_type
        if pd.notna(as_) and float(as_) >= threshold and (pd.isna(hs) or float(as_) > float(hs)):
            return True, "away", float(as_), metric_type

        return False, None, None, None

    chosen = df.apply(choose_side, axis=1, result_type="expand")
    df["bet"] = chosen[0].astype(bool)
    df["bet_side"] = chosen[1]
    df["metric_used"] = chosen[2]
    df["metric_type"] = chosen[3]

    bets = df[df["bet"]].copy()
    diag: Dict[str, Any] = {
        "eval_window": eval_info,
        "total_games": total_games,
        "usable_games": usable_games,
        "coverage": coverage,
        "pretrim_bets": int(len(bets)),
        "disable_bet_cap": bool(cfg.disable_bet_cap),
        "max_bet_rate": cfg.max_bet_rate,
        "mode": mode,
        "ml_side": ml_side,
        "metric_type": metric_type,
    }

    if bets.empty:
        return bets.reset_index(drop=True), diag

    # Stake 1u (selection-only; sizing is not part of ML v3 yet)
    bets["stake"] = 1.0

    # Pick odds price
    bets["odds_price"] = bets.apply(
        lambda r: r["ml_home_consensus"] if str(r["bet_side"]).lower() == "home" else r["ml_away_consensus"],
        axis=1,
    )
    bets["odds_price"] = pd.to_numeric(bets["odds_price"], errors="coerce")

    # Contract assertion on bet odds (belt + suspenders)
    invalid = bets["odds_price"].apply(lambda x: pd.notna(x) and (0 < abs(float(x)) < 100))
    if bool(invalid.any()):
        sample = bets.loc[invalid, ["game_date", "home_team", "away_team", "bet_side", "odds_price"]].head(10)
        raise RuntimeError(
            "[ml_roi] American-only ML contract violation: found bet rows with 0 < abs(odds) < 100. Sample:\n"
            + sample.to_string(index=False)
        )

    # -------------------------
    # ML Selector v3: EV-trim bet-rate cap (like Totals v3)
    # -------------------------
    if not cfg.disable_bet_cap:
        max_bets = max(1, int(math.floor(cfg.max_bet_rate * max(total_games, 1))))
        bets = bets.sort_values("metric_used", ascending=False)
        if len(bets) > max_bets:
            print(f"[ml_roi] Bet-rate capped: trimming {len(bets)} → {max_bets} (cap={cfg.max_bet_rate:.2f})")
            bets = bets.head(max_bets).copy()

    diag["posttrim_bets"] = int(len(bets))
    diag["bet_rate"] = float(len(bets) / max(total_games, 1))

    # -------------------------
    # Settlement
    # -------------------------
    def settle_result(r) -> str:
        side = str(r["bet_side"]).lower()
        hwa = r["home_win_actual"]
        if pd.isna(hwa):
            return "unknown"
        if side == "home":
            return "win" if hwa == 1 else "loss"
        if side == "away":
            return "win" if hwa == 0 else "loss"
        return "unknown"

    bets["result"] = bets.apply(settle_result, axis=1).astype(str)

    def profit_units(r) -> float:
        stake = float(r["stake"])
        res = str(r["result"]).lower()
        if res == "push":
            return 0.0
        if res != "win":
            return -stake
        ppu = win_profit_per_unit_american(r["odds_price"])
        if ppu is None:
            return 0.0
        return stake * float(ppu)

    bets["profit"] = bets.apply(profit_units, axis=1)
    bets["market"] = "moneyline"

    # Profit sanity
    max_abs = float(pd.to_numeric(bets["profit"], errors="coerce").abs().max())
    if max_abs > cfg.max_profit_abs:
        sample = bets.sort_values("profit", ascending=False).head(10)[
            ["game_date", "home_team", "away_team", "bet_side", "odds_price", "result", "stake", "profit", "metric_used", "metric_type"]
        ]
        raise RuntimeError(
            f"[ml_roi] Profit sanity failure: max |profit|={max_abs}u (limit={cfg.max_profit_abs}). Sample:\n"
            + sample.to_string(index=False)
        )

    return bets.reset_index(drop=True), diag


def main() -> None:
    ap = argparse.ArgumentParser("ml_roi_analysis.py (American-only, bet-gated, EV-trimmed ML Selector v3)")

    ap.add_argument("--per_game", required=True, help="Path to outputs/backtest_per_game.csv")
    ap.add_argument("--mode", default="prob", choices=["prob", "ev", "ev_cal"], help="Selection mode: prob | ev | ev_cal")

    # ML Selector v3 default: away_only
    ap.add_argument("--ml-side", default="away_only", choices=["both", "away_only", "home_only"], help="Moneyline side policy")

    ap.add_argument("--edge", type=float, default=0.04, help="Probability-edge threshold (prob mode).")
    ap.add_argument("--ev", type=float, default=0.01, help="EV threshold in units (ev/ev_cal).")

    ap.add_argument("--calibrator", default=None, help="Optional absolute calibrator joblib (diagnostics only)")
    ap.add_argument("--delta-calibrator", default=None, help="Delta calibrator joblib (required for ev_cal)")

    # Eval window (optional)
    ap.add_argument("--eval-start", default=None, help="Eval start date (YYYY-MM-DD). Optional.")
    ap.add_argument("--eval-end", default=None, help="Eval end date (YYYY-MM-DD). Optional.")

    # Guards
    ap.add_argument("--max-profit-abs", type=float, default=10.0)
    ap.add_argument("--max-bet-rate", type=float, default=0.15, help="Cap bet volume via EV trimming (ML Selector v3).")

    # Diagnostic override
    ap.add_argument(
        "--disable-bet-cap",
        action="store_true",
        help="If set, do NOT trim by --max-bet-rate (diagnostic mode). Default: trimming enabled.",
    )

    ap.add_argument("--no-ppu-gate", action="store_true", help="Disable pre-bet profit-per-unit gate (not recommended)")

    args = ap.parse_args()

    cfg = MLROIConfig(
        per_game_path=args.per_game,
        mode=str(args.mode),
        ml_side=str(args.ml_side),
        prob_edge_threshold=float(args.edge),
        ev_threshold=float(args.ev),
        calibrator_path=args.calibrator if args.calibrator else None,
        delta_calibrator_path=args.delta_calibrator if args.delta_calibrator else None,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        out_dir="outputs",
        max_profit_abs=float(args.max_profit_abs),
        max_bet_rate=float(args.max_bet_rate),
        disable_bet_cap=bool(args.disable_bet_cap),
        enforce_max_profit_per_unit_gate=(not bool(args.no_ppu_gate)),
    )

    print(f"[ml_roi] version={ML_ROI_VERSION}")
    print(f"[ml_roi] __file__={__file__}")
    print(f"[ml_roi] cwd={os.getcwd()}")
    print(f"[ml_roi] per_game={cfg.per_game_path}")
    print(f"[ml_roi] mode={cfg.mode} ml_side={cfg.ml_side} prob_edge_threshold={cfg.prob_edge_threshold:.4f} ev_threshold={cfg.ev_threshold:.4f}")
    print(f"[ml_roi] guards: max_profit_abs={cfg.max_profit_abs} max_bet_rate={cfg.max_bet_rate}")
    print(f"[ml_roi] disable_bet_cap={bool(cfg.disable_bet_cap)}")

    if not os.path.exists(cfg.per_game_path):
        raise FileNotFoundError(f"[ml_roi] per_game not found: {cfg.per_game_path}")

    df = pd.read_csv(cfg.per_game_path)

    # Quick input contract checks (decimal-like odds)
    for c in REQUIRED_ODDS_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            decimal_like = int(((s.abs() < 100) & (s.abs() > 0)).sum())
            print(f"[ml_roi] input_contract_check {c}: decimal_like_count={decimal_like}")

    bets, diag = build_bets(df, cfg)

    overall = summarize(bets)
    home_bets = bets[bets["bet_side"].astype(str).str.lower() == "home"].copy() if not bets.empty else bets
    away_bets = bets[bets["bet_side"].astype(str).str.lower() == "away"].copy() if not bets.empty else bets
    home_sum = summarize(home_bets)
    away_sum = summarize(away_bets)

    bet_rate = diag.get("bet_rate", None)

    print(f"[ml_roi] eval_window: {diag.get('eval_window')}")
    print(f"[ml_roi] overall: {overall}")
    print(f"[ml_roi] home_only: {home_sum}")
    print(f"[ml_roi] away_only: {away_sum}")
    print(f"[ml_roi] bet_rate: {bet_rate:.3f}" if bet_rate is not None else "[ml_roi] bet_rate: n/a")

    os.makedirs(cfg.out_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.out_dir, "ml_roi_metrics.json")
    bets_path = os.path.join(cfg.out_dir, "ml_roi_bets.csv")

    metrics: Dict[str, Any] = {
        "version": ML_ROI_VERSION,
        "mode": cfg.mode,
        "ml_side": cfg.ml_side,
        "prob_edge_threshold": cfg.prob_edge_threshold,
        "ev_threshold": cfg.ev_threshold,
        "overall": overall,
        "home_only": home_sum,
        "away_only": away_sum,
        "bet_rate": bet_rate,
        "calibrator": cfg.calibrator_path,
        "delta_calibrator": cfg.delta_calibrator_path,
        "diagnostics": diag,
        "schema_contract": {
            "required_odds_cols": REQUIRED_ODDS_COLS,
            "model_prob_col_used": pick_model_prob_col(df),
            "american_only": True,
        },
        "guards": {
            "max_profit_abs": cfg.max_profit_abs,
            "max_bet_rate": cfg.max_bet_rate,
            "disable_bet_cap": bool(cfg.disable_bet_cap),
        },
        "eval_window": diag.get("eval_window"),
        "coverage": {
            "coverage": diag.get("coverage"),
            "usable_games": diag.get("usable_games"),
            "total_games": diag.get("total_games"),
            "warn": cfg.coverage_warn,
            "fail": cfg.coverage_fail,
        },
    }

    write_json(metrics_path, metrics)
    bets.to_csv(bets_path, index=False)

    print(f"[ml_roi] wrote: {metrics_path}")
    print(f"[ml_roi] wrote: {bets_path}")


if __name__ == "__main__":
    main()

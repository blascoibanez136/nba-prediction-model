"""
Moneyline ROI analysis with probability, EV, and calibrated EV modes
(American-only, contract-first, bet-gated).

Modes:
  prob   : select by probability edge vs de-vig market
  ev     : select by EV using raw model probability
  ev_cal : select by EV using market-relative calibrated probability (delta-calibrator)

Policy controls:
  --ml-side {both, away_only, home_only}
    - both (default): legacy behavior, best-of home/away
    - away_only: DISABLE home ML bets entirely
    - home_only: DISABLE away ML bets entirely

Hardening added:
- eval window filtering + NaT drop
- probability clipping to (1e-6, 1-1e-6)
- ML odds coverage diagnostics (warn <95%, fail <80%)
- ML-specific output filenames (avoid collisions)

Critical invariants preserved:
- American-only moneyline contract (abs(odds) >= 100, no decimals)
- Fail-loud bet-rate and profit sanity guards
- No change to daily prediction outputs (this is ROI/selection only)
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
from src.model.market_relative_calibration import (
    load_delta_calibrator,
    apply_delta_calibrator,
)
from src.utils.odds_math import (
    clean_american_ml,
    devig_home_prob,
    win_profit_per_unit_american,
    expected_value_units,
)

ML_ROI_VERSION = "ml_roi_v8_prob_ev_evcal_ml_policy_optional_eval_window_outputs_coverage_2025-12-16"
REQUIRED_ODDS_COLS = ["ml_home_consensus", "ml_away_consensus"]

PROB_EPS = 1e-6


# -----------------------------
# Helpers
# -----------------------------
def _get_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["game_date", "date", "gamedate"]:
        if c in df.columns:
            return c
    return None


def _clip_prob(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(PROB_EPS, 1.0 - PROB_EPS)


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


def _col_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def summarize(bets: pd.DataFrame) -> Dict[str, Any]:
    if bets is None or bets.empty:
        return {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": None, "win_rate": None}
    stake = float(pd.to_numeric(bets["stake"], errors="coerce").fillna(0.0).sum())
    profit = float(pd.to_numeric(bets["profit"], errors="coerce").fillna(0.0).sum())
    roi = (profit / stake) if stake > 0 else None
    win_rate = float((bets["result"].astype(str).str.lower() == "win").mean())
    return {"bets": int(len(bets)), "stake": stake, "profit": profit, "roi": roi, "win_rate": win_rate}


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class ROIConfig:
    per_game_path: str
    mode: str = "prob"              # prob | ev | ev_cal
    ml_side: str = "both"           # both | away_only | home_only

    prob_edge_threshold: float = 0.04
    ev_threshold: float = 0.01

    calibrator_path: Optional[str] = None          # diagnostics only
    delta_calibrator_path: Optional[str] = None    # required for ev_cal

    out_dir: str = "outputs"

    min_market_weight: float = 0.35
    max_market_weight: float = 0.75
    max_dispersion: float = 2.25
    require_market_weight: bool = False
    require_dispersion: bool = False

    max_profit_abs: float = 10.0
    max_bet_rate: float = 0.35

    enforce_max_profit_per_unit_gate: bool = True

    # Integrity thresholds
    warn_coverage: float = 0.95
    fail_coverage: float = 0.80


# -----------------------------
# Core: build bets
# -----------------------------
def build_bets(per_game: pd.DataFrame, cfg: ROIConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if per_game is None or per_game.empty:
        raise RuntimeError("[ml_roi] per_game is empty")

    missing = [c for c in REQUIRED_ODDS_COLS if c not in per_game.columns]
    if missing:
        raise RuntimeError(f"[ml_roi] Missing required odds columns: {missing}")

    df = ensure_home_win_actual(per_game)

    # Sanitize odds again (belt + suspenders)
    df["ml_home_consensus"] = pd.to_numeric(df["ml_home_consensus"], errors="coerce").apply(clean_american_ml)
    df["ml_away_consensus"] = pd.to_numeric(df["ml_away_consensus"], errors="coerce").apply(clean_american_ml)

    # Coverage diagnostics (both sides valid)
    valid_ml = df["ml_home_consensus"].notna() & df["ml_away_consensus"].notna()
    total_games = int(df["merge_key"].nunique()) if "merge_key" in df.columns else int(len(df))
    valid_ml_games = int(valid_ml.sum()) if "merge_key" not in df.columns else int(df.loc[valid_ml, "merge_key"].nunique())
    coverage = float(valid_ml_games) / max(total_games, 1)

    diag: Dict[str, Any] = {
        "total_games": total_games,
        "valid_ml_games": valid_ml_games,
        "ml_coverage": coverage,
    }

    if coverage < cfg.fail_coverage:
        raise RuntimeError(f"[ml_roi] ML odds coverage too low: {coverage:.3f} (<{cfg.fail_coverage:.2f})")
    if coverage < cfg.warn_coverage:
        print(f"[ml_roi] WARNING: ML odds coverage {coverage:.3f} (<{cfg.warn_coverage:.2f})")

    # Market prob (devig)
    probs = df.apply(
        lambda r: devig_home_prob(r["ml_home_consensus"], r["ml_away_consensus"]),
        axis=1,
        result_type="expand",
    )
    df["market_prob_home"] = probs[0]
    df["market_prob_method"] = probs[1]
    df["market_prob_home"] = _clip_prob(df["market_prob_home"])

    # Raw model probability
    model_col = pick_model_prob_col(df)
    df[model_col] = pd.to_numeric(df[model_col], errors="coerce")
    df["model_prob_home_raw"] = _clip_prob(df[model_col])
    df["model_prob_away_raw"] = 1.0 - df["model_prob_home_raw"]

    # Optional absolute calibrator for diagnostics only
    if cfg.calibrator_path:
        cal = load_calibrator(cfg.calibrator_path)
        diag_target = "home_win_prob" if "home_win_prob" in df.columns else model_col
        df[diag_target] = pd.to_numeric(df[diag_target], errors="coerce")
        df[f"{diag_target}_calibrated"] = _clip_prob(apply_calibrator(df[diag_target], cal))
        print(f"[ml_roi] Applied calibrator from {cfg.calibrator_path} to column '{diag_target}' (diagnostics only).")

    # Prob edges vs market (diagnostics + prob mode)
    df["home_edge_prob"] = df["model_prob_home_raw"] - df["market_prob_home"]
    df["away_edge_prob"] = df["model_prob_away_raw"] - (1.0 - df["market_prob_home"])

    # EV using raw probabilities
    df["home_ev_raw"] = df.apply(lambda r: expected_value_units(r["model_prob_home_raw"], r["ml_home_consensus"]), axis=1)
    df["away_ev_raw"] = df.apply(lambda r: expected_value_units(r["model_prob_away_raw"], r["ml_away_consensus"]), axis=1)

    # ev_cal: calibrated probabilities via delta calibrator (market-relative)
    if str(cfg.mode).strip().lower() == "ev_cal":
        if not cfg.delta_calibrator_path:
            raise RuntimeError("[ml_roi] mode=ev_cal requires --delta-calibrator")
        cal_obj = load_delta_calibrator(cfg.delta_calibrator_path)

        df["delta_home"] = df["model_prob_home_raw"] - df["market_prob_home"]
        df["model_prob_home_cal"] = df.apply(
            lambda r: apply_delta_calibrator(
                delta=(float(r["delta_home"]) if pd.notna(r["delta_home"]) else None),
                ml_home_odds=(float(r["ml_home_consensus"]) if pd.notna(r["ml_home_consensus"]) else None),
                calibrator=cal_obj,
            ),
            axis=1,
        )
        df["model_prob_home_cal"] = _clip_prob(df["model_prob_home_cal"])
        df["model_prob_away_cal"] = 1.0 - df["model_prob_home_cal"]

        df["home_ev_cal"] = df.apply(lambda r: expected_value_units(r["model_prob_home_cal"], r["ml_home_consensus"]), axis=1)
        df["away_ev_cal"] = df.apply(lambda r: expected_value_units(r["model_prob_away_cal"], r["ml_away_consensus"]), axis=1)

    # -------------------------
    # Bet gating
    # -------------------------
    base_gate = (
        df["model_prob_home_raw"].between(0, 1, inclusive="neither")
        & df["market_prob_home"].notna()
        & df["ml_home_consensus"].notna()
        & df["ml_away_consensus"].notna()
    )

    # market weight gate
    if _col_exists(df, "market_weight"):
        mw = pd.to_numeric(df["market_weight"], errors="coerce")
        mw_gate = mw.between(cfg.min_market_weight, cfg.max_market_weight, inclusive="both")
        if cfg.require_market_weight:
            base_gate = base_gate & mw_gate
        df["gate_market_weight_ok"] = mw_gate
    else:
        df["gate_market_weight_ok"] = pd.NA

    # dispersion gate (optional; uses existing columns if present)
    dispersion_col = "home_spread_dispersion" if _col_exists(df, "home_spread_dispersion") else (
        "book_dispersion" if _col_exists(df, "book_dispersion") else None
    )
    if dispersion_col:
        disp = pd.to_numeric(df[dispersion_col], errors="coerce")
        disp_gate = disp.le(cfg.max_dispersion) | disp.isna()
        if cfg.require_dispersion:
            base_gate = base_gate & disp_gate & disp.notna()
        df["gate_dispersion_ok"] = disp_gate
        df["dispersion_used_col"] = dispersion_col
    else:
        df["gate_dispersion_ok"] = pd.NA
        df["dispersion_used_col"] = pd.NA

    # Pre-bet payout sanity gate
    if cfg.enforce_max_profit_per_unit_gate:
        home_ppu = df["ml_home_consensus"].apply(win_profit_per_unit_american)
        away_ppu = df["ml_away_consensus"].apply(win_profit_per_unit_american)
        ppu_gate = (
            (home_ppu.isna() | (home_ppu <= cfg.max_profit_abs))
            & (away_ppu.isna() | (away_ppu <= cfg.max_profit_abs))
        )
        df["gate_ppu_ok"] = ppu_gate
        base_gate = base_gate & ppu_gate
    else:
        df["gate_ppu_ok"] = pd.NA

    df["eligible"] = base_gate
    diag["eligible_rows"] = int(df["eligible"].sum())

    mode = str(cfg.mode).strip().lower()
    if mode not in ("prob", "ev", "ev_cal"):
        raise RuntimeError(f"[ml_roi] Invalid mode='{cfg.mode}'. Expected 'prob', 'ev', or 'ev_cal'.")

    ml_side = str(cfg.ml_side).strip().lower()
    if ml_side not in ("both", "away_only", "home_only"):
        raise RuntimeError(f"[ml_roi] Invalid ml_side='{cfg.ml_side}'. Expected 'both', 'away_only', or 'home_only'.")

    # Select score columns based on mode
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
    if bets.empty:
        return bets.reset_index(drop=True), diag

    bets["stake"] = 1.0
    bets["odds_price"] = bets.apply(
        lambda r: r["ml_home_consensus"] if str(r["bet_side"]).lower() == "home" else r["ml_away_consensus"],
        axis=1,
    )
    bets["odds_price"] = pd.to_numeric(bets["odds_price"], errors="coerce")

    # Contract assertion on bet odds
    invalid_bet_odds = bets["odds_price"].apply(
        lambda x: (x is not None)
        and (not math.isnan(float(x)))
        and (abs(float(x)) < 100)
        and (abs(float(x)) > 0)
    )
    if bool(invalid_bet_odds.any()):
        sample = bets.loc[invalid_bet_odds, ["game_date", "home_team", "away_team", "bet_side", "odds_price",
                                            "ml_home_consensus", "ml_away_consensus"]].head(10)
        raise RuntimeError(
            "[ml_roi] American-only ML contract violation: found bet rows with 0 < abs(odds) < 100. Sample:\n"
            + sample.to_string(index=False)
        )

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

    # Fail-loud: realized profit sanity
    max_abs = float(pd.to_numeric(bets["profit"], errors="coerce").abs().max())
    if max_abs > cfg.max_profit_abs:
        sample = bets.sort_values("profit", ascending=False).head(10)[
            ["game_date", "home_team", "away_team", "bet_side", "odds_price", "result", "stake", "profit", "metric_used", "metric_type"]
        ]
        raise RuntimeError(
            f"[ml_roi] Profit sanity failure: max |profit|={max_abs}u (limit={cfg.max_profit_abs}). Sample:\n"
            + sample.to_string(index=False)
        )

    # Fail-loud: bet-rate regression detector
    bet_rate = float(len(bets) / max(total_games, 1))
    diag["bets"] = int(len(bets))
    diag["bet_rate"] = bet_rate
    if bet_rate > cfg.max_bet_rate:
        raise RuntimeError(
            f"[ml_roi] Bet-rate too high: bets={len(bets)} total_games={total_games} bet_rate={bet_rate:.3f} "
            f"(cap={cfg.max_bet_rate})."
        )

    return bets.reset_index(drop=True), diag


def main() -> None:
    ap = argparse.ArgumentParser("ml_roi_analysis.py (American-only, bet-gated, optional ML side policy)")

    ap.add_argument("--per_game", required=True, help="Path to outputs/backtest_per_game.csv")

    ap.add_argument("--mode", default="prob", choices=["prob", "ev", "ev_cal"], help="Selection mode: prob | ev | ev_cal")
    ap.add_argument("--ml-side", default="both", choices=["both", "away_only", "home_only"], help="Moneyline side policy")

    ap.add_argument("--edge", type=float, default=0.04, help="Probability-edge threshold (prob mode).")
    ap.add_argument("--ev", type=float, default=0.01, help="EV threshold in units (ev/ev_cal).")

    ap.add_argument("--calibrator", default=None, help="Optional absolute calibrator joblib (diagnostics only)")
    ap.add_argument("--delta-calibrator", default=None, help="Delta calibrator joblib (required for ev_cal)")

    ap.add_argument("--min-market-weight", type=float, default=0.35)
    ap.add_argument("--max-market-weight", type=float, default=0.75)
    ap.add_argument("--max-dispersion", type=float, default=2.25)
    ap.add_argument("--require-market-weight", action="store_true")
    ap.add_argument("--require-dispersion", action="store_true")

    ap.add_argument("--max-profit-abs", type=float, default=10.0)
    ap.add_argument("--max-bet-rate", type=float, default=0.35)

    ap.add_argument("--no-ppu-gate", action="store_true", help="Disable pre-bet profit-per-unit gate (not recommended)")

    # New: eval window filtering
    ap.add_argument("--eval-start", default=None, help="Filter eval window start (YYYY-MM-DD). Optional.")
    ap.add_argument("--eval-end", default=None, help="Filter eval window end (YYYY-MM-DD). Optional.")

    args = ap.parse_args()

    cfg = ROIConfig(
        per_game_path=args.per_game,
        mode=str(args.mode),
        ml_side=str(args.ml_side),
        prob_edge_threshold=float(args.edge),
        ev_threshold=float(args.ev),
        calibrator_path=args.calibrator if args.calibrator else None,
        delta_calibrator_path=args.delta_calibrator if args.delta_calibrator else None,
        out_dir="outputs",
        min_market_weight=float(args.min_market_weight),
        max_market_weight=float(args.max_market_weight),
        max_dispersion=float(args.max_dispersion),
        require_market_weight=bool(args.require_market_weight),
        require_dispersion=bool(args.require_dispersion),
        max_profit_abs=float(args.max_profit_abs),
        max_bet_rate=float(args.max_bet_rate),
        enforce_max_profit_per_unit_gate=(not bool(args.no_ppu_gate)),
    )

    print(f"[ml_roi] version={ML_ROI_VERSION}")
    print(f"[ml_roi] __file__={__file__}")
    print(f"[ml_roi] cwd={os.getcwd()}")
    print(f"[ml_roi] per_game={cfg.per_game_path}")
    print(f"[ml_roi] mode={cfg.mode} ml_side={cfg.ml_side} prob_edge_threshold={cfg.prob_edge_threshold:.4f} ev_threshold={cfg.ev_threshold:.4f}")
    print(f"[ml_roi] guards: max_profit_abs={cfg.max_profit_abs} max_bet_rate={cfg.max_bet_rate}")

    if not os.path.exists(cfg.per_game_path):
        raise FileNotFoundError(f"[ml_roi] per_game not found: {cfg.per_game_path}")

    df = pd.read_csv(cfg.per_game_path)
    if df.empty:
        raise RuntimeError("[ml_roi] per_game is empty")

    # Optional eval window filtering
    date_col = _get_date_col(df)
    eval_meta: Dict[str, Any] = {"enabled": False, "date_col": date_col}
    if (args.eval_start or args.eval_end) and not date_col:
        raise RuntimeError("[ml_roi] eval-start/eval-end provided but no date column found (game_date/date/gamedate)")

    if date_col and (args.eval_start or args.eval_end):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).copy()
        es = pd.to_datetime(args.eval_start, errors="coerce") if args.eval_start else None
        ee = pd.to_datetime(args.eval_end, errors="coerce") if args.eval_end else None
        if args.eval_start and pd.isna(es):
            raise RuntimeError("[ml_roi] invalid eval-start")
        if args.eval_end and pd.isna(ee):
            raise RuntimeError("[ml_roi] invalid eval-end")
        if es is not None:
            df = df[df[date_col] >= es].copy()
        if ee is not None:
            df = df[df[date_col] <= ee].copy()
        eval_meta = {
            "enabled": True,
            "date_col": date_col,
            "start": str(es.date()) if es is not None else None,
            "end": str(ee.date()) if ee is not None else None,
            "rows": int(len(df)),
        }
        print(f"[ml_roi] eval_window: {eval_meta}")

    # Contract check (diagnostics)
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

    total_games = diag.get("total_games", int(df["merge_key"].nunique()) if "merge_key" in df.columns else int(len(df)))
    bet_rate = diag.get("bet_rate", float(len(bets) / max(total_games, 1)))

    print(f"[ml_roi] overall: {overall}")
    print(f"[ml_roi] home_only: {home_sum}")
    print(f"[ml_roi] away_only: {away_sum}")
    print(f"[ml_roi] bet_rate: {bet_rate:.3f}")

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
        "schema_contract": {
            "required_odds_cols": REQUIRED_ODDS_COLS,
            "model_prob_col_used": pick_model_prob_col(df),
            "american_only": True,
        },
        "gating": {
            "min_market_weight": cfg.min_market_weight,
            "max_market_weight": cfg.max_market_weight,
            "max_dispersion": cfg.max_dispersion,
            "require_market_weight": cfg.require_market_weight,
            "require_dispersion": cfg.require_dispersion,
            "ppu_gate_enabled": cfg.enforce_max_profit_per_unit_gate,
        },
        "guards": {
            "max_profit_abs": cfg.max_profit_abs,
            "max_bet_rate": cfg.max_bet_rate,
            "coverage_warn_lt": cfg.warn_coverage,
            "coverage_fail_lt": cfg.fail_coverage,
        },
        "eval_window": eval_meta,
        "diagnostics": diag,
    }

    _write_json(metrics_path, metrics)
    bets.to_csv(bets_path, index=False)

    print(f"[ml_roi] wrote: {metrics_path}")
    print(f"[ml_roi] wrote: {bets_path}")


if __name__ == "__main__":
    main()

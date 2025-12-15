"""
Moneyline ROI analysis (American-only, contract-first, bet-gated).

This module assumes upstream has enforced an American-only contract for:
  - ml_home_consensus (American odds)
  - ml_away_consensus (American odds)

Key principles (Elite discipline):
- Selection uses RAW model probability (pre-calibration).
- Optional calibrator is applied for diagnostics only.
- Market probability is computed by de-vigging implied probs from BOTH sides.
- Bet selection is gated to prevent overtrading and reduce noise:
    * require both ML odds valid (abs >= 100)
    * require market weight within [min_market_weight, max_market_weight] if available
    * require spread dispersion <= max_dispersion if available
    * edge threshold required
    * optional max_bet_rate cap (fail loud)

Outputs:
- outputs/roi_metrics.json
- outputs/roi_buckets.csv
- outputs/roi_bets.csv

Run:
  PYTHONPATH=. python -m src.eval.roi_analysis \
    --per_game outputs/backtest_per_game.csv \
    --edge 0.04 \
    --calibrator models/calibrator_winprob_isotonic.joblib
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

ROI_ANALYSIS_VERSION = "roi_analysis_v4_american_only_bet_gated_2025-12-15"

REQUIRED_ODDS_COLS = ["ml_home_consensus", "ml_away_consensus"]


# -----------------------------
# Helpers
# -----------------------------
def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v) or v == 0.0:
        return None
    return v


def clean_american_ml(x) -> Optional[float]:
    """
    Strict American-only sanitizer:
    - numeric, finite, non-zero
    - abs(x) >= 100
    """
    v = _to_float(x)
    if v is None:
        return None
    if abs(v) < 100:
        return None
    return v


def american_to_prob(o: Optional[float]) -> Optional[float]:
    """
    Implied probability from American odds.
    Returns None for invalid odds.
    """
    o = clean_american_ml(o)
    if o is None:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def devig_home_prob(ml_home: Optional[float], ml_away: Optional[float]) -> Tuple[Optional[float], str]:
    """
    De-vig using both sides:
      p_home = ph / (ph + pa)
    Returns (p_home, method).
    """
    ph = american_to_prob(ml_home)
    pa = american_to_prob(ml_away)
    if ph is None or pa is None:
        return None, "missing_or_invalid"
    s = ph + pa
    if s <= 0:
        return None, "missing_or_invalid"
    return ph / s, "devig_two_sided"


def win_profit_per_unit_american(o: Optional[float]) -> Optional[float]:
    """
    For 1u stake, profit if win:
      +odds: odds/100
      -odds: 100/abs(odds)
    """
    o = clean_american_ml(o)
    if o is None:
        return None
    if o > 0:
        return float(o) / 100.0
    return 100.0 / abs(float(o))


def pick_model_prob_col(df: pd.DataFrame) -> str:
    """
    Prefer raw model probability for selection.
    Falls back through known columns.
    """
    candidates = [
        "home_win_prob_model_raw",
        "home_win_prob_model",
        "home_win_prob",          # blended or model depending on upstream
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"[roi] No model prob column found. Tried: {candidates}")


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
    raise RuntimeError("[roi] Missing home_win_actual and cannot infer from scores.")


def _col_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class ROIConfig:
    per_game_path: str
    edge_threshold: float
    calibrator_path: Optional[str]
    out_dir: str = "outputs"

    # Bet gating (Elite discipline)
    min_market_weight: float = 0.35
    max_market_weight: float = 0.75
    max_dispersion: float = 2.25   # spreads std dev threshold (if available)
    require_market_weight: bool = False
    require_dispersion: bool = False

    # Fail-loud guardrails
    max_profit_abs: float = 10.0   # per 1u stake, ML profit > 10u is suspicious
    max_bet_rate: float = 0.35     # fail loud if betting > 35% of games (regression detector)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


# -----------------------------
# Core: build bets
# -----------------------------
def build_bets(per_game: pd.DataFrame, cfg: ROIConfig) -> pd.DataFrame:
    if per_game is None or per_game.empty:
        raise RuntimeError("[roi] per_game is empty")

    missing = [c for c in REQUIRED_ODDS_COLS if c not in per_game.columns]
    if missing:
        raise RuntimeError(f"[roi] Missing required odds columns: {missing}")

    df = ensure_home_win_actual(per_game)

    # Sanitize odds upstream contract again (belt + suspenders)
    df["ml_home_consensus"] = pd.to_numeric(df["ml_home_consensus"], errors="coerce")
    df["ml_away_consensus"] = pd.to_numeric(df["ml_away_consensus"], errors="coerce")
    df["ml_home_consensus"] = df["ml_home_consensus"].apply(clean_american_ml)
    df["ml_away_consensus"] = df["ml_away_consensus"].apply(clean_american_ml)

    # Compute market prob (devig) and method
    probs = df.apply(
        lambda r: devig_home_prob(r["ml_home_consensus"], r["ml_away_consensus"]),
        axis=1,
        result_type="expand",
    )
    df["market_prob_home"] = probs[0]
    df["market_prob_method"] = probs[1]

    # Select model probability (RAW) for selection
    model_col = pick_model_prob_col(df)
    df[model_col] = pd.to_numeric(df[model_col], errors="coerce")
    df["model_prob_home_raw"] = df[model_col]
    df["model_prob_away_raw"] = 1.0 - df["model_prob_home_raw"]

    # Optional calibrator for diagnostics only (NOT used for selection)
    if cfg.calibrator_path:
        cal = load_calibrator(cfg.calibrator_path)
        # Apply to whichever column exists for diagnostics (prefer home_win_prob if present)
        diag_target = "home_win_prob" if "home_win_prob" in df.columns else model_col
        df[diag_target] = pd.to_numeric(df[diag_target], errors="coerce")
        df[f"{diag_target}_calibrated"] = apply_calibrator(df[diag_target], cal)
        print(f"[roi_analysis] Applied calibrator from {cfg.calibrator_path} to column '{diag_target}'.")

    # Edges vs de-vig market
    df["home_edge"] = df["model_prob_home_raw"] - df["market_prob_home"]
    df["away_edge"] = df["model_prob_away_raw"] - (1.0 - df["market_prob_home"])

    # -------------------------
    # Bet gating
    # -------------------------
    # Gate: require market prob and both odds
    base_gate = (
        df["model_prob_home_raw"].between(0, 1, inclusive="neither")
        & df["market_prob_home"].notna()
        & df["ml_home_consensus"].notna()
        & df["ml_away_consensus"].notna()
    )

    # Optional gate: market weight bounds
    if _col_exists(df, "market_weight"):
        mw = pd.to_numeric(df["market_weight"], errors="coerce")
        mw_gate = mw.between(cfg.min_market_weight, cfg.max_market_weight, inclusive="both")
        if cfg.require_market_weight:
            base_gate = base_gate & mw_gate
        df["gate_market_weight_ok"] = mw_gate
    else:
        df["gate_market_weight_ok"] = pd.NA

    # Optional gate: dispersion threshold
    dispersion_col = "home_spread_dispersion" if _col_exists(df, "home_spread_dispersion") else ("book_dispersion" if _col_exists(df, "book_dispersion") else None)
    if dispersion_col:
        disp = pd.to_numeric(df[dispersion_col], errors="coerce")
        disp_gate = disp.le(cfg.max_dispersion) | disp.isna()  # allow missing dispersion by default
        if cfg.require_dispersion:
            base_gate = base_gate & disp_gate & disp.notna()
        df["gate_dispersion_ok"] = disp_gate
        df["dispersion_used_col"] = dispersion_col
    else:
        df["gate_dispersion_ok"] = pd.NA
        df["dispersion_used_col"] = pd.NA

    df["eligible"] = base_gate

    # Choose side: take the larger edge above threshold
    def choose_side(r) -> Tuple[bool, Optional[str], Optional[float]]:
        if not bool(r["eligible"]):
            return False, None, None
        he = r["home_edge"]
        ae = r["away_edge"]
        if pd.isna(he) or pd.isna(ae):
            return False, None, None
        if he >= cfg.edge_threshold and he >= ae:
            return True, "home", float(he)
        if ae >= cfg.edge_threshold and ae > he:
            return True, "away", float(ae)
        return False, None, None

    chosen = df.apply(choose_side, axis=1, result_type="expand")
    df["bet"] = chosen[0].astype(bool)
    df["bet_side"] = chosen[1]
    df["edge_used"] = chosen[2]

    bets = df[df["bet"]].copy()
    if bets.empty:
        return bets.reset_index(drop=True)

    bets["stake"] = 1.0

    # Odds price for the bet
    bets["odds_price"] = bets.apply(
        lambda r: r["ml_home_consensus"] if str(r["bet_side"]).lower() == "home" else r["ml_away_consensus"],
        axis=1,
    )
    bets["odds_price"] = pd.to_numeric(bets["odds_price"], errors="coerce")

    # Contract assertion: any bet must have valid American odds
    invalid_bet_odds = bets["odds_price"].apply(lambda x: (x is not None) and (not math.isnan(float(x))) and (abs(float(x)) < 100) and (abs(float(x)) > 0))
    if bool(invalid_bet_odds.any()):
        sample = bets.loc[invalid_bet_odds, ["game_date", "home_team", "away_team", "bet_side", "odds_price", "ml_home_consensus", "ml_away_consensus"]].head(10)
        raise RuntimeError(
            "[roi] American-only ML contract violation: found bet rows with 0 < abs(odds) < 100. Sample:\n"
            + sample.to_string(index=False)
        )

    # Result label (string) + boolean win flag
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
    bets["result_win"] = bets["result"].astype(str).str.lower().eq("win")

    # Profit in units for 1u stake
    def profit_units(r) -> float:
        stake = float(r["stake"])
        res = str(r["result"]).lower()
        if res == "push":
            return 0.0
        if res != "win":
            return -stake
        ppu = win_profit_per_unit_american(_to_float(r["odds_price"]))
        if ppu is None:
            # should not happen if contract holds, but keep deterministic
            return 0.0
        return stake * float(ppu)

    bets["profit"] = bets.apply(profit_units, axis=1)
    bets["market"] = "moneyline"

    # Fail-loud: profit sanity
    max_abs = float(pd.to_numeric(bets["profit"], errors="coerce").abs().max())
    if max_abs > cfg.max_profit_abs:
        sample = bets.sort_values("profit", ascending=False).head(10)[
            ["game_date", "home_team", "away_team", "bet_side", "odds_price", "result", "stake", "profit", "edge_used"]
        ]
        raise RuntimeError(
            f"[roi] Profit sanity failure: max |profit|={max_abs}u (limit={cfg.max_profit_abs}). "
            "This indicates corrupted odds or wrong settlement. Sample high-profit rows:\n"
            + sample.to_string(index=False)
        )

    # Fail-loud: overtrading regression detector
    total_games = int(df["merge_key"].nunique()) if "merge_key" in df.columns else int(len(df))
    bet_rate = float(len(bets) / max(total_games, 1))
    if bet_rate > cfg.max_bet_rate:
        raise RuntimeError(
            f"[roi] Bet-rate too high: bets={len(bets)} total_games={total_games} bet_rate={bet_rate:.3f} "
            f"(cap={cfg.max_bet_rate}). This indicates edge logic or gating regression."
        )

    # Summary-friendly column order
    front = [
        "game_date", "home_team", "away_team", "merge_key",
        "bet_side", "odds_price", "stake", "result", "result_win", "profit",
        "edge_used", "home_edge", "away_edge",
        "model_prob_home_raw", "market_prob_home", "market_prob_method",
        "market_weight", "home_spread_dispersion", "book_dispersion",
        "gate_market_weight_ok", "gate_dispersion_ok", "eligible",
        "market",
    ]
    front = [c for c in front if c in bets.columns]
    bets = bets[front + [c for c in bets.columns if c not in front]]

    return bets.reset_index(drop=True)


# -----------------------------
# Summaries
# -----------------------------
def summarize(bets: pd.DataFrame) -> Dict[str, Any]:
    if bets is None or bets.empty:
        return {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": None, "win_rate": None}
    stake = float(pd.to_numeric(bets["stake"], errors="coerce").fillna(0).sum())
    profit = float(pd.to_numeric(bets["profit"], errors="coerce").fillna(0).sum())
    roi = (profit / stake) if stake > 0 else None
    win_rate = float(pd.to_numeric(bets.get("result_win", False), errors="coerce").fillna(0).mean()) if "result_win" in bets.columns else float((bets["result"].astype(str).str.lower() == "win").mean())
    return {"bets": int(len(bets)), "stake": stake, "profit": profit, "roi": roi, "win_rate": win_rate}


def bucketize(bets: pd.DataFrame) -> pd.DataFrame:
    if bets is None or bets.empty:
        return pd.DataFrame(columns=["bucket", "bets", "stake", "profit", "roi", "win_rate", "avg_edge", "avg_odds"])

    b = bets.copy()
    b["edge_used"] = pd.to_numeric(b["edge_used"], errors="coerce")

    # Default buckets (tune later once spread/totals ROI is added)
    bins = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 10.0]
    labels = ["<0.02", "0.02-0.03", "0.03-0.04", "0.04-0.05", "0.05-0.06", "0.06-0.08", "0.08-0.10", ">=0.10"]
    b["bucket"] = pd.cut(b["edge_used"].fillna(-1.0), bins=bins, labels=labels, right=False, include_lowest=True)

    # observed=False to avoid pandas FutureWarning behavior changes
    g = b.groupby("bucket", dropna=False, observed=False)

    out = g.agg(
        bets=("profit", "size"),
        stake=("stake", "sum"),
        profit=("profit", "sum"),
        win_rate=("result_win", "mean") if "result_win" in b.columns else ("result", lambda s: (s.astype(str).str.lower() == "win").mean()),
        avg_edge=("edge_used", "mean"),
        avg_odds=("odds_price", lambda s: pd.to_numeric(s, errors="coerce").mean()),
    ).reset_index()

    out["roi"] = out.apply(lambda r: (r["profit"] / r["stake"]) if r["stake"] else None, axis=1)
    return out


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser("roi_analysis.py (American-only, bet-gated)")
    ap.add_argument("--per_game", required=True, help="Path to outputs/backtest_per_game.csv")
    ap.add_argument("--edge", required=True, type=float, help="Edge threshold (e.g., 0.04)")

    ap.add_argument("--calibrator", default=None, help="Optional calibrator joblib (diagnostics only)")

    # Gating knobs
    ap.add_argument("--min-market-weight", type=float, default=0.35)
    ap.add_argument("--max-market-weight", type=float, default=0.75)
    ap.add_argument("--max-dispersion", type=float, default=2.25)
    ap.add_argument("--require-market-weight", action="store_true")
    ap.add_argument("--require-dispersion", action="store_true")

    # Guardrails
    ap.add_argument("--max-profit-abs", type=float, default=10.0)
    ap.add_argument("--max-bet-rate", type=float, default=0.35)

    args = ap.parse_args()

    cfg = ROIConfig(
        per_game_path=args.per_game,
        edge_threshold=float(args.edge),
        calibrator_path=args.calibrator if args.calibrator else None,
        out_dir="outputs",
        min_market_weight=float(args.min_market_weight),
        max_market_weight=float(args.max_market_weight),
        max_dispersion=float(args.max_dispersion),
        require_market_weight=bool(args.require_market_weight),
        require_dispersion=bool(args.require_dispersion),
        max_profit_abs=float(args.max_profit_abs),
        max_bet_rate=float(args.max_bet_rate),
    )

    print(f"[roi] version={ROI_ANALYSIS_VERSION}")
    print(f"[roi] __file__={__file__}")
    print(f"[roi] cwd={os.getcwd()}")
    print(f"[roi] per_game={cfg.per_game_path}")
    print(f"[roi] edge_threshold={cfg.edge_threshold:.4f}")
    print(f"[roi] gating: market_weight in [{cfg.min_market_weight:.2f},{cfg.max_market_weight:.2f}] require={cfg.require_market_weight} "
          f"dispersion<= {cfg.max_dispersion:.2f} require={cfg.require_dispersion}")
    print(f"[roi] guards: max_profit_abs={cfg.max_profit_abs} max_bet_rate={cfg.max_bet_rate}")

    if not os.path.exists(cfg.per_game_path):
        raise FileNotFoundError(f"[roi] per_game not found: {cfg.per_game_path}")

    df = pd.read_csv(cfg.per_game_path)

    # Quick contract visibility
    for c in REQUIRED_ODDS_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            decimal_like = int(((s.abs() < 100) & (s.abs() > 0)).sum())
            print(f"[roi] input_contract_check {c}: decimal_like_count={decimal_like}")

    bets = build_bets(df, cfg)

    os.makedirs(cfg.out_dir, exist_ok=True)

    overall = summarize(bets)
    home_bets = bets[bets["bet_side"].astype(str).str.lower() == "home"].copy() if not bets.empty else bets
    away_bets = bets[bets["bet_side"].astype(str).str.lower() == "away"].copy() if not bets.empty else bets
    home_sum = summarize(home_bets)
    away_sum = summarize(away_bets)

    # Metrics
    total_games = int(df["merge_key"].nunique()) if "merge_key" in df.columns else int(len(df))
    bet_rate = float(len(bets) / max(total_games, 1)) if total_games else None

    print(f"[roi] overall: {overall}")
    print(f"[roi] home_only: {home_sum}")
    print(f"[roi] away_only: {away_sum}")
    print(f"[roi] bet_rate: {bet_rate:.3f}" if bet_rate is not None else "[roi] bet_rate: n/a")

    buckets = bucketize(bets)

    metrics: Dict[str, Any] = {
        "version": ROI_ANALYSIS_VERSION,
        "edge_threshold": cfg.edge_threshold,
        "overall": overall,
        "home_only": home_sum,
        "away_only": away_sum,
        "bet_rate": bet_rate,
        "calibrator": cfg.calibrator_path,
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
        },
        "guards": {
            "max_profit_abs": cfg.max_profit_abs,
            "max_bet_rate": cfg.max_bet_rate,
        },
    }

    metrics_path = os.path.join(cfg.out_dir, "roi_metrics.json")
    buckets_path = os.path.join(cfg.out_dir, "roi_buckets.csv")
    bets_path = os.path.join(cfg.out_dir, "roi_bets.csv")

    write_json(metrics_path, metrics)
    buckets.to_csv(buckets_path, index=False)
    bets.to_csv(bets_path, index=False)

    print(f"[roi] wrote: {metrics_path}")
    print(f"[roi] wrote: {buckets_path}")
    print(f"[roi] wrote: {bets_path}")


if __name__ == "__main__":
    main()

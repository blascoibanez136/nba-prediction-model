"""
Moneyline ROI analysis (fail-loud, market-anchored, bet-side correct).

Hard contract:
  - ml_home_consensus (American odds)
  - ml_away_consensus (American odds)

This version adds strong guardrails so the script will FAIL loudly if:
  - it somehow bets every game with a positive edge threshold
  - it produces nonsensical profit magnitudes (e.g., -40001 units)

Run:
  PYTHONPATH=. python src/eval/roi_analysis.py \
    --per_game outputs/backtest_per_game.csv \
    --edge 0.02 \
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


ROI_ANALYSIS_VERSION = "roi_analysis_v2_fail_loud_2025-12-15"


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


def american_to_implied_prob(odds: float) -> Optional[float]:
    o = _to_float(odds)
    if o is None:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def devig_probs(home_odds: float, away_odds: float) -> Tuple[Optional[float], Optional[float]]:
    ph = american_to_implied_prob(home_odds)
    pa = american_to_implied_prob(away_odds)
    if ph is None or pa is None:
        return None, None
    s = ph + pa
    if s <= 0:
        return None, None
    return ph / s, pa / s


def american_win_profit_per_unit(odds: float) -> Optional[float]:
    o = _to_float(odds)
    if o is None:
        return None
    if o > 0:
        return o / 100.0
    return 100.0 / abs(o)


@dataclass(frozen=True)
class ROIConfig:
    per_game_path: str
    edge_threshold: float
    calibrator_path: Optional[str]
    out_dir: str = "outputs"


REQUIRED_ODDS_COLS = ["ml_home_consensus", "ml_away_consensus"]


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def pick_model_prob_col(df: pd.DataFrame) -> str:
    candidates = [
        "home_win_prob_model_raw",
        "home_win_prob_model",
        "home_win_prob",
        "home_win_prob_market",
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


def build_bets(per_game: pd.DataFrame, edge_threshold: float, calibrator_path: Optional[str]) -> pd.DataFrame:
    if per_game.empty:
        raise RuntimeError("[roi] per_game is empty")

    missing = [c for c in REQUIRED_ODDS_COLS if c not in per_game.columns]
    if missing:
        raise RuntimeError(f"[roi] Missing required odds columns: {missing}. Found cols: {list(per_game.columns)}")

    df = ensure_home_win_actual(per_game)

    # Parse odds
    df["ml_home_consensus"] = pd.to_numeric(df["ml_home_consensus"], errors="coerce")
    df["ml_away_consensus"] = pd.to_numeric(df["ml_away_consensus"], errors="coerce")

    # Market probabilities (devig)
    mh, ma, mm = [], [], []
    for ho, ao in zip(df["ml_home_consensus"].tolist(), df["ml_away_consensus"].tolist()):
        ph, pa = devig_probs(ho, ao)
        if ph is None or pa is None:
            mh.append(None); ma.append(None); mm.append("missing_odds")
        else:
            mh.append(ph); ma.append(pa); mm.append("devig_implied")
    df["market_prob_home"] = mh
    df["market_prob_away"] = ma
    df["market_prob_method"] = mm

    # Model probabilities (RAW for selection)
    model_col = pick_model_prob_col(df)
    df[model_col] = pd.to_numeric(df[model_col], errors="coerce")
    df["model_prob_home_raw"] = df[model_col]
    df["model_prob_away_raw"] = 1.0 - df["model_prob_home_raw"]

    # Optional calibrator for diagnostics only
    if calibrator_path:
        cal = load_calibrator(calibrator_path)
        target = "home_win_prob" if "home_win_prob" in df.columns else model_col
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df[f"{target}_calibrated"] = apply_calibrator(cal, df[target])
        print(f"[roi_analysis] Applied calibrator from {calibrator_path} to column '{target}'.")

    # Edges
    df["home_edge"] = df["model_prob_home_raw"] - df["market_prob_home"]
    df["away_edge"] = df["model_prob_away_raw"] - df["market_prob_away"]

    def choose_side(r) -> Tuple[bool, Optional[str], Optional[float]]:
        he = r["home_edge"]
        ae = r["away_edge"]
        if pd.isna(he) or pd.isna(ae):
            return False, None, None
        if he >= edge_threshold and he >= ae:
            return True, "home", float(he)
        if ae >= edge_threshold and ae > he:
            return True, "away", float(ae)
        return False, None, None

    chosen = df.apply(choose_side, axis=1, result_type="expand")
    df["bet"] = chosen[0].astype(bool)
    df["bet_side"] = chosen[1]
    df["edge_used"] = chosen[2]

    bets = df[df["bet"]].copy()

    # Guardrail: If edge_threshold > 0 and we bet EVERYTHING, something is wrong.
    if edge_threshold > 0 and len(bets) == len(df) and len(df) > 50:
        # Print quick diagnostics
        he_desc = pd.to_numeric(df["home_edge"], errors="coerce").describe().to_dict()
        ae_desc = pd.to_numeric(df["away_edge"], errors="coerce").describe().to_dict()
        raise RuntimeError(
            "[roi] Sanity failure: bet count equals total games despite positive edge threshold. "
            f"edge_threshold={edge_threshold}. home_edge_desc={he_desc}. away_edge_desc={ae_desc}."
        )

    if bets.empty:
        return bets

    # Stake default 1u
    if "stake" in bets.columns:
        bets["stake"] = pd.to_numeric(bets["stake"], errors="coerce").fillna(1.0)
    else:
        bets["stake"] = 1.0

    # Odds for bet side
    bets["odds_price"] = bets.apply(
        lambda r: r["ml_home_consensus"] if str(r["bet_side"]).lower() == "home" else r["ml_away_consensus"],
        axis=1,
    )
    bets["odds_price"] = pd.to_numeric(bets["odds_price"], errors="coerce")

    # Result as STRING labels
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

    # Profit in units using correct American odds
    def profit_units(r) -> float:
        stake = float(r["stake"])
        res = str(r["result"]).lower()
        if res == "push":
            return 0.0
        if res != "win":
            return -stake
        ppu = american_win_profit_per_unit(r["odds_price"])
        if ppu is None:
            return 0.0
        return stake * float(ppu)

    bets["profit"] = bets.apply(profit_units, axis=1)

    # Guardrail: profit magnitude sanity on 1u bets
    # For NBA moneylines, >25u profit magnitude per bet is effectively impossible unless odds are corrupt.
    max_abs = float(pd.to_numeric(bets["profit"], errors="coerce").abs().max())
    if max_abs > 25.0:
        sample = bets.sort_values("profit").head(3)[["game_date","home_team","away_team","bet_side","odds_price","result","stake","profit"]]
        raise RuntimeError(
            f"[roi] Profit sanity failure: max |profit|={max_abs}u. "
            "This indicates corrupted odds or wrong settlement. Sample worst rows:\n"
            + sample.to_string(index=False)
        )

    bets["market"] = "moneyline"

    # Ensure merge_key present if possible
    if "merge_key" not in bets.columns and {"home_team","away_team","game_date"} <= set(bets.columns):
        bets["merge_key"] = (
            bets["home_team"].astype(str).str.lower().str.strip()
            + "__"
            + bets["away_team"].astype(str).str.lower().str.strip()
            + "__"
            + bets["game_date"].astype(str).str[:10]
        )

    # Column order
    front = [
        "game_date","home_team","away_team","merge_key",
        "bet_side","odds_price","stake","result","profit",
        "edge_used","home_edge","away_edge",
        "market_prob_home","market_prob_away","market_prob_method",
        "model_prob_home_raw","model_prob_away_raw",
        "market",
    ]
    front = [c for c in front if c in bets.columns]
    bets = bets[front + [c for c in bets.columns if c not in front]]

    return bets


def summarize(bets: pd.DataFrame) -> Dict[str, Any]:
    if bets.empty:
        return {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": None, "win_rate": None}
    stake = float(bets["stake"].sum())
    profit = float(bets["profit"].sum())
    roi = (profit / stake) if stake > 0 else None
    win_rate = float((bets["result"].astype(str).str.lower() == "win").mean())
    return {"bets": int(len(bets)), "stake": stake, "profit": profit, "roi": roi, "win_rate": win_rate}


def bucketize(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame(columns=["bucket","bets","stake","profit","roi","win_rate","avg_edge","avg_odds"])
    b = bets.copy()
    b["edge_used"] = pd.to_numeric(b["edge_used"], errors="coerce")

    bins = [0.0,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,10.0]
    labels = ["<0.02","0.02-0.03","0.03-0.04","0.04-0.05","0.05-0.06",
              "0.06-0.07","0.07-0.08","0.08-0.09","0.09-0.10",">=0.10"]
    b["bucket"] = pd.cut(b["edge_used"].fillna(-1.0), bins=bins, labels=labels, right=False, include_lowest=True)

    g = b.groupby("bucket", dropna=False, observed=False)
    out = g.agg(
        bets=("profit","size"),
        stake=("stake","sum"),
        profit=("profit","sum"),
        win_rate=("result", lambda s: (s.astype(str).str.lower()=="win").mean()),
        avg_edge=("edge_used","mean"),
        avg_odds=("odds_price", lambda s: pd.to_numeric(s, errors="coerce").mean()),
    ).reset_index()
    out["roi"] = out.apply(lambda r: (r["profit"]/r["stake"]) if r["stake"] else None, axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser("roi_analysis.py")
    ap.add_argument("--per_game", required=True)
    ap.add_argument("--edge", required=True, type=float)
    ap.add_argument("--calibrator", default=None)
    args = ap.parse_args()

    cfg = ROIConfig(
        per_game_path=args.per_game,
        edge_threshold=float(args.edge),
        calibrator_path=args.calibrator if args.calibrator else None,
        out_dir="outputs",
    )

    print(f"[roi] version={ROI_ANALYSIS_VERSION}")
    print(f"[roi] __file__={__file__}")
    print(f"[roi] cwd={os.getcwd()}")
    print(f"[roi] per_game={cfg.per_game_path}")

    if not os.path.exists(cfg.per_game_path):
        raise FileNotFoundError(f"[roi] per_game not found: {cfg.per_game_path}")

    df = pd.read_csv(cfg.per_game_path)
    bets = build_bets(df, cfg.edge_threshold, cfg.calibrator_path)

    os.makedirs(cfg.out_dir, exist_ok=True)

    overall = summarize(bets)
    home_bets = bets[bets["bet_side"].astype(str).str.lower()=="home"].copy() if not bets.empty else bets
    away_bets = bets[bets["bet_side"].astype(str).str.lower()=="away"].copy() if not bets.empty else bets

    home_sum = summarize(home_bets)
    away_sum = summarize(away_bets)

    print(f"[roi] edge_threshold={cfg.edge_threshold:.4f}")
    print(f"[roi] overall: {overall}")
    print(f"[roi] home_only: {home_sum}")
    print(f"[roi] away_only: {away_sum}")

    buckets = bucketize(bets)

    metrics = {
        "version": ROI_ANALYSIS_VERSION,
        "edge_threshold": cfg.edge_threshold,
        "overall": overall,
        "home_only": home_sum,
        "away_only": away_sum,
        "calibrator": cfg.calibrator_path,
        "schema_contract": {"required_odds_cols": REQUIRED_ODDS_COLS, "model_prob_col_used": pick_model_prob_col(df)},
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

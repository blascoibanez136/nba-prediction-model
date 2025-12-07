"""
Edge Finder & Pick Sheet
------------------------
Reads today's model predictions (market-adjusted if available) and either:
  A) the latest CLOSE snapshot from The Odds API (preferred, includes prices), or
  B) the dispersion/consensus file (outputs/odds_dispersion_latest.csv; no prices)

Builds a composite merge key to join predictions and odds reliably across ID systems:
    merge_key = norm(home_team) + "__" + norm(away_team) + "__" + game_date

Outputs:
    outputs/picks_<YYYY-MM-DD>.csv
    picks_report.html

Run locally:
    PYTHONPATH=. python src/eval/edge_picker.py
"""

from __future__ import annotations

import os
import glob
import json
from datetime import date, datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

SNAP_DIR = "data/_snapshots"


# -----------------------
# TEAM NORMALIZATION
# -----------------------
def _norm_team(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    aliases = {
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "ny knicks": "new york knicks",
    }
    return aliases.get(s, s)


def _merge_key(home_team: str, away_team: str, game_date: str) -> str:
    return f"{_norm_team(home_team)}__{_norm_team(away_team)}__{game_date}"


def _date_utc_from_commence(ts: str) -> str:
    ts = str(ts)
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d")


# -----------------------
# PRICE + IMPLIED PROB HELPERS
# -----------------------
def _american_to_prob(american: float) -> float:
    try:
        a = float(american)
    except Exception:
        return np.nan
    if a > 0:
        return 100.0 / (a + 100.0)
    return (-a) / ((-a) + 100.0)


def _price_to_prob(price: float) -> float:
    if pd.isna(price):
        return np.nan
    try:
        p = float(price)
    except Exception:
        return np.nan

    # Heuristics:
    # - If |p| >= 10, treat as American
    # - If 1.01 <= p <= 10, treat as decimal
    # - Otherwise, default to American for safety
    if abs(p) >= 10:
        # American odds (e.g. -110, +150)
        return _american_to_prob(p)
    elif 1.01 <= p <= 10.0:
        # Decimal odds (e.g. 1.90)
        return 1.0 / p
    else:
        # Fallback
        return _american_to_prob(p)



def _kelly_fraction(p: float, odds_american: float) -> float:
    if pd.isna(odds_american):
        return 0.0
    b = (abs(odds_american) / 100.0) if odds_american > 0 else (100.0 / abs(odds_american))
    q = 1.0 - p
    k = (b * p - q) / b
    return max(0.0, float(k))


# -----------------------
# LOADING PREDICTIONS
# -----------------------
def load_predictions_for_today() -> (str, pd.DataFrame):
    today = os.getenv("RUN_DATE") or date.today().strftime("%Y-%m-%d")
    base = f"outputs/predictions_{today}.csv"
    market = f"outputs/predictions_{today}_market.csv"
    path = market if os.path.exists(market) else base
    if not os.path.exists(path):
        raise FileNotFoundError(f"No predictions found for {today}")

    df = pd.read_csv(path)

    required_cols = {"home_team", "away_team", "game_date", "fair_spread", "home_win_prob"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {missing}")

    df["game_id"] = df.get("game_id", "").astype(str)
    df["merge_key"] = df.apply(lambda r: _merge_key(r["home_team"], r["away_team"], r["game_date"]), axis=1)

    return today, df


# -----------------------
# CLOSE SNAPSHOT LOADING
# -----------------------
def _latest_close_snapshot() -> Optional[str]:
    os.makedirs(SNAP_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SNAP_DIR, "close_*.csv")))
    return files[-1] if files else None


def _to_list(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            try:
                import ast
                return ast.literal_eval(obj)
            except Exception:
                return []
    return []


# -----------------------
# FLATTEN SNAPSHOT SPREADS
# -----------------------
def flatten_spreads_from_snapshot(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    rows: List[Dict[str, Any]] = []

    for _, row in raw.iterrows():
        game_id = str(row.get("id") or row.get("game_id") or "")
        if not game_id:
            continue

        home = row.get("home_team")
        away = row.get("away_team")
        commence = row.get("commence_time")
        if not home or not away:
            continue

        books = _to_list(row.get("bookmakers", []))
        game_date = _date_utc_from_commence(commence) if pd.notna(commence) else ""

        for bk in books:
            book_key = bk.get("key") or bk.get("title") or "unknown"

            for m in bk.get("markets", []):
                if (m.get("key") or "").lower() != "spreads":
                    continue

                for outc in m.get("outcomes", []):
                    team = outc.get("name")
                    spread = pd.to_numeric(outc.get("point"), errors="coerce")
                    price = pd.to_numeric(outc.get("price"), errors="coerce")

                    rows.append({
                        "game_id": game_id,
                        "book": str(book_key),
                        "team": team,
                        "spread": spread,
                        "price": price,
                        "home_team": home,
                        "away_team": away,
                        "game_date": game_date,
                        "merge_key": _merge_key(home, away, game_date),
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["game_id"] = df["game_id"].astype(str)
    return df


# -----------------------
# BUILD HOME/AWAY SIDE TABLE
# -----------------------
def build_side_table(spreads_df: pd.DataFrame) -> pd.DataFrame:
    if spreads_df.empty:
        return pd.DataFrame(columns=[
            "merge_key", "game_id", "book", "home_team", "away_team",
            "market_side", "book_spread_home", "market_price"
        ])

    home_rows = spreads_df[spreads_df["team"] == spreads_df["home_team"]].copy()
    away_rows = spreads_df[spreads_df["team"] == spreads_df["away_team"]].copy()

    home_rows["market_side"] = "HOME"
    home_rows["book_spread_home"] = home_rows["spread"]
    home_rows["market_price"] = home_rows["price"]

    away_rows["market_side"] = "AWAY"
    away_rows["book_spread_home"] = -away_rows["spread"]
    away_rows["market_price"] = away_rows["price"]

    merged = pd.concat([home_rows, away_rows], ignore_index=True, sort=False)

    keep = [
        "merge_key", "game_id", "book", "home_team", "away_team",
        "market_side", "book_spread_home", "market_price"
    ]
    return merged[keep]


# -----------------------
# PICKS WITH PRICES (FULL)
# -----------------------
def make_picks_with_prices(preds: pd.DataFrame, side_table: pd.DataFrame) -> pd.DataFrame:
    if side_table.empty:
        return pd.DataFrame(columns=[
            "game_id", "game_date", "home_team", "away_team",
            "book", "market_side", "market_price",
            "book_spread_home", "model_fair_spread",
            "model_edge_pts", "book_implied_prob",
            "model_side_prob", "suggested_kelly",
            "suggested_stake_units"
        ])

    # Prediction fields to pull in (only if present)
    pred_cols = [
        "merge_key", "home_team", "away_team", "game_date",
        "fair_spread", "fair_spread_market",
        "home_win_prob", "home_win_prob_market",
        "consensus_close", "book_dispersion"
    ]

    available_pred_cols = [c for c in pred_cols if c in preds.columns]
    df = side_table.merge(preds[available_pred_cols], on="merge_key", how="left")

    # Model fair spread (prefer market-adjusted)
    df["model_fair_spread"] = pd.to_numeric(
        df.get("fair_spread_market", df.get("fair_spread")),
        errors="coerce"
    )

    # Price handling
    df["market_price"] = pd.to_numeric(df["market_price"], errors="coerce")
    df["book_implied_prob"] = df["market_price"].apply(_price_to_prob)

    # Choose correct win prob
    use_wp = np.where(
        df.get("home_win_prob_market").notna()
        if "home_win_prob_market" in df.columns else False,
        df.get("home_win_prob_market", df.get("home_win_prob")),
        df.get("home_win_prob")
    ).astype(float)

    df["model_side_prob"] = np.where(df["market_side"] == "HOME", use_wp, 1 - use_wp)

    # Edge
    df["model_edge_pts"] = df["model_fair_spread"] - df["book_spread_home"]

    # Kelly sizing
    k_raw = [
        _kelly_fraction(p, price) if not pd.isna(price) else 0.0
        for p, price in zip(df["model_side_prob"], df["market_price"])
    ]
    df["kelly_raw"] = k_raw

    k_cap = float(os.getenv("KELLY_CAP", "0.25"))
    df["suggested_kelly"] = np.clip(df["kelly_raw"], 0.0, k_cap)

    bankroll = float(os.getenv("BANKROLL_UNITS", "100.0"))
    df["suggested_stake_units"] = df["suggested_kelly"] * bankroll

    # Final output ordering
    keep = [
        "game_id", "game_date", "home_team", "away_team",
        "book", "market_side", "market_price",
        "book_spread_home", "model_fair_spread",
        "model_edge_pts", "book_implied_prob",
        "model_side_prob", "suggested_kelly",
        "suggested_stake_units", "consensus_close", "book_dispersion"
    ]

    available = [c for c in keep if c in df.columns]

    df = df[available].sort_values(
        ["suggested_kelly", "model_edge_pts"],
        ascending=[False, False]
    )

    return df


# -----------------------
# PICKS WITHOUT PRICES (DISPERSION MODE)
# -----------------------
def make_picks_reduced(preds: pd.DataFrame, dispersion: pd.DataFrame) -> pd.DataFrame:
    need_cols = {"merge_key", "consensus_close", "book_dispersion", "home_team", "away_team", "game_date"}
    missing = need_cols - set(dispersion.columns)

    if missing:
        if {"home_team", "away_team", "game_date"}.issubset(dispersion.columns):
            dispersion = dispersion.copy()
            dispersion["merge_key"] = dispersion.apply(
                lambda r: _merge_key(r["home_team"], r["away_team"], r["game_date"]), axis=1
            )
        else:
            return pd.DataFrame()

    df = preds.merge(
        dispersion[["merge_key", "consensus_close", "book_dispersion"]],
        on="merge_key", how="left"
    )

    df["model_edge_pts"] = df["fair_spread"] - df["consensus_close"]

    keep = [
        "game_id", "home_team", "away_team",
        "fair_spread", "consensus_close",
        "model_edge_pts", "book_dispersion"
    ]

    out = df[keep].dropna(subset=["consensus_close"]).copy()
    return out.sort_values(["model_edge_pts", "book_dispersion"], ascending=[False, True])


# -----------------------
# HTML REPORT
# -----------------------
def render_html(picks: pd.DataFrame, out_html: str, today: str):
    if picks.empty:
        html = f"<html><body><h1>Picks for {today}</h1><p>No edges today.</p></body></html>"
        open(out_html, "w").write(html)
        return

    fmt = {col: "{:.4f}".format for col in picks.select_dtypes(include=[float]).columns}

    html = f"""
    <html><head><meta charset="utf-8">
    <style>
      body {{ font-family: system-ui, sans-serif; padding: 16px; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ padding: 6px 8px; border: 1px solid #ddd; }}
      th {{ background: #f6f6f6; }}
    </style>
    </head><body>
      <h1>Picks for {today}</h1>
      <p>If no price snapshot exists, showing reduced pick sheet.</p>
      {picks.to_html(index=False, formatters=fmt)}
    </body></html>
    """

    open(out_html, "w").write(html)


# -----------------------
# MAIN
# -----------------------
def main():
    today, preds = load_predictions_for_today()
    os.makedirs("outputs", exist_ok=True)

    snap = _latest_close_snapshot()
    disp_path = "outputs/odds_dispersion_latest.csv"

    have_snap = bool(snap and os.path.exists(snap))
    have_disp = os.path.exists(disp_path)

    if have_snap:
        spreads = flatten_spreads_from_snapshot(snap)
        side_table = build_side_table(spreads)
        picks = make_picks_with_prices(preds, side_table)

    elif have_disp:
        disp = pd.read_csv(disp_path)
        picks = make_picks_reduced(preds, disp)

    else:
        print("No CLOSE snapshot or dispersion file available; returning empty pick sheet.")
        picks = pd.DataFrame()

    out_csv = f"outputs/picks_{today}.csv"
    picks.to_csv(out_csv, index=False)
    print(f"✅ wrote {out_csv} ({len(picks)} rows)")

    render_html(picks, "picks_report.html", today)


if __name__ == "__main__":
    main()

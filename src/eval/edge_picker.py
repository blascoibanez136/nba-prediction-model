"""
Edge Finder & Pick Sheet
------------------------
Reads today's model predictions (market-adjusted if available) and either:
  A) the latest CLOSE snapshot from The Odds API (preferred; has prices), or
  B) the dispersion/consensus file (outputs/odds_dispersion_latest.csv; no prices)

Builds a composite merge key to join predictions and odds reliably across ID systems:
  merge_key = norm(home_team) + "__" + norm(away_team) + "__" + game_date (UTC)

Outputs:
- outputs/picks_<YYYY-MM-DD>.csv
- picks_report.html

Run locally:
    PYTHONPATH=. python src/eval/edge_picker.py
"""

from __future__ import annotations

import os
import glob
import json
import math
from datetime import date, datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

SNAP_DIR = "data/_snapshots"


# -----------------------
# helpers (normalization)
# -----------------------
def _norm_team(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    aliases = {
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "ny knicks": "new york knicks",
        # extend as you discover more book/team variations
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
# odds price/prob helpers
# -----------------------
def _american_to_prob(american: float) -> float:
    try:
        a = float(american)
    except Exception:
        return np.nan
    if a > 0:
        return 100.0 / (a + 100.0)
    else:
        return (-a) / ((-a) + 100.0)


def _price_to_prob(price: float) -> float:
    if pd.isna(price):
        return np.nan
    try:
        p = float(price)
    except Exception:
        return np.nan
    # Heuristic: treat typical +/- prices as American; otherwise assume decimal
    if -5000 <= p <= 5000:
        return _american_to_prob(p)
    return 1.0 / p


def _kelly_fraction(p: float, odds_american: float) -> float:
    """Kelly stake fraction for a single outcome with American odds."""
    if pd.isna(odds_american):
        return 0.0
    b = (abs(odds_american) / 100.0) if odds_american > 0 else (100.0 / abs(odds_american))
    q = 1.0 - p
    k = (b * p - q) / b
    return max(0.0, float(k))


# -----------------------
# predictions & snapshots
# -----------------------
def load_predictions_for_today() -> Tuple[str, pd.DataFrame]:
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
        raise ValueError(f"predictions missing columns: {missing}")

    df["game_id"] = df["game_id"].astype(str) if "game_id" in df.columns else ""
    df["merge_key"] = df.apply(lambda r: _merge_key(r["home_team"], r["away_team"], r["game_date"]), axis=1)
    return today, df


def _latest_close_snapshot() -> Optional[str]:
    os.makedirs(SNAP_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SNAP_DIR, "close_*.csv")))
    return files[-1] if files else None


# -----------------------
# flatten spreads (snapshot path)
# -----------------------
def _to_list(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str):
        # try JSON, fallback to literal_eval for old CSVs
        try:
            return json.loads(obj)
        except Exception:
            try:
                import ast
                return ast.literal_eval(obj)
            except Exception:
                return []
    return []


def flatten_spreads_from_snapshot(csv_path: str) -> pd.DataFrame:
    """
    Input: raw snapshot CSV with nested 'bookmakers'
    Output: tidy spreads with teams, prices, and derived game_date + merge_key
    """
    raw = pd.read_csv(csv_path)
    rows: List[Dict[str, Any]] = []

    for _, row in raw.iterrows():
        game_id = str(row.get("id") or row.get("game_id") or "")
        if not game_id:
            continue
        home = row.get("home_team")
        away = row.get("away_team")
        commence = row.get("commence_time")
        books = _to_list(row.get("bookmakers", []))
        if not home or not away:
            continue
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


def build_side_table(spreads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce two rows per book/game: HOME and AWAY sides from home perspective,
    carrying merge_key for a reliable join with predictions.
    """
    if spreads_df.empty:
        return pd.DataFrame(columns=[
            "merge_key", "game_id", "book", "home_team", "away_team",
            "side", "book_spread_home", "book_price_for_side"
        ])

    home_rows = spreads_df[spreads_df["team"] == spreads_df["home_team"]].copy()
    away_rows = spreads_df[spreads_df["team"] == spreads_df["away_team"]].copy()

    home_rows["side"] = "HOME"
    home_rows["book_spread_home"] = home_rows["spread"]
    home_rows["book_price_for_side"] = home_rows["price"]

    away_rows["side"] = "AWAY"
    away_rows["book_spread_home"] = -away_rows["spread"]
    away_rows["book_price_for_side"] = away_rows["price"]

    merged = pd.concat([home_rows, away_rows], ignore_index=True, sort=False)
    keep = [
        "merge_key", "game_id", "book", "home_team", "away_team",
        "side", "book_spread_home", "book_price_for_side"
    ]
    return merged[keep].dropna(subset=["book_spread_home"])


# -----------------------
# picks: priced (snapshot) or reduced (dispersion)
# -----------------------
def make_picks_with_prices(preds: pd.DataFrame, side_table: pd.DataFrame) -> pd.DataFrame:
    """
    Full picks (with price, implied prob, Kelly stake).
    Join on merge_key to avoid cross-provider ID mismatches.
    """
    if side_table.empty:
        return pd.DataFrame(columns=[
            "game_id","home_team","away_team","book","side",
            "model_fair_spread","book_spread_home","model_edge_pts",
            "book_price","book_implied_prob",
            "model_side_prob","suggested_kelly","suggested_stake_units"
        ])

    df = side_table.merge(
        preds[["merge_key","home_team","away_team","fair_spread","home_win_prob_market","home_win_prob"]],
        on="merge_key", how="left"
    )
    df["model_fair_spread"] = pd.to_numeric(df["fair_spread"], errors="coerce")
    df["book_price"] = pd.to_numeric(df["book_price_for_side"], errors="coerce")
    df["book_implied_prob"] = df["book_price"].apply(_price_to_prob)

    use_wp = np.where(df["home_win_prob_market"].notna(), df["home_win_prob_market"], df["home_win_prob"]).astype(float)
    df["model_side_prob"] = np.where(df["side"].eq("HOME"), use_wp, 1.0 - use_wp)

    df["model_edge_pts"] = df["model_fair_spread"] - df["book_spread_home"]

    # Kelly (capped)
    k_raw = [
        _kelly_fraction(p, price if not pd.isna(price) else np.nan)
        if not pd.isna(price) else 0.0
        for p, price in zip(df["model_side_prob"], df["book_price"])
    ]
    df["kelly_raw"] = k_raw
    kelly_cap = float(os.getenv("KELLY_CAP", "0.25"))  # default cap 25%
    df["suggested_kelly"] = np.clip(df["kelly_raw"], 0.0, kelly_cap)
    bankroll_units = float(os.getenv("BANKROLL_UNITS", "100.0"))
    df["suggested_stake_units"] = df["suggested_kelly"] * bankroll_units

    keep = [
        "game_id","home_team","away_team","book","side",
        "model_fair_spread","book_spread_home","model_edge_pts",
        "book_price","book_implied_prob",
        "model_side_prob","suggested_kelly","suggested_stake_units"
    ]
    df = df[keep].sort_values(["suggested_kelly","model_edge_pts"], ascending=[False, False])
    return df


def make_picks_reduced(preds: pd.DataFrame, dispersion: pd.DataFrame) -> pd.DataFrame:
    """
    Reduced pick sheet (no prices/stakes). Uses consensus close and dispersion.
    """
    need_cols = {"merge_key","consensus_close","book_dispersion","home_team","away_team","game_date"}
    missing = need_cols - set(dispersion.columns)
    if missing:
        # Try to build merge_key if teams/date exist
        if {"home_team","away_team","game_date"}.issubset(dispersion.columns) and "merge_key" not in dispersion.columns:
            dispersion = dispersion.copy()
            dispersion["merge_key"] = dispersion.apply(
                lambda r: _merge_key(r["home_team"], r["away_team"], r["game_date"]), axis=1
            )
        else:
            return pd.DataFrame(columns=[
                "game_id","home_team","away_team",
                "model_fair_spread","consensus_close","model_edge_pts","book_dispersion"
            ])

    df = preds.merge(
        dispersion[["merge_key","consensus_close","book_dispersion"]],
        on="merge_key", how="left"
    )
    df["model_edge_pts"] = df["fair_spread"] - df["consensus_close"]
    keep = [
        "game_id","home_team","away_team",
        "fair_spread","consensus_close","model_edge_pts","book_dispersion"
    ]
    out = df[keep].dropna(subset=["consensus_close"]).copy()
    out = out.sort_values(["model_edge_pts","book_dispersion"], ascending=[False, True])
    return out


# -----------------------
# HTML report
# -----------------------
def render_html(picks: pd.DataFrame, out_html: str, today: str):
    if picks.empty:
        html = f"<html><body><h1>Picks for {today}</h1><p>No edges today.</p></body></html>"
        open(out_html, "w").write(html)
        return
    floats = {col: "{:.4f}".format for col in picks.select_dtypes(include=[float]).columns}
    table_html = picks.to_html(index=False, formatters=floats)
    html = f"""
    <html><head><meta charset="utf-8">
    <style>
      body {{ font-family: system-ui, sans-serif; padding: 16px; }}
      table {{ border-collapse: collapse; }}
      th, td {{ padding: 6px 8px; border: 1px solid #ddd; }}
      th {{ background: #f6f6f6; }}
    </style>
    </head><body>
      <h1>Picks for {today}</h1>
      <p>If prices are unavailable, this report shows a reduced pick sheet using consensus close (no stakes).</p>
      {table_html}
    </body></html>
    """
    open(out_html, "w").write(html)


# -----------------------
# main
# -----------------------
def main():
    today, preds = load_predictions_for_today()
    os.makedirs("outputs", exist_ok=True)

    # Prefer snapshot (priced picks). If missing, fallback to dispersion (reduced picks).
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
        print("No CLOSE snapshot or dispersion file available; writing empty pick sheet.")
        picks = pd.DataFrame()

    out_csv = f"outputs/picks_{today}.csv"
    picks.to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv} ({len(picks)} rows)")
    render_html(picks, "picks_report.html", today)


if __name__ == "__main__":
    main()

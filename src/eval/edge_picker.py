"""
Edge Finder & Pick Sheet
------------------------
Reads today's model predictions (market-adjusted if available) and the latest
CLOSE odds snapshot from The Odds API (nested bookmakers), flattens spreads,
and computes simple edges + capped Kelly stake suggestions.

Outputs:
- outputs/picks_<YYYY-MM-DD>.csv
- picks_report.html

Run locally:
    PYTHONPATH=. python src/eval/edge_picker.py
"""

from __future__ import annotations
import os, glob, json, math
from datetime import date
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

SNAP_DIR = "data/_snapshots"

def _latest_close_snapshot() -> str | None:
    os.makedirs(SNAP_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SNAP_DIR, "close_*.csv")))
    return files[-1] if files else None

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
    # Some providers store decimal odds; attempt to detect
    # If |price| > 1000, probably decimal? Keep it simple:
    # Better: if price looks like +/-110 (abs<500), treat as American
    if pd.isna(price):
        return np.nan
    try:
        p = float(price)
    except Exception:
        return np.nan
    if -5000 <= p <= 5000:
        return _american_to_prob(p)
    # else decimal odds
    return 1.0 / p

def _kelly_fraction(p: float, odds_american: float) -> float:
    """Kelly for American odds (single outcome)."""
    b = (abs(odds_american) / 100.0) if odds_american > 0 else (100.0 / abs(odds_american))
    q = 1.0 - p
    k = (b * p - q) / b
    return max(0.0, float(k))

def load_predictions_for_today() -> Tuple[str, pd.DataFrame]:
    today = os.getenv("RUN_DATE") or date.today().strftime("%Y-%m-%d")
    base = f"outputs/predictions_{today}.csv"
    market = f"outputs/predictions_{today}_market.csv"
    path = market if os.path.exists(market) else base
    if not os.path.exists(path):
        raise FileNotFoundError(f"No predictions found for {today}")
    df = pd.read_csv(path)
    if "game_id" not in df.columns:
        raise ValueError("predictions missing game_id")
    df["game_id"] = df["game_id"].astype(str)
    return today, df

def flatten_spreads_from_snapshot(csv_path: str) -> pd.DataFrame:
    """
    Input is the raw CSV you saved (nested bookmakers as JSON in a column).
    Expected columns: id, commence_time, home_team, away_team, bookmakers (JSON)
    We extract SPREAD market rows into tidy rows:
      game_id, book, team, spread, price, home_team, away_team
    """
    raw = pd.read_csv(csv_path)
    out_rows: List[Dict[str, Any]] = []

    for _, row in raw.iterrows():
        game_id = str(row.get("id") or row.get("game_id") or "")
        if not game_id:
            continue
        home = row.get("home_team")
        away = row.get("away_team")
        bks = row.get("bookmakers")
        if pd.isna(bks):
            continue
        try:
            books = json.loads(bks) if isinstance(bks, str) else bks
        except Exception:
            continue
        if not isinstance(books, list):
            continue
        for bk in books:
            book_key = bk.get("key") or bk.get("title") or "unknown"
            for m in bk.get("markets", []):
                if (m.get("key") or "").lower() != "spreads":
                    continue
                for outc in m.get("outcomes", []):
                    team = outc.get("name")
                    spread = outc.get("point")
                    price = outc.get("price")
                    out_rows.append({
                        "game_id": game_id,
                        "book": str(book_key),
                        "team": team,
                        "spread": spread,
                        "price": price,
                        "home_team": home,
                        "away_team": away,
                    })

    df = pd.DataFrame(out_rows)
    # normalize types
    if not df.empty:
        df["game_id"] = df["game_id"].astype(str)
        for c in ["spread", "price"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_side_table(spreads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce two rows per game_id per book: one for HOME side, one for AWAY side,
    with the book's spread from home perspective.
    """
    if spreads_df.empty:
        return pd.DataFrame(columns=[
            "game_id","book","home_team","away_team",
            "side","book_spread_home","book_price_for_side"
        ])
    # Determine home and away rows
    home_rows = spreads_df[spreads_df["team"] == spreads_df["home_team"]].copy()
    away_rows = spreads_df[spreads_df["team"] == spreads_df["away_team"]].copy()

    home_rows["side"] = "HOME"
    home_rows["book_spread_home"] = home_rows["spread"]
    home_rows["book_price_for_side"] = home_rows["price"]

    away_rows["side"] = "AWAY"
    # If book lists away spread from away perspective, home perspective = -(away spread)
    away_rows["book_spread_home"] = -away_rows["spread"]
    away_rows["book_price_for_side"] = away_rows["price"]

    merged = pd.concat([home_rows, away_rows], ignore_index=True, sort=False)
    keep = [
        "game_id","book","home_team","away_team",
        "side","book_spread_home","book_price_for_side"
    ]
    return merged[keep].dropna(subset=["book_spread_home","book_price_for_side"])

def make_picks(preds: pd.DataFrame, side_table: pd.DataFrame) -> pd.DataFrame:
    """
    Compare model fair_spread (home perspective) to book spread/prices per side.
    Returns a pick row per (game_id, book, side) with edges and suggested stake.
    """
    if side_table.empty:
        return pd.DataFrame(columns=[
            "game_id","home_team","away_team","book","side",
            "model_fair_spread","book_spread_home","model_edge_pts",
            "book_price","book_implied_prob",
            "suggested_kelly","suggested_stake_units"
        ])

    df = side_table.merge(
        preds[["game_id","home_team","away_team","fair_spread","home_win_prob_market","home_win_prob"]],
        on="game_id", how="left"
    )
    df["model_fair_spread"] = pd.to_numeric(df["fair_spread"], errors="coerce")
    df["book_price"] = pd.to_numeric(df["book_price_for_side"], errors="coerce")
    df["book_implied_prob"] = df["book_price"].apply(_price_to_prob)

    # model edge in points (positive means model likes HOME more than the book)
    df["model_edge_pts"] = df["model_fair_spread"] - df["book_spread_home"]

    # pick win-prob to use
    use_wp = np.where(
        df["home_win_prob_market"].notna(),
        df["home_win_prob_market"],
        df["home_win_prob"]
    ).astype(float)

    # convert to side-specific probability
    df["model_side_prob"] = np.where(
        df["side"].eq("HOME"), use_wp, 1.0 - use_wp
    )

    # Kelly (capped)
    k_raw = [
        _kelly_fraction(p, price if not pd.isna(price) else np.nan)
        if not pd.isna(price) else 0.0
        for p, price in zip(df["model_side_prob"], df["book_price"])
    ]
    df["kelly_raw"] = k_raw
    kelly_cap = float(os.getenv("KELLY_CAP", "0.25"))  # cap at 25% by default
    df["suggested_kelly"] = np.clip(df["kelly_raw"], 0.0, kelly_cap)
    bankroll_units = float(os.getenv("BANKROLL_UNITS", "100.0"))
    df["suggested_stake_units"] = df["suggested_kelly"] * bankroll_units

    # Simple filter columns
    keep = [
        "game_id","home_team","away_team","book","side",
        "model_fair_spread","book_spread_home","model_edge_pts",
        "book_price","book_implied_prob",
        "model_side_prob","suggested_kelly","suggested_stake_units"
    ]
    df = df[keep].sort_values(["suggested_kelly","model_edge_pts"], ascending=[False, False])
    return df

def render_html(picks: pd.DataFrame, out_html: str, today: str):
    if picks.empty:
        html = f"<html><body><h1>Picks for {today}</h1><p>No edges today.</p></body></html>"
        open(out_html, "w").write(html)
        return
    # light style
    table_html = picks.to_html(index=False, float_format=lambda x: f"{x:.4f}")
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
      <p>Capped Kelly (env <code>KELLY_CAP</code>, default 0.25); bankroll units (env <code>BANKROLL_UNITS</code>, default 100).</p>
      {table_html}
    </body></html>
    """
    open(out_html, "w").write(html)

def main():
    today, preds = load_predictions_for_today()
    snap = _latest_close_snapshot()
    if not snap or not os.path.exists(snap):
        print("No CLOSE snapshot found; cannot make a pick sheet.")
        # still emit an empty report for CI
        os.makedirs("outputs", exist_ok=True)
        out_csv = f"outputs/picks_{today}.csv"
        pd.DataFrame().to_csv(out_csv, index=False)
        render_html(pd.DataFrame(), "picks_report.html", today)
        return

    spreads = flatten_spreads_from_snapshot(snap)
    side_table = build_side_table(spreads)
    picks = make_picks(preds, side_table)

    os.makedirs("outputs", exist_ok=True)
    out_csv = f"outputs/picks_{today}.csv"
    picks.to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv} ({len(picks)} rows)")

    render_html(picks, "picks_report.html", today)

if __name__ == "__main__":
    main()

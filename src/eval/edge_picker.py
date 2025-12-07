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
    """
    Convert a price (American or decimal) to implied probability.

    Heuristics:
      - If |p| >= 10, treat as American odds (e.g. -110, +150).
      - If 1.01 <= p <= 10, treat as decimal odds (e.g. 1.90).
      - Otherwise, fallback to American.
    """
    if pd.isna(price):
        return np.nan
    try:
        p = float(price)
    except Exception:
        return np.nan

    if abs(p) >= 10:
        # American odds
        return _american_to_prob(p)
    elif 1.01 <= p <= 10.0:
        # Decimal odds
        return 1.0 / p
    else:
        # Fallback: treat as American
        return _american_to_prob(p)


def _kelly_fraction(p: float, price: float) -> float:
    """
    Kelly stake fraction for a single outcome with a given price.

    We support both American and decimal odds using the same heuristic
    as _price_to_prob:

      - If |price| >= 10, treat as American.
      - If 1.01 <= price <= 10, treat as decimal.
    """
    if pd.isna(price):
        return 0.0
    try:
        p = float(p)
        pr = float(price)
    except Exception:
        return 0.0

    if p <= 0.0 or p >= 1.0:
        return 0.0

    # Determine 'b' = net odds (profit per 1 unit staked)
    if abs(pr) >= 10:
        # American odds
        if pr > 0:
            b = pr / 100.0
        else:
            b = 100.0 / abs(pr)
    elif 1.01 <= pr <= 10.0:
        # Decimal odds
        b = pr - 1.0
    else:
        # Fallback, treat as American
        if pr > 0:
            b = pr / 100.0
        else:
            b = 100.0 / abs(pr)

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

    # Prefer market-adjusted file if present
    if os.path.exists(market):
        path = market
    else:
        path = base

    if not os.path.exists(path):
        raise FileNotFoundError(f"No predictions found for {today} (looked for {path})")

    df = pd.read_csv(path)

    # Ensure basic columns
    required_cols = {"home_team", "away_team", "game_date"}
    missing_basic = required_cols - set(df.columns)
    if missing_basic:
        raise ValueError(f"predictions missing columns: {missing_basic}")

    # Ensure we have win prob + fair_spread
    if "home_win_prob" not in df.columns and "home_win_prob_market" not in df.columns:
        raise ValueError("predictions missing 'home_win_prob' / 'home_win_prob_market'.")

    if "fair_spread" not in df.columns and "fair_spread_market" not in df.columns:
        raise ValueError("predictions missing 'fair_spread' / 'fair_spread_market'.")

    # Build merge_key if missing
    if "merge_key" not in df.columns:
        df["merge_key"] = df.apply(
            lambda r: _merge_key(r["home_team"], r["away_team"], r["game_date"]), axis=1
        )

    # Ensure game_id exists (even synthetic)
    if "game_id" not in df.columns:
        df["game_id"] = df["merge_key"]

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
            "market_side", "book_spread_home", "market_price", "game_date"
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
        "game_date", "market_side", "book_spread_home", "market_price"
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
            "game_id", "game_date", "book", "market_side", "market_price",
            "book_spread_home", "model_fair_spread", "model_edge_pts",
            "book_implied_prob", "model_side_prob", "suggested_kelly",
            "suggested_stake_units", "consensus_close", "book_dispersion"
        ])

    # Merge odds with predictions via merge_key
    df = side_table.merge(
        preds[
            [
                "merge_key",
                "game_id",
                "game_date",
                "home_team",
                "away_team",
                "fair_spread",
                "fair_spread_market",
                "home_win_prob",
                "home_win_prob_market",
                "consensus_close",
                "book_dispersion",
            ]
            if set(["fair_spread_market", "home_win_prob_market"]).issubset(preds.columns)
            else [
                "merge_key",
                "game_id",
                "game_date",
                "home_team",
                "away_team",
                "fair_spread",
                "home_win_prob",
                "consensus_close",
                "book_dispersion",
            ]
        ].drop_duplicates("merge_key"),
        on="merge_key",
        how="left",
        suffixes=("", "_pred"),
    )

    # Choose market-adjusted or raw model spread
    if "fair_spread_market" in df.columns and df["fair_spread_market"].notna().any():
        df["model_fair_spread"] = pd.to_numeric(df["fair_spread_market"], errors="coerce")
    else:
        df["model_fair_spread"] = pd.to_numeric(df["fair_spread"], errors="coerce")

    # Choose market-adjusted or raw win prob
    if "home_win_prob_market" in df.columns and df["home_win_prob_market"].notna().any():
        use_wp = pd.to_numeric(df["home_win_prob_market"], errors="coerce")
    else:
        use_wp = pd.to_numeric(df["home_win_prob"], errors="coerce")

    # Model side prob from home perspective
    df["model_side_prob"] = np.where(
        df["market_side"].eq("HOME"),
        use_wp,
        1.0 - use_wp,
    )

    # Implied probability from market price
    df["market_price"] = pd.to_numeric(df["market_price"], errors="coerce")
    df["book_implied_prob"] = df["market_price"].apply(_price_to_prob)

    # Edge in points (model - market)
    df["model_edge_pts"] = df["model_fair_spread"] - df["book_spread_home"]

    # Kelly (capped)
    k_raw = [
        _kelly_fraction(p, price if not pd.isna(price) else np.nan)
        if not pd.isna(price) and not pd.isna(p) else 0.0
        for p, price in zip(df["model_side_prob"], df["market_price"])
    ]
    df["kelly_raw"] = k_raw
    kelly_cap = float(os.getenv("KELLY_CAP", "0.25"))  # default cap 25%
    df["suggested_kelly"] = np.clip(df["kelly_raw"], 0.0, kelly_cap)
    bankroll_units = float(os.getenv("BANKROLL_UNITS", "100.0"))
    df["suggested_stake_units"] = df["suggested_kelly"] * bankroll_units

    # Drop rows where we have no model opinion
    df = df[
        df["model_fair_spread"].notna() &
        df["model_side_prob"].notna()
    ].copy()

    keep = [
        "game_id",
        "game_date",
        "book",
        "market_side",
        "market_price",
        "book_spread_home",
        "model_fair_spread",
        "model_edge_pts",
        "book_implied_prob",
        "model_side_prob",
        "suggested_kelly",
        "suggested_stake_units",
        "consensus_close",
        "book_dispersion",
    ]

    # Only keep columns that actually exist
    available = [c for c in keep if c in df.columns]
    missing = [c for c in keep if c not in df.columns]

    if missing:
        print(f"[edge_picker] Warning: missing columns in pick sheet output: {missing}")

    # Sort by Kelly and edge
    sort_cols = ["suggested_kelly", "model_edge_pts"]
    for col in sort_cols:
        if col not in df.columns:
            raise KeyError(
                f"[edge_picker] Required column '{col}' not found in DataFrame "
                f"columns: {list(df.columns)}"
            )

    out = df[available].sort_values(sort_cols, ascending=[False, False])
    return out


def make_picks_reduced(preds: pd.DataFrame, dispersion: pd.DataFrame) -> pd.DataFrame:
    """
    Reduced pick sheet (no prices/stakes). Uses consensus close and dispersion.
    """
    if dispersion is None or dispersion.empty:
        return pd.DataFrame(columns=[
            "game_id", "home_team", "away_team",
            "model_fair_spread", "consensus_close", "model_edge_pts", "book_dispersion"
        ])

    # Ensure merge_key in dispersion
    if "merge_key" not in dispersion.columns:
        if {"home_team", "away_team", "game_date"}.issubset(dispersion.columns):
            dispersion = dispersion.copy()
            dispersion["merge_key"] = dispersion.apply(
                lambda r: _merge_key(r["home_team"], r["away_team"], r["game_date"]), axis=1
            )
        else:
            return pd.DataFrame(columns=[
                "game_id", "home_team", "away_team",
                "model_fair_spread", "consensus_close", "model_edge_pts", "book_dispersion"
            ])

    # Ensure merge_key in preds
    if "merge_key" not in preds.columns:
        preds = preds.copy()
        preds["merge_key"] = preds.apply(
            lambda r: _merge_key(r["home_team"], r["away_team"], r["game_date"]), axis=1
        )

    df = preds.merge(
        dispersion[["merge_key", "consensus_close", "book_dispersion"]],
        on="merge_key", how="left"
    )

    # Use market-adjusted spread if present, else model fair_spread
    if "fair_spread_market" in df.columns and df["fair_spread_market"].notna().any():
        df["model_fair_spread"] = df["fair_spread_market"]
    else:
        df["model_fair_spread"] = df["fair_spread"]

    df["model_edge_pts"] = df["model_fair_spread"] - df["consensus_close"]

    keep = [
        "game_id", "home_team", "away_team",
        "model_fair_spread", "consensus_close", "model_edge_pts", "book_dispersion"
    ]
    out = df[keep].dropna(subset=["consensus_close"]).copy()
    out = out.sort_values(["model_edge_pts", "book_dispersion"], ascending=[False, True])
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

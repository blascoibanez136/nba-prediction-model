"""
Backtest picks (supports two modes):

Mode A (embedded odds):
- picks contain an odds column (american or decimal)

Mode B (snapshot odds):
- picks have team+side; odds are resolved from market snapshots by date + merge_key

Hard rules:
- Never silently succeed on empty data
- Always write audit JSON on failure in CLI
- Keep merge_key contract intact
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Optional import: reuse project team normalizer if available
# ---------------------------------------------------------------------

def _try_normalize_team(name: object) -> str:
    s = str(name).strip() if name is not None else ""
    if not s:
        return ""
    try:
        from src.utils.team_name_normalizer import normalize_team_name  # type: ignore
        out = normalize_team_name(s)
        return str(out).strip() if out is not None else s
    except Exception:
        return s


def _norm_team(x: object) -> str:
    return _try_normalize_team(x).strip().lower()


def _merge_key(home: object, away: object, game_date: object) -> str:
    return f"{_norm_team(home)}__{_norm_team(away)}__{str(game_date).strip()}"


# ---------------------------------------------------------------------
# Odds conversion
# ---------------------------------------------------------------------

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


def _coerce_decimal_odds(x: object) -> float:
    # Accept decimal odds as-is (>=1.01), or american odds like -110 / +125
    if x is None:
        return np.nan
    try:
        v = float(x)
    except Exception:
        return np.nan
    if np.isnan(v) or v == 0:
        return np.nan
    # heuristic: decimal odds usually between ~1.01 and 50; american often <= -100 or >= +100
    if v >= 1.01 and v <= 100:
        return v
    # treat as american otherwise
    return _american_to_decimal(v)


# ---------------------------------------------------------------------
# Snapshot loading + column discovery
# ---------------------------------------------------------------------

def _load_snapshot_for_date(snapshot_dir: Path, game_date: str) -> Optional[pd.DataFrame]:
    if snapshot_dir is None or not snapshot_dir.exists():
        return None
    ymd = str(game_date).replace("-", "")
    candidates = [
        snapshot_dir / f"close_{ymd}.csv",
        snapshot_dir / f"{ymd}.csv",
        snapshot_dir / f"snapshot_{ymd}.csv",
        snapshot_dir / f"market_{ymd}.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df["_snapshot_file"] = p.name
            return df
    return None


def _score_col(name: str, must: List[str], nice: List[str], avoid: List[str]) -> int:
    n = name.lower()
    for m in must:
        if m not in n:
            return -10_000
    score = 0
    for a in avoid:
        if a in n:
            score -= 50
    for w in nice:
        if w in n:
            score += 10
    score -= max(0, len(n) - 24) // 4
    return score


def _best_match(columns: List[str], must: List[str], nice: List[str], avoid: List[str]) -> Optional[str]:
    best = None
    best_score = -10_000
    for c in columns:
        s = _score_col(c, must=must, nice=nice, avoid=avoid)
        if s > best_score:
            best_score = s
            best = c
    if best_score < 0:
        return None
    return best


def _discover_team_cols(snap: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = list(snap.columns)
    home = _best_match(
        cols,
        must=["home"],
        nice=["team", "name"],
        avoid=["score", "ml", "money", "line", "spread", "total", "odds", "price"],
    )
    away = _best_match(
        cols,
        must=["away"],
        nice=["team", "name"],
        avoid=["score", "ml", "money", "line", "spread", "total", "odds", "price"],
    )
    return home, away


def _discover_ml_cols(snap: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(snap.columns)

    moneyline_like = [
        c for c in cols
        if re.search(r"(moneyline|(^|_)ml(_|$)|american|odds|price|line)", c.lower())
        and not re.search(r"(spread|total|team|name|score)", c.lower())
    ]

    def pick(side: str, timing: str) -> Optional[str]:
        must = [side]
        nice = ["moneyline", "ml", "american", "odds", "price", "line"]
        avoid = ["spread", "total", "team", "name", "score"]

        if timing == "close":
            nice += ["close", "closing", "cl"]
            avoid += ["open", "opening"]
        else:
            nice += ["open", "opening", "op"]
            avoid += ["close", "closing"]

        return _best_match(moneyline_like, must=must, nice=nice, avoid=avoid)

    return {
        "home_ml_close": pick("home", "close"),
        "away_ml_close": pick("away", "close"),
        "home_ml_open": pick("home", "open"),
        "away_ml_open": pick("away", "open"),
    }


# ---------------------------------------------------------------------
# Picks normalization + mode detection
# ---------------------------------------------------------------------

def _normalize_picks(picks_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = picks_df.copy()
    audit: Dict = {"rows_in": int(len(df)), "inferred": {}, "mode": None}

    if "game_date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "game_date"})
        audit["inferred"]["game_date"] = "renamed from date"

    need = {"game_date", "home_team", "away_team"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"[picks_backtest] Picks missing required columns: {sorted(miss)}")

    if "merge_key" not in df.columns:
        df["merge_key"] = [
            _merge_key(h, a, d) for h, a, d in zip(df["home_team"], df["away_team"], df["game_date"])
        ]
        audit["inferred"]["merge_key"] = "built from teams+date (normalized)"

    if "bet_side" not in df.columns:
        if "pick_side" in df.columns:
            df["bet_side"] = df["pick_side"].astype(str).str.upper()
            audit["inferred"]["bet_side"] = "from pick_side"
        elif "side" in df.columns:
            df["bet_side"] = df["side"].astype(str).str.upper()
            audit["inferred"]["bet_side"] = "from side"
        else:
            raise RuntimeError("[picks_backtest] Cannot infer bet_side (need bet_side/pick_side/side)")

    df["bet_side"] = df["bet_side"].astype(str).str.upper()
    df = df[df["bet_side"].isin(["HOME", "AWAY"])].copy()

    # Detect embedded odds column candidates
    embedded_cols = [c for c in ["odds_decimal", "odds", "american_odds", "price", "line"] if c in df.columns]
    if embedded_cols:
        # pick first in priority order
        src = embedded_cols[0]
        df["odds_decimal"] = df[src].apply(_coerce_decimal_odds)
        audit["mode"] = "embedded_odds"
        audit["inferred"]["odds_decimal"] = f"from {src}"
    else:
        df["odds_decimal"] = np.nan
        audit["mode"] = "snapshot_odds"

    audit["rows_out"] = int(len(df))
    return df, audit


# ---------------------------------------------------------------------
# Attach odds from snapshots (only when needed)
# ---------------------------------------------------------------------

def _attach_odds_from_snapshots(
    picks_df: pd.DataFrame,
    snapshot_dir: Optional[Path],
) -> Tuple[pd.DataFrame, Dict]:
    df = picks_df.copy()

    audit: Dict = {
        "snapshot_dir": str(snapshot_dir) if snapshot_dir else None,
        "dates_seen": int(df["game_date"].nunique()),
        "snapshots_found": 0,
        "rows_with_snapshot_match": 0,
        "odds_filled": 0,
        "odds_missing_after": 0,
        "team_cols": None,
        "ml_cols": None,
        "example_snapshot_columns": None,
        "note": None,
    }

    # If odds already present, skip snapshots entirely
    if int(df["odds_decimal"].notna().sum()) > 0:
        audit["note"] = "embedded odds detected; snapshot odds attach skipped"
        audit["odds_missing_after"] = int(df["odds_decimal"].isna().sum())
        return df, audit

    if snapshot_dir is None:
        audit["note"] = "snapshot_dir not provided; cannot attach odds"
        audit["odds_missing_after"] = int(df["odds_decimal"].isna().sum())
        return df, audit

    filled = 0
    matched_rows = 0
    found_snaps = 0
    first_cols_dumped = False

    for game_date, idx in df.groupby("game_date").groups.items():
        snap = _load_snapshot_for_date(snapshot_dir, str(game_date))
        if snap is None or snap.empty:
            continue

        found_snaps += 1

        if not first_cols_dumped:
            audit["example_snapshot_columns"] = list(snap.columns)[:120]
            first_cols_dumped = True

        home_team_col, away_team_col = _discover_team_cols(snap)
        ml_cols = _discover_ml_cols(snap)

        if audit["team_cols"] is None:
            audit["team_cols"] = {"home_team_col": home_team_col, "away_team_col": away_team_col}
        if audit["ml_cols"] is None:
            audit["ml_cols"] = ml_cols

        if not home_team_col or not away_team_col:
            continue

        # Build merge-key index for snapshot
        snap["_mk"] = [
            _merge_key(h, a, game_date) for h, a in zip(snap[home_team_col], snap[away_team_col])
        ]
        snap = snap.set_index("_mk")

        for i in idx:
            mk = df.at[i, "merge_key"]
            if mk not in snap.index:
                continue

            matched_rows += 1
            row = snap.loc[mk]

            # close preferred, open fallback
            if df.at[i, "bet_side"] == "HOME":
                am = row.get(ml_cols["home_ml_close"], np.nan) if ml_cols["home_ml_close"] else np.nan
                if pd.isna(am) and ml_cols["home_ml_open"]:
                    am = row.get(ml_cols["home_ml_open"], np.nan)
            else:
                am = row.get(ml_cols["away_ml_close"], np.nan) if ml_cols["away_ml_close"] else np.nan
                if pd.isna(am) and ml_cols["away_ml_open"]:
                    am = row.get(ml_cols["away_ml_open"], np.nan)

            dec = _american_to_decimal(am)
            if pd.notna(dec):
                df.at[i, "odds_decimal"] = float(dec)
                filled += 1

    audit["snapshots_found"] = int(found_snaps)
    audit["rows_with_snapshot_match"] = int(matched_rows)
    audit["odds_filled"] = int(filled)
    audit["odds_missing_after"] = int(df["odds_decimal"].isna().sum())
    return df, audit


# ---------------------------------------------------------------------
# History score detection
# ---------------------------------------------------------------------

def _find_score_cols(df: pd.DataFrame) -> Tuple[str, str]:
    candidates = [
        ("home_score", "away_score"),
        ("home_points", "away_points"),
        ("pts_home", "pts_away"),
        ("home_team_score", "away_team_score"),
        ("home_score_final", "away_score_final"),
        ("home_pts", "away_pts"),
    ]
    for h, a in candidates:
        if h in df.columns and a in df.columns:
            return h, a
    raise RuntimeError(
        "[picks_backtest] Could not find home/away score columns in joined data. "
        f"Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------
# Backtest core
# ---------------------------------------------------------------------

def backtest_picks(
    picks_df: pd.DataFrame,
    history_df: pd.DataFrame,
    *,
    snapshot_dir: Optional[Path],
    stake: float,
) -> Tuple[pd.DataFrame, Dict]:

    picks_norm, norm_audit = _normalize_picks(picks_df)
    picks_w_odds, odds_audit = _attach_odds_from_snapshots(picks_norm, snapshot_dir)

    hist = history_df.copy()
    if "game_date" not in hist.columns and "date" in hist.columns:
        hist = hist.rename(columns={"date": "game_date"})

    if "merge_key" not in hist.columns:
        need = {"game_date", "home_team", "away_team"}
        miss = need - set(hist.columns)
        if miss:
            raise RuntimeError(f"[picks_backtest] History missing required columns: {sorted(miss)}")
        hist["merge_key"] = [
            _merge_key(h, a, d) for h, a, d in zip(hist["home_team"], hist["away_team"], hist["game_date"])
        ]

    joined = picks_w_odds.merge(
        hist,
        on="merge_key",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    join_audit = {
        "picks_rows": int(len(picks_w_odds)),
        "joined_rows": int(len(joined)),
        "missing_history": int((joined["_merge"] != "both").sum()),
    }

    joined = joined[joined["_merge"] == "both"].drop(columns="_merge")
    if joined.empty:
        raise RuntimeError("[picks_backtest] All picks dropped after history join")

    home_score_col, away_score_col = _find_score_cols(joined)
    joined["home_win"] = (joined[home_score_col] > joined[away_score_col]).astype(int)
    joined["result"] = joined.apply(
        lambda r: r["home_win"] if r["bet_side"] == "HOME" else 1 - r["home_win"],
        axis=1,
    )

    joined["stake"] = float(stake)
    joined["pnl"] = np.where(
        joined["odds_decimal"].notna(),
        np.where(joined["result"] == 1, (joined["odds_decimal"] - 1.0) * joined["stake"], -joined["stake"]),
        np.nan,
    )

    resolved = joined[joined["pnl"].notna()].copy()

    audit = {
        "normalize": norm_audit,
        "odds_attach": odds_audit,
        "history_join": join_audit,
        "resolved_bets": int(len(resolved)),
        "unresolved_missing_odds": int(joined["pnl"].isna().sum()),
    }

    if resolved.empty:
        raise RuntimeError(
            "[picks_backtest] No resolved bets. "
            "Likely: snapshot join mismatch OR moneyline columns not detected. "
            "Inspect outputs/picks_backtest_audit.json -> odds_attach.team_cols/ml_cols/example_snapshot_columns."
        )

    total_pnl = float(resolved["pnl"].sum())
    total_staked = float(resolved["stake"].sum())
    roi = (total_pnl / total_staked) if total_staked > 0 else float("nan")

    audit["summary"] = {
        "bets": int(len(resolved)),
        "wins": int((resolved["result"] == 1).sum()),
        "losses": int((resolved["result"] == 0).sum()),
        "total_pnl": total_pnl,
        "total_staked": total_staked,
        "roi": roi,
    }

    return resolved, audit


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest picks (embedded odds OR snapshot odds).")
    ap.add_argument("--picks-dir", required=True)
    ap.add_argument("--pattern", default="picks_*.csv")
    ap.add_argument("--history", required=True)
    ap.add_argument("--snapshot-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stake", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "picks_backtest_audit.json"

    try:
        picks_dir = Path(args.picks_dir)
        files = sorted(picks_dir.glob(args.pattern))
        if not files:
            raise RuntimeError(f"[picks_backtest] No picks files matched: {picks_dir}/{args.pattern}")

        # Avoid concat warning: skip empty/all-NA frames
        dfs = []
        for f in files:
            df = pd.read_csv(f)
            if df is None or df.empty:
                continue
            # drop fully-empty columns that cause dtype warnings
            df = df.dropna(axis=1, how="all")
            df["_source_file"] = f.name
            dfs.append(df)

        if not dfs:
            raise RuntimeError("[picks_backtest] All picks files were empty")

        picks = pd.concat(dfs, ignore_index=True)
        history = pd.read_csv(args.history)
        snap_dir = Path(args.snapshot_dir) if args.snapshot_dir else None

        joined, audit = backtest_picks(
            picks,
            history,
            snapshot_dir=snap_dir,
            stake=float(args.stake),
        )

        joined_path = out_dir / "picks_backtest.csv"
        summary_path = out_dir / "picks_backtest_summary.csv"

        joined.to_csv(joined_path, index=False)
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        pd.DataFrame([audit["summary"]]).to_csv(summary_path, index=False)

        print(f"[picks_backtest] wrote {joined_path}")
        print(f"[picks_backtest] wrote {summary_path}")
        print(f"[picks_backtest] wrote {audit_path}")
        return 0

    except Exception as e:
        # Always write failure audit
        try:
            audit_path.write_text(json.dumps({"error": str(e)}, indent=2), encoding="utf-8")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())

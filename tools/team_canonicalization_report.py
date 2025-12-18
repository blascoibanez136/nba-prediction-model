# tools/team_canonicalization_report.py
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from src.utils.team_name_normalizer import normalize_team_name


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str  # "predictions" | "backtest_joined" | "snapshots"
    paths: List[str]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_date_from_path(p: str) -> Optional[str]:
    m = DATE_RE.search(os.path.basename(p))
    return m.group(1) if m else None


def _date_in_range(date_str: str, date_min: Optional[str], date_max: Optional[str]) -> bool:
    if not date_str:
        return True
    if date_min and date_str < date_min:
        return False
    if date_max and date_str > date_max:
        return False
    return True


def _safe_read_csv(path: str) -> pd.DataFrame:
    # Keep strings stable; avoid dtype inference surprises
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])


def _detect_team_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (home_col, away_col, date_col) if detected.
    Date col is best-effort; may be None.
    """
    cols = [c for c in df.columns]
    lower = {c.lower(): c for c in cols}

    # Most likely (locked contract expectation)
    home_col = lower.get("home_team") or lower.get("hometeam")
    away_col = lower.get("away_team") or lower.get("awayteam")

    # Common date column variants
    date_col = (
        lower.get("game_date")
        or lower.get("date")
        or lower.get("game_day")
        or lower.get("gamedate")
    )

    # If not found, try other common fields (report-only heuristics)
    if home_col is None and "home" in lower:
        home_col = lower["home"]
    if away_col is None and "away" in lower:
        away_col = lower["away"]

    return home_col, away_col, date_col


def _extract_team_strings_from_df(
    df: pd.DataFrame,
    source_name: str,
    file_path: str,
    date_min: Optional[str],
    date_max: Optional[str],
) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Returns:
      - list of team strings (raw)
      - list of example rows dicts for audit (bounded later)
    """
    home_col, away_col, date_col = _detect_team_columns(df)

    if home_col is None or away_col is None:
        # Not an error: snapshots may not contain expected columns; we report it.
        return [], [{
            "source": source_name,
            "file": file_path,
            "issue": "missing_home_or_away_columns",
            "columns": ",".join(df.columns),
        }]

    # Optional filtering by date if a date column exists
    if date_col is not None:
        mask = df[date_col].apply(lambda d: _date_in_range(str(d), date_min, date_max))
        df = df.loc[mask]

    teams = []
    # Extend with both columns
    teams.extend(df[home_col].astype(str).tolist())
    teams.extend(df[away_col].astype(str).tolist())

    examples = []
    # Small sample for audit context
    if len(df) > 0:
        sample = df[[home_col, away_col] + ([date_col] if date_col else [])].head(5)
        for _, row in sample.iterrows():
            r = {
                "source": source_name,
                "file": file_path,
                "home": str(row[home_col]),
                "away": str(row[away_col]),
            }
            if date_col:
                r["date"] = str(row[date_col])
            examples.append(r)

    return teams, examples


def _scan_predictions(pred_dir: str, date_min: Optional[str], date_max: Optional[str]) -> SourceSpec:
    pattern = os.path.join(pred_dir, "predictions_*.csv")
    paths = sorted(glob.glob(pattern))
    # Date filtering based on filename if present
    filtered = []
    for p in paths:
        d = _parse_date_from_path(p)
        if d is None or _date_in_range(d, date_min, date_max):
            filtered.append(p)
    return SourceSpec(name="predictions", kind="predictions", paths=filtered)


def _scan_snapshots(snapshots_dir: str, date_min: Optional[str], date_max: Optional[str]) -> SourceSpec:
    # intentionally broad: snapshot naming may vary
    pattern = os.path.join(snapshots_dir, "*.csv")
    paths = sorted(glob.glob(pattern))
    filtered = []
    for p in paths:
        d = _parse_date_from_path(p)
        if d is None or _date_in_range(d, date_min, date_max):
            filtered.append(p)
    return SourceSpec(name="snapshots", kind="snapshots", paths=filtered)


def _scan_backtest_joined(backtest_joined_path: str) -> SourceSpec:
    return SourceSpec(name="backtest_joined", kind="backtest_joined", paths=[backtest_joined_path])


def _build_vocab_from_sources(
    sources: Sequence[SourceSpec],
    date_min: Optional[str],
    date_max: Optional[str],
    max_examples: int = 200,
) -> Tuple[Dict[str, Counter], Dict[str, Dict[str, Counter]], List[Dict[str, str]]]:
    """
    Returns:
      - counts_by_source: {source_name: Counter(raw_name -> count)}
      - normalized_buckets_by_source: {source_name: {normalized -> Counter(raw -> count)}}
      - issues/examples: list of audit issue dicts
    """
    counts_by_source: Dict[str, Counter] = {}
    normalized_buckets_by_source: Dict[str, Dict[str, Counter]] = {}
    issues: List[Dict[str, str]] = []

    for src in sources:
        c = Counter()
        buckets: Dict[str, Counter] = defaultdict(Counter)

        for path in src.paths:
            try:
                df = _safe_read_csv(path)
            except Exception as e:
                issues.append({
                    "source": src.name,
                    "file": path,
                    "issue": "csv_read_failed",
                    "error": repr(e),
                })
                continue

            teams, ex = _extract_team_strings_from_df(
                df=df,
                source_name=src.name,
                file_path=path,
                date_min=date_min,
                date_max=date_max,
            )
            issues.extend(ex)

            for t in teams:
                raw = str(t).strip()
                if raw == "":
                    continue
                c[raw] += 1
                nn = normalize_team_name(raw).normalized
                buckets[nn][raw] += 1

        counts_by_source[src.name] = c
        normalized_buckets_by_source[src.name] = buckets

    # bound issues/examples
    issues = issues[:max_examples]
    return counts_by_source, normalized_buckets_by_source, issues


def _collisions(buckets: Dict[str, Counter], min_distinct_raw: int = 2) -> List[Dict[str, object]]:
    """
    For a single source: list normalized buckets that map to multiple raw strings.
    """
    out = []
    for norm, raw_counter in buckets.items():
        if len(raw_counter) >= min_distinct_raw:
            out.append({
                "normalized": norm,
                "distinct_raw": len(raw_counter),
                "raw_variants": raw_counter.most_common(20),
            })
    out.sort(key=lambda x: (-int(x["distinct_raw"]), x["normalized"]))
    return out


def _cross_source_missing(
    counts_by_source: Dict[str, Counter],
    src_a: str,
    src_b: str,
) -> Dict[str, object]:
    a = set(counts_by_source.get(src_a, Counter()).keys())
    b = set(counts_by_source.get(src_b, Counter()).keys())
    return {
        "a": src_a,
        "b": src_b,
        "a_only": sorted(a - b),
        "b_only": sorted(b - a),
        "a_only_count": len(a - b),
        "b_only_count": len(b - a),
        "intersection_count": len(a & b),
    }


def _suggest_aliases(
    buckets_a: Dict[str, Counter],
    buckets_b: Dict[str, Counter],
    max_suggestions: int = 50,
) -> List[Dict[str, object]]:
    """
    Suggest alias mappings between two sources based on shared normalized buckets.
    This is intentionally conservative: it suggests pairs inside the same normalized bucket.
    """
    suggestions = []

    shared_norms = set(buckets_a.keys()) & set(buckets_b.keys())
    for norm in shared_norms:
        a_vars = buckets_a[norm].most_common()
        b_vars = buckets_b[norm].most_common()

        # If both have exactly one dominant variant and they differ, that's a strong alias candidate.
        if len(a_vars) >= 1 and len(b_vars) >= 1:
            a_top, a_cnt = a_vars[0]
            b_top, b_cnt = b_vars[0]
            if a_top != b_top:
                suggestions.append({
                    "normalized": norm,
                    "source_a_top": (a_top, a_cnt),
                    "source_b_top": (b_top, b_cnt),
                    "confidence": "high" if (len(a_vars) == 1 and len(b_vars) == 1) else "medium",
                })

    # Rank by combined frequency, then confidence, then name
    def score(s: Dict[str, object]) -> Tuple[int, int, str]:
        a_cnt = int(s["source_a_top"][1])
        b_cnt = int(s["source_b_top"][1])
        conf = 2 if s["confidence"] == "high" else 1
        return (a_cnt + b_cnt, conf, str(s["normalized"]))

    suggestions.sort(key=score, reverse=True)
    return suggestions[:max_suggestions]


def _render_markdown(report: Dict[str, object]) -> str:
    lines: List[str] = []
    meta = report["meta"]

    lines.append("# Team Canonicalization Report (Audit-Only)")
    lines.append("")
    lines.append(f"- Generated: `{meta['generated_at_utc']}`")
    lines.append(f"- pred_dir: `{meta.get('pred_dir')}`")
    lines.append(f"- snapshots_dir: `{meta.get('snapshots_dir')}`")
    lines.append(f"- backtest_joined: `{meta.get('backtest_joined_path')}`")
    lines.append(f"- date_min/date_max: `{meta.get('date_min')}` / `{meta.get('date_max')}`")
    lines.append("")

    lines.append("## Vocabularies (unique raw team strings)")
    lines.append("")
    for src_name, vocab in report["vocabularies"].items():
        lines.append(f"### {src_name}")
        lines.append(f"- Unique teams: **{vocab['unique_count']}**")
        top = vocab["top_teams"][:15]
        if top:
            lines.append("")
            lines.append("| Team | Count |")
            lines.append("|---|---:|")
            for team, cnt in top:
                lines.append(f"| {team} | {cnt} |")
        lines.append("")

    lines.append("## Normalized collisions (alias candidates within a source)")
    lines.append("")
    for src_name, cols in report["collisions"].items():
        lines.append(f"### {src_name}")
        lines.append(f"- Buckets with >=2 variants: **{len(cols)}**")
        show = cols[:10]
        for c in show:
            lines.append(f"- `{c['normalized']}` → {c['raw_variants'][:5]}")
        lines.append("")

    lines.append("## Cross-source missing sets (raw string mismatches)")
    lines.append("")
    for x in report["cross_source_missing"]:
        lines.append(f"- **{x['a']} vs {x['b']}**: "
                     f"{x['a_only_count']} only-in-{x['a']}, "
                     f"{x['b_only_count']} only-in-{x['b']}, "
                     f"intersection {x['intersection_count']}")
    lines.append("")

    lines.append("## Suggested alias mappings (NOT applied)")
    lines.append("")
    sugg = report["suggested_aliases"]
    if sugg:
        lines.append("| Normalized bucket | A top | B top | Confidence |")
        lines.append("|---|---|---|---|")
        for s in sugg[:25]:
            a = f"{s['source_a_top'][0]} ({s['source_a_top'][1]})"
            b = f"{s['source_b_top'][0]} ({s['source_b_top'][1]})"
            lines.append(f"| `{s['normalized']}` | {a} | {b} | {s['confidence']} |")
    else:
        lines.append("_No alias suggestions produced (insufficient overlap)._")
    lines.append("")

    lines.append("## Issues / detection notes (sample)")
    lines.append("")
    issues = report.get("issues_sample", [])
    if issues:
        for it in issues[:30]:
            if it.get("issue") == "missing_home_or_away_columns":
                lines.append(f"- {it['source']}: `{Path(it['file']).name}` missing home/away columns; columns=`{it['columns']}`")
            elif it.get("issue") == "csv_read_failed":
                lines.append(f"- {it['source']}: `{Path(it['file']).name}` read failed: `{it['error']}`")
            else:
                # example row
                if "home" in it and "away" in it:
                    lines.append(f"- {it['source']}: `{Path(it['file']).name}` e.g. {it['home']} vs {it['away']}" + (f" ({it.get('date')})" if it.get("date") else ""))
    else:
        lines.append("_No issues detected._")

    lines.append("")
    lines.append("## Next action")
    lines.append("")
    lines.append("- Review suggested aliases and decide whether to promote a *gated* alias map into production canonicalization.")
    lines.append("- Do **not** change merge_key contract; any enforcement must be optional and auditable.")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit-only team canonicalization report (does not change pipeline behavior).")
    ap.add_argument("--pred-dir", default="outputs", help="Directory containing predictions_YYYY-MM-DD.csv files.")
    ap.add_argument("--snapshots-dir", default=None, help="Directory containing market snapshot CSVs (optional).")
    ap.add_argument("--backtest-joined", default=None, help="Path to backtest_joined.csv (optional).")
    ap.add_argument("--date-min", default=None, help="YYYY-MM-DD (optional).")
    ap.add_argument("--date-max", default=None, help="YYYY-MM-DD (optional).")
    ap.add_argument("--out-dir", default="outputs/audits", help="Directory to write report artifacts.")
    args = ap.parse_args()

    sources: List[SourceSpec] = []
    pred = _scan_predictions(args.pred_dir, args.date_min, args.date_max)
    sources.append(pred)

    if args.snapshots_dir:
        sources.append(_scan_snapshots(args.snapshots_dir, args.date_min, args.date_max))

    if args.backtest_joined:
        sources.append(_scan_backtest_joined(args.backtest_joined))

    # Require at least one existing input file
    total_files = sum(len(s.paths) for s in sources)
    if total_files == 0:
        raise SystemExit("No input CSVs found. Check --pred-dir / --snapshots-dir / --backtest-joined paths.")

    counts_by_source, buckets_by_source, issues = _build_vocab_from_sources(
        sources=sources,
        date_min=args.date_min,
        date_max=args.date_max,
    )

    vocabularies = {}
    for src_name, counter in counts_by_source.items():
        vocabularies[src_name] = {
            "unique_count": len(counter),
            "top_teams": counter.most_common(50),
        }

    collisions_by_source = {}
    for src_name, buckets in buckets_by_source.items():
        collisions_by_source[src_name] = _collisions(buckets)

    cross_missing = []
    src_names = list(counts_by_source.keys())
    for i in range(len(src_names)):
        for j in range(i + 1, len(src_names)):
            cross_missing.append(_cross_source_missing(counts_by_source, src_names[i], src_names[j]))

    # Alias suggestions: prioritize predictions <-> backtest_joined, else predictions <-> snapshots if present
    suggested_aliases: List[Dict[str, object]] = []
    if "predictions" in buckets_by_source and "backtest_joined" in buckets_by_source:
        suggested_aliases = _suggest_aliases(buckets_by_source["predictions"], buckets_by_source["backtest_joined"])
    elif "predictions" in buckets_by_source and "snapshots" in buckets_by_source:
        suggested_aliases = _suggest_aliases(buckets_by_source["predictions"], buckets_by_source["snapshots"])

    report: Dict[str, object] = {
        "meta": {
            "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pred_dir": args.pred_dir,
            "snapshots_dir": args.snapshots_dir,
            "backtest_joined_path": args.backtest_joined,
            "date_min": args.date_min,
            "date_max": args.date_max,
            "inputs": {s.name: s.paths for s in sources},
        },
        "vocabularies": vocabularies,
        "collisions": collisions_by_source,
        "cross_source_missing": cross_missing,
        "suggested_aliases": suggested_aliases,
        "issues_sample": issues,
    }

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    json_path = out_dir / "team_canonicalization_report.json"
    md_path = out_dir / "team_canonicalization_report.md"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

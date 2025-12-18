from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class SnapshotFileQuality:
    file: str
    date: Optional[str]
    rows: int
    cols: int
    columns: List[str]
    missing_required_columns: List[str]
    duplicate_rows: int
    duplicate_key_rows: Optional[int]
    null_rates: Dict[str, float]
    nunique: Dict[str, int]
    constant_flags: Dict[str, bool]
    notes: List[str]


def parse_date_from_filename(path: str) -> Optional[str]:
    m = DATE_RE.search(Path(path).name)
    return m.group(1) if m else None


def safe_read_csv(path: str) -> pd.DataFrame:
    # Keep strings stable; prevent dtype inference surprises
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])


def compute_snapshot_quality(
    df: pd.DataFrame,
    *,
    file_path: str,
    required_columns: List[str],
    key_columns_candidates: List[Tuple[str, str, str]],
    columns_to_profile: List[str],
) -> SnapshotFileQuality:
    notes: List[str] = []
    date = parse_date_from_filename(file_path)

    cols = list(df.columns)
    missing_required = [c for c in required_columns if c not in cols]

    dup_rows = int(df.duplicated().sum()) if len(df) else 0

    # Attempt to compute duplicates on a key (home/away/date) if present
    dup_key_rows: Optional[int] = None
    key_cols_used: Optional[Tuple[str, str, str]] = None
    for hc, ac, dc in key_columns_candidates:
        if hc in cols and ac in cols and dc in cols:
            key_cols_used = (hc, ac, dc)
            dup_key_rows = int(df.duplicated(subset=[hc, ac, dc]).sum()) if len(df) else 0
            break
    if key_cols_used is None:
        notes.append("Could not find a standard (home, away, date) key column triplet to check key-duplicates.")

    # Profile null rates + nunique for selected columns if present
    null_rates: Dict[str, float] = {}
    nunique: Dict[str, int] = {}
    constant_flags: Dict[str, bool] = {}

    for c in columns_to_profile:
        if c not in cols:
            continue
        ser = df[c]
        # Treat "" as missing (since we read empty strings for missing values)
        missing = (ser.astype(str).str.strip() == "").sum()
        null_rate = float(missing) / float(max(len(df), 1))
        null_rates[c] = round(null_rate, 6)

        # nunique excluding blanks
        nonblank = ser.astype(str).str.strip()
        nun = int(nonblank[nonblank != ""].nunique(dropna=True))
        nunique[c] = nun

        constant_flags[c] = (nun <= 1 and (1.0 - null_rate) > 0.5)  # constant but actually populated

    return SnapshotFileQuality(
        file=Path(file_path).as_posix(),
        date=date,
        rows=int(len(df)),
        cols=int(len(cols)),
        columns=cols,
        missing_required_columns=missing_required,
        duplicate_rows=dup_rows,
        duplicate_key_rows=dup_key_rows,
        null_rates=null_rates,
        nunique=nunique,
        constant_flags=constant_flags,
        notes=notes,
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def audit_left_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    right_tag: str,
    required_right_cols: Optional[List[str]] = None,
    sample_n: int = 10,
) -> Dict[str, Any]:
    """
    Behavior-preserving: does not modify data.
    Produces coverage + missing key samples for observability.
    """
    if required_right_cols is None:
        required_right_cols = []

    left_keys = left[key].astype(str)
    right_keys = set(right[key].astype(str).unique().tolist())

    missing_mask = ~left_keys.isin(right_keys)
    missing_keys = left.loc[missing_mask, key].astype(str).head(sample_n).tolist()

    coverage = 1.0 - (missing_mask.mean() if len(left) else 0.0)

    col_coverage = {}
    for c in required_right_cols:
        if c in left.columns:
            # if merged frame already, non-null rate tells if join filled it
            col_coverage[c] = float(left[c].notna().mean())
        else:
            col_coverage[c] = None

    return {
        "key": key,
        "right_tag": right_tag,
        "rows_left": int(len(left)),
        "rows_right": int(len(right)),
        "coverage": float(coverage),
        "missing_keys_sample": missing_keys,
        "required_right_cols_nonnull_rate": col_coverage,
    }

from __future__ import annotations

import math
from typing import Optional, Tuple


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
    - abs(x) >= 100  (reject decimal-like odds)
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


def expected_value_units(p_win: Optional[float], american_odds: Optional[float]) -> Optional[float]:
    """
    EV in units for a 1u stake.
      EV = p * profit_if_win - (1-p) * 1
    """
    if p_win is None:
        return None
    try:
        p = float(p_win)
    except Exception:
        return None
    if not (0.0 < p < 1.0) or math.isnan(p) or math.isinf(p):
        return None

    ppu = win_profit_per_unit_american(american_odds)
    if ppu is None:
        return None

    return p * float(ppu) - (1.0 - p)

from typing import Optional
import math

PPU_TOTAL_MINUS_110 = 100.0 / 110.0


def _to_float(x):
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def expected_value_total(p: Optional[float]) -> Optional[float]:
    p = _to_float(p)
    if p is None or not (0.0 < p < 1.0):
        return None
    return p * PPU_TOTAL_MINUS_110 - (1.0 - p)

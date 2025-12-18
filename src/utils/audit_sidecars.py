# src/utils/audit_sidecars.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_unseen_teams_sidecar(
    out_dir: str,
    game_date: str,
    *,
    unseen_rows: List[Dict[str, Any]],
    history_team_universe: Optional[Iterable[str]] = None,
    history_rows: Optional[int] = None,
) -> str:
    """
    Writes a deterministic sidecar JSON describing unseen-team events.
    Does not alter core outputs or schemas.

    Returns the path written.
    """
    p = Path(out_dir) / "audits"
    p.mkdir(parents=True, exist_ok=True)

    universe_list: List[str] = []
    if history_team_universe is not None:
        universe_list = sorted({str(x).strip().lower() for x in history_team_universe if str(x).strip()})

    payload: Dict[str, Any] = {
        "game_date": game_date,
        "unseen_rows_count": len(unseen_rows),
        "unseen_rows": unseen_rows[:500],  # hard cap for safety
        "history_rows": history_rows,
        "history_team_universe_count": len(universe_list) if universe_list else None,
        "history_team_universe_sample": universe_list[:50] if universe_list else None,
        "history_team_universe_sha256": _sha256_text("|".join(universe_list)) if universe_list else None,
    }

    out_path = p / f"unseen_teams_{game_date}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(out_path)

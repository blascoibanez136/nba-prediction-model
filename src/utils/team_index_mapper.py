from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from src.ingest.team_normalizer import normalize_team_name as normalize_franchise_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeamIndexMapping:
    """
    Mapping between:
      - franchise_canonical: canonical franchise identity (lowercase full name)
      - team_index_key: exact key used by models/team_index.json
    """
    franchise_canonical: str
    team_index_key: str


class TeamIndexMapper:
    """
    Deterministic mapper from arbitrary raw team names -> exact team_index.json keys.

    Strategy:
      raw_name -> franchise_canonical (via src.ingest.team_normalizer.normalize_team_name)
      franchise_canonical -> team_index_key (reverse map built from team_index keys)
    """

    def __init__(self, team_index_path: Path):
        self.team_index_path = team_index_path
        self._team_index: Dict[str, int] = {}
        self._key_by_franchise: Dict[str, str] = {}

    def load(self) -> None:
        if not self.team_index_path.exists():
            raise FileNotFoundError(f"Missing team_index.json: {self.team_index_path}")

        self._team_index = json.loads(self.team_index_path.read_text(encoding="utf-8"))

        # Build reverse map: franchise_canonical -> exact team_index key
        key_by_franchise: Dict[str, str] = {}
        collisions: Dict[str, list[str]] = {}

        for k in self._team_index.keys():
            franchise = normalize_franchise_name(k)  # e.g. "LA Clippers" -> "los angeles clippers"
            if not franchise:
                continue

            if franchise in key_by_franchise and key_by_franchise[franchise] != k:
                # Collision is extremely unlikely in NBA, but we record it explicitly.
                collisions.setdefault(franchise, []).extend([key_by_franchise[franchise], k])
                # Keep first seen for determinism.
                continue

            key_by_franchise[franchise] = k

        self._key_by_franchise = key_by_franchise

        if collisions:
            # This should never happen for NBA; log loudly if it does.
            logger.error("[team_index_mapper] collisions detected in franchise->key mapping: %s", collisions)

        logger.info(
            "[team_index_mapper] loaded team_index=%d franchise_keys=%d",
            len(self._team_index),
            len(self._key_by_franchise),
        )

    @property
    def team_index(self) -> Dict[str, int]:
        return self._team_index

    def to_team_index_key(self, raw_name: object) -> Optional[str]:
        """
        Convert raw team name to exact key used by team_index.json.
        Returns None if empty input.
        Returns a deterministic fallback (canonical franchise string) if not found,
        so callers can treat it as 'unseen' in the model universe.
        """
        if raw_name is None:
            return None
        raw = str(raw_name).strip()
        if not raw:
            return None

        franchise = normalize_franchise_name(raw)  # canonical lowercase full name
        if not franchise:
            return None

        mapped = self._key_by_franchise.get(franchise)
        if mapped is not None:
            return mapped

        # Deterministic fallback (will be unseen in team_index)
        return franchise

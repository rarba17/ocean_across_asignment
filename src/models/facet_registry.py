"""
facet_registry.py — Loads and caches facets from the processed JSON file.

Design principle: the registry is fully data-driven. Adding 5000 facets
requires only updating the JSON file — no code changes needed anywhere.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Optional, Dict
from functools import lru_cache

from src.models.schemas import FacetMeta

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "facets_processed.json"


class FacetRegistry:
    def __init__(self, json_path: Path):
        self._facets: Dict[int, FacetMeta] = {}
        self._load(json_path)

    def _load(self, json_path: Path) -> None:
        if not json_path.exists():
            raise FileNotFoundError(
                f"Facets JSON not found at {json_path}. "
                "Set FACETS_JSON_PATH env var to the correct path."
            )
        with open(json_path, "r", encoding="utf-8") as f:
            raw: list = json.load(f)
        for item in raw:
            meta = FacetMeta(**item)
            self._facets[meta.facet_id] = meta

    # ── Query helpers ──────────────────────────────────────────────

    def get(self, facet_id: int) -> Optional[FacetMeta]:
        return self._facets.get(facet_id)

    def get_all(self) -> List[FacetMeta]:
        return list(self._facets.values())

    def get_by_ids(self, ids: List[int]) -> List[FacetMeta]:
        return [self._facets[i] for i in ids if i in self._facets]

    def get_by_category(self, category: str) -> List[FacetMeta]:
        return [f for f in self._facets.values() if f.category == category]

    def observable_facets(self) -> List[FacetMeta]:
        return [f for f in self._facets.values() if f.is_observable_in_text]

    def count(self) -> int:
        return len(self._facets)


@lru_cache(maxsize=1)
def get_registry() -> FacetRegistry:
    path_str = os.environ.get("FACETS_JSON_PATH", str(_DEFAULT_PATH))
    return FacetRegistry(Path(path_str))

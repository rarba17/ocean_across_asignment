"""
facet_registry.py Loads and caches facets from the processed JSON file.

Design principle: the registry is fully data-driven. Adding 5000 facets
requires only updating the JSON file  no code changes needed anywhere.
"""


from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Optional, Dict
from functools import lru_cache

from src.models.schemas import FacetMeta
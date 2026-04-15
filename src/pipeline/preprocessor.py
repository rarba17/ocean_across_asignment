"""
preprocessor.py — Cleans conversation turns and prepares facet batches.

Key responsibility: chunk the (potentially large) facet list into batches
of ≤50 so each LLM call stays within a safe context window. This is what
allows the system to scale to 5000+ facets without redesign.
"""

from __future__ import annotations
import re
from typing import List, Tuple

from src.models.schemas import ConversationTurn, FacetMeta
from src.models.facet_registry import get_registry


# Maximum number of facets to send in a single LLM call.
# Keep this ≤50 to stay under ~4096 tokens per prompt.
BATCH_SIZE = int(50)


def clean_text(text: str) -> str:
    """
    Basic text normalisation:
    - Collapse excessive whitespace
    - Strip control characters
    - Normalize unicode quotes
    """
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return text


def resolve_facets(turn: ConversationTurn) -> List[FacetMeta]:
    """
    Resolve which facets to evaluate for a given turn.
    - If turn.facet_ids is provided → use those
    - Otherwise → use all registered facets
    """
    registry = get_registry()
    if turn.facet_ids:
        return registry.get_by_ids(turn.facet_ids)
    return registry.get_all()


def chunk_facets(
    facets: List[FacetMeta], batch_size: int = BATCH_SIZE
) -> List[List[FacetMeta]]:
    """
    Split a list of facets into sub-lists of at most `batch_size`.
    This is the core mechanism that lets us handle arbitrarily many facets:
    we simply generate more batches, each processed in parallel.
    """
    return [facets[i : i + batch_size] for i in range(0, len(facets), batch_size)]


def prepare_turn(turn: ConversationTurn) -> Tuple[ConversationTurn, List[List[FacetMeta]]]:
    """
    Clean the turn text and split its facets into batches.
    Returns (cleaned_turn, list_of_facet_batches).
    """
    cleaned = turn.copy(update={
        "text": clean_text(turn.text),
        "context": [clean_text(c) for c in turn.context],
    })
    facets = resolve_facets(cleaned)
    batches = chunk_facets(facets)
    return cleaned, batches


def build_context_block(turn: ConversationTurn, max_context_turns: int = 3) -> str:
    """
    Format the conversation context for inclusion in the prompt.
    Limits to the most recent `max_context_turns` to control token use.
    """
    recent = turn.context[-max_context_turns:]
    if not recent:
        return "No prior context."
    lines = [f"[Prior turn {i+1}]: {c}" for i, c in enumerate(recent)]
    return "\n".join(lines)

"""
dispatcher.py — Async parallel dispatcher for facet batch evaluation.

For N facets split into B batches of ≤50, launches all B LLM calls
concurrently (bounded by MAX_CONCURRENT_BATCHES to avoid overwhelming
the model server).

This is the core of the scalability story: adding more facets just means
more batches, and each batch is an independent async task.
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import List

from src.models.schemas import (
    ConversationTurn, FacetScore, TurnEvaluationResult, LLMFacetOutput
)
from src.models.facet_registry import get_registry
from src.pipeline.preprocessor import prepare_turn
from src.pipeline.scorer import LLMScorer

logger = logging.getLogger(__name__)

# Limit concurrent batch calls to avoid saturating the LLM server
MAX_CONCURRENT_BATCHES = 5

SCORE_LABELS = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}


async def evaluate_turn(
    turn: ConversationTurn,
    scorer: LLMScorer,
) -> TurnEvaluationResult:
    """
    Full pipeline for a single conversation turn:
    1. Clean text + split into facet batches
    2. Dispatch all batches concurrently (semaphore-bounded)
    3. Aggregate results into a TurnEvaluationResult
    """
    t0 = time.time()
    registry = get_registry()

    # Step 1: preprocess
    cleaned_turn, facet_batches = prepare_turn(turn)

    # Step 2: concurrent batch dispatch
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)

    async def _score_one_batch(batch_facets):
        async with semaphore:
            return await scorer.score_batch(cleaned_turn, batch_facets)

    tasks = [_score_one_batch(batch) for batch in facet_batches]
    batch_results: List[List[LLMFacetOutput]] = await asyncio.gather(*tasks)

    # Step 3: flatten + enrich with facet metadata
    all_outputs = [item for sublist in batch_results for item in sublist]

    facet_scores: List[FacetScore] = []
    for output in all_outputs:
        meta = registry.get(output.facet_id)
        if meta is None:
            continue
        facet_scores.append(FacetScore(
            facet_id=output.facet_id,
            name=meta.name,
            category=meta.category,
            score=output.score,
            confidence=output.confidence,
            rationale=output.rationale,
            score_label=SCORE_LABELS[output.score],
        ))

    # Sort by facet_id for deterministic output
    facet_scores.sort(key=lambda s: s.facet_id)

    latency_ms = (time.time() - t0) * 1000

    logger.info(
        "Evaluated turn %s: %d facets in %d batches (%.0f ms)",
        turn.turn_id, len(facet_scores), len(facet_batches), latency_ms,
    )

    return TurnEvaluationResult(
        turn_id=cleaned_turn.turn_id,
        speaker=cleaned_turn.speaker,
        text_preview=cleaned_turn.text[:120],
        scores=facet_scores,
        total_facets_evaluated=len(facet_scores),
        latency_ms=round(latency_ms, 1),
        model=scorer.model,
        batch_count=len(facet_batches),
    )


async def evaluate_batch(
    turns: List[ConversationTurn],
    scorer: LLMScorer,
    max_concurrent_turns: int = 3,
) -> List[TurnEvaluationResult]:
    """
    Evaluate multiple turns, with bounded concurrency across turns.
    Each turn still fans out to concurrent batch calls internally.
    """
    semaphore = asyncio.Semaphore(max_concurrent_turns)

    async def _eval_one(t):
        async with semaphore:
            return await evaluate_turn(t, scorer)

    results = await asyncio.gather(*[_eval_one(t) for t in turns])
    return list(results)

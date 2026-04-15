"""
aggregator.py — Post-processing utilities for evaluation results.

Provides summary statistics, category breakdowns, and comparison
utilities useful for benchmarking across multiple conversations.
"""

from __future__ import annotations
from typing import List, Dict
from collections import defaultdict
import statistics

from src.models.schemas import TurnEvaluationResult, FacetScore


def summarize_turn(result: TurnEvaluationResult) -> Dict:
    """
    Produce a summary dict for a single turn's evaluation result.
    Includes per-category averages and overall statistics.
    """
    scores = result.scores
    if not scores:
        return {"turn_id": result.turn_id, "summary": "No scores"}

    all_scores = [s.score for s in scores]
    all_confs = [s.confidence for s in scores]

    by_category: Dict[str, List[FacetScore]] = defaultdict(list)
    for s in scores:
        by_category[s.category].append(s)

    cat_summary = {}
    for cat, cat_scores in by_category.items():
        vals = [s.score for s in cat_scores]
        cat_summary[cat] = {
            "count": len(vals),
            "mean": round(statistics.mean(vals), 2),
            "max": max(vals),
            "min": min(vals),
            "top_facets": sorted(cat_scores, key=lambda x: -x.score)[:3],
        }

    return {
        "turn_id": result.turn_id,
        "speaker": result.speaker,
        "overall": {
            "mean_score": round(statistics.mean(all_scores), 2),
            "mean_confidence": round(statistics.mean(all_confs), 2),
            "std_score": round(statistics.stdev(all_scores), 2) if len(all_scores) > 1 else 0.0,
            "high_signal_facets": [
                s for s in scores if s.score >= 4
            ],
            "low_signal_facets": [
                s for s in scores if s.score <= 2
            ],
        },
        "by_category": cat_summary,
        "latency_ms": result.latency_ms,
        "model": result.model,
    }


def compare_turns(results: List[TurnEvaluationResult]) -> Dict:
    """
    Compare multiple turn results side-by-side.
    Returns a facet-aligned comparison dict.
    """
    if not results:
        return {}

    # Build facet_id ->list of (turn_id, score) for all turns
    facet_map: Dict[int, List[Dict]] = defaultdict(list)
    for r in results:
        for s in r.scores:
            facet_map[s.facet_id].append({
                "turn_id": r.turn_id,
                "score": s.score,
                "confidence": s.confidence,
            })

    # Find facets with highest variance across turns (most discriminative)
    discriminative = []
    for fid, entries in facet_map.items():
        if len(entries) > 1:
            vals = [e["score"] for e in entries]
            var = statistics.variance(vals)
            if var > 0:
                discriminative.append({"facet_id": fid, "variance": round(var, 3), "entries": entries})

    discriminative.sort(key=lambda x: -x["variance"])

    return {
        "turn_count": len(results),
        "total_facets": len(facet_map),
        "most_discriminative_facets": discriminative[:20],
    }

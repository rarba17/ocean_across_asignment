"""
main.py — FastAPI application for the Ahoum conversation evaluation benchmark.

Endpoints:
  POST /evaluate          → Score a single conversation turn
  POST /evaluate/batch    → Score multiple turns
  GET  /facets            → List all facets
  GET  /facets/{id}       → Get a single facet
  GET  /health            → Health check
"""

from __future__ import annotations
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.models.schemas import (
    ConversationTurn,
    BatchEvaluationRequest,
    TurnEvaluationResult,
    BatchEvaluationResult,
    FacetMeta,
)
from src.models.facet_registry import get_registry
from src.pipeline.scorer import LLMScorer
from src.pipeline.dispatcher import evaluate_turn, evaluate_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# App lifecycle — scorer singleton
# ─────────────────────────────────────────────

scorer: Optional[LLMScorer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer
    scorer = LLMScorer(
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
        model=os.environ.get("LLM_MODEL", "qwen2:7b-instruct"),
        api_key=os.environ.get("LLM_API_KEY", "ollama"),
        request_logprobs=os.environ.get("REQUEST_LOGPROBS", "true").lower() == "true",
    )
    logger.info("LLMScorer initialised: model=%s", scorer.model)
    yield
    if scorer:
        await scorer.close()


app = FastAPI(
    title="Ahoum Conversation Evaluation API",
    description="Score conversation turns across 300–5000 psychological/behavioral facets.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    registry = get_registry()
    return {
        "status": "ok",
        "facets_loaded": registry.count(),
        "model": scorer.model if scorer else "not initialised",
    }


@app.get("/facets", response_model=List[FacetMeta])
async def list_facets(
    category: Optional[str] = Query(default=None, description="Filter by category"),
    observable_only: bool = Query(default=False),
):
    registry = get_registry()
    if category:
        facets = registry.get_by_category(category)
    elif observable_only:
        facets = registry.observable_facets()
    else:
        facets = registry.get_all()
    return facets


@app.get("/facets/{facet_id}", response_model=FacetMeta)
async def get_facet(facet_id: int):
    registry = get_registry()
    meta = registry.get(facet_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Facet {facet_id} not found")
    return meta


@app.post("/evaluate", response_model=TurnEvaluationResult)
async def evaluate_single(turn: ConversationTurn):
    if scorer is None:
        raise HTTPException(status_code=503, detail="Scorer not ready")
    try:
        result = await evaluate_turn(turn, scorer)
        return result
    except Exception as e:
        logger.exception("Error evaluating turn %s", turn.turn_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/batch", response_model=BatchEvaluationResult)
async def evaluate_batch_endpoint(request: BatchEvaluationRequest):
    if scorer is None:
        raise HTTPException(status_code=503, detail="Scorer not ready")
    if len(request.turns) > 20:
        raise HTTPException(
            status_code=422,
            detail="Batch endpoint accepts at most 20 turns. Split larger batches."
        )
    t0 = time.time()
    try:
        results = await evaluate_batch(request.turns, scorer)
        total_ms = (time.time() - t0) * 1000
        return BatchEvaluationResult(
            results=results,
            total_turns=len(results),
            total_facets_evaluated=sum(r.total_facets_evaluated for r in results),
            total_latency_ms=round(total_ms, 1),
        )
    except Exception as e:
        logger.exception("Error in batch evaluation")
        raise HTTPException(status_code=500, detail=str(e))

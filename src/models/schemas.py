from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class FacetMeta(BaseModel):
    facet_id: int
    name: str
    category: str
    is_observable_in_text: bool
    prompt_hint: str
    score_scale: Dict[str, str] = Field(
        default={
            "1": "Very Low",
            "2": "Low",
            "3": "Moderate",
            "4": "High",
            "5": "Very High",
        }
    )

# input models

class ConversationTurn(BaseModel):
    turn_id: str = Field(..., description="Unique identifier for this turn")
    speaker: str = Field(..., description="'user' or 'assistant' or speaker name")
    text: str = Field(..., description="The spoken/written text of this turn")
    context: List[str] = Field(
        default_factory=list,
        description="Prior turn texts (oldest first) for context window"
    )
    facet_ids: Optional[List[int]] = Field(
        default=None,
        description="Subset of facet IDs to evaluate; None means all"
    )

    @validator("text")
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Turn text must not be empty")
        return v.strip()


class BatchEvaluationRequest(BaseModel):
    turns: List[ConversationTurn]

# output / score models

class FacetScore(BaseModel):
    facet_id: int
    name: str
    category: str
    score: int = Field(..., ge=1, le=5, description="Score 1–5")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence 0–1")
    rationale: str = Field(..., description="Brief reasoning for this score")
    score_label: str = Field(..., description="Human-readable label for the score")


class TurnEvaluationResult(BaseModel):
    turn_id: str
    speaker: str
    text_preview: str = Field(..., description="First 120 chars of the turn text")
    scores: List[FacetScore]
    total_facets_evaluated: int
    latency_ms: float
    model: str
    batch_count: int = Field(..., description="Number of LLM batches used")


class BatchEvaluationResult(BaseModel):
    results: List[TurnEvaluationResult]
    total_turns: int
    total_facets_evaluated: int
    total_latency_ms: float


# llm internal schema -> structured output

class LLMFacetOutput(BaseModel):
    """Schema the LLM must return for each facet in a batch."""
    facet_id: int
    score: int = Field(..., ge=1, le=5)
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str


class LLMBatchOutput(BaseModel):
    """Top-level JSON the LLM returns per batch call."""
    scores: List[LLMFacetOutput]

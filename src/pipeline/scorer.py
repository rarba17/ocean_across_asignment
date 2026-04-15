"""
scorer.py — LLM-based scoring for a single facet batch.

Prompting strategy (satisfies "no one-shot" constraint):
  Turn 0  [system]   → Role definition + scoring rubric
  Turn 1  [user]     → Conversation text + context
  Turn 2  [assistant]→ (Few-shot example response, hardcoded)
  Turn 3  [user]     → Actual facet batch to score
  Turn 4  [assistant]→ Model's structured JSON output

Confidence is derived from the logprobs of each score digit token.
If logprobs are unavailable (e.g., Ollama), confidence defaults to 0.5.
"""

from __future__ import annotations
import json
import time
import logging
import math
from typing import List, Optional, Dict, Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.schemas import (
    ConversationTurn, FacetMeta, LLMFacetOutput, LLMBatchOutput
)
from src.pipeline.preprocessor import build_context_block

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional conversation analyst with deep expertise in psychology, linguistics, and behavioral science.

Your task is to evaluate conversation turns on specific psychological and behavioral facets.

## Scoring Scale
| Score | Label      | Description                                              |
|-------|------------|----------------------------------------------------------|
| 1     | Very Low   | Trait barely or not present; absent from this turn       |
| 2     | Low        | Slightly present; faint traces detectable                |
| 3     | Moderate   | Clearly present to a typical degree                      |
| 4     | High       | Strongly and consistently expressed in this turn         |
| 5     | Very High  | Extremely pronounced; a defining feature of this turn    |

## Instructions
1. Read the conversation context and target turn carefully.
2. For each facet you are given, reason briefly about the evidence, then assign a score 1–5.
3. Also provide a confidence value between 0.0 and 1.0 reflecting how certain you are.
4. Return ONLY valid JSON matching this exact schema — no markdown fences, no preamble:
   {"scores": [{"facet_id": <int>, "score": <1-5>, "confidence": <0.0-1.0>, "rationale": "<brief reason>"}]}

## Important Notes
- Base your score ONLY on what is observable in the provided text.
- If a facet is not inferable from the text at all, score it 3 (Moderate) with low confidence (0.2).
- Rationale must be 1–2 concise sentences.
- Never refuse or skip a facet — always produce a score.
"""

FEW_SHOT_USER = """Here is an example evaluation request.

[Context]: "I've been really stressed about the project deadline."

[Target Turn]: Speaker: user | Text: "I've decided to just wing it and submit whatever I have. Can't keep worrying."

[Facets to Score]:
- facet_id=1, name="Risktaking", hint="Evaluate the degree of Risktaking expressed in this conversation turn."
- facet_id=6, name="Hesitation", hint="Evaluate the degree of Hesitation expressed in this conversation turn."
"""

FEW_SHOT_ASSISTANT = """{"scores": [{"facet_id": 1, "score": 4, "confidence": 0.88, "rationale": "Speaker explicitly chooses to submit without preparation, indicating willingness to accept uncertain outcomes."}, {"facet_id": 6, "score": 2, "confidence": 0.82, "rationale": "Speaker has resolved their uncertainty and commits to action, showing little residual hesitation."}]}"""


def build_batch_user_prompt(turn: ConversationTurn, facets: List[FacetMeta]) -> str:
    context_block = build_context_block(turn)
    facet_lines = "\n".join(
        f'- facet_id={f.facet_id}, name="{f.name}", hint="{f.prompt_hint}"'
        for f in facets
    )
    return f"""[Context]:
{context_block}

[Target Turn]:
Speaker: {turn.speaker} | Text: "{turn.text}"

[Facets to Score]:
{facet_lines}"""


# ─────────────────────────────────────────────
# LLM Client
# ─────────────────────────────────────────────

class LLMScorer:
    """
    Scores a batch of facets for a single conversation turn.
    Connects to any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",  # default: Ollama
        model: str = "qwen2:7b-instruct",
        api_key: str = "ollama",  # vLLM requires a key; Ollama accepts anything
        timeout: float = 120.0,
        request_logprobs: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.request_logprobs = request_logprobs
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def score_batch(
        self,
        turn: ConversationTurn,
        facets: List[FacetMeta],
    ) -> List[LLMFacetOutput]:
        """
        Run multi-turn prompting for a single batch of facets.
        Returns a list of LLMFacetOutput (one per facet in the batch).
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",      "content": FEW_SHOT_USER},
            {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
            {"role": "user",      "content": build_batch_user_prompt(turn, facets)},
        ]

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,      # near-deterministic for reliability
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},  # vLLM structured output
        }

        if self.request_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5

        t0 = time.time()
        resp = await self.client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        elapsed = (time.time() - t0) * 1000

        raw_content = data["choices"][0]["message"]["content"]
        logprobs_data = data["choices"][0].get("logprobs")

        parsed = self._parse_response(raw_content, facets)
        parsed = self._inject_confidence(parsed, logprobs_data)

        logger.debug(
            "Scored %d facets in %.0f ms (model=%s)", len(parsed), elapsed, self.model
        )
        return parsed

    def _parse_response(
        self, raw: str, facets: List[FacetMeta]
    ) -> List[LLMFacetOutput]:
        """Parse the LLM JSON output; fall back to default scores on failure."""
        facet_id_set = {f.facet_id for f in facets}
        try:
            data = json.loads(raw)
            batch = LLMBatchOutput(**data)
            # Keep only scores for facets we asked about
            valid = [s for s in batch.scores if s.facet_id in facet_id_set]
            # Fill any missing facets with a default
            scored_ids = {s.facet_id for s in valid}
            for f in facets:
                if f.facet_id not in scored_ids:
                    valid.append(LLMFacetOutput(
                        facet_id=f.facet_id, score=3, confidence=0.2,
                        rationale="Facet not inferable from text; default applied."
                    ))
            return valid
        except Exception as e:
            logger.warning("Failed to parse LLM output: %s — using defaults. Raw: %s", e, raw[:200])
            return [
                LLMFacetOutput(
                    facet_id=f.facet_id, score=3, confidence=0.1,
                    rationale="Parse error; default score applied."
                )
                for f in facets
            ]

    def _inject_confidence(
        self,
        outputs: List[LLMFacetOutput],
        logprobs_data: Optional[Dict],
    ) -> List[LLMFacetOutput]:
        """
        If logprobs are available, compute confidence from the probability
        distribution over score tokens ("1"–"5") at each score position.
        
        If not available (e.g., Ollama without logprobs), keep existing confidence.
        """
        if not logprobs_data:
            return outputs

        # Extract per-token logprobs from the response
        try:
            tokens = logprobs_data.get("content", [])
            score_logprobs = [
                t for t in tokens
                if t.get("token", "").strip() in {"1", "2", "3", "4", "5"}
            ]
            # Map score position index → confidence
            for i, output in enumerate(outputs):
                if i < len(score_logprobs):
                    top = score_logprobs[i].get("top_logprobs", [])
                    if top:
                        chosen_logp = next(
                            (t["logprob"] for t in top if t["token"].strip() == str(output.score)),
                            None,
                        )
                        if chosen_logp is not None:
                            # Convert log-prob to probability
                            prob = math.exp(chosen_logp)
                            output = output.copy(update={"confidence": round(prob, 3)})
                            outputs[i] = output
        except Exception as e:
            logger.debug("Could not extract logprob confidence: %s", e)

        return outputs

    async def close(self):
        await self.client.aclose()

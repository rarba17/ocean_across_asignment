# Ahoum — Conversation Evaluation Benchmark

A production-ready system that scores every conversation turn across **399 psychological, linguistic, and behavioural facets** using a local open-weight LLM. Designed to scale to 5 000+ facets without any architectural changes.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Facet Dataset](#facet-dataset)
- [Hard Constraints Checklist](#hard-constraints-checklist)
- [Quick Start (Docker)](#quick-start-docker)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)
- [Deliverables](#deliverables)

---

## Overview

Given a conversation turn (and optional prior context), the system:

1. Resolves which facets to evaluate (all 399 by default, or a caller-specified subset)
2. Splits them into batches of ≤ 50 to stay within the LLM's context window
3. Dispatches all batches **concurrently** via an async semaphore
4. Returns a structured score (1–5), confidence (0–1), and rationale for every facet

The architecture is fully data-driven: adding more facets means appending rows to `data/processed/facets_processed.json` — no code changes required.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  FastAPI  (src/api/main.py)                             │
│  POST /evaluate · POST /evaluate/batch · GET /facets   │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │  Dispatcher         │  ← async parallel fan-out
          │  (dispatcher.py)    │
          └──────────┬──────────┘
                     │  asyncio.gather  (≤5 concurrent)
          ┌──────────▼──────────┐
          │  LLMScorer          │  ← multi-turn prompting
          │  (scorer.py)        │     system + few-shot + query
          └──────────┬──────────┘
                     │  HTTP  (OpenAI-compatible)
          ┌──────────▼──────────┐
          │  Ollama / vLLM      │  ← Qwen2-7B-Instruct (default)
          └─────────────────────┘
```

**Scaling story:** 5 000 facets → 100 batches of 50 → all 100 dispatched concurrently (bounded by `MAX_CONCURRENT_BATCHES=5`). No redesign needed; just more batches.

---

## Facet Dataset

Source: `data/raw/Facets_Assignment.csv` — 399 facets preprocessed into `data/processed/facets_processed.json`.

### Added columns (beyond the raw name)

| Column | Type | Description |
|---|---|---|
| `facet_id` | int | Stable numeric ID for indexing and API lookup |
| `category` | str | Domain grouping (see below) |
| `is_observable_in_text` | bool | Whether the facet can be inferred from speech/text alone |
| `prompt_hint` | str | Facet-specific instruction injected into the LLM prompt |
| `score_scale` | dict | Human-readable label for each integer score (1–5) |

### Category breakdown

| Category | Count | Observable |
|---|---|---|
| personality | 135 | mostly yes |
| cognition | 55 | mixed |
| lifestyle | 42 | **no** — behavioural counts, not inferable from text |
| spiritual | 39 | mixed — practice *counts* are not observable |
| social | 33 | mostly yes |
| emotion | 31 | yes |
| pragmatics | 17 | yes |
| biological | 15 | **no** — lab values, physiological measurements |
| motivation | 12 | yes |
| leadership | 8 | yes |
| linguistic_quality | 7 | yes |
| safety | 5 | mostly yes |

The `is_observable_in_text` flag lets callers filter to facets that are actually inferable from a transcript (`GET /facets?observable_only=true`), avoiding noise from biological or lifestyle metrics.

---

## Quick Start (Docker)

**Prerequisites:** Docker Engine 25+ with the Compose v2 plugin.

```bash
# Clone and enter the repo
git clone <repo-url>
cd ocean_across

# Bring up Ollama + API + UI
docker compose up --build
```

On first run Ollama will pull `qwen2:7b-instruct` (~4 GB). The API starts after Ollama passes its healthcheck.

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Interactive docs | http://localhost:8000/docs |
| UI | http://localhost:3000 |
| Ollama (host access) | http://localhost:11435 |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://ollama:11434/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen2:7b-instruct` | Model name |
| `LLM_API_KEY` | `ollama` | API key (Ollama ignores this; vLLM requires it) |
| `REQUEST_LOGPROBS` | `false` | Set `true` for vLLM to get logprob-based confidence |
| `FACETS_JSON_PATH` | `/app/data/processed/facets_processed.json` | Path to facets file |

### Switching to vLLM (GPU)

Uncomment the `vllm` service in `docker-compose.yml` and set `LLM_BASE_URL=http://vllm:8000/v1`. Requires NVIDIA GPU with CUDA 12+.

---

## API Reference

### `POST /evaluate`

Score a single conversation turn on all (or a subset of) facets.

```json
{
  "turn_id": "t1",
  "speaker": "user",
  "text": "I've decided to just wing it and submit whatever I have.",
  "context": ["I've been really stressed about the project deadline."],
  "facet_ids": [1, 6, 10]
}
```

**Response** — `TurnEvaluationResult`:

```json
{
  "turn_id": "t1",
  "speaker": "user",
  "text_preview": "I've decided to just wing it...",
  "scores": [
    {
      "facet_id": 1,
      "name": "Risktaking",
      "category": "personality",
      "score": 4,
      "confidence": 0.88,
      "rationale": "Speaker explicitly chooses to submit without preparation.",
      "score_label": "High"
    }
  ],
  "total_facets_evaluated": 3,
  "latency_ms": 820.4,
  "model": "qwen2:7b-instruct",
  "batch_count": 1
}
```

### `POST /evaluate/batch`

Score up to 20 turns in one request. Turns are evaluated concurrently (bounded by `max_concurrent_turns=3`).

### `GET /facets`

List all facets. Optional query params:

- `?category=emotion` — filter by category
- `?observable_only=true` — only facets inferable from text

### `GET /facets/{id}`

Get a single facet by ID.

### `GET /health`

Returns `{"status": "ok", "facets_loaded": 399, "model": "qwen2:7b-instruct"}`.

---

## Project Structure

```
ocean_across/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI app + routes
│   ├── models/
│   │   ├── schemas.py           # Pydantic models
│   │   └── facet_registry.py    # Facet loader + LRU-cached singleton
│   ├── pipeline/
│   │   ├── preprocessor.py      # Text cleaning + facet batching
│   │   ├── scorer.py            # LLM client + multi-turn prompting
│   │   ├── dispatcher.py        # Async parallel batch dispatcher
│   │   └── aggregators.py       # Per-category summaries + turn comparison
│   └── ui/
│       └── index.html           # Single-page evaluator UI
├── data/
│   ├── raw/
│   │   └── Facets_Assignment.csv
│   ├── processed/
│   │   └── facets_processed.json  # Enriched facet definitions
│   └── conversations/
│       ├── conv_001.json … conv_050.json
│       └── sample_conversations_scored.json
├── configs/
│   └── model_config.yaml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Design Decisions

### Multi-turn prompting (satisfies "no one-shot" constraint)

Every LLM call uses a 4-message conversation:

```
[system]    Role definition + scoring rubric
[user]      Few-shot example request
[assistant] Few-shot example response (hardcoded gold output)
[user]      Actual facet batch to score
```

This gives the model a concrete output format to imitate before it sees real data, which substantially improves JSON compliance and score calibration compared to a single-shot prompt.

### Batching (satisfies "≥ 5 000 facets" constraint)

Facets are chunked into batches of ≤ 50 (`BATCH_SIZE` in `preprocessor.py`). Each batch is an independent async task. Adding more facets increases the number of batches linearly with no code changes. `MAX_CONCURRENT_BATCHES=5` prevents saturating the model server.

### Confidence

When `REQUEST_LOGPROBS=true` (vLLM), confidence is computed from the probability distribution over score tokens (`"1"`–`"5"`) at each score position in the output, converted from log-probability via `math.exp()`. When logprobs are unavailable (Ollama), the model's self-reported confidence in its JSON output is used.

### `is_observable_in_text` flag

123 of the 399 facets are not inferable from a text transcript (biological measurements, practice frequency counts, astrological attributes, etc.). Marking them `is_observable_in_text: false` allows the UI and API callers to skip them, preventing the model from producing meaningless scores on unobservable dimensions.

---

## Deliverables

| Deliverable | Location |
|---|---|
| GitHub repository | this repo |
| 50+ scored conversations | `data/conversations/` (50 files) + `data/conversations/sample_conversations_scored.json` |
| Dockerised baseline | `Dockerfile` + `docker-compose.yml` |
| Sample UI | `src/ui/index.html` (served on port 3000) |

# Mumzworld Returns Intelligence

**Track A — AI Engineering Intern**

A production-ready return reason classifier for Mumzworld that processes free-text customer messages in English and Arabic, classifies them into `refund | exchange | store_credit | escalate`, validates against a strict Pydantic schema, expresses uncertainty explicitly, and drafts a contextual reply in the customer's language.

---

## Summary

Mumzworld serves Arabic and English-speaking mothers across the GCC. Return requests arrive as free-text messages in both languages — often vague, sometimes angry, occasionally off-topic. This system:

- **Classifies** the return intent with a confidence score
- **Extracts** structured fields (product issue, sentiment, urgency, resolution preference)
- **Escalates honestly** when confidence is below 0.65 rather than guessing
- **Replies** in the customer's language with a warm, drafted message
- **Validates** every output against a Pydantic schema — no silent failures

---

## Setup & Run (under 5 minutes)

### Prerequisites
- Python 3.11+
- A free OpenRouter account → [openrouter.ai](https://openrouter.ai) (takes ~2 minutes)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/mumzworld-returns.git
cd mumzworld-returns
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and paste your OPENROUTER_API_KEY
export OPENROUTER_API_KEY=your_key_here
```

### 3. Run the server

```bash
python main.py
```

Open [http://localhost:8000](http://localhost:8000) — the UI loads immediately.

### 4. Try it via curl

```bash
# English refund
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "The stroller arrived damaged. I want a full refund."}'

# Arabic exchange
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "طلبت مقاس صغير بس وصلني مقاس كبير. أبغى أبدل المنتج."}'

# Ambiguous (should escalate)
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I want to return this."}'
```

### 5. Run evals

```bash
# Logic + schema tests only (no API key required)
python evals/run_evals.py --skip-llm

# Full LLM eval suite (requires OPENROUTER_API_KEY)
python evals/run_evals.py

# Single case
python evals/run_evals.py --case AR-001
```

---

## API Reference

### `POST /classify`

**Request:**
```json
{
  "text": "Customer's free-text return reason (EN or AR)",
  "order_id": "optional-order-id"
}
```

**Response:**
```json
{
  "classification": "refund | exchange | store_credit | escalate",
  "confidence": 0.92,
  "reasoning": "Customer explicitly states a defective item and requests a refund.",
  "extracted": {
    "product_issue": "broken wheel on stroller",
    "customer_sentiment": "frustrated",
    "urgency": "medium",
    "item_mentioned": "stroller",
    "resolution_preference": "full refund"
  },
  "suggested_reply": "We're so sorry to hear your stroller arrived damaged...",
  "language_detected": "en",
  "escalate_reason": null
}
```

When `classification` is `escalate` and confidence was the trigger:
```json
{
  "classification": "escalate",
  "confidence": 0.51,
  "escalate_reason": "Low confidence (51%) on predicted class 'exchange'. Routed to human review."
}
```

---

## Architecture

```
Customer text (EN/AR)
        │
        ▼
 detect_language()          ← heuristic: Arabic char ratio > 20% → ar
        │
        ▼
 build_system_prompt()      ← language-specific prompt (native Arabic, not translated)
        │
        ▼
 OpenRouter LLM call        ← qwen/qwen3-8b, response_format: json_object, temp=0.1
        │
        ▼
 JSON parse + strip fences
        │
        ▼
 ClassificationResult       ← Pydantic validation; raises ValidationError on bad output
 (Pydantic model)
        │
        ▼
 _apply_confidence_gate()   ← confidence < 0.65 → force escalate, log reason
        │
        ▼
 ReturnResponse (FastAPI)   ← returned to caller / rendered in UI
```

Key design decisions are documented in [TRADEOFFS.md](./TRADEOFFS.md).

---

## Evals

Full rubric, test cases, failure modes, and scoring in [EVALS.md](./EVALS.md).

**20 test cases** covering:
- 10 English (5 clear, 5 adversarial)
- 8 Arabic (4 clear, 4 adversarial)
- 1 mixed-language
- 1 edge case (symbols only)

Run the eval suite: `python evals/run_evals.py`

---

## Tradeoffs

Full analysis in [TRADEOFFS.md](./TRADEOFFS.md). Short version:

- **Model**: `qwen/qwen3-8b` on OpenRouter free tier — strong Arabic, reliable JSON output, zero cost
- **Language detection**: heuristic (Arabic char ratio), not a library — transparent, zero deps
- **Confidence gate**: separate post-processing step so it can be tuned without touching the prompt
- **What was cut**: RAG over policy docs, feedback loop, fine-tuning, batch endpoint

---

## Tooling

| Tool | Role |
|---|---|
| Claude (Anthropic) | Initial architecture scoping, prompt design, code review |
| OpenRouter | Model gateway — `qwen/qwen3-8b` for free-tier LLM calls |
| FastAPI + Pydantic | API framework + schema validation |
| httpx | Async HTTP client for OpenRouter calls |

**How AI was used**: Claude was used as a pair-programmer for the classifier pipeline and eval rubric design. The Arabic system prompt was written and iterated manually (3 rounds) to ensure native fluency rather than translated English. The confidence gate logic and Pydantic schema were my own design — Claude was used to check edge cases. All prompts are committed in `prompts.py`.

**What the agent got wrong / where I overruled**: The first version of the Arabic prompt produced replies that were technically correct Arabic but sounded like a translation ("نأسف جداً لسماع ذلك" is valid but stilted). I rewrote the tone guidance to specify Gulf Arabic register, which the model then followed well.

---

## File Structure

```
mumzworld-returns/
├── main.py                   # FastAPI app + endpoints
├── classifier.py             # LLM call, Pydantic schema, confidence gate
├── prompts.py                # EN and AR system prompts
├── requirements.txt
├── .env.example
├── frontend/
│   └── index.html            # Single-file UI (no build step)
├── data/
│   └── synthetic_returns.json  # 20 labelled test cases
├── evals/
│   └── run_evals.py          # Eval harness with scoring + report
├── EVALS.md                  # Rubric, test cases, failure modes
└── TRADEOFFS.md              # Architecture decisions, model choice, what was cut
```

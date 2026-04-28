# TRADEOFFS.md — Mumzworld Returns Intelligence

## Why This Problem

Returns processing is a high-volume, repetitive task with real financial stakes. Every misclassified
return either costs an agent time (if routed incorrectly) or costs the business money (if auto-refunded
when an exchange was appropriate). In a GCC e-commerce platform serving Arabic and English
customers, the language dimension adds an additional layer of complexity that off-the-shelf solutions
handle poorly.

I evaluated five problem candidates before settling on this one:

| Problem | Rejected reason |
|---|---|
| Product image → PDP content | Multimodal adds complexity without adding AI judgment insight |
| Voice memo → shopping list | Audio transcription dependency; harder to evaluate without audio hardware |
| Product comparison blog post | Output quality is hard to eval rigorously in 5 hours |
| Gift finder | Requires product catalog data; synthetic data is unconvincing |
| **Return reason classifier** | ✓ Natural uncertainty handling, real business value, easy synthetic data, bilingual by nature |

The return classifier wins because:
- The **uncertainty case is natural** — vague returns, off-topic messages, conflicting signals are all things the model should refuse rather than classify confidently
- **Synthetic data is fully realistic** — customers really do write these things
- It requires **structured output + schema validation + evals + multilingual** — hitting the stated engineering bar
- It is **immediately productionisable** — this could replace a rule-based triage system today

---

## Architecture Decisions

### Separation of concerns: `classifier.py` vs `prompts.py`

The LLM call, schema validation, and confidence gate are in `classifier.py`. The system prompts are
in `prompts.py`. This separation means you can iterate on prompts without touching the validation
logic, and vice versa. If you swap to a different model, only `classifier.py` changes.

### Pydantic for schema validation, not JSON Schema directly

I could have passed a JSON Schema spec to OpenRouter's `response_format` parameter and validated
the raw response manually. I chose Pydantic because:
- Validation errors are field-level and explicit (which failure mode, which field)
- `field_validator` lets me apply business logic (confidence clamping, allowed classification values)
- The `ClassificationResult` model doubles as the FastAPI response type, giving us OpenAPI docs for free

### Confidence gate as a separate post-processing step

The confidence gate (`_apply_confidence_gate`) is decoupled from the LLM call. This matters because:
- It is deterministic and auditable — you can see exactly why a case was escalated
- It can be tuned without touching the prompt
- It provides an explicit `escalate_reason` string rather than a silent classification change

### Language detection: heuristic, not a library

Arabic is detected by checking if >20% of non-space characters fall in the Unicode Arabic block
(U+0600–U+06FF). I chose this over `langdetect` or `fasttext` because:
- Zero dependencies
- Works reliably for the EN/AR binary case relevant to Mumzworld
- Transparent and auditable — no black-box language model for a 10-line check
- Handles mixed-language inputs gracefully (the MIXED-001 test case)

The tradeoff: it will misclassify Urdu or Farsi (also Arabic-script languages) as Arabic. For
Mumzworld's GCC context this is acceptable — the suggested reply in Arabic will still be
comprehensible to Urdu speakers familiar with Arabic script.

---

## Model Choice: `qwen/qwen3-8b`

Available free on OpenRouter. Reasons:
- **Strong bilingual Arabic + English** — trained on large Arabic corpora, not just translated English
- **Reliable instruction following** — consistently returns JSON-only when asked
- **8B parameter size** — fast enough for interactive use, no GPU required on the client

Alternatives considered:
- `meta-llama/llama-3.3-70b-instruct:free` — stronger reasoning but slower; better for ambiguous cases
- `deepseek/deepseek-chat:free` — excellent structured output but Arabic quality is weaker than Qwen
- GPT-4o / Claude Sonnet — best quality but not free; this prototype is designed to run at zero cost

If this moves to production, I would A/B test Qwen 3 8B (fast/cheap) against Llama 3.3 70B
(higher quality) on the escalation and Arabic cases specifically.

---

## Uncertainty Handling

Three layers of uncertainty handling:

1. **Prompt-level**: The system prompt explicitly tells the model to classify as `escalate` when
confidence is below 0.65, and lists the exact scenarios that trigger escalation (vague input,
off-topic, conflicting signals).

2. **Confidence gate** (code-level): After the LLM responds, `_apply_confidence_gate` checks the
returned confidence score. If < 0.65 and the model didn't already return `escalate`, we override
it and record the reason. This catches cases where the model says "exchange, 60%" — it thought it
knew but wasn't confident enough.

3. **Eval-level**: The escalation precision dimension in the rubric rewards honest uncertainty.
A model that escalates a genuinely ambiguous case scores *higher* than one that confidently
picks the wrong class.

---

## What I Cut

| Cut feature | Reason | Would add with more time |
|---|---|---|
| Authentication / API keys on the endpoint | Out of scope for prototype | Add JWT or API key middleware |
| Database logging of classifications | Not needed for proof-of-concept | PostgreSQL with order_id index |
| Feedback loop (agent marks right/wrong) | Scope | Use logged corrections to fine-tune or few-shot prompt |
| Batch processing endpoint | Single-case is sufficient to demonstrate | Add `/classify/batch` for CSV upload |
| Streaming response | FastAPI supports it; adds complexity | Server-sent events for the "thinking" state in the UI |
| Return policy document RAG | Would let the reply cite policy | Add a small vector store of Mumzworld policy docs |
| Fine-tuning on real returns data | No real data available | Fine-tune Qwen on 1,000 labelled returns for 2-3× accuracy gain |

---

## What I Would Build Next

1. **Policy-grounded replies using RAG**: Mumzworld has return policies by product category,
   delivery zone, and seller type. Embedding those docs and retrieving the relevant chunk into the
   reply prompt would make suggested replies legally accurate, not just warm.

2. **Feedback loop**: Let agents mark each classification as correct/incorrect. After 500 labelled
   examples, fine-tune the model on your specific return language patterns — GCC Arabic slang,
   brand names, product categories. This would push accuracy from ~85% to 95%+.

3. **Anomaly detection on escalation rate**: If escalation rate spikes above baseline (e.g. a batch
   of defective products from one supplier), surface that in an ops dashboard before agents are overwhelmed.

---

## Time Log

| Phase | Time |
|---|---|
| Problem selection + scoping | ~45 min |
| Prompt design + iteration | ~60 min |
| Core classifier + schema | ~60 min |
| FastAPI + frontend | ~60 min |
| Eval harness + 20 test cases | ~60 min |
| README + EVALS.md + TRADEOFFS.md | ~45 min |
| **Total** | **~5.5 hours** |

Went ~30 minutes over budget — the Arabic system prompt and the eval rubric took longer than expected.
The prompt needed three iterations to produce native-sounding Arabic replies rather than translated ones.

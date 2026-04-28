"""
classifier.py
-------------
Core classification logic.

Pipeline:
  1. Detect language (EN / AR / other) from input text
  2. Call OpenRouter LLM with a strict JSON schema in the system prompt
  3. Parse + validate the response against ClassificationResult (Pydantic)
  4. Apply confidence gate: if confidence < CONFIDENCE_THRESHOLD, force escalate
  5. Return validated result or raise a structured error

Model choice: qwen/qwen3-8b (free on OpenRouter)
  - Strong Arabic + English bilingual capability
  - Reliable instruction-following for JSON output
  - Free tier, no key needed beyond OpenRouter account (free signup)
"""

import os
import json
import re
import httpx
from pydantic import BaseModel, field_validator
from typing import Literal
from prompts import build_system_prompt, SUPPORTED_CLASSIFICATIONS

# --- Config -----------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.getenv("MODEL", "qwen/qwen3-8b")          # free on OpenRouter
CONFIDENCE_THRESHOLD = 0.65                            # below this → escalate
MAX_TOKENS = 1200
TIMEOUT = 30


# --- Pydantic Schemas -------------------------------------------------

class ExtractedFields(BaseModel):
    """Structured fields extracted from the return reason text."""
    product_issue: str | None = None       # What is wrong with the product
    customer_sentiment: Literal["positive", "neutral", "frustrated", "angry"] = "neutral"
    urgency: Literal["low", "medium", "high"] = "medium"
    item_mentioned: str | None = None      # Product name / SKU if mentioned
    resolution_preference: str | None = None  # What the customer explicitly asked for


class ClassificationResult(BaseModel):
    """Full validated output of one classification run."""
    classification: Literal["refund", "exchange", "store_credit", "escalate"]
    confidence: float
    reasoning: str
    extracted: ExtractedFields
    suggested_reply: str
    language_detected: Literal["en", "ar", "other"]
    escalate_reason: str | None = None     # populated when forced to escalate

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return round(max(0.0, min(1.0, v)), 2)

    @field_validator("classification")
    @classmethod
    def validate_classification(cls, v: str) -> str:
        if v not in SUPPORTED_CLASSIFICATIONS:
            raise ValueError(f"Unknown classification: {v}")
        return v


# --- Language detection (lightweight, no external library) -----------

_AR_RANGE = re.compile(r'[\u0600-\u06FF]')

def detect_language(text: str) -> Literal["en", "ar", "other"]:
    """Heuristic: if >20% of chars are Arabic Unicode block → ar, else en."""
    if not text:
        return "en"
    arabic_chars = len(_AR_RANGE.findall(text))
    ratio = arabic_chars / max(len(text.replace(" ", "")), 1)
    if ratio > 0.2:
        return "ar"
    # basic latin check
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if latin > 0:
        return "en"
    return "other"


# --- LLM Call ---------------------------------------------------------

async def _call_llm(user_text: str, language: str) -> dict:
    """
    Call OpenRouter with a strict JSON-only system prompt.
    Returns the parsed dict or raises ValueError with a clear message.
    """
    system_prompt = build_system_prompt(language)

    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": "https://mumzworld.com",
        "X-Title": "Mumzworld Returns Intelligence",
    }
    if OPENROUTER_API_KEY:
        headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"

    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,   # low temp for deterministic structured output
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(OPENROUTER_URL, json=payload, headers=headers)

    if response.status_code == 401:
        raise ValueError(
            "OpenRouter authentication failed. "
            "Set OPENROUTER_API_KEY env variable. "
            "Free signup at https://openrouter.ai"
        )
    if response.status_code != 200:
        raise ValueError(
            f"OpenRouter returned HTTP {response.status_code}: {response.text[:300]}"
        )

    data = response.json()

    # Extract text content from response
    try:
        raw = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected OpenRouter response shape: {e}\n{data}")

    # Strip markdown fences if model wrapped JSON in ```
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned non-JSON output: {e}\nRaw: {raw[:400]}")


# --- Confidence Gate --------------------------------------------------

def _apply_confidence_gate(result: ClassificationResult) -> ClassificationResult:
    """
    If model confidence is below threshold, override classification to 'escalate'
    and record the reason. This makes uncertainty handling explicit and auditable.
    """
    if result.confidence < CONFIDENCE_THRESHOLD and result.classification != "escalate":
        original = result.classification
        result = result.model_copy(update={
            "classification": "escalate",
            "escalate_reason": (
                f"Low confidence ({result.confidence:.0%}) on predicted class '{original}'. "
                "Routed to human review."
            ),
        })
    return result


# --- Main entry point -------------------------------------------------

async def classify_return(text: str) -> ClassificationResult:
    """
    Full pipeline: detect language → call LLM → validate schema → apply gate.

    Raises:
        ValueError: if LLM output cannot be parsed or validated.
        httpx.TimeoutException: if OpenRouter call times out.
    """
    language = detect_language(text)

    raw_dict = await _call_llm(text, language)

    # Inject language_detected if model didn't include it (fallback)
    if "language_detected" not in raw_dict or not raw_dict["language_detected"]:
        raw_dict["language_detected"] = language

    # Normalize nested extracted dict if model returned flat keys
    extracted_keys = ["product_issue", "customer_sentiment", "urgency",
                      "item_mentioned", "resolution_preference"]

    existing = raw_dict.get("extracted")
    if not isinstance(existing, dict):
        # Model returned flat keys or a string — pull them up into a nested dict
        raw_dict["extracted"] = {
            k: raw_dict.pop(k, None)
            for k in extracted_keys
        }

    # Pydantic validation — raises ValidationError with field-level detail
    result = ClassificationResult.model_validate(raw_dict)

    # Confidence gate
    result = _apply_confidence_gate(result)

    return result
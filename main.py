"""
Mumzworld Returns Intelligence API
-----------------------------------
Classifies free-text return reasons (EN/AR) into:
  refund | exchange | store_credit | escalate
Produces structured JSON output validated against a Pydantic schema,
a confidence score, and a drafted reply in the customer's language.
Low-confidence results (<0.65) are always routed to `escalate`.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

from classifier import classify_return, ClassificationResult

# Resolve paths relative to this file so the app works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_PATH = os.path.join(BASE_DIR, "frontend", "index.html")

app = FastAPI(
    title="Mumzworld Returns Intelligence",
    description="AI-powered return reason classifier with EN/AR support",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReturnRequest(BaseModel):
    text: str
    order_id: str | None = None


class ReturnResponse(BaseModel):
    classification: str
    confidence: float
    reasoning: str
    extracted: dict
    suggested_reply: str
    language_detected: str
    escalate_reason: str | None = None


@app.get("/")
async def serve_frontend():
    if not os.path.exists(FRONTEND_PATH):
        return HTMLResponse("<h2>frontend/index.html not found. Make sure the frontend/ folder exists next to main.py.</h2>", status_code=500)
    return FileResponse(FRONTEND_PATH, media_type="text/html")


@app.post("/classify", response_model=ReturnResponse)
async def classify(request: ReturnRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Return text cannot be empty.")
    if len(request.text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Return text is too short to classify.")

    result: ClassificationResult = await classify_return(request.text.strip())

    return ReturnResponse(
        classification=result.classification,
        confidence=result.confidence,
        reasoning=result.reasoning,
        extracted=result.extracted.model_dump(),
        suggested_reply=result.suggested_reply,
        language_detected=result.language_detected,
        escalate_reason=result.escalate_reason,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
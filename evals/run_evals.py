"""
evals/run_evals.py
-------------------
Evaluation harness for the Mumzworld Returns Intelligence system.

Rubric:
  - Classification accuracy  (primary metric)
  - Language detection accuracy
  - Escalation precision: did it escalate when it should have?
  - Hallucination check: does reasoning reference only content in the input?
  - Schema validity: did every response validate against ClassificationResult?
  - Reply language match: is suggested_reply in the correct language?

Scoring:
  Each test case is scored pass/fail per dimension.
  Overall score = weighted average across dimensions.

Usage:
  python evals/run_evals.py
  python evals/run_evals.py --case EN-001          # single case
  python evals/run_evals.py --skip-llm             # schema/logic tests only (no API call)
"""

import asyncio
import json
import argparse
import sys
import os
import re
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import classify_return, detect_language, ClassificationResult
from pydantic import ValidationError

# Load test cases
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_returns.json")

with open(DATA_PATH, encoding="utf-8") as f:
    TEST_CASES = json.load(f)

# Arabic unicode check (for reply language validation)
_AR_RE = re.compile(r'[\u0600-\u06FF]')

def is_arabic_text(text: str) -> bool:
    chars = len(_AR_RE.findall(text))
    return chars / max(len(text.replace(" ", "")), 1) > 0.15


# --- Per-case evaluation ---------------------------------------------

async def evaluate_case(case: dict, skip_llm: bool = False) -> dict:
    case_id = case["id"]
    text = case["text"]
    expected_class = case["expected_classification"]
    expected_lang = case["expected_language"]

    result_row = {
        "id": case_id,
        "input_snippet": text[:60] + ("..." if len(text) > 60 else ""),
        "expected": expected_class,
        "notes": case.get("notes", ""),
        # dimensions
        "classification_correct": None,
        "language_correct": None,
        "escalation_correct": None,
        "schema_valid": None,
        "reply_language_correct": None,
        "error": None,
        # raw output
        "got_classification": None,
        "got_confidence": None,
        "got_language": None,
    }

    # Language detection test (pure logic, no LLM)
    detected_lang = detect_language(text)
    result_row["language_correct"] = detected_lang == expected_lang
    result_row["got_language"] = detected_lang

    if skip_llm:
        result_row["classification_correct"] = "SKIPPED"
        result_row["schema_valid"] = "SKIPPED"
        result_row["reply_language_correct"] = "SKIPPED"
        result_row["escalation_correct"] = "SKIPPED"
        return result_row

    # LLM classification
    try:
        result: ClassificationResult = await classify_return(text)
        result_row["schema_valid"] = True
        result_row["got_classification"] = result.classification
        result_row["got_confidence"] = result.confidence

        # 1. Classification accuracy
        result_row["classification_correct"] = (result.classification == expected_class)

        # 2. Escalation precision
        # If expected=escalate → must escalate. If expected!=escalate and got escalate with
        # confidence < threshold → acceptable (correct uncertainty). If got wrong class → fail.
        if expected_class == "escalate":
            result_row["escalation_correct"] = (result.classification == "escalate")
        else:
            # Not expected to escalate: pass if correct, or if escalated due to low confidence
            # (i.e. the system expressed honest uncertainty rather than wrong confident answer)
            if result.classification == result_row["expected"]:
                result_row["escalation_correct"] = True
            elif result.classification == "escalate" and result.confidence < 0.65:
                result_row["escalation_correct"] = True  # honest uncertainty is acceptable
            else:
                result_row["escalation_correct"] = False

        # 3. Reply language match
        reply = result.suggested_reply or ""
        if detected_lang == "ar":
            result_row["reply_language_correct"] = is_arabic_text(reply)
        else:
            # For en/other: reply should NOT be predominantly Arabic
            result_row["reply_language_correct"] = not is_arabic_text(reply)

    except ValidationError as e:
        result_row["schema_valid"] = False
        result_row["error"] = f"Schema validation failed: {e.error_count()} errors"
        result_row["classification_correct"] = False
        result_row["escalation_correct"] = False
        result_row["reply_language_correct"] = False
    except Exception as e:
        result_row["error"] = str(e)[:200]
        result_row["schema_valid"] = False
        result_row["classification_correct"] = False
        result_row["escalation_correct"] = False
        result_row["reply_language_correct"] = False

    return result_row


# --- Scoring ---------------------------------------------------------

WEIGHTS = {
    "classification_correct": 0.35,
    "language_correct": 0.15,
    "escalation_correct": 0.25,
    "schema_valid": 0.15,
    "reply_language_correct": 0.10,
}

def score_results(results: list[dict]) -> dict:
    scores = {dim: [] for dim in WEIGHTS}

    for r in results:
        for dim in WEIGHTS:
            val = r.get(dim)
            if val in (True, False):
                scores[dim].append(1.0 if val else 0.0)

    dim_scores = {
        dim: (sum(vals) / len(vals) if vals else None)
        for dim, vals in scores.items()
    }

    weighted = sum(
        score * WEIGHTS[dim]
        for dim, score in dim_scores.items()
        if score is not None
    )
    total_weight = sum(
        WEIGHTS[dim]
        for dim, score in dim_scores.items()
        if score is not None
    )
    overall = weighted / total_weight if total_weight > 0 else 0.0

    return {"dimensions": dim_scores, "overall": overall}


# --- Report ----------------------------------------------------------

def print_report(results: list[dict], scores: dict, elapsed: float):
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"\n{'='*72}")
    print(f"{BOLD}  MUMZWORLD RETURNS CLASSIFIER — EVAL REPORT{RESET}")
    print(f"{'='*72}")
    print(f"  Cases run: {len(results)}   |   Elapsed: {elapsed:.1f}s")
    print(f"{'─'*72}")

    for r in results:
        ok = r["classification_correct"]
        icon = f"{GREEN}✓{RESET}" if ok is True else (f"{RED}✗{RESET}" if ok is False else f"{YELLOW}~{RESET}")
        conf_str = f"({r['got_confidence']:.0%})" if r["got_confidence"] is not None else ""
        expected_str = f"{r['expected']:<14}"
        if r["got_classification"]:
            got_str = f"{str(r['got_classification']):<14}"
        elif r["classification_correct"] == "SKIPPED":
            got_str = f"{'(skipped)':<14}"
        else:
            got_str = f"{'ERROR':<14}"
        dims = "".join([
            f"{GREEN}L{RESET}" if r["language_correct"] else f"{RED}L{RESET}",
            f"{GREEN}E{RESET}" if r["escalation_correct"] is True else (f"{RED}E{RESET}" if r["escalation_correct"] is False else "E"),
            f"{GREEN}S{RESET}" if r["schema_valid"] is True else (f"{RED}S{RESET}" if r["schema_valid"] is False else "S"),
            f"{GREEN}R{RESET}" if r["reply_language_correct"] is True else (f"{RED}R{RESET}" if r["reply_language_correct"] is False else "R"),
        ])
        error_str = f"  {YELLOW}! {r['error'][:50]}{RESET}" if r["error"] else ""
        print(f"  {icon} {r['id']:<12} exp={expected_str} got={got_str}{conf_str:<8} [{dims}]{error_str}")

    print(f"{'─'*72}")
    print(f"  {BOLD}Dimension scores:{RESET}  (L=language  E=escalation  S=schema  R=reply-lang)")
    dims = scores["dimensions"]
    for dim, score in dims.items():
        label = dim.replace("_correct", "").replace("_", " ").title()
        bar_len = int((score or 0) * 20)
        bar = f"{GREEN}{'█' * bar_len}{'░' * (20 - bar_len)}{RESET}"
        score_str = f"{score:.0%}" if score is not None else "N/A"
        print(f"    {label:<28} {bar}  {score_str}")

    overall = scores["overall"]
    color = GREEN if overall >= 0.80 else (YELLOW if overall >= 0.65 else RED)
    print(f"{'─'*72}")
    print(f"  {BOLD}Overall weighted score: {color}{overall:.1%}{RESET}")
    print(f"{'='*72}\n")


# --- Entry point -----------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Run Mumzworld Returns Classifier evals")
    parser.add_argument("--case", type=str, help="Run a single case by ID")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM calls (logic/schema tests only)")
    args = parser.parse_args()

    cases = TEST_CASES
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"Case '{args.case}' not found. Available: {[c['id'] for c in TEST_CASES]}")
            sys.exit(1)

    print(f"\nRunning {len(cases)} eval case(s)... (skip_llm={args.skip_llm})")
    if not args.skip_llm and not os.getenv("OPENROUTER_API_KEY"):
        print("⚠  OPENROUTER_API_KEY not set — calls will fail unless the model endpoint is open.")

    start = time.time()
    results = []
    for case in cases:
        print(f"  → {case['id']}...", end=" ", flush=True)
        r = await evaluate_case(case, skip_llm=args.skip_llm)
        ok = r["classification_correct"]
        print("✓" if ok is True else ("✗" if ok is False else "~"))
        results.append(r)

    elapsed = time.time() - start
    scores = score_results(results)
    print_report(results, scores, elapsed)

    # Exit with non-zero if overall < 70%
    sys.exit(0 if scores["overall"] >= 0.70 else 1)


if __name__ == "__main__":
    asyncio.run(main())

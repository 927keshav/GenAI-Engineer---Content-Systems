"""Rubric, evaluator JSON, and the rules that decide pass, retry, or best effort.

A lesson is passed only when all six checks are present and each one is PASS.
An overall flag that disagrees with the checks is corrected.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from config import MAX_RETRIES
from errors import EvaluationError

CRITERIA: list[tuple[str, str]] = [
    ("accuracy_and_groundedness", "Accuracy and groundedness"),
    ("beginner_friendly_language", "Beginner-friendly language"),
    ("teaches_by_examples", "Teaches by examples"),
    ("clear_language_no_unexplained_jargon", "Clear language without unexplained jargon"),
    ("coverage_of_key_points", "Coverage of key points"),
    ("coherent_teaching_flow", "Coherent teaching flow"),
]

CRITERION_KEYS: list[str] = [key for key, _label in CRITERIA]
LABELS: dict[str, str] = dict(CRITERIA)

CRITERION_GUIDE: dict[str, str] = {
    "accuracy_and_groundedness": (
        "The central facts are true at a high-school level, the lesson does not contradict itself, "
        "and it does not invent studies, quotations, dates, or measurements. "
        "A simpler explanation still passes when it remains true."
    ),
    "beginner_friendly_language": (
        "Sentences are short and the words are common enough for a 12th-grade student with limited English. "
        "The writing does not sound like a college textbook."
    ),
    "teaches_by_examples": (
        "The lesson includes at least two concrete everyday examples, and each example shows the idea "
        "instead of only naming it."
    ),
    "clear_language_no_unexplained_jargon": (
        "Every specialist word is defined in plain English the first time it appears. "
        "A word a student with no background would not know counts as jargon."
    ),
    "coverage_of_key_points": (
        "The lesson teaches the core ideas a beginner needs in order to understand the topic. "
        "It does not skip a central idea, and it does not wander into unnecessary side topics."
    ),
    "coherent_teaching_flow": (
        "The lesson moves in a clear order: what the idea is, why it matters, how it works, examples, "
        "a short check, and a summary. Later parts build on earlier parts."
    ),
}

PASSED = "passed"
BEST_EFFORT = "best_effort"
UNEVALUATED = "unevaluated"
ERROR = "error"


class CriterionCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")

    result: str = Field(description="PASS only if this criterion is fully met. Otherwise FAIL.")
    reason: str = Field(description="Cite what you saw in the lesson, or say what was missing.")


class ChecksModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    accuracy_and_groundedness: CriterionCheck = Field(
        description=CRITERION_GUIDE["accuracy_and_groundedness"]
    )
    beginner_friendly_language: CriterionCheck = Field(
        description=CRITERION_GUIDE["beginner_friendly_language"]
    )
    teaches_by_examples: CriterionCheck = Field(description=CRITERION_GUIDE["teaches_by_examples"])
    clear_language_no_unexplained_jargon: CriterionCheck = Field(
        description=CRITERION_GUIDE["clear_language_no_unexplained_jargon"]
    )
    coverage_of_key_points: CriterionCheck = Field(description=CRITERION_GUIDE["coverage_of_key_points"])
    coherent_teaching_flow: CriterionCheck = Field(description=CRITERION_GUIDE["coherent_teaching_flow"])


class EvaluationReport(BaseModel):
    """Schema sent to Gemini. The app still recomputes overall from the six checks."""

    model_config = ConfigDict(extra="ignore")

    checks: ChecksModel
    failed_checks: list[str] = Field(
        description="Keys of checks whose result is FAIL. Empty only when every check is PASS."
    )
    rejection_reasons: list[str] = Field(
        description="One plain-language reason for each failed check. Empty only when every check is PASS."
    )
    overall: str = Field(description="PASS only if every check is PASS. Otherwise FAIL.")


def as_verdict(value: Any) -> str:
    token = str(value or "").strip().upper().rstrip(".")
    if token in {"PASS", "PASSED"}:
        return "PASS"
    return "FAIL"


def extract_json(text: str) -> dict[str, Any]:
    body = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", body, re.DOTALL)
    if fence:
        body = fence.group(1)
    else:
        start = body.find("{")
        end = body.rfind("}")
        if start < 0 or end <= start:
            raise EvaluationError("The evaluator response did not contain JSON.")
        body = body[start : end + 1]
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise EvaluationError("The evaluator response was not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise EvaluationError("The evaluator JSON was not an object.")
    return parsed


def _as_dict(payload: Any) -> dict[str, Any]:
    if payload is None:
        raise EvaluationError("The evaluator returned nothing.")
    if isinstance(payload, str):
        return extract_json(payload)
    if hasattr(payload, "model_dump"):
        dumped = payload.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if isinstance(payload, dict):
        parsed = payload.get("parsed") if "parsed" in payload and "raw" in payload else None
        if parsed is not None:
            return _as_dict(parsed)
        return payload
    raise EvaluationError("The evaluator returned an unsupported result.")


def _raw_checks(data: dict[str, Any]) -> dict[str, Any]:
    raw_checks = data.get("checks")
    if hasattr(raw_checks, "model_dump"):
        raw_checks = raw_checks.model_dump()
    if raw_checks is None and all(key in data for key in CRITERION_KEYS):
        raw_checks = {key: data[key] for key in CRITERION_KEYS}
    if isinstance(raw_checks, list):
        mapped: dict[str, Any] = {}
        for item in raw_checks:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("criterion") or item.get("key")
            if isinstance(name, str):
                mapped[name] = item
        raw_checks = mapped
    if not isinstance(raw_checks, dict):
        raise EvaluationError("Evaluator JSON did not include the six checks.")
    missing = [key for key in CRITERION_KEYS if key not in raw_checks]
    if missing:
        raise EvaluationError(
            "Evaluator JSON omitted checks: " + ", ".join(missing)
        )
    return raw_checks


def normalize_evaluation(payload: Any) -> dict[str, Any]:
    """Build the stored JSON from the model's checks.

    ``overall``, ``failed_checks``, and ``rejection_reasons`` are recomputed.
    A model that says PASS while a check is FAIL is recorded as FAIL.
    """
    data = _as_dict(payload)
    raw_checks = _raw_checks(data)
    checks: dict[str, dict[str, str]] = {}
    failed: list[str] = []
    reasons: list[str] = []
    for key in CRITERION_KEYS:
        item = raw_checks[key]
        if isinstance(item, str):
            result = as_verdict(item)
            reason = ""
        elif isinstance(item, dict):
            result = as_verdict(item.get("result", item.get("verdict")))
            reason = str(item.get("reason") or item.get("explanation") or "").strip()
        else:
            raise EvaluationError(f"Check '{key}' was not readable.")
        if not reason:
            reason = (
                "No specific reason was given."
                if result == "PASS"
                else "The evaluator marked this check as failed without a specific reason."
            )
        checks[key] = {"result": result, "reason": reason}
        if result != "PASS":
            failed.append(key)
            reasons.append(f"{LABELS[key]}: {reason}")
    raw_overall = str(data.get("overall") or "").strip()
    model_overall = as_verdict(raw_overall) if raw_overall else None
    overall = "PASS" if not failed else "FAIL"
    corrected = model_overall is not None and model_overall != overall
    # A second pass must keep the flag. The stored overall already matches the checks.
    if data.get("overall_corrected") is True:
        corrected = True
    return {
        "checks": checks,
        "failed_checks": failed,
        "rejection_reasons": reasons,
        "overall": overall,
        "pass_count": len(CRITERION_KEYS) - len(failed),
        "overall_corrected": corrected,
    }


def is_complete_pass(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    checks = report.get("checks")
    if not isinstance(checks, dict):
        return False
    if report.get("failed_checks"):
        return False
    if report.get("overall") != "PASS":
        return False
    for key in CRITERION_KEYS:
        item = checks.get(key)
        if not isinstance(item, dict) or item.get("result") != "PASS":
            return False
    return True


def pass_count_of(report: dict[str, Any] | None) -> int:
    if not isinstance(report, dict):
        return 0
    checks = report.get("checks")
    if isinstance(checks, dict) and checks:
        return sum(
            1
            for key in CRITERION_KEYS
            if isinstance(checks.get(key), dict) and checks[key].get("result") == "PASS"
        )
    return int(report.get("pass_count") or 0)


def select_best(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [item for item in history if isinstance(item.get("evaluation"), dict)]
    if not usable:
        return None

    def sort_key(item: dict[str, Any]) -> tuple[int, int]:
        # Keep the earliest draft when scores tie; a later rewrite should only
        # replace it when the evaluator finds more passing checks.
        return (pass_count_of(item.get("evaluation")), -int(item.get("attempt") or 0))

    return max(usable, key=sort_key)


def failure_details(report: dict[str, Any]) -> list[dict[str, str]]:
    checks = report.get("checks") or {}
    details: list[dict[str, str]] = []
    for key in report.get("failed_checks") or []:
        item = checks.get(key) or {}
        reason = item.get("reason") if isinstance(item, dict) else ""
        details.append(
            {
                "key": key,
                "label": LABELS.get(key, key),
                "reason": reason or "No reason was given.",
            }
        )
    return details


def decide_route(state: dict[str, Any]) -> str:
    """Stop on a real pass, an error, or after the initial draft plus two retries.

    ``attempt`` is 1-based. Attempt 1 uses zero retries. Attempts 2 and 3 are the
    two retries. The next step is ``retry`` only while a failed review can still
    be rewritten.
    """
    if state.get("error"):
        return "finalize"
    evaluation = state.get("evaluation")
    if is_complete_pass(evaluation):
        return "finalize"
    if not evaluation:
        return "finalize"
    attempt = int(state.get("attempt") or 0)
    limit = state.get("max_retries")
    if limit is None:
        limit = MAX_RETRIES
    if attempt > int(limit):
        return "finalize"
    return "retry"


def build_disclosure(
    *,
    outcome: str,
    retry_count: int = 0,
    best_attempt: int | None = None,
    pass_count: int = 0,
    remaining: list[dict[str, str]] | None = None,
    error: str | None = None,
) -> str:
    if outcome == ERROR:
        return error or "The lesson could not be generated."
    if outcome == UNEVALUATED:
        base = (
            "The evaluator did not return usable checks, so this lesson is not marked as passed."
        )
        if error:
            return f"{base} {error}"
        return base
    if outcome == PASSED:
        if retry_count == 0:
            return (
                f"All 6 checks passed on attempt {best_attempt}. No retries were needed."
            )
        noun = "retry" if retry_count == 1 else "retries"
        return f"All 6 checks passed on attempt {best_attempt} after {retry_count} {noun}."
    labels = ", ".join(item["label"] for item in (remaining or [])) or "one or more checks"
    if retry_count == 0:
        lead = "The lesson failed review on the only completed attempt."
    elif retry_count == 1:
        lead = "After 1 retry the lesson still failed review."
    else:
        lead = f"After {retry_count} retries the lesson still failed review."
    return (
        f"{lead} The version shown is attempt {best_attempt}, which passed {pass_count} of 6 checks. "
        f"Remaining failures: {labels}. This lesson did not pass."
    )


def assemble_final(state: dict[str, Any]) -> dict[str, Any]:
    """Choose the lesson to show from completed reviews. Does not call a model."""
    history = [item for item in (state.get("history") or []) if isinstance(item, dict)]
    error = state.get("error")
    lesson_text = (state.get("lesson") or "").strip()

    if history:
        best = select_best(history)
        if best is None:
            history = []
        else:
            report = best["evaluation"]
            remaining = [] if is_complete_pass(report) else failure_details(report)
            outcome = PASSED if is_complete_pass(report) else BEST_EFFORT
            retry_count = max(
                0,
                int(state.get("attempt") or len(history)) - 1,
            )
            disclosure = build_disclosure(
                outcome=outcome,
                retry_count=retry_count,
                best_attempt=int(best.get("attempt") or 0),
                pass_count=pass_count_of(report),
                remaining=remaining,
                error=None,
            )
            if error and outcome == BEST_EFFORT:
                disclosure = f"{disclosure} A later attempt did not finish. {error}"
            return {
                "final_lesson": best.get("lesson") or "",
                "final_evaluation": report,
                "best_attempt_number": int(best.get("attempt") or 0),
                "retry_count": retry_count,
                "outcome": outcome,
                "remaining_failures": remaining,
                "pass_count": pass_count_of(report),
                "disclosure": disclosure,
                "overall": "PASS" if outcome == PASSED else "FAIL",
            }

    if lesson_text:
        disclosure = build_disclosure(outcome=UNEVALUATED, error=error if isinstance(error, str) else None)
        return {
            "final_lesson": lesson_text,
            "final_evaluation": None,
            "best_attempt_number": None,
            "retry_count": 0,
            "outcome": UNEVALUATED,
            "remaining_failures": [],
            "pass_count": 0,
            "disclosure": disclosure,
            "overall": "UNEVALUATED",
        }

    disclosure = build_disclosure(outcome=ERROR, error=error if isinstance(error, str) else None)
    return {
        "final_lesson": "",
        "final_evaluation": None,
        "best_attempt_number": None,
        "retry_count": 0,
        "outcome": ERROR,
        "remaining_failures": [],
        "pass_count": 0,
        "disclosure": disclosure,
        "overall": "ERROR",
    }


def trusted_outcome(outcome: str, evaluation: dict[str, Any] | None) -> str:
    """Refuse a stored PASS that does not have six passing checks."""
    if outcome == PASSED and not is_complete_pass(evaluation):
        return BEST_EFFORT if evaluation else UNEVALUATED
    return outcome

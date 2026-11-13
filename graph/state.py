"""State passed between the writer, reviewer, and retry nodes."""

from __future__ import annotations

import re
from typing import Any, TypedDict

from config import MAX_RETRIES


class LessonState(TypedDict):
    topic: str
    lesson: str
    attempt: int
    max_retries: int
    evaluation: dict[str, Any] | None
    history: list[dict[str, Any]]
    memory_context: str
    memory_hits: list[dict[str, Any]]
    memory_warning: str | None
    rejection_reasons: list[str]
    status: str
    status_log: list[str]
    error: str | None
    generation_id: int | None
    final_lesson: str
    final_evaluation: dict[str, Any] | None
    best_attempt_number: int | None
    retry_count: int
    outcome: str
    remaining_failures: list[dict[str, str]]
    disclosure: str
    model_name: str
    pass_count: int
    overall: str
    demo_mode: bool


def clean_topic(topic: str) -> str:
    cleaned = re.sub(r"[\x00-\x1f]+", " ", topic or "")
    cleaned = " ".join(cleaned.split())
    if len(cleaned) < 3:
        raise ValueError("Enter a topic of at least 3 characters.")
    if len(cleaned) > 200:
        raise ValueError("Use a topic of 200 characters or fewer.")
    return cleaned


def initial_state(topic: str, model_name: str = "") -> LessonState:
    return {
        "topic": clean_topic(topic),
        "lesson": "",
        "attempt": 0,
        "max_retries": MAX_RETRIES,
        "evaluation": None,
        "history": [],
        "memory_context": "",
        "memory_hits": [],
        "memory_warning": None,
        "rejection_reasons": [],
        "status": "Starting.",
        "status_log": [],
        "error": None,
        "generation_id": None,
        "final_lesson": "",
        "final_evaluation": None,
        "best_attempt_number": None,
        "retry_count": 0,
        "outcome": "",
        "remaining_failures": [],
        "disclosure": "",
        "model_name": model_name,
        "pass_count": 0,
        "overall": "",
        "demo_mode": False,
    }


def push_status(state: dict[str, Any], message: str) -> list[str]:
    log = list(state.get("status_log") or [])
    log.append(message)
    return log

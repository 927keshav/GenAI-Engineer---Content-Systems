"""Generator, evaluator, and conditional retry loop.

The graph stops as soon as every check passes. It rewrites at most twice.
If checks still fail, the best completed attempt is returned and the failures stay visible.
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from config import MAX_RETRIES
from errors import GenerationError
from evaluation import (
    assemble_final,
    decide_route,
    is_complete_pass,
    normalize_evaluation,
)
from graph.state import LessonState, initial_state, push_status
from memory.store import MemoryStore
from prompts import evaluation_messages, format_memory_context, generation_messages
from safety import public_error

GenerateFn = Callable[[list[dict[str, str]]], str]
EvaluateFn = Callable[[str, str], Any]
StatusFn = Callable[[dict[str, Any]], None]

_MAX_LESSON_CHARS = 20000


def _save_unreviewed_attempt(
    store: MemoryStore, state: LessonState, *, lesson: str, error: str
) -> str | None:
    """Persist an API/evaluator failure as UNEVALUATED, never as a rubric FAIL/PASS."""
    generation_id = state.get("generation_id")
    if generation_id is None:
        return state.get("memory_warning")
    record = {
        "checks": {},
        "failed_checks": [],
        "rejection_reasons": [],
        "overall": "UNEVALUATED",
        "pass_count": 0,
        "error": error,
    }
    try:
        store.save_attempt(
            generation_id=generation_id,
            attempt_number=int(state.get("attempt") or 1),
            lesson=lesson,
            evaluation=record,
        )
    except Exception as exc:
        return public_error(exc)
    return state.get("memory_warning")


def _trim_lesson(lesson: str) -> str:
    text = lesson.strip()
    if len(text) <= _MAX_LESSON_CHARS:
        return text
    return text[:_MAX_LESSON_CHARS].rstrip() + "\n\n[The lesson was cut because it was too long.]"


def build_lesson_graph(
    store: MemoryStore,
    generate_fn: GenerateFn,
    evaluate_fn: EvaluateFn,
):
    def load_memory(state: LessonState) -> dict[str, Any]:
        topic = state["topic"]
        warning = None
        try:
            hits = store.relevant_failures(topic)
            # Controlled demonstrations never write altered results to memory.
            generation_id = None if state.get("demo_mode") else store.start_generation(
                topic, state.get("model_name") or ""
            )
        except Exception as exc:
            hits = []
            generation_id = None
            warning = public_error(exc)
        if warning:
            message = f"Memory could not be read ({warning}). Continuing without it."
        elif hits:
            message = f"Loaded {len(hits)} related past failures."
        else:
            message = "No related past failures were found."
        return {
            "memory_hits": hits,
            "memory_context": format_memory_context(hits),
            "generation_id": generation_id,
            "memory_warning": warning,
            "status": message,
            "status_log": push_status(state, message),
        }

    def generator(state: LessonState) -> dict[str, Any]:
        if state.get("error"):
            return {}
        attempt = int(state.get("attempt") or 0) + 1
        max_attempts = int(state.get("max_retries") or MAX_RETRIES) + 1
        writing = f"Writing attempt {attempt} of {max_attempts}."
        log = push_status(state, writing)
        previous_lesson = ""
        reasons: list[str] = []
        if attempt > 1:
            previous_lesson = state.get("lesson") or ""
            reasons = list(state.get("rejection_reasons") or [])
        try:
            messages = generation_messages(
                topic=state["topic"],
                memory_context=state.get("memory_context") or "",
                previous_lesson=previous_lesson,
                rejection_reasons=reasons,
                attempt=attempt,
                max_attempts=max_attempts,
            )
            lesson = generate_fn(messages)
            if not isinstance(lesson, str) or not lesson.strip():
                raise GenerationError("The model returned an empty lesson.")
            lesson = _trim_lesson(lesson)
        except Exception as exc:
            failed = f"Generation failed on attempt {attempt}."
            return {
                "attempt": attempt,
                "error": public_error(exc),
                "status": failed,
                "status_log": log + [failed],
            }
        return {
            "attempt": attempt,
            "lesson": lesson,
            "error": None,
            "status": writing,
            "status_log": log,
        }

    def evaluator(state: LessonState) -> dict[str, Any]:
        if state.get("error"):
            message = "Evaluation was skipped because generation did not finish."
            warning = _save_unreviewed_attempt(
                store, state, lesson=state.get("lesson") or "", error=state["error"]
            )
            history = list(state.get("history") or [])
            history.append({
                "attempt": state.get("attempt"),
                "lesson": state.get("lesson") or "",
                "evaluation": None,
                "pass_count": 0,
                "error": state["error"],
            })
            return {
                "history": history,
                "status": message,
                "status_log": push_status(state, message),
                "memory_warning": warning,
            }
        try:
            report = normalize_evaluation(evaluate_fn(state["topic"], state["lesson"]))
        except Exception as exc:
            message = "Evaluation could not be completed. The lesson is not marked as passed."
            error = public_error(exc)
            warning = _save_unreviewed_attempt(
                store, state, lesson=state.get("lesson") or "", error=error
            )
            history = list(state.get("history") or [])
            history.append({
                "attempt": state.get("attempt"),
                "lesson": state.get("lesson") or "",
                "evaluation": None,
                "pass_count": 0,
                "error": error,
            })
            return {
                "history": history,
                "error": error,
                "evaluation": None,
                "status": message,
                "status_log": push_status(state, message),
                "memory_warning": warning,
            }

        if state.get("demo_mode") and state.get("attempt") == 1:
            # The Gemini review above still runs. This labeled override exists only
            # to make the retry path reproducible; it is never stored in SQLite.
            demo_key = "beginner_friendly_language"
            report["checks"][demo_key] = {
                "result": "FAIL",
                "reason": (
                    "DEMO ONLY: a controlled failure was inserted to demonstrate the retry path. "
                    "This is not a genuine evaluator finding."
                ),
            }
            report["failed_checks"] = [
                key for key, item in report["checks"].items() if item.get("result") != "PASS"
            ]
            report["rejection_reasons"] = [
                f"{key}: {report['checks'][key]['reason']}" for key in report["failed_checks"]
            ]
            report["overall"] = "FAIL"
            report["overall_corrected"] = True
            report["pass_count"] = len(report["checks"]) - len(report["failed_checks"])

        history = list(state.get("history") or [])
        history.append(
            {
                "attempt": state["attempt"],
                "lesson": state["lesson"],
                "evaluation": report,
                "pass_count": report["pass_count"],
            }
        )
        warning = state.get("memory_warning")
        if state.get("generation_id") is not None:
            try:
                store.save_attempt(
                    generation_id=state["generation_id"],
                    attempt_number=state["attempt"],
                    lesson=state["lesson"],
                    evaluation=report,
                )
                store.save_feedback(
                    generation_id=state["generation_id"],
                    attempt_number=state["attempt"],
                    topic=state["topic"],
                    evaluation=report,
                )
            except Exception as exc:
                warning = public_error(exc)

        if is_complete_pass(report):
            message = f"Checked attempt {state['attempt']}: PASS."
        else:
            message = (
                f"Checked attempt {state['attempt']}: FAIL "
                f"({len(report['failed_checks'])} checks)."
            )
        return {
            "evaluation": report,
            "history": history,
            "rejection_reasons": list(report["rejection_reasons"]),
            "status": message,
            "status_log": push_status(state, message),
            "memory_warning": warning,
            "error": None,
        }

    def retry(state: LessonState) -> dict[str, Any]:
        count = len(state.get("rejection_reasons") or [])
        message = (
            f"Retry {state['attempt']}: sending {count} rejection reasons to the generator."
        )
        return {"status": message, "status_log": push_status(state, message)}

    def finalize(state: LessonState) -> dict[str, Any]:
        final = assemble_final(state)
        warning = state.get("memory_warning")
        if state.get("generation_id") is not None:
            try:
                store.finish_generation(
                    state["generation_id"],
                    final_lesson=final["final_lesson"],
                    final_evaluation=final["final_evaluation"],
                    outcome=final["outcome"],
                    retry_count=final["retry_count"],
                    overall=final["overall"],
                    best_attempt=final["best_attempt_number"],
                    pass_count=final["pass_count"],
                    remaining_failures=final["remaining_failures"],
                    disclosure=final["disclosure"],
                    status=final["disclosure"],
                )
            except Exception as exc:
                warning = public_error(exc)
        return {
            **final,
            "evaluation": final["final_evaluation"],
            "status": final["disclosure"],
            "status_log": push_status(state, final["disclosure"]),
            "memory_warning": warning,
        }

    graph = StateGraph(LessonState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("generator", generator)
    graph.add_node("evaluator", evaluator)
    graph.add_node("retry", retry)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "generator")
    graph.add_edge("generator", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        decide_route,
        {"retry": "retry", "finalize": "finalize"},
    )
    graph.add_edge("retry", "generator")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_lesson_graph(
    graph,
    topic: str,
    model_name: str = "",
    on_status: StatusFn | None = None,
    demo_mode: bool = False,
) -> dict[str, Any]:
    state = initial_state(topic, model_name=model_name)
    state["demo_mode"] = demo_mode
    if demo_mode:
        state["status_log"].append(
            "DEMO MODE: attempt 1 includes a labeled controlled failure, not a genuine evaluator finding. "
            "Demo data is not saved in SQLite."
        )
    last: dict[str, Any] | None = None
    for snapshot in graph.stream(state, config={"recursion_limit": 40}, stream_mode="values"):
        last = snapshot
        if on_status is not None:
            on_status(snapshot)
    if last is None:
        raise RuntimeError("The lesson graph did not return a result.")
    return last

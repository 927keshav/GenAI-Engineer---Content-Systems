"""Offline tests for the real LangGraph flow; Gemini calls are replaced with fakes."""

from __future__ import annotations

import json

import pytest

from evaluation import CRITERION_KEYS, EvaluationError, is_complete_pass, normalize_evaluation
from graph.workflow import build_lesson_graph, run_lesson_graph
from memory.store import MemoryStore
from safety import public_error


def report(*, failed: tuple[str, ...] = (), overall: str | None = None) -> dict:
    checks = {
        key: {"result": "FAIL" if key in failed else "PASS",
              "reason": f"Needs work on {key}." if key in failed else "Evidence present."}
        for key in CRITERION_KEYS
    }
    return {"checks": checks, "failed_checks": list(failed),
            "rejection_reasons": [f"Needs work on {key}." for key in failed],
            "overall": overall or ("FAIL" if failed else "PASS")}


class FakeModels:
    def __init__(self, evaluations: list[dict]) -> None:
        self.evaluations = iter(evaluations)
        self.generated_prompts: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.generated_prompts.append(messages)
        return f"Lesson attempt {len(self.generated_prompts)} with clear examples."

    def evaluate(self, _topic: str, _lesson: str) -> dict:
        return next(self.evaluations)


def run(tmp_path, models: FakeModels, *, demo: bool = False):
    store = MemoryStore(tmp_path / "lessons.db")
    graph = build_lesson_graph(store, models.generate, models.evaluate)
    return run_lesson_graph(graph, "Retrieval-Augmented Generation (RAG)", demo_mode=demo), store


def test_all_six_checks_must_pass_even_if_overall_claims_pass():
    result = normalize_evaluation(report(failed=("teaches_by_examples",), overall="PASS"))
    assert result["overall"] == "FAIL"
    assert result["failed_checks"] == ["teaches_by_examples"]
    assert "Needs work" in result["rejection_reasons"][0]


def test_invalid_evaluator_json_is_an_error_not_a_pass():
    with pytest.raises(EvaluationError):
        normalize_evaluation("not valid JSON")


def test_graph_stops_immediately_when_all_checks_pass(tmp_path):
    models = FakeModels([report()])
    result, store = run(tmp_path, models)
    try:
        assert len(models.generated_prompts) == 1
        assert len(result["history"]) == 1
        assert result["outcome"] == "passed"
        assert is_complete_pass(result["final_evaluation"])
    finally:
        store.close()


def test_graph_retries_twice_then_returns_best_attempt(tmp_path):
    models = FakeModels([
        report(failed=("accuracy_and_groundedness", "teaches_by_examples")),
        report(failed=("teaches_by_examples",)),
        report(failed=("teaches_by_examples",)),
    ])
    result, store = run(tmp_path, models)
    try:
        assert len(result["history"]) == len(models.generated_prompts) == 3
        assert result["retry_count"] == 2
        assert result["best_attempt_number"] == 2
        assert result["outcome"] == "best_effort"
        assert result["remaining_failures"][0]["key"] == "teaches_by_examples"
        assert "Needs work on accuracy_and_groundedness" in json.dumps(models.generated_prompts[1])
    finally:
        store.close()


def test_successful_retry_stops_after_second_attempt(tmp_path):
    models = FakeModels([report(failed=("teaches_by_examples",)), report()])
    result, store = run(tmp_path, models)
    try:
        assert len(models.generated_prompts) == 2
        assert result["outcome"] == "passed"
        assert result["retry_count"] == 1
        assert result["best_attempt_number"] == 2
    finally:
        store.close()


def test_memory_survives_reopen_and_is_used_in_future_prompt(tmp_path):
    db_path = tmp_path / "persistent.db"
    first = MemoryStore(db_path)
    generation_id = first.start_generation("RAG retrieval", "fake-model")
    failed = normalize_evaluation(report(failed=("teaches_by_examples",)))
    first.save_attempt(generation_id=generation_id, attempt_number=1, lesson="Draft", evaluation=failed)
    first.save_feedback(generation_id=generation_id, attempt_number=1, topic="RAG retrieval", evaluation=failed)
    first.finish_generation(
        generation_id, final_lesson="Draft", final_evaluation=failed, outcome="best_effort",
        retry_count=0, overall="FAIL", best_attempt=1, pass_count=5,
        remaining_failures=[{"key": "teaches_by_examples", "label": "Examples", "reason": "Needs work."}],
        disclosure="Did not pass", status="Did not pass")
    first.close()

    reopened = MemoryStore(db_path)
    try:
        assert reopened.stats()["generations"] == 1
        assert reopened.stats()["failures"] == 1
        hits = reopened.relevant_failures("RAG retrieval augmented generation")
        assert hits and hits[0]["category"] == "teaches_by_examples"
        prompts: list[list[dict[str, str]]] = []

        def generate(messages):
            prompts.append(messages)
            return "A beginner RAG lesson."

        graph = build_lesson_graph(reopened, generate, lambda *_: report())
        result = run_lesson_graph(graph, "RAG retrieval augmented generation")
        assert result["memory_hits"]
        assert "Teaches by examples" in json.dumps(prompts[0])
    finally:
        reopened.close()


def test_demo_inserts_labeled_failure_and_does_not_persist_demo_rows(tmp_path):
    models = FakeModels([report(), report()])
    result, store = run(tmp_path, models, demo=True)
    try:
        assert len(models.generated_prompts) == 2
        failed_check = result["history"][0]["evaluation"]["checks"]["beginner_friendly_language"]
        assert failed_check["result"] == "FAIL"
        assert "DEMO ONLY" in failed_check["reason"]
        assert result["outcome"] == "passed"
        assert store.stats()["generations"] == 0
    finally:
        store.close()


def test_invalid_review_is_saved_as_unevaluated_not_as_a_pass_or_failure(tmp_path):
    store = MemoryStore(tmp_path / "review-error.db")
    graph = build_lesson_graph(store, lambda _messages: "A draft lesson.", lambda *_: "not JSON")
    try:
        result = run_lesson_graph(graph, "A beginner science topic")
        attempts = store.list_attempts(1)
        assert result["outcome"] == "unevaluated"
        assert len(attempts) == 1
        assert attempts[0]["evaluation"]["overall"] == "UNEVALUATED"
        assert attempts[0]["lesson"] == "A draft lesson."
        assert attempts[0]["evaluation"]["failed_checks"] == []
    finally:
        store.close()


def test_failed_retry_is_counted_and_logged_without_inventing_review(tmp_path):
    calls = 0

    def generate(_messages):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("temporary provider failure")
        return "First draft."

    store = MemoryStore(tmp_path / "retry-error.db")
    graph = build_lesson_graph(
        store, generate, lambda *_: report(failed=("teaches_by_examples",))
    )
    try:
        result = run_lesson_graph(graph, "A beginner science topic")
        attempts = store.list_attempts(1)
        assert result["retry_count"] == 1
        assert result["best_attempt_number"] == 1
        assert len(attempts) == 2
        assert attempts[1]["evaluation"]["overall"] == "UNEVALUATED"
    finally:
        store.close()


def test_generation_error_is_logged_as_unevaluated(tmp_path):
    def generate(_messages):
        raise RuntimeError("simulated provider outage")

    def evaluator_must_not_run(*_args):
        pytest.fail("The evaluator must not run when lesson generation fails.")

    store = MemoryStore(tmp_path / "generation-error.db")
    graph = build_lesson_graph(store, generate, evaluator_must_not_run)
    try:
        result = run_lesson_graph(graph, "A beginner science topic")
        attempts = store.list_attempts(1)
        assert result["outcome"] == "error"
        assert len(result["history"]) == 1
        assert result["history"][0]["evaluation"] is None
        assert len(attempts) == 1
        assert attempts[0]["evaluation"]["overall"] == "UNEVALUATED"
        assert attempts[0]["lesson"] == ""
    finally:
        store.close()


def test_public_error_redacts_configured_key(monkeypatch):
    fake_key = "AIza" + "TEST_ONLY_1234567890"
    monkeypatch.setenv("GEMINI_API_KEY", fake_key)
    message = public_error(RuntimeError(f"request failed with {fake_key}"))
    assert fake_key not in message
    assert "[redacted]" in message

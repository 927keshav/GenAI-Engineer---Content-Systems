"""Gemini calls through LangChain. The key is passed to the client and never logged."""

from __future__ import annotations

import logging
from typing import Any

from config import NO_KEY_MESSAGE, get_api_key, get_model
from errors import EvaluationError, LessonAppError, MissingApiKeyError
from evaluation import EvaluationReport, normalize_evaluation
from prompts import evaluation_messages
from safety import public_error

logger = logging.getLogger("lesson_generator")

_REQUEST_MARKERS = (
    "api key",
    "api_key",
    "permission",
    "unauthorized",
    "unauthenticated",
    "quota",
    "resource exhausted",
    "429",
    "401",
    "403",
    "timeout",
    "timed out",
    "unavailable",
    "connection",
    "dns",
    "ssl",
)


def _request_failed(exc: BaseException) -> bool:
    text = f"{exc.__class__.__name__} {exc}".lower()
    return any(marker in text for marker in _REQUEST_MARKERS)


def _message_types():
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        from langchain.messages import HumanMessage, SystemMessage
    return SystemMessage, HumanMessage


def to_langchain_messages(messages: list[dict[str, str]]) -> list[Any]:
    system_cls, human_cls = _message_types()
    converted = []
    for message in messages:
        content = message["content"]
        if message["role"] == "system":
            converted.append(system_cls(content=content))
        elif message["role"] == "human":
            converted.append(human_cls(content=content))
        else:
            raise LessonAppError(f"Unsupported prompt role: {message['role']}")
    return converted


def message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                if block.get("type") in {"thinking", "reasoning"}:
                    continue
                text = block.get("text") or block.get("content") or ""
                if text:
                    parts.append(str(text))
                continue
            text = getattr(block, "text", "")
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


def _build_model(max_tokens: int) -> Any:
    key = get_api_key()
    if not key:
        raise MissingApiKeyError(NO_KEY_MESSAGE)
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise LessonAppError(
            "The Gemini client is not installed. Run: pip install -r requirements.txt"
        ) from exc
    # Temperature is left unset. Gemini 3 models are tuned for the default.
    attempts = (
        {
            "model": get_model(),
            "google_api_key": key,
            "max_tokens": max_tokens,
            "timeout": 120,
            "max_retries": 1,
        },
        {
            "model": get_model(),
            "google_api_key": key,
            "max_output_tokens": max_tokens,
        },
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return ChatGoogleGenerativeAI(**kwargs)
        except TypeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            logger.warning("Could not start the Gemini client: %s", public_error(exc))
            raise LessonAppError(public_error(exc)) from None
    logger.warning("Could not start the Gemini client: %s", public_error(last_error) if last_error else "unknown")
    raise LessonAppError(public_error(last_error) if last_error else "Could not start the Gemini client.")


def _structured_reviewer(model: Any) -> Any | None:
    schema = EvaluationReport.model_json_schema()
    builders = (
        lambda: model.with_structured_output(schema=schema, method="json_schema"),
        lambda: model.with_structured_output(schema),
        lambda: model.with_structured_output(EvaluationReport),
    )
    for build in builders:
        try:
            return build()
        except TypeError:
            continue
        except Exception as exc:
            if _request_failed(exc):
                _raise_request_failure(exc)
            logger.warning("Structured output is unavailable: %s", public_error(exc))
            return None
    return None


def _raise_request_failure(exc: BaseException) -> None:
    logger.warning("Gemini request failed: %s", public_error(exc))
    raise LessonAppError(public_error(exc)) from None


def make_callables():
    """Return (generate, evaluate) functions that call the configured Gemini model."""
    writer = _build_model(max_tokens=8192)
    reviewer = _build_model(max_tokens=4096)
    structured = _structured_reviewer(reviewer)

    def generate(messages: list[dict[str, str]]) -> str:
        try:
            response = writer.invoke(to_langchain_messages(messages))
        except LessonAppError:
            raise
        except Exception as exc:
            if _request_failed(exc):
                _raise_request_failure(exc)
            logger.warning("Generation failed: %s", public_error(exc))
            raise LessonAppError(public_error(exc)) from None
        return message_text(response)

    def evaluate(topic: str, lesson: str) -> dict[str, Any]:
        messages = evaluation_messages(topic, lesson)
        lc_messages = to_langchain_messages(messages)
        if structured is not None:
            try:
                raw = structured.invoke(lc_messages)
                return normalize_evaluation(raw)
            except EvaluationError:
                logger.warning("Structured review did not include every check. Trying JSON text.")
            except LessonAppError:
                raise
            except Exception as exc:
                if _request_failed(exc):
                    _raise_request_failure(exc)
                logger.warning("Structured review failed, trying JSON text: %s", public_error(exc))
        retry_messages = list(messages) + [
            {
                "role": "human",
                "content": (
                    "Return only a JSON object with all six checks, failed_checks, "
                    "rejection_reasons, and overall. Use PASS or FAIL for every check."
                ),
            }
        ]
        try:
            response = reviewer.invoke(to_langchain_messages(retry_messages))
            return normalize_evaluation(message_text(response))
        except EvaluationError:
            raise
        except LessonAppError:
            raise
        except Exception as exc:
            if _request_failed(exc):
                _raise_request_failure(exc)
            logger.warning("Review failed: %s", public_error(exc))
            raise EvaluationError(
                "The evaluator did not return usable PASS/FAIL checks for all six criteria."
            ) from None

    return generate, evaluate

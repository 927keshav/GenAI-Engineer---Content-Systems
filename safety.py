"""Turn exceptions into text that is safe to show in the browser and the database."""

from __future__ import annotations

import re

from errors import LessonAppError

_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z\-_]{10,}")
_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*")
_AUTH_MARKERS = (
    "api key",
    "api_key",
    "invalid key",
    "permission denied",
    "unauthorized",
    "unauthenticated",
)

API_KEY_FAILURE = (
    "Gemini refused the request. Check that GEMINI_API_KEY in your .env file is valid. "
    "The key is not shown."
)


def redact(text: str) -> str:
    from config import get_api_key

    redacted = text or ""
    key = get_api_key()
    if key:
        redacted = redacted.replace(key, "[redacted]")
    redacted = _KEY_PATTERN.sub("[redacted]", redacted)
    redacted = _BEARER_PATTERN.sub("Bearer [redacted]", redacted)
    return redacted


def public_error(exc: BaseException) -> str:
    """A short message for the UI. Authentication failures drop the original text."""
    if isinstance(exc, LessonAppError):
        text = redact(str(exc))
    else:
        text = redact(f"{exc.__class__.__name__}: {exc}")
    lowered = text.lower()
    if any(marker in lowered for marker in _AUTH_MARKERS):
        return API_KEY_FAILURE
    text = " ".join(text.split())
    if len(text) > 700:
        return text[:700].rstrip() + "..."
    return text or "Something went wrong."

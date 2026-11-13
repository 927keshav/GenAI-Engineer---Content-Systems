"""Paths, model name, and API-key presence. This module never prints the key."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Do not turn on tracing unless the user already chose it. Tracing can send prompts off-machine.
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

MAX_RETRIES = 2
DEFAULT_MODEL = "gemini-3.5-flash-lite"
_PLACEHOLDERS = {"", "your_gemini_api_key_here", "changeme"}

NO_KEY_MESSAGE = (
    "Set GEMINI_API_KEY in a .env file in the project folder. "
    "The app does not generate or pass a lesson without a real evaluator."
)


def get_api_key() -> str:
    """Return the configured key, or an empty string. Callers must not display it."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(name, "").strip()
        if value and value not in _PLACEHOLDERS:
            return value
    return ""


def api_key_configured() -> bool:
    return bool(get_api_key())


def get_model() -> str:
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip()
    if (
        not model
        or any(char.isspace() for char in model)
        or model.startswith("AIza")
        or len(model) > 80
    ):
        return DEFAULT_MODEL
    return model


def sqlite_path() -> Path:
    raw = os.getenv("SQLITE_PATH", "data/lessons.db").strip() or "data/lessons.db"
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def settings_summary() -> dict[str, object]:
    """Public settings only. The API key is intentionally absent."""
    return {
        "api_key_configured": api_key_configured(),
        "model": get_model(),
        "database": str(sqlite_path()),
        "max_retries": MAX_RETRIES,
    }

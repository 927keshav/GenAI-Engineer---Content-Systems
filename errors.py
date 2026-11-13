"""Errors the lesson app can show without revealing secrets."""


class LessonAppError(Exception):
    """Safe, user-facing application error."""


class MissingApiKeyError(LessonAppError):
    """Raised when no Gemini key is configured."""


class GenerationError(LessonAppError):
    """Raised when the writer returns nothing usable."""


class EvaluationError(LessonAppError):
    """Raised when the reviewer does not return all six checks."""

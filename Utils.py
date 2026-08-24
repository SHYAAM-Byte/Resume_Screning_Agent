"""
utils.py — Shared helper functions for the AI Interviewer.

Framework-agnostic utilities used across app.py, interviewer.py, ai_engine.py
and database.py. Nothing here talks to Streamlit, Ollama, Whisper, or the
database directly — these are small, pure, well-tested building blocks.

Sections:
  - Resume text cleaning
  - JSON helpers (validation + safe parsing/extraction)
  - Timestamp & ID generation
  - Score formatting
  - Session helpers
  - File validation
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "clean_resume_text",
    "truncate_text",
    "is_valid_json",
    "safe_json_loads",
    "validate_json_keys",
    "extract_json_from_text",
    "current_timestamp",
    "current_iso_timestamp",
    "unique_id",
    "clamp",
    "format_score",
    "average_score",
    "score_to_grade",
    "new_session_id",
    "is_valid_session_id",
    "Stopwatch",
    "get_file_extension",
    "is_allowed_file",
    "is_valid_resume_file",
    "is_valid_audio_file",
    "ALLOWED_RESUME_EXTENSIONS",
    "ALLOWED_AUDIO_EXTENSIONS",
]


# --------------------------------------------------------------------------
# Resume text cleaning
# --------------------------------------------------------------------------
def clean_resume_text(text: str) -> str:
    """
    Normalize raw text extracted from a PDF/DOCX resume.

    Handles common extraction artifacts: form-feed page breaks, stray
    control characters, inconsistent whitespace, and excessive blank lines.

    Args:
        text: Raw extracted resume text.

    Returns:
        A cleaned, whitespace-normalized string. Returns "" for falsy input.
    """
    if not text:
        return ""

    # Normalize unicode so visually-identical characters compare/match consistently.
    text = unicodedata.normalize("NFKC", text)

    # PDF extraction often leaves form-feed characters at page boundaries.
    text = text.replace("\x0c", "\n")

    # Drop control characters (category "C*") but keep newlines and tabs.
    text = "".join(
        ch for ch in text if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )

    # Collapse runs of spaces/tabs, and runs of 3+ blank lines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Trim trailing whitespace on each line.
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def truncate_text(text: str, max_length: int = 3000, suffix: str = "…") -> str:
    """Truncate text to `max_length` characters, appending `suffix` if cut."""
    if not text or len(text) <= max_length:
        return text or ""
    return text[: max(0, max_length - len(suffix))].rstrip() + suffix


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------
def is_valid_json(payload: str) -> bool:
    """Return True if `payload` parses as valid JSON."""
    try:
        json.loads(payload)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def safe_json_loads(payload: str, default: Optional[Any] = None) -> Any:
    """Parse JSON, returning `default` instead of raising on failure."""
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse JSON payload; returning default.")
        return default


def validate_json_keys(data: Any, required_keys: Iterable[str]) -> Tuple[bool, List[str]]:
    """
    Check that a dict contains all required keys.

    Args:
        data: The object to check (expected to be a dict).
        required_keys: Keys that must be present.

    Returns:
        (is_valid, missing_keys) — missing_keys is empty when is_valid is True.
    """
    required = list(required_keys)
    if not isinstance(data, dict):
        return False, required
    missing = [key for key in required if key not in data]
    return (len(missing) == 0, missing)


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Extract the first valid JSON object embedded in a larger text blob.

    Useful for parsing LLM responses that wrap JSON in prose or markdown
    code fences (e.g. an ai_engine.py evaluation response).

    Args:
        text: Raw text that may contain a JSON object.

    Returns:
        The parsed dict, or None if no valid JSON object could be found.
    """
    if not text:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced_match.group(1) if fenced_match else None

    if candidate is None:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace_match.group(0) if brace_match else None

    if candidate is None:
        return None

    parsed = safe_json_loads(candidate)
    return parsed if isinstance(parsed, dict) else None


# --------------------------------------------------------------------------
# Timestamp & ID generation
# --------------------------------------------------------------------------
def current_timestamp(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return the current local time formatted as a string (for display/logging)."""
    return datetime.now().strftime(fmt)


def current_iso_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string (for storage in SQLite)."""
    return datetime.now(timezone.utc).isoformat()


def unique_id(prefix: str = "") -> str:
    """Generate a short unique identifier, optionally prefixed (e.g. 'q_a1b2c3d4e5f6')."""
    suffix = uuid.uuid4().hex[:12]
    return f"{prefix}{suffix}" if prefix else suffix


# --------------------------------------------------------------------------
# Score formatting
# --------------------------------------------------------------------------
def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp `value` into the inclusive [min_value, max_value] range."""
    return max(min_value, min(max_value, value))


def format_score(score: Any, scale: float = 10.0, decimals: int = 1) -> str:
    """
    Format a numeric score for display, e.g. 7.333 -> "7.3 / 10".

    Non-numeric or out-of-range input is coerced to a safe 0-scale value
    rather than raising, since this is primarily a UI-facing helper.
    """
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = 0.0
    clamped = clamp(numeric_score, 0.0, scale)
    return f"{clamped:.{decimals}f} / {scale:g}"


def average_score(scores: Iterable[Any]) -> float:
    """Compute the average of a list of numeric scores, ignoring non-numeric entries."""
    numeric_scores = [s for s in scores if isinstance(s, (int, float))]
    if not numeric_scores:
        return 0.0
    return sum(numeric_scores) / len(numeric_scores)


def score_to_grade(score: Any, scale: float = 10.0) -> str:
    """Map a numeric score to a coarse qualitative grade label."""
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = 0.0
    pct = clamp(numeric_score, 0.0, scale) / scale if scale else 0.0
    if pct >= 0.85:
        return "Excellent"
    if pct >= 0.70:
        return "Good"
    if pct >= 0.50:
        return "Fair"
    return "Needs Improvement"


# --------------------------------------------------------------------------
# Session helpers
# --------------------------------------------------------------------------
def new_session_id() -> str:
    """Generate a new unique interview/session identifier (UUID4 string)."""
    return str(uuid.uuid4())


def is_valid_session_id(session_id: Any) -> bool:
    """Check whether a value is a syntactically valid UUID session id."""
    try:
        uuid.UUID(str(session_id))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


class Stopwatch:
    """
    Minimal elapsed-time tracker for timing an interview or a single answer.

    Example:
        timer = Stopwatch()
        timer.start()
        ...
        elapsed = timer.stop()  # seconds, float
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._elapsed: float = 0.0

    def start(self) -> None:
        """Start (or resume) timing."""
        if self._start is None:
            self._start = time.monotonic()

    def stop(self) -> float:
        """Pause timing and return the total elapsed seconds so far."""
        if self._start is not None:
            self._elapsed += time.monotonic() - self._start
            self._start = None
        return self._elapsed

    def reset(self) -> None:
        """Reset the stopwatch to zero."""
        self._start = None
        self._elapsed = 0.0

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed seconds, including any currently-running interval."""
        if self._start is not None:
            return self._elapsed + (time.monotonic() - self._start)
        return self._elapsed


# --------------------------------------------------------------------------
# File validation
# --------------------------------------------------------------------------
ALLOWED_RESUME_EXTENSIONS = {"pdf", "docx"}
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3"}


def get_file_extension(filename: str) -> str:
    """Return a filename's extension in lowercase, without the leading dot."""
    return Path(filename or "").suffix.lower().lstrip(".")


def is_allowed_file(filename: str, allowed_extensions: Iterable[str]) -> bool:
    """Check whether `filename`'s extension is in `allowed_extensions`."""
    normalized = {ext.lower().lstrip(".") for ext in allowed_extensions}
    return get_file_extension(filename) in normalized


def is_valid_resume_file(
    filename: str, size_bytes: int, max_size_mb: float = 10.0
) -> Tuple[bool, str]:
    """
    Validate an uploaded resume file's extension and size.

    Returns:
        (is_valid, error_message). error_message is "" when the file is valid.
    """
    if not filename:
        return False, "No filename was provided."
    if not is_allowed_file(filename, ALLOWED_RESUME_EXTENSIONS):
        allowed = ", ".join(sorted(ALLOWED_RESUME_EXTENSIONS))
        return False, f"Unsupported file type. Allowed types: {allowed}."
    if size_bytes <= 0:
        return False, "The file is empty."
    if size_bytes > max_size_mb * 1024 * 1024:
        return False, f"File exceeds the {max_size_mb:g}MB size limit."
    return True, ""


def is_valid_audio_file(
    filename: str, size_bytes: int, max_size_mb: float = 25.0
) -> Tuple[bool, str]:
    """
    Validate an uploaded/recorded audio file's extension and size.

    Returns:
        (is_valid, error_message). error_message is "" when the file is valid.
    """
    if not filename:
        return False, "No filename was provided."
    if not is_allowed_file(filename, ALLOWED_AUDIO_EXTENSIONS):
        allowed = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        return False, f"Unsupported audio format. Allowed types: {allowed}."
    if size_bytes <= 0:
        return False, "The audio file is empty."
    if size_bytes > max_size_mb * 1024 * 1024:
        return False, f"Audio file exceeds the {max_size_mb:g}MB size limit."
    return True, ""
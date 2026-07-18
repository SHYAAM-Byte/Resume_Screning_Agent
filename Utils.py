"""
utils.py
--------
Generic, reusable helper utilities shared across the Resume Screening Agent.

Scope of this file (and ONLY this file):
    - Clean/normalize extracted text
    - Safely parse JSON returned by the LLM (Ollama)
    - Sort candidate results by score
    - Small, generic, dependency-free helper functions

This file contains NO AI/LLM calls and NO prompt engineering — it only
provides pure, stateless helper functions used by other modules.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Normalize extracted resume/job-description text for consistent downstream use.

    - Normalizes line endings (\\r\\n, \\r -> \\n)
    - Strips non-printable/control characters (keeps newlines, tabs, and
      international characters such as accents)
    - Collapses 3+ blank lines into a single blank line
    - Collapses runs of spaces/tabs into a single space
    - Trims trailing whitespace on each line and the string as a whole

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned plain text. Returns "" for falsy input.
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip control/non-printable characters, but keep newlines and tabs.
    # Unicode letters (e.g. accented names) are preserved via str.isprintable().
    text = "".join(ch for ch in text if ch in ("\n", "\t") or ch.isprintable())

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse repeated spaces/tabs into a single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Trim trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()


def truncate_text(text: str, max_length: int = 6000, suffix: str = "\n...[truncated]") -> str:
    """
    Truncate text to a maximum character length.

    Useful for keeping large resumes/job descriptions within an LLM's
    effective context window before they're handed off to ai_engine.py.

    Args:
        text: The text to truncate.
        max_length: Maximum number of characters to keep.
        suffix: Text appended when truncation occurs.

    Returns:
        The original text if within the limit, otherwise a truncated copy.
    """
    if not text or len(text) <= max_length:
        return text or ""
    return text[:max_length].rstrip() + suffix


# ---------------------------------------------------------------------------
# Safe JSON parsing (for LLM output)
# ---------------------------------------------------------------------------
def safe_json_parse(raw_text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Safely parse a JSON object/array out of raw LLM output.

    Local LLMs (including Ollama models) often wrap JSON in markdown code
    fences, or add stray preamble/explanation text before or after the
    actual JSON payload. This function tries several increasingly
    permissive strategies before giving up, and never raises — callers can
    always rely on getting either a parsed object or None back.

    Args:
        raw_text: The raw string returned by the LLM.

    Returns:
        The parsed JSON (dict or list), or None if parsing failed at every stage.
    """
    if not raw_text or not raw_text.strip():
        return None

    candidates: List[str] = [raw_text.strip()]

    # Strategy 2: strip markdown code fences, e.g. ```json { ... } ```
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    # Strategy 3: grab the widest {...} or [...] block as a last resort
    brace_match = re.search(r"(\{.*\}|\[.*\])", raw_text, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(1).strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

    return None


# ---------------------------------------------------------------------------
# Candidate sorting
# ---------------------------------------------------------------------------
def sort_candidates(
    candidates: List[Dict[str, Any]],
    score_key: str = "Match Score",
    descending: bool = True,
) -> List[Dict[str, Any]]:
    """
    Sort a list of candidate result dicts by a numeric score field.

    Missing, None, or non-numeric scores are treated as the lowest possible
    value so malformed entries sink to the bottom instead of raising an error.

    Args:
        candidates: List of candidate result dictionaries.
        score_key: The dict key holding the numeric score to sort by.
        descending: If True (default), the highest scores come first.

    Returns:
        A new, sorted list. The input list is not mutated.
    """
    if not candidates:
        return []

    def _score(candidate: Dict[str, Any]) -> float:
        value = candidate.get(score_key)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")

    return sorted(candidates, key=_score, reverse=descending)


# ---------------------------------------------------------------------------
# Small generic helpers
# ---------------------------------------------------------------------------
def safe_get(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely traverse nested dictionaries without raising KeyError/TypeError.

    Example:
        safe_get(result, "candidate", "name", default="Unknown")

    Args:
        data: The dictionary to traverse.
        *keys: Sequence of keys describing the path to walk.
        default: Value returned if any key is missing or a non-dict is encountered.

    Returns:
        The resolved value, or `default` if the path doesn't exist.
    """
    current: Any = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def is_supported_extension(filename: str, allowed_extensions: tuple = ("pdf", "docx")) -> bool:
    """
    Check whether a filename has one of the allowed extensions (case-insensitive).

    Args:
        filename: The filename or path to check.
        allowed_extensions: Tuple of allowed extensions, without dots (e.g. ("pdf", "docx")).

    Returns:
        True if the extension is allowed, False otherwise.
    """
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in allowed_extensions


def normalize_score(value: Any, min_value: float = 0.0, max_value: float = 100.0) -> float:
    """
    Coerce a raw score value (which may come from an LLM as str/int/float/None)
    into a float clamped within [min_value, max_value].

    Args:
        value: The raw score value to normalize.
        min_value: Lower clamp bound.
        max_value: Upper clamp bound.

    Returns:
        A float clamped to [min_value, max_value]. Defaults to min_value if
        `value` cannot be interpreted as a number.
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        return min_value
    return max(min_value, min(max_value, score))
"""
utils.py

Enterprise AI Research & Decision Intelligence Agent
------------------------------------------------------
General-purpose helper functions reusable across the project.

Responsibilities (and ONLY these):
    - Clean text extracted from documents/web pages.
    - Validate simple inputs (non-empty strings, file extensions,
      numeric ranges).
    - Safely parse JSON, tolerating minor formatting noise.
    - Remove duplicate items/sources.
    - Normalize heterogeneous source/citation metadata into one shape.
    - Format confidence scores and structured research results into
      human-readable text.
    - Provide small, generic error-handling helpers (custom exception
      types, a "call this and degrade gracefully" wrapper, and a
      clean exception-to-string formatter).

This module does NOT implement:
    - Any Streamlit or other UI code        (see app.py)
    - Any Ollama / LLM API calls             (see ai_engine.py)
    - Web search                             (see web_search.py)
    - Document parsing / RAG                 (see rag.py)
    - Business reasoning of any kind (no scoring, ranking, or
      decision-making beyond simple, literal data transformations --
      that belongs in research_agent.py / ai_engine.py).

Every function here is small, pure (no side effects beyond optional
logging), and dependency-free (standard library only), so any module
in this project can import from it safely.

Contract with research_agent.py
---------------------------------
research_agent.py imports and calls exactly these three functions:

    safe_json_extract(text) -> dict | None
    dedupe_by_key(items, key) -> list[dict]
    clamp(value, lo, hi) -> float

Note on `safe_json_extract`: it returns `None` (not `{}`) when no
valid JSON object can be recovered, so callers can distinguish "failed
to parse" from "parsed to an empty object" with a simple
`isinstance(result, dict)` check -- which is exactly how
research_agent.py uses it.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, TypeVar
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

__all__ = [
    # errors
    "UtilsError",
    "ValidationError",
    "safe_call",
    "format_error_message",
    # text cleaning
    "remove_control_characters",
    "normalize_whitespace",
    "clean_text",
    # validation
    "is_non_empty_string",
    "require_non_empty_string",
    "validate_file_extension",
    "clamp",
    # JSON parsing
    "safe_json_loads",
    "safe_json_extract",
    # source metadata
    "normalize_url",
    "domain_label",
    "normalize_source_metadata",
    # deduplication
    "dedupe_by_key",
    "dedupe_sources",
    # confidence formatting
    "normalize_confidence",
    "confidence_label",
    "format_confidence",
    # result formatting
    "format_bullet_list",
    "truncate_text",
    "format_result_as_markdown",
]

T = TypeVar("T")


# ----------------------------------------------------------------------
# Handling common errors
# ----------------------------------------------------------------------
class UtilsError(Exception):
    """Base class for errors raised by this module's own helpers."""


class ValidationError(UtilsError):
    """Raised by the validation helpers below when input is invalid."""


def safe_call(
    func: Callable[..., T],
    *args: Any,
    default: Optional[T] = None,
    log_errors: bool = True,
    **kwargs: Any,
) -> Optional[T]:
    """Call `func(*args, **kwargs)` and return its result, or `default`
    if it raises anything. Centralizes the "log it and keep going"
    pattern used throughout this project instead of duplicating
    try/except blocks in every caller.

    Example:
        value = safe_call(json.loads, raw_text, default={})
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - intentionally broad; this
        # helper exists precisely to contain unknown failures.
        if log_errors:
            name = getattr(func, "__name__", repr(func))
            logger.warning("safe_call: %s raised %s: %s", name, type(exc).__name__, exc)
        return default


def format_error_message(exc: BaseException) -> str:
    """Render an exception as a short, user-facing message: its class
    name plus its message, with no traceback. Useful for showing a
    clean error to end users instead of a raw stack trace.
    """
    message = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {message}" if message else name


# ----------------------------------------------------------------------
# Cleaning extracted text
# ----------------------------------------------------------------------
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def remove_control_characters(text: str) -> str:
    """Strip non-printable control characters that sometimes leak in
    from PDF/DOCX extraction, while preserving normal whitespace
    (space, tab, newline). Safe on None/empty input.
    """
    return _CONTROL_CHARS_RE.sub("", text or "")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs into one space, collapse three or
    more consecutive newlines into two, and trim leading/trailing
    whitespace. Safe on None/empty input.
    """
    text = _INLINE_WHITESPACE_RE.sub(" ", text or "")
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Clean text extracted from an uploaded document or web page:
    remove control characters and normalize whitespace. Safe on
    None/empty input (returns "").
    """
    return normalize_whitespace(remove_control_characters(text or ""))


# ----------------------------------------------------------------------
# Validating input
# ----------------------------------------------------------------------
def is_non_empty_string(value: Any) -> bool:
    """True if `value` is a string with non-whitespace content."""
    return isinstance(value, str) and bool(value.strip())


def require_non_empty_string(value: Any, field_name: str = "value") -> str:
    """Validate that `value` is a non-empty string and return it
    stripped of leading/trailing whitespace.

    Raises:
        ValidationError: if `value` is missing, not a string, or
            empty/whitespace-only.
    """
    if not is_non_empty_string(value):
        raise ValidationError(f"{field_name} must be a non-empty string.")
    return value.strip()


def validate_file_extension(path: str, allowed_extensions: Iterable[str]) -> str:
    """Validate that `path` ends in one of `allowed_extensions`
    (case-insensitive; extensions may be given with or without a
    leading dot) and return the normalized, lowercased extension
    (e.g. ".pdf").

    Raises:
        ValidationError: if `path` has no extension, or its extension
            is not in `allowed_extensions`.
    """
    normalized_allowed: Set[str] = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in allowed_extensions
    }

    match = re.search(r"\.[A-Za-z0-9]+$", path or "")
    ext = match.group(0).lower() if match else ""

    if not ext or ext not in normalized_allowed:
        allowed_display = ", ".join(sorted(normalized_allowed)) or "(none configured)"
        raise ValidationError(
            f"Unsupported file extension '{ext or '(none)'}' for '{path}'. "
            f"Allowed: {allowed_display}."
        )
    return ext


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp `value` into the inclusive range [lo, hi]."""
    return max(lo, min(hi, value))


# ----------------------------------------------------------------------
# Safely parsing JSON
# ----------------------------------------------------------------------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def safe_json_loads(text: Any, default: Any = None) -> Any:
    """Parse `text` as JSON, returning `default` instead of raising if
    it isn't valid JSON. Unlike `safe_json_extract`, this does not try
    to recover a JSON object embedded in surrounding text -- use it
    when `text` is expected to already be clean JSON.
    """
    if not isinstance(text, (str, bytes, bytearray)):
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def safe_json_extract(text: Any) -> Optional[Dict[str, Any]]:
    """Defensively extract a JSON *object* from `text`.

    Tries, in order: parsing `text` directly, parsing the contents of
    a markdown code fence, and parsing the first "{...}" substring
    found. This tolerates the kind of minor formatting noise LLMs
    sometimes add around otherwise-valid JSON (stray commentary,
    code fences).

    Returns:
        The parsed dict on success. `None` if `text` is already a
        dict (returned as-is), or `None` if nothing could be parsed
        into a dict at all -- callers should treat any non-dict
        result as "parsing failed" via `isinstance(result, dict)`.
    """
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None

    candidates = [text.strip()]

    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    braced = _JSON_BRACE_RE.search(text)
    if braced:
        candidates.append(braced.group(0).strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


# ----------------------------------------------------------------------
# Normalizing source metadata
# ----------------------------------------------------------------------
def normalize_url(url: Optional[str]) -> str:
    """Normalize a URL for comparison/deduplication: lowercase the
    scheme and host, strip a trailing slash, and drop the query
    string and fragment. Returns "" for falsy/unparseable input.
    """
    if not url or not isinstance(url, str):
        return ""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return url.strip().lower()

    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return f"{netloc}{path}".lower()


def domain_label(url: Optional[str]) -> str:
    """Human-readable domain for a URL, e.g. 'example.com' from
    'https://www.example.com/page?x=1'. Returns 'Unknown source' for
    falsy or unparseable input.
    """
    if not url or not isinstance(url, str):
        return "Unknown source"
    try:
        netloc = urlparse(url.strip()).netloc.lower()
    except ValueError:
        return "Unknown source"
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or "Unknown source"


def normalize_source_metadata(source: Dict[str, Any]) -> Dict[str, str]:
    """Normalize a heterogeneous source/citation dict into a
    consistent shape: {"title": str, "url": str, "domain": str}.

    Accepts common key variants across this project's modules
    (title/name/source, url/href/link) so callers don't need to know
    which module a given source dict originally came from.
    """
    if not isinstance(source, dict):
        return {"title": "Unknown source", "url": "", "domain": "Unknown source"}

    title = source.get("title") or source.get("name") or source.get("source") or "Unknown source"
    url = source.get("url") or source.get("href") or source.get("link") or ""
    url = url.strip() if isinstance(url, str) else ""

    return {
        "title": str(title).strip() or "Unknown source",
        "url": url,
        "domain": domain_label(url) if url else "Unknown source",
    }


# ----------------------------------------------------------------------
# Removing duplicate sources
# ----------------------------------------------------------------------
def dedupe_by_key(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """Remove duplicate dicts from `items` by exact equality of
    `item.get(key)`, keeping the first occurrence of each value.

    Items where the key is missing, `None`, or otherwise falsy are
    always kept (there's no reliable value to compare them by, so
    they're never treated as duplicates of one another).
    """
    seen: Set[Any] = set()
    unique: List[Dict[str, Any]] = []

    for item in items or []:
        value = item.get(key) if isinstance(item, dict) else None
        if value:
            if value in seen:
                continue
            seen.add(value)
        unique.append(item)

    return unique


def dedupe_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate sources by normalized URL, falling back to a
    normalized title when a source has no URL. Keeps the first
    occurrence of each unique source and preserves the original dicts
    (only the comparison key is normalized, not the returned data).
    """
    seen: Set[str] = set()
    unique: List[Dict[str, Any]] = []

    for source in sources or []:
        normalized = normalize_source_metadata(source)
        key = normalize_url(normalized["url"]) or normalized["title"].strip().lower()
        if key:
            if key in seen:
                continue
            seen.add(key)
        unique.append(source)

    return unique


# ----------------------------------------------------------------------
# Formatting confidence scores
# ----------------------------------------------------------------------
def normalize_confidence(score: Any) -> Optional[float]:
    """Normalize a confidence score given on either a 0-1 or 0-100
    scale into a 0.0-1.0 float. Returns `None` if `score` can't be
    interpreted as a number.
    """
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if value > 1:
        value = value / 100.0
    return clamp(value, 0.0, 1.0)


def confidence_label(score: float) -> str:
    """Qualitative label for a 0.0-1.0 confidence score."""
    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def format_confidence(score: Any) -> str:
    """Human-readable confidence string, e.g. '78% (High)'. Returns
    'Unknown' if `score` can't be interpreted as a number.
    """
    normalized = normalize_confidence(score)
    if normalized is None:
        return "Unknown"
    percent = round(normalized * 100)
    return f"{percent}% ({confidence_label(normalized)})"


# ----------------------------------------------------------------------
# Formatting research results
# ----------------------------------------------------------------------
def format_bullet_list(items: Optional[List[Any]], empty_label: str = "None identified.") -> str:
    """Render `items` as a Markdown bullet list, one per line.
    Returns `empty_label` if there are no non-empty items.
    """
    cleaned = [str(item).strip() for item in (items or []) if str(item).strip()]
    if not cleaned:
        return empty_label
    return "\n".join(f"- {item}" for item in cleaned)


def truncate_text(text: str, max_length: int = 280, suffix: str = "...") -> str:
    """Shorten `text` to at most `max_length` characters, appending
    `suffix` if it was truncated. Safe on None/empty input.
    """
    text = (text or "").strip()
    if len(text) <= max_length:
        return text
    cutoff = max(0, max_length - len(suffix))
    return text[:cutoff].rstrip() + suffix


def format_result_as_markdown(result: Dict[str, Any]) -> str:
    """Render a research_agent.py-style result dict (executive_summary,
    key_findings, pros, cons, risks, recommendation, confidence_score,
    sources, ...) as a readable Markdown document.

    Purely presentational: it only formats what's already in `result`
    and performs no analysis, scoring, or decision-making. Missing
    fields are simply omitted; an empty/invalid `result` yields "".
    """
    if not isinstance(result, dict):
        return ""

    sections: List[str] = []

    summary = result.get("executive_summary")
    if summary:
        sections.append(f"## Executive Summary\n\n{summary}")

    findings = result.get("key_findings")
    if findings:
        sections.append(f"## Key Findings\n\n{format_bullet_list(findings)}")

    pros = result.get("pros")
    cons = result.get("cons")
    if pros or cons:
        pros_block = format_bullet_list(pros)
        cons_block = format_bullet_list(cons)
        sections.append(f"## Pros and Cons\n\n**Pros**\n{pros_block}\n\n**Cons**\n{cons_block}")

    risks = result.get("risks")
    if risks:
        sections.append(f"## Risks\n\n{format_bullet_list(risks)}")

    recommendation = result.get("recommendation")
    if recommendation:
        sections.append(f"## Recommendation\n\n{recommendation}")

    if result.get("confidence_score") is not None:
        sections.append(f"## Confidence\n\n{format_confidence(result['confidence_score'])}")

    sources = result.get("sources")
    if sources:
        lines = []
        for source in sources:
            normalized = (
                normalize_source_metadata(source)
                if isinstance(source, dict)
                else {"title": str(source), "url": ""}
            )
            if normalized["url"]:
                lines.append(f"- [{normalized['title']}]({normalized['url']})")
            else:
                lines.append(f"- {normalized['title']}")
        sections.append("## Sources\n\n" + "\n".join(lines))

    return "\n\n".join(sections)
"""
ai_engine.py
------------
Low-level interface to the local Ollama LLM (Llama 3) for the Resume
Screening Agent.

Scope of this file (and ONLY this file):
    - Connect to Ollama and send prompts to a configurable model
    - Receive raw text responses
    - Validate that responses are well-formed JSON
    - Automatically retry when the model returns invalid/malformed JSON
    - Return plain Python dicts/lists (never raw strings) to callers

This file contains NO prompt text (see prompts.py) and NO business logic
(scoring rules, thresholds, hiring decisions) — that belongs to screener.py.
ai_engine.py is a generic "ask the LLM for JSON, get a dict back" layer that
prompts.py and screener.py build on top of.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import ollama

from Prompts import (
    SYSTEM_PROMPT,
    build_resume_extraction_prompt,
    build_comparison_prompt,
)
from Utils import safe_json_parse

logger = logging.getLogger(__name__)

JSONResult = Union[Dict[str, Any], List[Any]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The model name is intentionally configurable (module default + per-call
# override) rather than hardcoded, so the same engine works against
# "llama3", "llama3:8b", "llama3:70b", or any other locally-pulled model.
DEFAULT_MODEL = "llama3"
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.2  # Low temperature favors consistent, structured output


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class AIEngineError(Exception):
    """Base exception for all ai_engine failures."""


class OllamaConnectionError(AIEngineError):
    """Raised when the Ollama server/model cannot be reached or used."""


class InvalidLLMResponseError(AIEngineError):
    """Raised when the LLM fails to return valid JSON after all retries."""


# ---------------------------------------------------------------------------
# Core: low-level call to Ollama
# ---------------------------------------------------------------------------
def _call_ollama(
    prompt: str,
    system_prompt: Optional[str],
    model: str,
    temperature: float,
) -> str:
    """
    Send a single prompt to Ollama and return the raw text response.

    Uses Ollama's native JSON output mode (`format="json"`) as a first
    line of defense for well-formed output, on top of the JSON-only
    instructions already baked into the prompt itself.

    Args:
        prompt: The user-role prompt content (built by prompts.py).
        system_prompt: Optional system-role instruction.
        model: Name of the Ollama model to use.
        temperature: Sampling temperature.

    Returns:
        The raw text content of the model's reply.

    Raises:
        OllamaConnectionError: If the server is unreachable, the model is
            not available, or the response has an unexpected shape.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            format="json",
            options={"temperature": temperature},
        )
    except ollama.ResponseError as exc:
        raise OllamaConnectionError(
            f"Ollama could not fulfill the request for model '{model}'. "
            f"If the model isn't pulled yet, run: ollama pull {model}. "
            f"Original error: {exc}"
        ) from exc
    except ollama.RequestError as exc:
        raise OllamaConnectionError(f"Invalid request sent to Ollama: {exc}") from exc
    except Exception as exc:
        raise OllamaConnectionError(
            f"Could not reach Ollama. Make sure the server is running "
            f"('ollama serve') and reachable. Original error: {exc}"
        ) from exc

    content = getattr(getattr(response, "message", None), "content", None)
    if content is None:
        raise OllamaConnectionError(f"Unexpected response shape from Ollama: {response!r}")

    return content


# ---------------------------------------------------------------------------
# Core: generic "ask for JSON, validate, retry" engine
# ---------------------------------------------------------------------------
def generate_json_response(
    prompt: str,
    system_prompt: Optional[str] = SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    temperature: float = DEFAULT_TEMPERATURE,
) -> JSONResult:
    """
    Send `prompt` to the configured Ollama model and return a validated
    Python object (dict or list) parsed from its JSON response.

    If the model returns invalid/malformed JSON, the request is retried —
    with an increasingly explicit reminder appended to the prompt — up to
    `max_retries` times before raising InvalidLLMResponseError.

    This function is intentionally generic: it has no knowledge of resumes,
    job descriptions, or scoring. It only turns "a prompt asking for JSON"
    into "a Python object", so it can be reused for any structured-output
    prompt built by prompts.py.

    Args:
        prompt: The fully-built prompt to send (see prompts.py builders).
        system_prompt: Optional system-role instruction. Defaults to the
            shared SYSTEM_PROMPT from prompts.py.
        model: Name of the Ollama model to use (configurable, not hardcoded).
        max_retries: Number of attempts before raising InvalidLLMResponseError.
        temperature: Sampling temperature passed to Ollama.

    Returns:
        A parsed Python dict (or list) representing the model's response.

    Raises:
        OllamaConnectionError: If Ollama cannot be reached at all.
        InvalidLLMResponseError: If every attempt fails to yield valid JSON.
    """
    last_raw_response = ""
    current_prompt = prompt

    for attempt in range(1, max_retries + 1):
        last_raw_response = _call_ollama(
            prompt=current_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
        )

        parsed = safe_json_parse(last_raw_response)
        if parsed is not None:
            return parsed

        logger.warning(
            "Attempt %d/%d: model '%s' returned invalid JSON. Retrying...",
            attempt, max_retries, model,
        )

        # Escalate the reminder on retry, but keep the original prompt intact.
        current_prompt = (
            f"{prompt}\n\n"
            f"REMINDER: Your previous response was not valid JSON. "
            f"Respond with ONLY a valid JSON object and nothing else — "
            f"no markdown, no code fences, no commentary."
        )

    raise InvalidLLMResponseError(
        f"Model '{model}' failed to return valid JSON after {max_retries} attempts. "
        f"Last raw response: {last_raw_response[:500]!r}"
    )


# ---------------------------------------------------------------------------
# Thin, named convenience wrappers (wiring only — no business logic)
# ---------------------------------------------------------------------------
def extract_resume_info(
    resume_text: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> JSONResult:
    """
    Extract structured information from raw resume text using the LLM.

    Thin wrapper combining `build_resume_extraction_prompt` (prompts.py)
    with `generate_json_response` (this file). Contains no interpretation
    of the result — that is screener.py's responsibility.

    Args:
        resume_text: Cleaned plain text of a candidate's resume.
        model: Ollama model name to use.
        max_retries: Number of retry attempts on invalid JSON.

    Returns:
        A dict of extracted resume fields (schema defined in prompts.py).
    """
    prompt = build_resume_extraction_prompt(resume_text)
    return generate_json_response(prompt, model=model, max_retries=max_retries)


def compare_resume_to_job(
    resume_data: str,
    job_description: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> JSONResult:
    """
    Compare a candidate's resume/profile against a job description using the LLM.

    Thin wrapper combining `build_comparison_prompt` (prompts.py) with
    `generate_json_response` (this file). Contains no scoring rules or
    decision-making of its own — the LLM produces the assessment, and
    screener.py decides what to do with it.

    Args:
        resume_data: Raw resume text or a JSON string of previously
            extracted resume information.
        job_description: Plain text of the job description.
        model: Ollama model name to use.
        max_retries: Number of retry attempts on invalid JSON.

    Returns:
        A dict describing match score, strengths, gaps, and recommendation
        (schema defined in prompts.py).
    """
    prompt = build_comparison_prompt(resume_data, job_description)
    return generate_json_response(prompt, model=model, max_retries=max_retries)
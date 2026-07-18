"""
screener.py
-----------
Orchestration layer for the Resume Screening Agent.

Scope of this file (and ONLY this file):
    - Accept multiple resume files and one job description
    - Use parser.py to extract raw text from each resume (and, if needed,
      from an uploaded job description file)
    - Use ai_engine.py to extract structured candidate info and to compare
      each candidate against the job description
    - Screen every resume independently, so one bad file/API call never
      stops the rest of the batch
    - Collect results, compute rankings from the returned match scores,
      and return a single sorted list ready for display

This file contains ONLY orchestration logic. It does not talk to Ollama
directly (see ai_engine.py), does not parse files directly (see parser.py),
and does not define prompts (see prompts.py).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import ai_engine
import Parser as resume_parser  # local module; aliased to avoid confusion with argparse-style naming
from Parser import ResumeParsingError
from Utils import clean_text, normalize_score, safe_get, sort_candidates, truncate_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Caps on how much text is sent to the LLM per document, to keep prompts
# within a reasonable context size regardless of how long a source file is.
MAX_RESUME_TEXT_CHARS = 8000
MAX_JOB_DESCRIPTION_CHARS = 4000

SCORE_KEY = "Match Score"  # Column used for ranking; matches app.py's expected schema


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class ScreeningError(Exception):
    """
    Raised for orchestration-level failures that prevent screening from
    starting at all (e.g. no resumes provided, no job description
    resolvable). Failures for an *individual* resume are never raised —
    they are captured as an error row in the results instead.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_display_name(file_obj: Any) -> str:
    """Best-effort resolution of a human-readable name for a path or file-like object."""
    name = getattr(file_obj, "name", None)
    if name:
        return os.path.basename(str(name))
    if isinstance(file_obj, (str, os.PathLike)):
        return os.path.basename(str(file_obj))
    return "unknown_file"


def _resolve_job_description(job_description: str, job_description_file: Optional[Any]) -> str:
    """
    Resolve the final job description text from either pasted text or an
    uploaded file. Pasted text takes precedence when both are supplied.

    Supports .pdf/.docx (via parser.py) and plain .txt files.

    Raises:
        ScreeningError: If a job description file is supplied but cannot be read.
    """
    if job_description and job_description.strip():
        return truncate_text(clean_text(job_description), max_length=MAX_JOB_DESCRIPTION_CHARS)

    if job_description_file is None:
        return ""

    file_name = _get_display_name(job_description_file)
    extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    try:
        if extension == "txt":
            if hasattr(job_description_file, "seek"):
                job_description_file.seek(0)
            raw = job_description_file.read()
            text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
        else:
            # .pdf / .docx handled by parser.py
            text = resume_parser.extract_text(job_description_file)
    except ResumeParsingError as exc:
        raise ScreeningError(f"Could not read job description file '{file_name}': {exc}") from exc
    except Exception as exc:
        raise ScreeningError(
            f"Unexpected error reading job description file '{file_name}': {exc}"
        ) from exc

    return truncate_text(clean_text(text), max_length=MAX_JOB_DESCRIPTION_CHARS)


def _build_error_result(file_name: str, error_message: str, candidate_name: str = "") -> Dict[str, Any]:
    """Build a result row for a resume that failed at any stage of screening."""
    logger.error("Screening failed for '%s': %s", file_name, error_message)
    return {
        "File Name": file_name,
        "Candidate Name": candidate_name or "Unknown",
        SCORE_KEY: None,  # None (not 0) so genuinely low-scoring candidates still rank above failures
        "Recommendation": "Error",
        "Key Strengths": "",
        "Key Gaps": error_message,
        "Matching Skills": "",
        "Missing Skills": "",
        "Status": "Failed",
    }


def _build_success_result(
    file_name: str,
    extracted_info: Dict[str, Any],
    comparison: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a result row for a resume that was successfully screened."""
    candidate_name = (
        safe_get(comparison, "candidate_name", default="")
        or safe_get(extracted_info, "full_name", default="")
        or file_name
    )

    matching_skills = safe_get(comparison, "matching_skills", default=[])
    missing_skills = safe_get(comparison, "missing_skills", default=[])

    return {
        "File Name": file_name,
        "Candidate Name": candidate_name,
        SCORE_KEY: normalize_score(safe_get(comparison, "match_score", default=0)),
        "Recommendation": safe_get(comparison, "recommendation", default="") or "Not Available",
        "Key Strengths": safe_get(comparison, "strengths", default="") or "",
        "Key Gaps": safe_get(comparison, "gaps", default="") or "",
        "Matching Skills": ", ".join(matching_skills) if isinstance(matching_skills, list) else str(matching_skills),
        "Missing Skills": ", ".join(missing_skills) if isinstance(missing_skills, list) else str(missing_skills),
        "Status": "Success",
    }


def _screen_single_resume(
    resume_file: Any,
    job_description_text: str,
    model: str,
    max_retries: int,
) -> Dict[str, Any]:
    """
    Run the full pipeline (parse -> extract -> compare) for a single resume.

    Any failure at any stage is caught here and converted into an error
    result row, so a single bad file never aborts the rest of the batch.
    """
    file_name = _get_display_name(resume_file)

    # --- Stage 1: extract raw text from the file ---
    try:
        raw_text = resume_parser.extract_text(resume_file)
    except ResumeParsingError as exc:
        return _build_error_result(file_name, f"Could not read resume: {exc}")
    except Exception as exc:  # Defensive: never let one bad file crash the batch
        return _build_error_result(file_name, f"Unexpected error while reading file: {exc}")

    resume_text = truncate_text(clean_text(raw_text), max_length=MAX_RESUME_TEXT_CHARS)

    # --- Stage 2: extract structured candidate info via the LLM ---
    try:
        extracted_info = ai_engine.extract_resume_info(
            resume_text, model=model, max_retries=max_retries
        )
    except ai_engine.AIEngineError as exc:
        return _build_error_result(file_name, f"AI extraction failed: {exc}")
    except Exception as exc:
        return _build_error_result(file_name, f"Unexpected error during AI extraction: {exc}")

    if not isinstance(extracted_info, dict):
        extracted_info = {}

    candidate_name_so_far = safe_get(extracted_info, "full_name", default="")

    # --- Stage 3: compare the extracted profile against the job description ---
    try:
        resume_payload = json.dumps(extracted_info, ensure_ascii=False)
        comparison = ai_engine.compare_resume_to_job(
            resume_payload, job_description_text, model=model, max_retries=max_retries
        )
    except ai_engine.AIEngineError as exc:
        return _build_error_result(file_name, f"AI comparison failed: {exc}", candidate_name_so_far)
    except Exception as exc:
        return _build_error_result(
            file_name, f"Unexpected error during AI comparison: {exc}", candidate_name_so_far
        )

    if not isinstance(comparison, dict):
        comparison = {}

    return _build_success_result(file_name, extracted_info, comparison)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def screen_candidates(
    resume_files: List[Any],
    job_description: str = "",
    job_description_file: Optional[Any] = None,
    model: str = ai_engine.DEFAULT_MODEL,
    max_retries: int = ai_engine.DEFAULT_MAX_RETRIES,
) -> List[Dict[str, Any]]:
    """
    Screen every uploaded resume independently against a single job
    description and return a ranked list of candidate results.

    Args:
        resume_files: List of resume file paths or file-like objects
            (e.g. Streamlit UploadedFile), PDF or DOCX.
        job_description: Plain text of the job description. Takes
            precedence over `job_description_file` if both are given.
        job_description_file: Optional uploaded job description file
            (.pdf, .docx, or .txt), used if `job_description` is empty.
        model: Ollama model name to use for every LLM call in this batch.
        max_retries: Retry attempts per LLM call on invalid JSON.

    Returns:
        A list of result dicts, one per resume, sorted by "Match Score"
        (highest first). Each dict includes a 1-based "Rank" key. Resumes
        that failed at any stage are included with Status="Failed" and a
        "Match Score" of None, sinking to the bottom of the ranking.

    Raises:
        ScreeningError: If no resumes were provided, or if no job
            description text could be resolved from either input.
    """
    if not resume_files:
        raise ScreeningError("No resume files were provided.")

    job_description_text = _resolve_job_description(job_description, job_description_file)
    if not job_description_text.strip():
        raise ScreeningError(
            "No job description text could be resolved. "
            "Provide job description text or a readable file."
        )

    results: List[Dict[str, Any]] = []
    for resume_file in resume_files:
        result = _screen_single_resume(resume_file, job_description_text, model, max_retries)
        results.append(result)

    ranked_results = sort_candidates(results, score_key=SCORE_KEY, descending=True)

    # Assign an explicit 1-based rank now that the final order is settled.
    for position, result in enumerate(ranked_results, start=1):
        result["Rank"] = position

    return ranked_results
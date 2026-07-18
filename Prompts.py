"""
prompts.py
----------
Reusable prompt templates for the Resume Screening Agent's LLM calls.

Scope of this file (and ONLY this file):
    - Define the prompt templates used for:
        1. Resume information extraction
        2. Resume vs. Job Description comparison
    - Both templates strictly instruct the LLM to return ONLY valid JSON,
      matching a fixed, documented schema.

This file contains NO API calls, NO HTTP/Ollama client code, and NO
response parsing/validation. It only builds prompt strings — ai_engine.py
is responsible for actually calling the LLM and handling its output.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Shared system prompt
# ---------------------------------------------------------------------------
# Sets the model's role/tone once; reused as the "system" message by
# ai_engine.py for every call so behavior stays consistent across prompts.
SYSTEM_PROMPT = (
    "You are a precise, detail-oriented AI assistant embedded in an "
    "automated resume screening pipeline. You always follow output-format "
    "instructions exactly. You never add explanations, greetings, or "
    "commentary outside the exact format requested."
)


# ---------------------------------------------------------------------------
# Shared JSON-only output contract
# ---------------------------------------------------------------------------
_JSON_ONLY_INSTRUCTIONS = """
IMPORTANT OUTPUT RULES:
- Respond with ONLY a single valid JSON object. Nothing else.
- Do NOT include any explanations, apologies, greetings, or commentary.
- Do NOT wrap the JSON in markdown code fences (no ``` of any kind).
- Do NOT include trailing commas or comments.
- All keys and string values must use double quotes, per the JSON spec.
- If a field's value is unknown or not present in the source text, use an
  empty string "" (for strings), an empty list [] (for lists), or 0 (for
  numbers) as appropriate — never omit a key that is part of the schema.
""".strip()


# ---------------------------------------------------------------------------
# Template 1: Resume information extraction
# ---------------------------------------------------------------------------
# Documented schema for extracted resume data. Exported so other modules
# (e.g. screener.py) can validate/reference the expected keys without
# duplicating them.
RESUME_EXTRACTION_SCHEMA = """{
  "full_name": "string",
  "email": "string",
  "phone": "string",
  "total_years_experience": 0,
  "skills": ["string", "..."],
  "education": [
    {"degree": "string", "institution": "string", "year": "string"}
  ],
  "work_experience": [
    {"title": "string", "company": "string", "duration": "string", "summary": "string"}
  ],
  "certifications": ["string", "..."]
}"""

RESUME_EXTRACTION_REQUIRED_KEYS = [
    "full_name",
    "email",
    "phone",
    "total_years_experience",
    "skills",
    "education",
    "work_experience",
    "certifications",
]


def build_resume_extraction_prompt(resume_text: str) -> str:
    """
    Build a prompt instructing the LLM to extract structured information
    from raw resume text and return it as JSON matching
    RESUME_EXTRACTION_SCHEMA.

    Args:
        resume_text: Cleaned plain text of a candidate's resume
            (see parser.py / utils.clean_text).

    Returns:
        A complete prompt string ready to send to the LLM.
    """
    return f"""You are an expert technical recruiter and resume parser.

Read the resume text below and extract structured information from it.

RESUME TEXT:
\"\"\"
{resume_text}
\"\"\"

Return the extracted information as a JSON object with EXACTLY this schema
(use these exact keys, in this structure):

{RESUME_EXTRACTION_SCHEMA}

{_JSON_ONLY_INSTRUCTIONS}
"""


# ---------------------------------------------------------------------------
# Template 2: Resume vs. Job Description comparison
# ---------------------------------------------------------------------------
COMPARISON_SCHEMA = """{
  "candidate_name": "string",
  "match_score": 0,
  "matching_skills": ["string", "..."],
  "missing_skills": ["string", "..."],
  "strengths": "string",
  "gaps": "string",
  "recommendation": "Strong Fit | Consider | Not a Fit"
}"""

COMPARISON_REQUIRED_KEYS = [
    "candidate_name",
    "match_score",
    "matching_skills",
    "missing_skills",
    "strengths",
    "gaps",
    "recommendation",
]


def build_comparison_prompt(resume_data: str, job_description: str) -> str:
    """
    Build a prompt instructing the LLM to compare a candidate's resume
    against a job description and return a structured match assessment
    as JSON matching COMPARISON_SCHEMA.

    Args:
        resume_data: Either raw resume text, or a JSON string of
            previously extracted resume information (the output of
            build_resume_extraction_prompt + ai_engine's extraction call).
        job_description: Plain text of the job description.

    Returns:
        A complete prompt string ready to send to the LLM.
    """
    return f"""You are an expert technical recruiter evaluating a candidate
against a specific job opening.

JOB DESCRIPTION:
\"\"\"
{job_description}
\"\"\"

CANDIDATE RESUME / PROFILE:
\"\"\"
{resume_data}
\"\"\"

Compare the candidate against the job description and assess fit.
- "match_score" must be an integer from 0 to 100 (100 = perfect match).
- "matching_skills" and "missing_skills" must be evaluated against the
  requirements stated or implied in the job description.
- "recommendation" must be exactly one of: "Strong Fit", "Consider", "Not a Fit".

Return your assessment as a JSON object with EXACTLY this schema
(use these exact keys, in this structure):

{COMPARISON_SCHEMA}

{_JSON_ONLY_INSTRUCTIONS}
"""
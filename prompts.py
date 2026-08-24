"""
prompts.py — Prompt templates for the AI Interviewer's LLM calls.

Every prompt built here instructs the model to return ONLY a single valid
JSON object — no markdown code fences, no commentary, no text outside the
JSON. This module builds prompt text ONLY; it never calls the model,
parses responses, or validates JSON — that's ai_engine.py's job.

Four tasks are covered, one prompt builder (+ matching system prompt) each:
  1. Resume analysis
  2. Interview question generation
  3. Answer evaluation
  4. Final interview report
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

__all__ = [
    "JSON_ONLY_INSTRUCTION",
    "resume_analysis_system_prompt",
    "resume_analysis_prompt",
    "question_generation_system_prompt",
    "question_generation_prompt",
    "answer_evaluation_system_prompt",
    "answer_evaluation_prompt",
    "final_report_system_prompt",
    "final_report_prompt",
    "retry_correction_prompt",
]

# Appended to the end of every task prompt so the instruction is the last
# thing the model reads (models tend to weight recency in the prompt).
JSON_ONLY_INSTRUCTION = (
    "Respond with ONLY a single valid JSON object. Do not include markdown "
    "code fences, explanations, disclaimers, or any text outside the JSON "
    "object. The response must start with '{' and end with '}'."
)


# --------------------------------------------------------------------------
# 1. Resume analysis
# --------------------------------------------------------------------------
def resume_analysis_system_prompt() -> str:
    return (
        "You are an expert technical recruiter analyzing a candidate's resume "
        "ahead of a mock interview. You are precise, extract only what the "
        "resume actually supports, and never fabricate details. "
        f"{JSON_ONLY_INSTRUCTION}"
    )


def resume_analysis_prompt(resume_text: str) -> str:
    """
    Build the prompt asking the model to analyze a resume into a structured
    candidate profile used to tailor interview questions.

    Expected JSON response shape:
        {
            "candidate_name": string | null,
            "seniority_level": "junior" | "mid" | "senior" | "lead" | "unknown",
            "primary_role": string,
            "skills": [string, ...],
            "years_of_experience": number | null,
            "key_projects": [string, ...],
            "summary": string
        }
    """
    return (
        "Analyze the following resume and extract structured information "
        "about the candidate.\n\n"
        f'Resume:\n"""\n{resume_text.strip()}\n"""\n\n'
        "Return a JSON object with EXACTLY these keys:\n"
        '- "candidate_name": string, or null if it cannot be determined\n'
        '- "seniority_level": one of "junior", "mid", "senior", "lead", "unknown"\n'
        '- "primary_role": string, the candidate\'s primary job title or field\n'
        '- "skills": array of strings, the key technical/professional skills found\n'
        '- "years_of_experience": number, or null if it cannot be estimated\n'
        '- "key_projects": array of strings, notable projects or achievements\n'
        '- "summary": string, a 2-3 sentence overview of the candidate\n\n'
        f"{JSON_ONLY_INSTRUCTION}"
    )


# --------------------------------------------------------------------------
# 2. Interview question generation
# --------------------------------------------------------------------------
def question_generation_system_prompt() -> str:
    return (
        "You are an experienced technical interviewer who writes thoughtful, "
        "role-relevant interview questions tailored to a specific candidate. "
        f"{JSON_ONLY_INSTRUCTION}"
    )


def question_generation_prompt(
    resume_analysis: Dict[str, Any],
    num_questions: int = 5,
    difficulty: str = "medium",
) -> str:
    """
    Build the prompt asking the model to generate a set of interview
    questions tailored to the candidate's profile.

    Expected JSON response shape:
        {"questions": [string, string, ...]}   # exactly `num_questions` items
    """
    return (
        f"Based on the candidate profile below, generate exactly {num_questions} "
        f"interview questions at {difficulty} difficulty. Mix behavioral questions "
        "with role-specific technical questions grounded in the candidate's actual "
        "skills and projects. Order them from warm-up to more in-depth. Each "
        "question must be self-contained and understandable without any other "
        "context.\n\n"
        f"Candidate profile (JSON):\n{json.dumps(resume_analysis, indent=2)}\n\n"
        "Return a JSON object with EXACTLY this key:\n"
        f'- "questions": an array of exactly {num_questions} strings, each a '
        "complete, standalone interview question\n\n"
        f"{JSON_ONLY_INSTRUCTION}"
    )


# --------------------------------------------------------------------------
# 3. Answer evaluation
# --------------------------------------------------------------------------
def answer_evaluation_system_prompt() -> str:
    return (
        "You are a fair, constructive interview evaluator. You score answers "
        "honestly based on clarity, relevance, and depth, and you never "
        "inflate scores to be encouraging. " + JSON_ONLY_INSTRUCTION
    )


def answer_evaluation_prompt(
    question: str,
    answer: str,
    resume_analysis: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build the prompt asking the model to score and critique one candidate
    answer to one interview question.

    Expected JSON response shape:
        {
            "score": number,          # 0-10, one decimal place is fine
            "feedback": string,       # 2-4 sentences, specific and constructive
            "strengths": [string, ...],
            "improvements": [string, ...]
        }
    """
    context = ""
    if resume_analysis:
        context = f"Candidate background (JSON):\n{json.dumps(resume_analysis, indent=2)}\n\n"

    return (
        f"{context}"
        f'Interview question:\n"{question.strip()}"\n\n'
        f'Candidate\'s answer:\n"""\n{answer.strip()}\n"""\n\n'
        "Evaluate this answer on a 0-10 scale, considering clarity, relevance "
        "to the question, accuracy, and depth of explanation. Be constructive "
        "but honest — do not inflate the score.\n\n"
        "Return a JSON object with EXACTLY these keys:\n"
        '- "score": a number from 0 to 10 (one decimal place is fine)\n'
        '- "feedback": string, 2-4 sentences of specific, constructive feedback\n'
        '- "strengths": array of strings, what the answer did well (can be empty)\n'
        '- "improvements": array of strings, what could be improved (can be empty)\n\n'
        f"{JSON_ONLY_INSTRUCTION}"
    )


# --------------------------------------------------------------------------
# 4. Final interview report
# --------------------------------------------------------------------------
def final_report_system_prompt() -> str:
    return (
        "You are an interview panel lead writing a final candidate assessment "
        "based on a completed mock interview. You weigh the full transcript, "
        "not just the average score. " + JSON_ONLY_INSTRUCTION
    )


def final_report_prompt(
    resume_analysis: Dict[str, Any],
    qa_history: List[Dict[str, Any]],
) -> str:
    """
    Build the prompt asking the model to synthesize a final interview report
    from the complete question/answer/feedback/score history.

    `qa_history` items are expected to look like:
        {"question": string, "answer": string, "feedback": string, "score": number}

    Expected JSON response shape:
        {
            "overall_score": number,             # 0-10
            "summary": string,                   # 3-5 sentences
            "strengths": [string, ...],
            "areas_for_improvement": [string, ...],
            "recommendation": "Strong Hire" | "Hire" | "No Hire" | "Strong No Hire"
        }
    """
    return (
        "Review the full mock interview below and write a final assessment.\n\n"
        f"Candidate profile (JSON):\n{json.dumps(resume_analysis, indent=2)}\n\n"
        "Interview transcript (JSON array, one entry per question):\n"
        f"{json.dumps(qa_history, indent=2)}\n\n"
        "Return a JSON object with EXACTLY these keys:\n"
        '- "overall_score": a number from 0 to 10, your overall assessment '
        "(this does not need to be a simple average of the individual scores)\n"
        '- "summary": string, 3-5 sentences summarizing overall performance\n'
        '- "strengths": array of strings, the candidate\'s top strengths across the interview\n'
        '- "areas_for_improvement": array of strings, key areas to work on\n'
        '- "recommendation": one of "Strong Hire", "Hire", "No Hire", "Strong No Hire"\n\n'
        f"{JSON_ONLY_INSTRUCTION}"
    )


# --------------------------------------------------------------------------
# Retry helper
# --------------------------------------------------------------------------
def retry_correction_prompt(original_prompt: str, error_message: str) -> str:
    """
    Wrap an original task prompt with a corrective instruction after an
    invalid JSON response, to be resent to the model on a retry attempt.

    Args:
        original_prompt: The exact prompt text sent on the failed attempt.
        error_message: A short, human-readable description of what was
            wrong with the previous response (e.g. "missing key 'score'").
    """
    return (
        f"{original_prompt}\n\n"
        "---\n"
        f"IMPORTANT: Your previous response was invalid ({error_message}). "
        "Respond again with ONLY a single valid JSON object exactly matching "
        "the schema requested above, and nothing else — no markdown, no "
        "commentary, no repeated explanation."
    )
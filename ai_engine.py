"""
ai_engine.py — LLM orchestration layer for the AI Interviewer (Ollama + Llama 3).

Two layers, kept deliberately separate:

  1. `OllamaClient` — a thin wrapper around the Ollama chat API. It knows
     how to send a prompt and get raw text back. It knows NOTHING about
     prompts, JSON schemas, or interview logic.

  2. `AIEngine` — the task-specific layer (resume analysis, question
     generation, answer evaluation, final report). It builds prompts via
     prompts.py, calls `OllamaClient`, validates the JSON response against
     an expected schema, and retries with a corrective prompt on failure.

Neither class knows about interview_id, sessions, Streamlit, or the
database — that orchestration belongs in interviewer.py. ai_engine.py's
only job is: build prompt -> call model -> return validated structured data.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import prompts
from Utils import clamp, extract_json_from_text, safe_json_loads, validate_json_keys

logger = logging.getLogger(__name__)

__all__ = [
    "AIEngine",
    "OllamaClient",
    "AIEngineError",
    "OllamaConnectionError",
    "InvalidResponseError",
    "DEFAULT_MODEL",
]

# Configurable via environment variable so the model can be changed without
# touching code (e.g. OLLAMA_MODEL=llama3:70b). Falls back to plain "llama3".
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST")  # None -> ollama-python's own default
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 120

# Required top-level keys for each task's JSON response. Used to validate
# the model's output before it's trusted by the rest of the application.
RESUME_ANALYSIS_KEYS = (
    "candidate_name",
    "seniority_level",
    "primary_role",
    "skills",
    "years_of_experience",
    "key_projects",
    "summary",
)
QUESTION_GENERATION_KEYS = ("questions",)
ANSWER_EVALUATION_KEYS = ("score", "feedback", "strengths", "improvements")
FINAL_REPORT_KEYS = (
    "overall_score",
    "summary",
    "strengths",
    "areas_for_improvement",
    "recommendation",
)

try:
    import ollama  # type: ignore
except ImportError:  # pragma: no cover - exercised only when dependency missing
    ollama = None  # noqa: N816


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------
class AIEngineError(Exception):
    """Base exception for all ai_engine failures."""


class OllamaConnectionError(AIEngineError):
    """Raised when the Ollama server/model can't be reached or errors out."""


class InvalidResponseError(AIEngineError):
    """Raised when the model's response isn't valid JSON after all retries."""


# --------------------------------------------------------------------------
# Layer 1: raw API access only — no prompt or schema knowledge
# --------------------------------------------------------------------------
class OllamaClient:
    """Minimal wrapper around the Ollama chat API. One job: text in, text out."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: Optional[str] = DEFAULT_HOST,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if ollama is None:
            raise OllamaConnectionError(
                "The 'ollama' package is not installed. Install it with `pip install ollama`."
            )
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._client = self._build_client(host, timeout)

    @staticmethod
    def _build_client(host: Optional[str], timeout: int):
        """Construct the underlying ollama.Client, tolerating version differences."""
        kwargs = {"host": host} if host else {}
        try:
            return ollama.Client(timeout=timeout, **kwargs)
        except TypeError:
            # Older/newer ollama-python releases may not accept `timeout` directly.
            return ollama.Client(**kwargs)

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Send one prompt to the model and return the raw text response."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": self.temperature},
            )
        except Exception as exc:  # noqa: BLE001 - normalize any client/HTTP error
            logger.exception("Ollama request failed (model=%s)", self.model)
            raise OllamaConnectionError(
                f"Could not reach Ollama or model '{self.model}': {exc}. "
                "Make sure Ollama is running (`ollama serve`) and the model "
                f"is pulled (`ollama pull {self.model}`)."
            ) from exc

        content = self._extract_content(response)
        if not content:
            raise OllamaConnectionError("Ollama returned an empty response.")
        return content

    @staticmethod
    def _extract_content(response: Any) -> str:
        # ollama-python has returned both a plain dict and a typed response
        # object across versions — handle both without assuming one shape.
        if isinstance(response, dict):
            message = response.get("message") or {}
            content = message.get("content") if isinstance(message, dict) else None
        else:
            message = getattr(response, "message", None)
            content = getattr(message, "content", None) if message is not None else None
        return (content or "").strip()


# --------------------------------------------------------------------------
# Layer 2: task-specific AI operations — prompts + validation + retry
# --------------------------------------------------------------------------
class AIEngine:
    """
    High-level AI operations for the interviewer: resume analysis, question
    generation, answer evaluation, and final report generation.

    Every method returns a plain Python dict/list validated against a fixed
    JSON schema (see the *_KEYS constants above). If the model's response is
    missing required keys or isn't valid JSON, the request is retried with a
    corrective prompt up to `max_retries` times before raising
    `InvalidResponseError`.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: Optional[str] = DEFAULT_HOST,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        client: Optional[OllamaClient] = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")
        self.max_retries = max_retries
        # A pre-built client can be injected — useful for tests or for
        # sharing one client across multiple AIEngine instances.
        self.client = client or OllamaClient(
            model=model, host=host, temperature=temperature, timeout=timeout
        )

    @property
    def model(self) -> str:
        """The configured Ollama model name (e.g. 'llama3', 'llama3:70b')."""
        return self.client.model

    # ---------------------------------------------------------------- #
    # Task 1: Resume analysis
    # ---------------------------------------------------------------- #
    def analyze_resume(self, resume_text: str) -> Dict[str, Any]:
        """
        Analyze resume text and return a structured candidate profile.
        See prompts.resume_analysis_prompt() for the exact JSON schema.
        """
        if not resume_text or not resume_text.strip():
            raise AIEngineError("Resume text is empty; nothing to analyze.")

        return self._call_and_validate_json(
            prompt=prompts.resume_analysis_prompt(resume_text),
            system=prompts.resume_analysis_system_prompt(),
            required_keys=RESUME_ANALYSIS_KEYS,
            task_name="resume analysis",
        )

    # ---------------------------------------------------------------- #
    # Task 2: Interview question generation
    # ---------------------------------------------------------------- #
    def generate_questions(
        self,
        resume_analysis: Dict[str, Any],
        num_questions: int = 5,
        difficulty: str = "medium",
    ) -> List[str]:
        """
        Generate interview questions tailored to the candidate's profile.

        Returns:
            A list of question strings. Callers should treat the count as
            best-effort (the model is asked for an exact count but may not
            always comply) rather than assuming `len(result) == num_questions`.
        """
        if num_questions < 1:
            raise ValueError("num_questions must be at least 1.")

        result = self._call_and_validate_json(
            prompt=prompts.question_generation_prompt(resume_analysis, num_questions, difficulty),
            system=prompts.question_generation_system_prompt(),
            required_keys=QUESTION_GENERATION_KEYS,
            task_name="question generation",
        )

        questions = result.get("questions")
        if not isinstance(questions, list) or not questions:
            raise InvalidResponseError(
                "Question generation returned an empty or malformed 'questions' list."
            )
        cleaned = [str(q).strip() for q in questions if str(q).strip()]
        if not cleaned:
            raise InvalidResponseError("Question generation returned no usable questions.")
        return cleaned

    # ---------------------------------------------------------------- #
    # Task 3: Answer evaluation
    # ---------------------------------------------------------------- #
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        resume_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Score and critique a single candidate answer.

        Returns a dict with "score" (float, 0-10), "feedback" (str),
        "strengths" (list[str]), and "improvements" (list[str]).
        See prompts.answer_evaluation_prompt() for the exact JSON schema.
        """
        if not question or not question.strip():
            raise AIEngineError("Question text is empty; cannot evaluate an answer to nothing.")
        if not answer or not answer.strip():
            raise AIEngineError("Answer text is empty; cannot evaluate an empty answer.")

        result = self._call_and_validate_json(
            prompt=prompts.answer_evaluation_prompt(question, answer, resume_analysis),
            system=prompts.answer_evaluation_system_prompt(),
            required_keys=ANSWER_EVALUATION_KEYS,
            task_name="answer evaluation",
        )
        result["score"] = self._coerce_score(result.get("score"))
        return result

    # ---------------------------------------------------------------- #
    # Task 4: Final interview report
    # ---------------------------------------------------------------- #
    def generate_summary(
        self,
        resume_analysis: Dict[str, Any],
        qa_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Synthesize the final interview report from the complete Q&A history.

        Args:
            resume_analysis: The candidate profile from analyze_resume().
            qa_history: List of {"question", "answer", "feedback", "score"}
                dicts, one per answered question, in order.

        Returns a dict with "overall_score" (float, 0-10), "summary" (str),
        "strengths" (list[str]), "areas_for_improvement" (list[str]), and
        "recommendation" (str). See prompts.final_report_prompt() for the
        exact JSON schema.
        """
        if not qa_history:
            raise AIEngineError("qa_history is empty; cannot generate a report with no answers.")

        result = self._call_and_validate_json(
            prompt=prompts.final_report_prompt(resume_analysis, qa_history),
            system=prompts.final_report_system_prompt(),
            required_keys=FINAL_REPORT_KEYS,
            task_name="final report generation",
        )
        result["overall_score"] = self._coerce_score(result.get("overall_score"))
        return result

    # ---------------------------------------------------------------- #
    # Internal: call the model, validate JSON, retry on failure
    # ---------------------------------------------------------------- #
    def _call_and_validate_json(
        self,
        prompt: str,
        system: str,
        required_keys: Iterable[str],
        task_name: str,
    ) -> Dict[str, Any]:
        """
        Call the model and parse/validate its response as JSON, retrying
        with a corrective prompt (see prompts.retry_correction_prompt) up
        to `self.max_retries` times.

        Raises:
            OllamaConnectionError: Propagated immediately (not retried) —
                a connection/model error won't be fixed by re-prompting.
            InvalidResponseError: If valid JSON matching the schema still
                hasn't been returned after all retries.
        """
        required_keys = tuple(required_keys)
        current_prompt = prompt
        last_error = "unknown error"
        last_raw_response = ""

        for attempt in range(1, self.max_retries + 1):
            last_raw_response = self.client.generate(prompt=current_prompt, system=system)

            parsed = extract_json_from_text(last_raw_response)
            if parsed is None:
                parsed = safe_json_loads(last_raw_response)

            if isinstance(parsed, dict):
                is_valid, missing_keys = validate_json_keys(parsed, required_keys)
                if is_valid:
                    return parsed
                last_error = f"missing required key(s): {missing_keys}"
            else:
                last_error = "response was not a valid JSON object"

            logger.warning(
                "%s: attempt %d/%d failed validation (%s)",
                task_name,
                attempt,
                self.max_retries,
                last_error,
            )
            if attempt < self.max_retries:
                current_prompt = prompts.retry_correction_prompt(prompt, last_error)

        raise InvalidResponseError(
            f"{task_name} did not return valid JSON after {self.max_retries} attempt(s) "
            f"({last_error}). Last raw response: {last_raw_response[:300]!r}"
        )

    @staticmethod
    def _coerce_score(score: Any) -> float:
        """Normalize a score value to a float clamped to [0, 10]; 0.0 if invalid."""
        try:
            value = float(score)
        except (TypeError, ValueError):
            return 0.0
        return clamp(value, 0.0, 10.0)
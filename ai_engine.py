"""
ai_engine.py

Enterprise AI Research & Decision Intelligence Agent
------------------------------------------------------
AI processing module. All communication with the local Llama 3 model
(served via Ollama) lives here, and nowhere else in the project.

Responsibilities (and ONLY these):
    - Send prompts to Ollama and return parsed, structured (JSON) results.
    - Provide one reusable function per stage of the research workflow:
      research analysis, evidence synthesis, alternative comparison,
      risk analysis, recommendation generation, and final report
      generation.
    - Request structured JSON output from the model and defensively
      parse it, tolerating minor formatting noise (markdown fences,
      stray text) without silently accepting garbage.
    - Distinguish "could not reach / use Ollama" failures from
      "Ollama responded but the output was not usable" failures, so
      callers (research_agent.py) can react appropriately.

This module does NOT implement:
    - Web search                       (see web_search.py)
    - Document parsing / RAG           (see rag.py)
    - Workflow orchestration           (see research_agent.py)
    - Any UI                           (see app.py)
    - Prompt text -- every prompt comes from prompts.py. This module
      never constructs or edits instructional/prompt text itself,
      including "avoid unsupported claims" style guardrails; those
      belong in every prompts.py builder function.

Configuration
--------------
The Ollama model, host, and generation options are configurable via
constructor arguments or environment variables:

    OLLAMA_MODEL   (default: "llama3")
    OLLAMA_HOST    (default: "http://localhost:11434")

Contract with prompts.py
--------------------------
prompts.py must expose one prompt-builder per stage, each returning a
(system_prompt, user_prompt) tuple, and each responsible for grounding
the model in the supplied context/evidence and instructing it to avoid
unsupported claims:

    build_research_analysis_prompt(question, context) -> (str, str)
    build_evidence_synthesis_prompt(question, evidence) -> (str, str)
    build_alternative_comparison_prompt(question, evidence, alternatives) -> (str, str)
    build_risk_analysis_prompt(question, context, evidence) -> (str, str)
    build_recommendation_prompt(question, findings, pros, cons, risks) -> (str, str)
    build_final_report_prompt(question, sections) -> (str, str)

Where:
    context      : str                     -- free-form research context
    evidence     : list[dict]               -- [{"id", "source", "url", "text"}, ...]
    alternatives : list[str]                -- named options, may be empty
    findings/pros/cons/risks : list[str]
    sections     : dict                     -- outputs of the earlier stages,
                                                keyed by stage name
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from ollama import Client, ResponseError

import prompts

logger = logging.getLogger(__name__)

__all__ = [
    "AIEngine",
    "AIEngineError",
    "OllamaConnectionError",
    "MalformedResponseError",
]

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_TEMPERATURE = 0.2
DEFAULT_JSON_RETRIES = 1  # extra attempts if the model's output isn't valid JSON


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
class AIEngineError(Exception):
    """Base class for all ai_engine failures."""


class OllamaConnectionError(AIEngineError):
    """Raised when Ollama cannot be reached, rejects the request, or
    otherwise fails to produce a response at all (network failure,
    server not running, unknown model, timeout, etc.).
    """


class MalformedResponseError(AIEngineError):
    """Raised when Ollama responded, but no valid JSON object could be
    extracted from its output after all parsing attempts.
    """


# ----------------------------------------------------------------------
# AIEngine
# ----------------------------------------------------------------------
class AIEngine:
    """Reusable client for running structured reasoning tasks against a
    local Llama 3 model served by Ollama.

    Each public method corresponds to one stage of the research
    workflow, accepts the research context / source information
    relevant to that stage, and returns a parsed JSON dict. Prompt text
    is never authored here -- every method delegates to a builder
    function imported from prompts.py.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        temperature: float = DEFAULT_TEMPERATURE,
        json_retries: int = DEFAULT_JSON_RETRIES,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.json_retries = max(0, json_retries)
        self._client = Client(host=self.host, timeout=self.timeout)

    # ------------------------------------------------------------------
    # Public, reusable, stage-specific functions
    # ------------------------------------------------------------------
    def analyze_research(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Initial pass over the question (and any free-form research
        context) to frame the research angle and key sub-questions
        before deeper evidence synthesis.
        """
        system_prompt, user_prompt = prompts.build_research_analysis_prompt(
            question=question, context=context or ""
        )
        return self._generate_json(system_prompt, user_prompt)

    def synthesize_evidence(
        self, question: str, evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine web and document evidence into key findings, each
        grounded in a specific evidence id (see EvidenceItem in
        research_agent.py) so citations can be validated afterwards.
        """
        system_prompt, user_prompt = prompts.build_evidence_synthesis_prompt(
            question=question, evidence=evidence
        )
        return self._generate_json(system_prompt, user_prompt)

    def compare_alternatives(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
        alternatives: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare named (or model-identified) alternatives when the
        question involves choosing between multiple options.
        """
        system_prompt, user_prompt = prompts.build_alternative_comparison_prompt(
            question=question,
            evidence=evidence,
            alternatives=alternatives or [],
        )
        return self._generate_json(system_prompt, user_prompt)

    def analyze_risks(
        self,
        question: str,
        context: str,
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Identify business, technical, financial, and operational
        risks relevant to the question, grounded in evidence where
        possible.
        """
        system_prompt, user_prompt = prompts.build_risk_analysis_prompt(
            question=question, context=context, evidence=evidence
        )
        return self._generate_json(system_prompt, user_prompt)

    def generate_recommendation(
        self,
        question: str,
        findings: List[str],
        pros: List[str],
        cons: List[str],
        risks: List[str],
    ) -> Dict[str, Any]:
        """Produce a final recommendation and confidence score from
        already-synthesized findings, pros, cons, and risks."""
        system_prompt, user_prompt = prompts.build_recommendation_prompt(
            question=question, findings=findings, pros=pros, cons=cons, risks=risks
        )
        return self._generate_json(system_prompt, user_prompt)

    def generate_final_report(
        self, question: str, sections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assemble an executive-ready report from the outputs of the
        earlier stages (research analysis, evidence synthesis,
        alternatives, risks, recommendation).
        """
        system_prompt, user_prompt = prompts.build_final_report_prompt(
            question=question, sections=sections
        )
        return self._generate_json(system_prompt, user_prompt)

    # ------------------------------------------------------------------
    # Low-level primitive, kept public for callers that already have a
    # ready-made (system_prompt, user_prompt) pair -- e.g. an
    # orchestrator composing its own prompt via prompts.py.
    # ------------------------------------------------------------------
    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Run a single structured-output call and return parsed JSON."""
        return self._generate_json(system_prompt, user_prompt)

    # ------------------------------------------------------------------
    # Internals: generation + defensive parsing
    # ------------------------------------------------------------------
    def _generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call Ollama and parse the result as JSON. If parsing fails,
        retry the generation (fresh sampling can produce valid output
        even when a previous attempt didn't) up to `json_retries` times
        before raising MalformedResponseError. Connection failures are
        not retried here -- they indicate Ollama itself is unreachable
        or misconfigured and should surface immediately.
        """
        last_error: Optional[MalformedResponseError] = None

        for attempt in range(self.json_retries + 1):
            raw_text = self._call_ollama(system_prompt, user_prompt)
            try:
                return self._safe_parse_json(raw_text)
            except MalformedResponseError as exc:
                last_error = exc
                logger.warning(
                    "Model response was not valid JSON (attempt %d/%d): %s",
                    attempt + 1,
                    self.json_retries + 1,
                    exc,
                )

        assert last_error is not None  # loop always runs at least once
        raise last_error

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat request to Ollama and return the raw text content.

        Raises:
            OllamaConnectionError: the request could not be completed at
                all (network failure, model not found, server error,
                timeout, ...).
            MalformedResponseError: Ollama responded but with no usable
                text content.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                format="json",
                options={"temperature": self.temperature},
                stream=False,
            )
        except ResponseError as exc:
            raise OllamaConnectionError(
                f"Ollama rejected the request for model '{self.model}': {exc}"
            ) from exc
        except Exception as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.host} (model='{self.model}'): {exc}"
            ) from exc

        content = self._extract_content(response)
        if not content or not content.strip():
            raise MalformedResponseError("Ollama returned an empty response.")
        return content

    @staticmethod
    def _extract_content(response: Any) -> str:
        """Pull the message text out of an Ollama chat response,
        tolerating both dict-style and attribute-style response objects
        across ollama-python client versions.
        """
        message: Any = None
        if isinstance(response, dict):
            message = response.get("message")
        else:
            message = getattr(response, "message", None)

        if message is None:
            return ""
        if isinstance(message, dict):
            return message.get("content", "") or ""
        return getattr(message, "content", "") or ""

    @staticmethod
    def _safe_parse_json(raw_text: str) -> Dict[str, Any]:
        """Defensively parse a JSON object out of model output.

        Even with `format="json"` requested, Llama 3 occasionally wraps
        the object in markdown fences or adds stray text around it, so
        this tries progressively looser extraction strategies before
        giving up.
        """
        candidates = [raw_text.strip()]

        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
        if fenced:
            candidates.append(fenced.group(1).strip())

        brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if brace_match:
            candidates.append(brace_match.group(0).strip())

        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed

        raise MalformedResponseError(
            "Could not parse a valid JSON object from the model's response."
        )
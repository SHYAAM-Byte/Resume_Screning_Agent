"""
research_agent.py

Enterprise AI Research & Decision Intelligence Agent
------------------------------------------------------
Main orchestration module.

Responsibilities (and ONLY these):
    - Coordinate the end-to-end research workflow: web search, document
      retrieval (RAG), and LLM-based reasoning/synthesis.
    - Merge evidence from multiple sources into a single, source-tagged
      evidence pool.
    - Drive the AI engine through a structured analysis (findings,
      alternatives, pros/cons, risks, recommendation, confidence).
    - Guard against hallucinated citations by validating every citation
      the AI produces against the real evidence pool.
    - Shape the final output into the dictionary contract expected by
      app.py.

This module does NOT implement:
    - The LLM client itself                  (see ai_engine.py)
    - Web search                              (see web_search.py)
    - Document parsing / vector retrieval     (see rag.py)
    - Prompt text                             (see prompts.py)
    - Low-level helpers (JSON extraction etc) (see utils.py)
    - Any UI                                  (see app.py)

Contract with app.py
---------------------
ResearchAgent().run(question, document_paths=None, progress_callback=None)
returns a dict:

    {
        "executive_summary": str,
        "key_findings": [str, ...],
        "evidence": [{"point": str, "source": str, "excerpt": str}, ...],
        "pros": [str, ...],
        "cons": [str, ...],
        "risks": [str, ...],
        "recommendation": str,
        "confidence_score": float,   # 0.0 - 1.0
        "sources": [{"title": str, "url": str}, ...],

        # Extra diagnostic fields -- not required by app.py, safe to ignore:
        "alternatives": [{"name": str, "summary": str}, ...],
        "warnings": [str, ...],
    }

Expected collaborator interfaces
---------------------------------
web_search.py:
    class WebSearchClient:
        def search(self, query: str, max_results: int = 6) -> list[dict]
            # each dict: {"title": str, "url": str, "snippet": str}

rag.py:
    class DocumentRetriever:
        def ingest(self, document_paths: list[str]) -> None
        def retrieve(self, query: str, top_k: int = 6) -> list[dict]
            # each dict: {"text": str, "source": str, "page": int | None,
            #             "score": float}

ai_engine.py:
    class AIEngine:
        def generate_json(self, system_prompt: str, user_prompt: str) -> dict
            # Calls the local Llama 3 model (via Ollama), parses, and
            # returns a JSON object matching the schema below. May raise
            # on unrecoverable failures (network/model errors); malformed
            # output should be handled by utils.safe_json_extract rather
            # than by this module reaching into the raw model text.

prompts.py:
    def build_synthesis_prompt(question, web_evidence, doc_evidence) -> tuple[str, str]
        # Returns (system_prompt, user_prompt). Responsible for instructing
        # the model to: ground every claim in the provided evidence ids,
        # categorize risks (business/technical/financial/operational) where
        # relevant, compare named alternatives when the question involves
        # multiple options, and never invent an evidence id.

utils.py:
    def safe_json_extract(text: str) -> dict
    def dedupe_by_key(items: list[dict], key: str) -> list[dict]
    def clamp(value: float, lo: float, hi: float) -> float

Expected AI JSON schema (produced by ai_engine.AIEngine.generate_json)
------------------------------------------------------------------------
    {
        "executive_summary": str,
        "key_findings": [str, ...],
        "alternatives": [{"name": str, "summary": str}, ...],
        "pros": [str, ...],
        "cons": [str, ...],
        "risks": [str, ...],                 # e.g. "Financial: ..."
        "recommendation": str,
        "confidence_score": float,            # 0.0 - 1.0, model's own estimate
        "citations": [{"evidence_id": str, "point": str}, ...],
    }

Evidence vs. AI output is intentionally kept as two separate data
structures throughout this module (EvidenceItem/ResearchContext vs. the
`analysis` dict) so that what was actually retrieved is never silently
merged with what the model concluded.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import ai_engine
import prompts
import rag
import utils
import web_search

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str, int], None]]


# ----------------------------------------------------------------------
# Internal data structures
# ----------------------------------------------------------------------
@dataclass
class EvidenceItem:
    """A single, source-tagged piece of evidence gathered before any AI
    reasoning happens. Kept separate from AI output so raw evidence and
    AI-generated conclusions are never conflated.
    """

    id: str
    kind: str  # "web" or "document"
    text: str
    source_label: str  # human-readable, e.g. page title or filename
    url: Optional[str] = None
    score: Optional[float] = None


@dataclass
class ResearchContext:
    """Everything gathered or produced during a single run() call."""

    question: str
    web_evidence: List[EvidenceItem] = field(default_factory=list)
    doc_evidence: List[EvidenceItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def all_evidence(self) -> List[EvidenceItem]:
        return self.web_evidence + self.doc_evidence


# ----------------------------------------------------------------------
# ResearchAgent
# ----------------------------------------------------------------------
class ResearchAgent:
    """Coordinates web search, document RAG, and LLM reasoning to turn a
    research question or business decision problem into a structured,
    citation-backed decision brief.
    """

    def __init__(
        self,
        max_web_results: int = 6,
        max_doc_results: int = 6,
        min_evidence_for_high_confidence: int = 4,
    ) -> None:
        self.max_web_results = max_web_results
        self.max_doc_results = max_doc_results
        self.min_evidence_for_high_confidence = min_evidence_for_high_confidence

        self._web_client = web_search.WebSearchClient()
        self._retriever = rag.DocumentRetriever()
        self._ai = ai_engine.AIEngine()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(
        self,
        question: str,
        document_paths: Optional[List[str]] = None,
        progress_callback: ProgressCallback = None,
    ) -> Dict[str, Any]:
        """Execute the full research workflow and return a dict that
        app.py can display directly.

        Expected failure modes (search down, no documents, model hiccup)
        are captured as warnings rather than raised, so the workflow
        degrades gracefully instead of crashing the UI. Only a missing
        question is treated as a hard error.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("A research question is required.")

        context = ResearchContext(question=question)

        self._notify(progress_callback, "Searching the web for relevant information...", 10)
        context.web_evidence = self._gather_web_evidence(question, context)

        self._notify(progress_callback, "Reading and retrieving from uploaded documents...", 30)
        context.doc_evidence = self._gather_document_evidence(question, document_paths, context)

        self._notify(progress_callback, "Synthesizing findings from all evidence...", 55)
        analysis = self._analyze(context)

        self._notify(progress_callback, "Validating citations against evidence...", 80)
        evidence_out, sources_out = self._build_citations(context, analysis)

        self._notify(progress_callback, "Finalizing recommendation and confidence score...", 92)
        confidence = self._resolve_confidence(analysis, context)

        self._notify(progress_callback, "Done.", 100)

        return {
            "executive_summary": analysis.get("executive_summary", ""),
            "key_findings": self._merge_alternatives_into_findings(analysis),
            "evidence": evidence_out,
            "pros": analysis.get("pros", []),
            "cons": analysis.get("cons", []),
            "risks": analysis.get("risks", []),
            "recommendation": analysis.get("recommendation", ""),
            "confidence_score": confidence,
            "sources": sources_out,
            "alternatives": analysis.get("alternatives", []),
            "warnings": context.warnings,
        }

    # ------------------------------------------------------------------
    # Step 1: Web evidence
    # ------------------------------------------------------------------
    def _gather_web_evidence(self, question: str, context: ResearchContext) -> List[EvidenceItem]:
        try:
            raw_results = self._web_client.search(question, max_results=self.max_web_results)
        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
            context.warnings.append(f"Web search unavailable: {exc}")
            return []

        raw_results = utils.dedupe_by_key(raw_results or [], key="url")

        items: List[EvidenceItem] = []
        for idx, r in enumerate(raw_results):
            text = (r.get("snippet") or r.get("content") or "").strip()
            if not text:
                continue
            items.append(
                EvidenceItem(
                    id=f"web-{idx}",
                    kind="web",
                    text=text,
                    source_label=r.get("title") or r.get("url") or "Web source",
                    url=r.get("url"),
                )
            )
        return items

    # ------------------------------------------------------------------
    # Step 2: Document evidence (RAG)
    # ------------------------------------------------------------------
    def _gather_document_evidence(
        self,
        question: str,
        document_paths: Optional[List[str]],
        context: ResearchContext,
    ) -> List[EvidenceItem]:
        if not document_paths:
            return []

        try:
            self._retriever.ingest(document_paths)
            raw_chunks = self._retriever.retrieve(question, top_k=self.max_doc_results)
        except Exception as exc:
            logger.warning("Document retrieval failed: %s", exc)
            context.warnings.append(f"Document retrieval unavailable: {exc}")
            return []

        items: List[EvidenceItem] = []
        for idx, chunk in enumerate(raw_chunks or []):
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            source = chunk.get("source") or "Uploaded document"
            page = chunk.get("page")
            label = f"{source} (p. {page})" if page else source
            items.append(
                EvidenceItem(
                    id=f"doc-{idx}",
                    kind="document",
                    text=text,
                    source_label=label,
                    score=chunk.get("score"),
                )
            )
        return items

    # ------------------------------------------------------------------
    # Step 3: AI reasoning / synthesis
    # ------------------------------------------------------------------
    def _analyze(self, context: ResearchContext) -> Dict[str, Any]:
        if not context.all_evidence:
            context.warnings.append(
                "No web or document evidence was found; analysis relies on "
                "general reasoning only and should be treated as low-confidence."
            )

        all_evidence = [
            self._evidence_as_prompt_dict(e)
            for e in (context.web_evidence + context.doc_evidence)
        ]

        system_prompt, user_prompt = prompts.build_evidence_synthesis_prompt(
            question=context.question,
            evidence=all_evidence,
        )

        try:
            raw = self._ai.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:
            logger.error("AI synthesis failed: %s", exc)
            context.warnings.append(f"AI reasoning step failed: {exc}")
            return self._fallback_analysis()

        analysis = utils.safe_json_extract(raw) if isinstance(raw, str) else raw
        if not isinstance(analysis, dict):
            context.warnings.append("AI response was not valid structured output; using fallback.")
            return self._fallback_analysis()

        return analysis

    @staticmethod
    def _evidence_as_prompt_dict(item: EvidenceItem) -> Dict[str, Any]:
        return {
            "id": item.id,
            "source": item.source_label,
            "url": item.url,
            "text": item.text,
        }

    @staticmethod
    def _fallback_analysis() -> Dict[str, Any]:
        """Used only if the AI engine call fails outright or returns
        unusable output, so run() still returns a well-formed (clearly
        low-confidence) result instead of raising and breaking the UI.
        """
        return {
            "executive_summary": (
                "Automated analysis could not be completed. The question and "
                "any gathered evidence are available, but no AI-generated "
                "synthesis was produced."
            ),
            "key_findings": [],
            "pros": [],
            "cons": [],
            "risks": ["Analysis incomplete due to a reasoning-engine failure."],
            "recommendation": "No recommendation could be generated. Please retry.",
            "confidence_score": 0.0,
            "citations": [],
            "alternatives": [],
        }

    # ------------------------------------------------------------------
    # Step 4: Citation validation (anti-hallucination guard)
    # ------------------------------------------------------------------
    def _build_citations(
        self, context: ResearchContext, analysis: Dict[str, Any]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Cross-check every citation the AI produced against the real
        evidence pool. Any citation referencing an unknown evidence id is
        dropped rather than trusted -- this is what prevents an invented
        source from ever reaching the UI.
        """
        evidence_by_id = {e.id: e for e in context.all_evidence}
        citations = analysis.get("citations", []) or []

        evidence_out: List[Dict[str, str]] = []
        used_sources: Dict[str, Dict[str, str]] = {}

        for c in citations:
            evidence_id = c.get("evidence_id") if isinstance(c, dict) else None
            item = evidence_by_id.get(evidence_id)
            if item is None:
                # The AI referenced evidence that doesn't exist -- skip it.
                continue

            point = (c.get("point") or "").strip() if isinstance(c, dict) else ""
            evidence_out.append(
                {
                    "point": point or item.text[:160],
                    "source": item.source_label,
                    "excerpt": item.text[:400],
                }
            )

            key = item.url or item.source_label
            if key not in used_sources:
                used_sources[key] = {"title": item.source_label, "url": item.url or ""}

        # If the AI produced no valid citations at all, fall back to
        # surfacing the raw evidence directly, clearly labeled as such,
        # so the user still sees what was found even without AI framing.
        if not evidence_out and context.all_evidence:
            for item in context.all_evidence:
                evidence_out.append(
                    {
                        "point": "Retrieved evidence (not directly cited by the AI analysis)",
                        "source": item.source_label,
                        "excerpt": item.text[:400],
                    }
                )
                key = item.url or item.source_label
                if key not in used_sources:
                    used_sources[key] = {"title": item.source_label, "url": item.url or ""}

        return evidence_out, list(used_sources.values())

    # ------------------------------------------------------------------
    # Step 5: Confidence scoring
    # ------------------------------------------------------------------
    def _resolve_confidence(self, analysis: Dict[str, Any], context: ResearchContext) -> float:
        """Blend the AI's self-reported confidence with an evidence-volume
        check, so confidence can't be artificially high when little or no
        real evidence was actually found.
        """
        try:
            ai_confidence = float(analysis.get("confidence_score", 0.0))
        except (TypeError, ValueError):
            ai_confidence = 0.0
        ai_confidence = utils.clamp(ai_confidence, 0.0, 1.0)

        evidence_count = len(context.all_evidence)
        if evidence_count == 0:
            return round(utils.clamp(min(ai_confidence, 0.25), 0.0, 1.0), 2)

        evidence_factor = utils.clamp(
            evidence_count / self.min_evidence_for_high_confidence, 0.0, 1.0
        )
        blended = (0.7 * ai_confidence) + (0.3 * evidence_factor)
        return round(utils.clamp(blended, 0.0, 1.0), 2)

    # ------------------------------------------------------------------
    # Step 6: Alternatives comparison folded into findings
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_alternatives_into_findings(analysis: Dict[str, Any]) -> List[str]:
        """Alternatives comparisons are produced by the AI as structured
        {name, summary} pairs (see module docstring) and folded into the
        findings list so app.py's existing "Key Findings" section can
        display them without needing a dedicated UI element.
        """
        findings = list(analysis.get("key_findings", []) or [])
        alternatives = analysis.get("alternatives", []) or []

        for alt in alternatives:
            if not isinstance(alt, dict):
                continue
            name = alt.get("name", "Option")
            summary = alt.get("summary", "")
            if summary:
                findings.append(f"{name}: {summary}")

        return findings

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _notify(progress_callback: ProgressCallback, stage: str, percent: int) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(stage, percent)
        except Exception:
            # Progress reporting must never break the actual workflow.
            logger.debug("progress_callback raised while reporting stage: %s", stage)
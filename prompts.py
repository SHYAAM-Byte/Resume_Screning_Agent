"""
prompts.py

Enterprise AI Research & Decision Intelligence Agent
------------------------------------------------------
Prompt templates for every stage of the research workflow.

Responsibilities (and ONLY these):
    - Build the (system_prompt, user_prompt) pairs that ai_engine.py
      sends to the model, for each stage of the workflow.
    - Encode, in one shared place, the ground rules every stage must
      follow: use only the provided evidence/context, avoid unsupported
      claims, surface uncertainty, keep facts separate from
      recommendations, note conflicting information, compare
      alternatives objectively, explain reasoning, cite evidence by
      id, and return a single valid JSON object matching a fixed
      schema.

This module does NOT implement:
    - Any Ollama / LLM API calls           (see ai_engine.py)
    - Any Streamlit or other UI code        (see app.py)
    - Web search or document retrieval      (see web_search.py, rag.py)
    - Workflow orchestration                (see research_agent.py)

Every function here is a pure string builder: given plain data
(question, evidence, prior-stage outputs, ...), it returns a
`(system_prompt, user_prompt)` tuple and has no side effects and no
dependencies beyond the standard library.

Provided builders
-------------------
    build_research_analysis_prompt(question, context)
        -- Stage 1: research question understanding.
    build_evidence_synthesis_prompt(question, evidence)
        -- Stage 2: evidence synthesis.
    build_alternative_comparison_prompt(question, evidence, alternatives)
        -- Stage 3: multi-source / alternative comparison.
    build_risk_analysis_prompt(question, context, evidence)
        -- Stage 4: risk analysis.
    build_pros_cons_prompt(question, evidence, context)
        -- Stage 5: pros and cons analysis.
    build_recommendation_prompt(question, findings, pros, cons, risks)
        -- Stage 6: business recommendation.
    build_final_report_prompt(question, sections)
        -- Stage 7: final decision report.

`evidence` is always a list of dicts shaped like:
    {"id": str, "source": str, "url": str | None, "text": str}
(this matches the evidence items research_agent.py / ai_engine.py pass
in). Every schema below that accepts evidence includes a "citations"
field so callers can validate references against real evidence ids
and drop anything the model may have invented.

Note: `build_pros_cons_prompt` is provided as a standalone reusable
stage per this module's requirements. ai_engine.py's current six
methods do not yet call it directly (pros/cons currently come out of
`compare_alternatives` / `generate_recommendation`); wire it in there
if a dedicated pros/cons stage is desired.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "build_research_analysis_prompt",
    "build_evidence_synthesis_prompt",
    "build_alternative_comparison_prompt",
    "build_risk_analysis_prompt",
    "build_pros_cons_prompt",
    "build_recommendation_prompt",
    "build_final_report_prompt",
]

Evidence = List[Dict[str, Any]]


# ----------------------------------------------------------------------
# Shared ground rules, reused (not duplicated) across every prompt
# ----------------------------------------------------------------------
_GUARDRAILS = (
    "You are an analytical research assistant supporting enterprise business "
    "decisions. Follow these rules in every response:\n"
    "1. Base every claim strictly on the evidence/context provided below. "
    "Never invent facts, statistics, sources, or evidence ids.\n"
    "2. If the evidence is insufficient, ambiguous, or conflicting, say so "
    "explicitly instead of guessing.\n"
    "3. Keep observed facts (from evidence) separate from your own analysis, "
    "inference, or recommendations -- never present one as the other.\n"
    "4. When sources disagree, note the disagreement rather than silently "
    "favoring one side.\n"
    "5. When comparing options, judge them on the same criteria; do not "
    "favor one option for reasons outside the evidence.\n"
    "6. Briefly explain the reasoning behind each conclusion, not just the "
    "conclusion itself.\n"
    "7. When referencing evidence, cite it using the exact evidence id given "
    "(e.g. \"web-0\", \"doc-2\"). Never fabricate an id.\n"
    "8. Respond with ONLY one valid JSON object -- no markdown fences, no "
    "text before or after it, no trailing commas.\n"
)


def _system_prompt(task: str, schema_example: str) -> str:
    """Assemble a full system prompt from the shared guardrails, a
    task-specific instruction, and the required JSON shape.
    """
    return (
        f"{_GUARDRAILS}\n"
        f"Task: {task}\n\n"
        "Return a single JSON object with exactly this shape (use empty "
        "arrays/strings where there is nothing to report; never omit a "
        f"key):\n{schema_example}"
    )


def _format_evidence(evidence: Optional[Evidence]) -> str:
    if not evidence:
        return "(no evidence provided)"

    lines = []
    for item in evidence:
        eid = item.get("id", "unknown")
        source = item.get("source") or "unknown source"
        url = item.get("url")
        label = f"{source} ({url})" if url else source
        text = (item.get("text") or "").strip()
        lines.append(f"[{eid}] {label}: {text}")
    return "\n".join(lines)


def _format_bullets(items: Optional[List[str]], empty_label: str = "(none provided)") -> str:
    cleaned = [str(item).strip() for item in (items or []) if str(item).strip()]
    if not cleaned:
        return empty_label
    return "\n".join(f"- {item}" for item in cleaned)


# ----------------------------------------------------------------------
# Stage 1: Research question understanding
# ----------------------------------------------------------------------
def build_research_analysis_prompt(question: str, context: str = "") -> Tuple[str, str]:
    """Frame the research question before evidence gathering: what is
    really being asked, useful sub-questions, whether it involves
    comparing named alternatives, and any assumptions or uncertainty
    already present in the question itself.
    """
    schema = (
        "{\n"
        '  "research_framing": "string",\n'
        '  "key_sub_questions": ["string", ...],\n'
        '  "detected_alternatives": ["string", ...],\n'
        '  "assumptions": ["string", ...],\n'
        '  "uncertainty_notes": ["string", ...]\n'
        "}"
    )
    system_prompt = _system_prompt(
        "Analyze the research question below. Identify what is really being "
        "asked, 2-5 useful sub-questions, whether it involves choosing between "
        "named alternatives (empty list if not), any assumptions the question "
        "relies on, and anything ambiguous or unclear about the question "
        "itself.",
        schema,
    )
    user_prompt = (
        f"Research question:\n{question.strip()}\n\n"
        f"Additional context (if any):\n{context.strip() or '(none provided)'}"
    )
    return system_prompt, user_prompt


# ----------------------------------------------------------------------
# Stage 2: Evidence synthesis
# ----------------------------------------------------------------------
def build_evidence_synthesis_prompt(question: str, evidence: Optional[Evidence]) -> Tuple[str, str]:
    """Synthesize the evidence pool into the most important findings
    relevant to the question, noting agreement, conflict, and gaps.
    """
    schema = (
        "{\n"
        '  "key_findings": [\n'
        '    {"finding": "string", "evidence_ids": ["string", ...], '
        '"confidence": "high|medium|low"}\n'
        "  ],\n"
        '  "conflicting_information": [\n'
        '    {"topic": "string", "description": "string", "evidence_ids": ["string", ...]}\n'
        "  ],\n"
        '  "gaps": ["string", ...],\n'
        '  "citations": [{"evidence_id": "string", "point": "string"}, ...]\n'
        "}"
    )
    system_prompt = _system_prompt(
        "Synthesize the evidence below into the most important findings "
        "relevant to the research question. Group related evidence into "
        "findings, note where sources agree or conflict, and flag any "
        "notable gaps where the evidence does not cover something important.",
        schema,
    )
    user_prompt = (
        f"Research question:\n{question.strip()}\n\n"
        f"Evidence:\n{_format_evidence(evidence)}"
    )
    return system_prompt, user_prompt


# ----------------------------------------------------------------------
# Stage 3: Multi-source / alternative comparison
# ----------------------------------------------------------------------
def build_alternative_comparison_prompt(
    question: str,
    evidence: Optional[Evidence],
    alternatives: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Compare information across the evidence sources (agreement vs.
    conflict) and, if the question involves choosing between options,
    compare those options objectively on the same criteria.
    """
    schema = (
        "{\n"
        '  "alternatives": [\n'
        '    {"name": "string", "summary": "string", '
        '"supporting_evidence_ids": ["string", ...], '
        '"opposing_evidence_ids": ["string", ...]}\n'
        "  ],\n"
        '  "cross_source_agreement": ["string", ...],\n'
        '  "cross_source_conflicts": [\n'
        '    {"topic": "string", "description": "string", "evidence_ids": ["string", ...]}\n'
        "  ],\n"
        '  "comparison_summary": "string",\n'
        '  "citations": [{"evidence_id": "string", "point": "string"}, ...]\n'
        "}"
    )
    system_prompt = _system_prompt(
        "Compare the evidence sources below: note where independent sources "
        "agree and where they conflict. If specific alternatives are listed, "
        "evaluate exactly those on the same criteria; if none are listed but "
        "the question implies choosing between options, infer them from the "
        "question and evidence; otherwise return an empty alternatives list.",
        schema,
    )
    named = _format_bullets(alternatives, empty_label="(none specified -- infer if applicable)")
    user_prompt = (
        f"Research question:\n{question.strip()}\n\n"
        f"Named alternatives (if any):\n{named}\n\n"
        f"Evidence:\n{_format_evidence(evidence)}"
    )
    return system_prompt, user_prompt


# ----------------------------------------------------------------------
# Stage 4: Risk analysis
# ----------------------------------------------------------------------
def build_risk_analysis_prompt(
    question: str, context: str = "", evidence: Optional[Evidence] = None
) -> Tuple[str, str]:
    """Identify business, technical, financial, and operational risks
    relevant to the question, grounded in the evidence/context given.
    """
    schema = (
        "{\n"
        '  "risks": [\n'
        '    {"category": "business|technical|financial|operational|other", '
        '"description": "string", "severity": "high|medium|low", '
        '"likelihood": "high|medium|low", "evidence_ids": ["string", ...]}\n'
        "  ],\n"
        '  "uncertain_areas": ["string", ...],\n'
        '  "citations": [{"evidence_id": "string", "point": "string"}, ...]\n'
        "}"
    )
    system_prompt = _system_prompt(
        "Identify risks relevant to the research question, using only the "
        "context and evidence given. Categorize each risk as business, "
        "technical, financial, or operational, and estimate severity and "
        "likelihood only where the evidence supports doing so. List any "
        "plausible risk areas the evidence does not cover under "
        "'uncertain_areas' instead of guessing at their severity.",
        schema,
    )
    user_prompt = (
        f"Research question:\n{question.strip()}\n\n"
        f"Context:\n{(context or '').strip() or '(none provided)'}\n\n"
        f"Evidence:\n{_format_evidence(evidence)}"
    )
    return system_prompt, user_prompt


# ----------------------------------------------------------------------
# Stage 5: Pros and cons analysis
# ----------------------------------------------------------------------
def build_pros_cons_prompt(
    question: str, evidence: Optional[Evidence] = None, context: str = ""
) -> Tuple[str, str]:
    """List advantages (pros) and disadvantages (cons) relevant to the
    question, grounded in the evidence/context given.
    """
    schema = (
        "{\n"
        '  "pros": [{"point": "string", "evidence_ids": ["string", ...]}],\n'
        '  "cons": [{"point": "string", "evidence_ids": ["string", ...]}],\n'
        '  "citations": [{"evidence_id": "string", "point": "string"}, ...]\n'
        "}"
    )
    system_prompt = _system_prompt(
        "List the advantages (pros) and disadvantages (cons) relevant to the "
        "research question, based only on the evidence and context given. "
        "Keep each point concise and, where possible, ground it in a "
        "specific evidence id. If pros or cons are genuinely not supported "
        "by the evidence, return an empty list for that side rather than "
        "inventing points.",
        schema,
    )
    user_prompt = (
        f"Research question:\n{question.strip()}\n\n"
        f"Context:\n{(context or '').strip() or '(none provided)'}\n\n"
        f"Evidence:\n{_format_evidence(evidence)}"
    )
    return system_prompt, user_prompt


# ----------------------------------------------------------------------
# Stage 6: Business recommendation
# ----------------------------------------------------------------------
def build_recommendation_prompt(
    question: str,
    findings: Optional[List[str]] = None,
    pros: Optional[List[str]] = None,
    cons: Optional[List[str]] = None,
    risks: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Produce a recommendation from already-synthesized findings,
    pros, cons, and risks -- no new evidence is introduced at this
    stage, and the recommendation is kept distinct from its reasoning.
    """
    schema = (
        "{\n"
        '  "recommendation": "string",\n'
        '  "reasoning": "string",\n'
        '  "confidence_score": 0.0,\n'
        '  "confidence_rationale": "string",\n'
        '  "key_uncertainties": ["string", ...]\n'
        "}"
    )
    system_prompt = _system_prompt(
        "Using ONLY the findings, pros, cons, and risks already synthesized "
        "below -- do not introduce new facts -- produce a clear "
        "recommendation for the research question. Keep the recommendation "
        "itself separate from the reasoning that supports it. Give an "
        "honest confidence_score (0.0-1.0) reflecting how strong and "
        "complete the input is, explain that score briefly, and list the "
        "most important remaining uncertainties.",
        schema,
    )
    user_prompt = (
        f"Research question:\n{question.strip()}\n\n"
        f"Key findings:\n{_format_bullets(findings)}\n\n"
        f"Pros:\n{_format_bullets(pros)}\n\n"
        f"Cons:\n{_format_bullets(cons)}\n\n"
        f"Risks:\n{_format_bullets(risks)}"
    )
    return system_prompt, user_prompt


# ----------------------------------------------------------------------
# Stage 7: Final decision report
# ----------------------------------------------------------------------
def build_final_report_prompt(question: str, sections: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """Assemble a polished, executive-ready report from the outputs of
    the earlier stages. This stage reorganizes and clarifies; it must
    not introduce facts, risks, or citations beyond what the provided
    sections already contain.
    """
    schema = (
        "{\n"
        '  "executive_summary": "string",\n'
        '  "key_findings": ["string", ...],\n'
        '  "alternatives": [{"name": "string", "summary": "string"}],\n'
        '  "pros": ["string", ...],\n'
        '  "cons": ["string", ...],\n'
        '  "risks": ["string", ...],\n'
        '  "recommendation": "string",\n'
        '  "confidence_score": 0.0,\n'
        '  "sources_used": ["string", ...],\n'
        '  "citations": [{"evidence_id": "string", "point": "string"}, ...]\n'
        "}"
    )
    system_prompt = _system_prompt(
        "Assemble a polished, executive-ready decision report from the "
        "prior-stage outputs given below. Reorganize, summarize, and "
        "clarify -- do not introduce any fact, risk, pro/con, or citation "
        "that is not already present in the provided sections.",
        schema,
    )
    sections_json = json.dumps(sections or {}, indent=2, ensure_ascii=False, default=str)
    user_prompt = (
        f"Research question:\n{question.strip()}\n\n"
        f"Prior-stage outputs (JSON):\n{sections_json}"
    )
    return system_prompt, user_prompt
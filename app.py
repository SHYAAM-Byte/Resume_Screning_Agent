"""
app.py

Enterprise AI Research & Decision Intelligence Agent
------------------------------------------------------
Streamlit front-end ONLY.

This file is intentionally free of any AI, RAG, or web-search logic.
All heavy lifting (LLM calls via Ollama/Llama 3, web search, document
parsing, RAG retrieval, and evidence synthesis) is delegated to
`research_agent.py`, which this UI treats as a black box.

Expected contract with research_agent.py
-----------------------------------------
research_agent.py must expose a `ResearchAgent` class with a method:

    ResearchAgent().run(
        question,
        document_paths=None,       # list[str] of saved upload paths
        progress_callback=None,    # callable(stage: str, percent: int)
    )  ->  dict | object

The return value should provide the following fields (as dict keys or
object attributes). Any missing field is simply hidden in the UI:

    executive_summary   : str
    key_findings         : list[str]
    evidence              : list[str] | list[dict]     (dict may have
                              'point'/'claim', 'source'/'url', 'excerpt')
    pros                   : list[str]
    cons                   : list[str]
    risks                  : list[str]
    recommendation       : str
    confidence_score     : float                        (0-1 or 0-100)
    sources                : list[str] | list[dict]     (dict may have
                              'title', 'url')

This UI is defensive: it accepts both dict-style and attribute-style
results, and degrades gracefully if fields are missing.
"""

import os
import tempfile
import traceback
from typing import Any, List, Optional

import streamlit as st

# ----------------------------------------------------------------------
# Import the backend workflow. Wrapped in try/except so the UI still
# renders (with a clear message) even if research_agent.py is missing
# or fails to import for some reason.
# ----------------------------------------------------------------------
AGENT_IMPORT_ERROR: Optional[str] = None
try:
    from research_agent import ResearchAgent
except Exception as exc:  # pragma: no cover
    ResearchAgent = None
    AGENT_IMPORT_ERROR = str(exc)


# ----------------------------------------------------------------------
# Page configuration & light styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise AI Research & Decision Intelligence Agent",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px;}
    h1 {font-weight: 700;}
    .stat-card {
        background: #f7f8fa;
        border: 1px solid #e2e5e9;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 1.75rem;
        margin-bottom: 0.5rem;
    }
    .source-pill {
        display: inline-block;
        background: #eef2ff;
        color: #3730a3;
        text-decoration: none;
        border-radius: 4px;
        padding: 0.2rem 0.6rem;
        font-size: 0.85rem;
        margin: 0.15rem 0.35rem 0.15rem 0;
    }
    .footer-note {color: #8a8f98; font-size: 0.8rem; margin-top: 3rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "error" not in st.session_state:
    st.session_state.error = None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from either a dict-like or attribute-style object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_percent(score: Any) -> Optional[int]:
    """Normalize a confidence score (0-1 or 0-100) into an int 0-100."""
    if score is None:
        return None
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None
    if score <= 1:
        score *= 100
    return max(0, min(100, round(score)))


def _confidence_label(pct: int) -> str:
    if pct >= 75:
        return "High"
    if pct >= 45:
        return "Medium"
    return "Low"


def _save_uploaded_files(uploaded_files) -> List[str]:
    """Persist uploaded files to a temp dir and return their paths."""
    if not uploaded_files:
        return []
    tmp_dir = tempfile.mkdtemp(prefix="research_uploads_")
    paths = []
    for f in uploaded_files:
        path = os.path.join(tmp_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        paths.append(path)
    return paths


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧭 About")
    st.write(
        "This tool researches a business question using web search, "
        "your uploaded documents (RAG), and a local Llama 3 model via "
        "Ollama, then returns a structured decision brief."
    )
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown(
        "1. Understand the question\n"
        "2. Search the web\n"
        "3. Retrieve from uploaded documents\n"
        "4. Synthesize findings\n"
        "5. Score confidence"
    )
    st.markdown("---")
    if AGENT_IMPORT_ERROR:
        st.warning(
            "Backend module `research_agent` could not be loaded. "
            "The UI will still run, but analysis will fail until "
            "`research_agent.py` is available.\n\n"
            f"Details: {AGENT_IMPORT_ERROR}"
        )
    if st.session_state.result is not None:
        if st.button("🔄 New Research", use_container_width=True):
            st.session_state.result = None
            st.session_state.last_question = ""
            st.session_state.error = None
            st.rerun()


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🧭 Enterprise AI Research & Decision Intelligence Agent")
st.caption(
    "Ask a research question or describe a business decision. The agent "
    "will search the web, read any documents you provide, and return an "
    "evidence-backed recommendation."
)

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
with st.form("research_form", clear_on_submit=False):
    question = st.text_area(
        "Research question or business decision problem",
        placeholder=(
            "e.g. Should we expand our SaaS product into the European "
            "market in the next 12 months?"
        ),
        height=120,
        value=st.session_state.last_question,
    )

    uploaded_files = st.file_uploader(
        "Optional supporting documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="These will be used as additional grounding evidence via RAG.",
    )

    col_a, _ = st.columns([1, 4])
    with col_a:
        submitted = st.form_submit_button(
            "🔍 Research / Analyze", use_container_width=True
        )

# ----------------------------------------------------------------------
# Run the research workflow
# ----------------------------------------------------------------------
if submitted:
    st.session_state.error = None

    if not question or not question.strip():
        st.warning(
            "Please enter a research question or decision problem before analyzing."
        )
    elif ResearchAgent is None:
        st.session_state.error = (
            "The research backend (`research_agent.py`) is not available: "
            f"{AGENT_IMPORT_ERROR}"
        )
    else:
        st.session_state.last_question = question
        document_paths = _save_uploaded_files(uploaded_files)

        progress_bar = st.progress(0, text="Starting research...")
        status = st.status("Initializing research agent...", expanded=True)

        def progress_callback(stage: str, percent: int = 0):
            try:
                status.update(label=stage)
                status.write(f"• {stage}")
                progress_bar.progress(min(max(int(percent), 0), 100), text=stage)
            except Exception:
                pass

        try:
            agent = ResearchAgent()

            try:
                result = agent.run(
                    question=question.strip(),
                    document_paths=document_paths,
                    progress_callback=progress_callback,
                )
            except TypeError:
                # Backend may not accept document_paths / progress_callback kwargs.
                result = agent.run(question.strip())

            st.session_state.result = result
            status.update(label="Research complete", state="complete", expanded=False)
            progress_bar.progress(100, text="Done")

        except Exception as exc:
            status.update(label="Research failed", state="error", expanded=True)
            st.session_state.result = None
            st.session_state.error = str(exc)
            with st.expander("Error details"):
                st.code(traceback.format_exc())

# ----------------------------------------------------------------------
# Error display (persists across reruns until the next submission)
# ----------------------------------------------------------------------
if st.session_state.error:
    st.error(f"Something went wrong while researching: {st.session_state.error}")

# ----------------------------------------------------------------------
# Results display
# ----------------------------------------------------------------------
result = st.session_state.result

if result is not None:
    st.markdown("---")

    exec_summary = _get(result, "executive_summary")
    key_findings = _get(result, "key_findings")
    evidence = _get(result, "evidence")
    pros = _get(result, "pros")
    cons = _get(result, "cons")
    risks = _get(result, "risks")
    recommendation = _get(result, "recommendation")
    pct = _as_percent(_get(result, "confidence_score"))
    sources = _get(result, "sources")

    # --- Executive Summary ---
    if exec_summary:
        st.markdown(
            '<div class="section-title">📋 Executive Summary</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="stat-card">{exec_summary}</div>', unsafe_allow_html=True)

    # --- Confidence Score ---
    if pct is not None:
        st.markdown(
            '<div class="section-title">📈 Confidence Score</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("Confidence", f"{pct}%", _confidence_label(pct))
        with c2:
            st.progress(pct / 100)

    # --- Key Findings ---
    if key_findings:
        st.markdown(
            '<div class="section-title">🔑 Key Findings</div>', unsafe_allow_html=True
        )
        for item in key_findings:
            st.markdown(f"- {item}")

    # --- Evidence ---
    if evidence:
        st.markdown(
            '<div class="section-title">🧪 Evidence</div>', unsafe_allow_html=True
        )
        for item in evidence:
            if isinstance(item, dict):
                point = item.get("point") or item.get("claim") or "Evidence item"
                source = item.get("source") or item.get("url")
                excerpt = item.get("excerpt")
                with st.expander(point):
                    if excerpt:
                        st.write(excerpt)
                    if source:
                        st.caption(f"Source: {source}")
            else:
                st.markdown(f"- {item}")

    # --- Pros and Cons ---
    if pros or cons:
        st.markdown(
            '<div class="section-title">⚖️ Pros and Cons</div>', unsafe_allow_html=True
        )
        col_p, col_c = st.columns(2)
        with col_p:
            st.markdown("**✅ Pros**")
            for item in pros or []:
                st.markdown(f"- {item}")
            if not pros:
                st.caption("None identified.")
        with col_c:
            st.markdown("**⚠️ Cons**")
            for item in cons or []:
                st.markdown(f"- {item}")
            if not cons:
                st.caption("None identified.")

    # --- Risks ---
    if risks:
        st.markdown(
            '<div class="section-title">🚨 Risks</div>', unsafe_allow_html=True
        )
        for item in risks:
            st.markdown(f"- {item}")

    # --- Recommendation ---
    if recommendation:
        st.markdown(
            '<div class="section-title">🎯 Recommendation</div>', unsafe_allow_html=True
        )
        st.success(recommendation)

    # --- Sources ---
    if sources:
        st.markdown(
            '<div class="section-title">📚 Sources</div>', unsafe_allow_html=True
        )
        pills = ""
        for src in sources:
            if isinstance(src, dict):
                title = src.get("title") or src.get("url") or "Source"
                url = src.get("url")
                if url:
                    pills += f'<a class="source-pill" href="{url}" target="_blank">{title}</a>'
                else:
                    pills += f'<span class="source-pill">{title}</span>'
            else:
                pills += f'<span class="source-pill">{src}</span>'
        st.markdown(pills, unsafe_allow_html=True)

    if not any(
        [
            exec_summary,
            key_findings,
            evidence,
            pros,
            cons,
            risks,
            recommendation,
            pct is not None,
            sources,
        ]
    ):
        st.info("The research agent returned an empty result.")

st.markdown(
    '<div class="footer-note">Enterprise AI Research &amp; Decision Intelligence '
    "Agent — powered by a local Llama 3 model via Ollama, live web search, and "
    "retrieval-augmented generation over your documents.</div>",
    unsafe_allow_html=True,
)
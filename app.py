

from typing import Optional
import traceback

import pandas as pd
import streamlit as st

try:
    from Screener import screen_candidates
    
except ImportError:
    screen_candidates = None  

st.set_page_config(
    page_title="Resume Screening Agent",
    page_icon="🧑‍💼",
    layout="wide",
)

def init_session_state() -> None:
    defaults = {
        "results_df": None,
        "screening_error": None,
        "jd_file": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_header() -> None:
    st.title("🧑‍💼 Resume Screning Agent")
    st.caption(
        "Upload candidate resumes and a job description, then let the "
        "local Llama 3 agent rank and shortlist candidates for you."
    )


def render_job_description_input() -> str:
    st.subheader("1. Job Description")

    jd_text = ""
    jd_input_mode = st.radio(
        "How would you like to provide the job description?",
        options=["Paste text", "Upload file"],
        horizontal=True,
        key="jd_input_mode",
    )

    if jd_input_mode == "Paste text":
        jd_text = st.text_area(
            "Paste the job description",
            height=200,
            placeholder="e.g. We are looking for a Senior Python Developer with 5+ years...",
            key="jd_text_area",
        )
        st.session_state["jd_file"] = None
    else:
        jd_file = st.file_uploader(
            "Upload job description (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False,
            key="jd_file_uploader",
        )
        st.session_state["jd_file"] = jd_file
        if jd_file is not None:
            st.success(f"Job description file uploaded: {jd_file.name}")

    return jd_text


def render_resume_uploader():
    st.subheader("2. Candidate Resumes")
    resume_files = st.file_uploader(
        "Upload one or more resumes (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="resume_uploader",
    )
    if resume_files:
        st.info(f"{len(resume_files)} resume(s) ready for screening.")
    return resume_files


def validate_inputs(resume_files, jd_text: str, jd_file) -> Optional[str]:
    if not resume_files:
        return "Please upload at least one resume (PDF or DOCX)."
    if not jd_text.strip() and jd_file is None:
        return "Please provide a job description (paste text or upload a file)."
    return None


def run_screening(resume_files, jd_text: str, jd_file) -> pd.DataFrame:
    if screen_candidates is None:
        raise RuntimeError(
            "Could not find 'screen_candidates' in screener.py. "
            "Make sure screener.py exists and exports that function."
        )

    try:
        results = screen_candidates(
            resume_files=resume_files,
            job_description=jd_text,
            job_description_file=jd_file,
        )
    except TypeError:
        results = screen_candidates(resume_files, jd_text)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def render_results(df: pd.DataFrame) -> None:
    st.subheader("3. Screening Results")

    if df.empty:
        st.warning("No results were returned by the screener.")
        return
    if "Match Score" in df.columns:
        df = df.sort_values(by="Match Score", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download results as CSV",
        data=csv_bytes,
        file_name="screening_results.csv",
        mime="text/csv",
    )

def main() -> None:
    init_session_state()
    render_header()

    jd_text = render_job_description_input()
    jd_file = st.session_state.get("jd_file")
    resume_files = render_resume_uploader()

    st.divider()

    if st.button("🔍 Screen Candidates", type="primary"):
        validation_error = validate_inputs(resume_files, jd_text, jd_file)

        if validation_error:
            st.error(validation_error)
        else:
            with st.spinner("Screening candidates with Llama 3... this may take a moment..."):
                try:
                    df = run_screening(resume_files, jd_text, jd_file)
                    st.session_state.results_df = df
                    st.session_state.screening_error = None
                except Exception as exc:
                    st.session_state.results_df = None
                    st.session_state.screening_error = str(exc)
                    traceback.print_exc()

    if st.session_state.screening_error:
        st.error(f"⚠️ Screening failed: {st.session_state.screening_error}")
        st.info(
            "Please verify that Ollama is running locally (e.g. `ollama serve`) "
            "and that the required model has been pulled, then try again."
        )

    if st.session_state.results_df is not None:
        render_results(st.session_state.results_df)


if __name__ == "__main__":
    main()
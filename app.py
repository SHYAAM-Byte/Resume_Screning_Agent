
import uuid
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="AI Interviewer", page_icon="🎤", layout="wide")
from Speech import speech_to_text
try:
    from interviewer import Interviewer, parse_resume
except ImportError as import_error:
    st.error(
        "**Could not import `interviewer.py`.**\n\n"
        "Make sure `interviewer.py` exists in the project root and exposes "
        "an `Interviewer` class with the methods documented at the top of "
        "`app.py`.\n\n"
        f"Import error: `{import_error}`"
    )
    st.stop()

SUPPORTED_RESUME_TYPES = ["pdf", "docx"]
SUPPORTED_AUDIO_TYPES = ["wav", "mp3", "m4a", "ogg", "webm"]
MAX_RESUME_SIZE_MB = 10
RECORDING_SUPPORTED = hasattr(st, "audio_input")


@st.cache_resource(show_spinner="Loading AI models (Llama 3 + Whisper)... this happens once.")
def get_interviewer_engine() -> "Interviewer":
    return Interviewer()

def init_session_state():
    defaults = {
        "interview_id": None,
        "resume_text": None,
        "resume_filename": None,
        "interview_started": False,
        "interview_complete": False,
        "current_question": None,
        "question_number": 0,
        "question_stage": "answering",
        "transcript": [],
        "current_score": 0.0,
        "final_report": None,
        "transcribed_answer": None,
        "last_feedback": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.interview_id is None:
        st.session_state.interview_id = str(uuid.uuid4())


def reset_interview():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.header("Session")
        if st.session_state.resume_filename:
            st.write(f"📄 **Resume:** {st.session_state.resume_filename}")
        if st.session_state.interview_started and not st.session_state.interview_complete:
            st.write(f"❓ **Question:** {st.session_state.question_number}")
        st.metric("Current Score", f"{st.session_state.current_score:.1f} / 10")
        st.divider()
        if st.button("🔄 Reset Interview", use_container_width=True):
            reset_interview()
            st.rerun()


# --------------------------------------------------------------------------
# Step 1 & 2: Resume upload + start interview
# --------------------------------------------------------------------------
def render_upload_section(engine: "Interviewer"):
    st.subheader("Step 1 · Upload Your Resume")
    resume_file = st.file_uploader(
        "Upload your resume (PDF or DOCX)", type=SUPPORTED_RESUME_TYPES
    )

    if resume_file is None:
        st.info("Upload a resume to generate personalized interview questions.")
        return

    if resume_file.size > MAX_RESUME_SIZE_MB * 1024 * 1024:
        st.error(f"File is too large. Please upload a resume under {MAX_RESUME_SIZE_MB}MB.")
        return

    if (
        st.session_state.resume_text is None
        or st.session_state.resume_filename != resume_file.name
    ):
        try:
            with st.spinner("Reading and parsing your resume..."):
                    st.session_state.resume_text = parse_resume(
                        resume_file.getvalue(),
                        resume_file.name,
                    )
                    st.session_state.resume_filename = resume_file.name
        except Exception as e:
            st.error(f"Could not read this resume: {e}")
            st.session_state.resume_text = None
            return

    st.success(f"Resume parsed: {resume_file.name}")
    with st.expander("Preview extracted resume text"):
        st.text((st.session_state.resume_text or "")[:3000] or "(no text extracted)")

    st.subheader("Step 2 · Start the Interview")
    if st.button("▶️ Start Interview", type="primary"):
        try:
            with st.spinner("Preparing your first question..."):
                interview = engine.start_interview(
                    resume_text=st.session_state.resume_text,
                    interview_id=st.session_state.interview_id,
                )
                st.write("DEBUG START:", interview)
                st.write("Saved Interview ID:", st.session_state.interview_id)

            # Save the interview id returned by interviewer.py
            st.session_state.interview_id = interview["interview_id"]

            # Save the first question
            st.session_state.current_question = interview["first_question"]

            st.session_state.question_number = 1
            st.session_state.question_stage = "answering"
            st.session_state.interview_started = True

            st.rerun()
        except Exception as e:
            st.error(f"Could not start the interview: {e}")
            st.info(
                "Make sure Ollama is running (`ollama serve`) and the Llama 3 "
                "model is available (`ollama pull llama3`)."
            )

def render_transcript_history():
    if not st.session_state.transcript:
        return
    with st.expander(
        f"📝 Interview Transcript ({len(st.session_state.transcript)} answered)",
        expanded=False,
    ):
        for i, item in enumerate(st.session_state.transcript, start=1):
            st.markdown(f"**Q{i}: {item['question']}**")
            st.write(item["answer"])
            st.caption(f"Feedback: {item['feedback']}  ·  Score: {item['score']}/10")
            st.divider()


def render_interview_section(engine: "Interviewer"):
    render_transcript_history()

    st.subheader(f"Question {st.session_state.question_number}")
    st.markdown(f"### {st.session_state.current_question}")

    if hasattr(engine, "synthesize_speech"):
        if st.button("🔊 Play Question Aloud"):
            try:
                with st.spinner("Generating audio..."):
                    audio_bytes = engine.synthesize_speech(st.session_state.current_question)
                st.audio(audio_bytes)
            except Exception as e:
                st.warning(f"Couldn't generate audio for this question: {e}")

    st.divider()

    widget_scope = f"{st.session_state.interview_id}_{st.session_state.question_number}"
    stage = st.session_state.question_stage

    if stage == "answering":
        _render_answering_stage(engine, widget_scope)
    elif stage == "transcribed":
        _render_transcribed_stage(engine, widget_scope)
    elif stage == "evaluated":
        _render_evaluated_stage(engine)


def _render_answering_stage(engine: "Interviewer", widget_scope: str):
    method_options = (["🎙️ Record Answer"] if RECORDING_SUPPORTED else []) + [
        "📁 Upload Audio File"
    ]

    if not RECORDING_SUPPORTED:
        st.info(
            "Live microphone recording isn't available in this Streamlit "
            "version — please upload an audio file instead."
        )

    if len(method_options) == 1:
        input_method = method_options[0]
    else:
        input_method = st.radio(
            "Answer input method", method_options, horizontal=True, key=f"method_{widget_scope}"
        )

    audio_bytes = None
    if input_method == "🎙️ Record Answer":
        audio_value = st.audio_input("Record your answer", key=f"rec_{widget_scope}")
        if audio_value is not None:
            audio_bytes = audio_value.getvalue()
    else:
        uploaded_audio = st.file_uploader(
            "Upload your answer", type=SUPPORTED_AUDIO_TYPES, key=f"upl_{widget_scope}"
        )
        if uploaded_audio is not None:
            audio_bytes = uploaded_audio.getvalue()

    if audio_bytes:
        st.audio(audio_bytes)
        if st.button("Transcribe Answer", type="primary"):
            try:
                with st.spinner("Transcribing your answer with Whisper..."):
                    text = speech_to_text(audio_bytes)
                if not text or not text.strip():
                    st.warning("No speech was detected. Please try recording again.")
                else:
                    st.session_state.transcribed_answer = text
                    st.session_state.question_stage = "transcribed"
                    st.rerun()
            except Exception as e:
                st.error(f"Transcription failed: {e}")


def _render_transcribed_stage(engine: "Interviewer", widget_scope: str):
    st.markdown("**Your Answer** (edit for accuracy if needed)")
    edited_text = st.text_area(
        "Transcript",
        value=st.session_state.transcribed_answer,
        height=150,
        key=f"edit_{widget_scope}",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("↩️ Re-record / Re-upload"):
            st.session_state.transcribed_answer = None
            st.session_state.question_stage = "answering"
            st.rerun()
    with col2:
        if st.button("✅ Submit Answer", type="primary"):
            if not edited_text.strip():
                st.warning("Your answer is empty. Please add some text before submitting.")
            else:
                try:
                    with st.spinner("Evaluating your answer..."):
                        st.write("DEBUG EVALUATE")
                        st.write("Interview ID:", st.session_state.interview_id)
                        st.write("Question:", st.session_state.current_question)
                        result = engine.evaluate_answer(
                            st.session_state.interview_id,
                            st.session_state.current_question,
                            edited_text,
                        )
                        current_score = engine.get_current_score(st.session_state.interview_id)
                    st.session_state.transcript.append(
                        {
                            "question": st.session_state.current_question,
                            "answer": edited_text,
                            "feedback": result.get("feedback", ""),
                            "score": result.get("score", 0),
                        }
                    )
                    st.session_state.last_feedback = result
                    st.session_state.current_score = current_score
                    st.session_state.question_stage = "evaluated"
                    st.rerun()
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")


def _render_evaluated_stage(engine: "Interviewer"):
    feedback = st.session_state.last_feedback or {}
    st.subheader("💬 AI Feedback")
    st.write(feedback.get("feedback", "No feedback available."))
    if feedback.get("score") is not None:
        st.metric("Score for this answer", f"{feedback['score']:.1f} / 10")

    if st.button("Next Question ➜", type="primary"):
        try:
            with st.spinner("Preparing the next question..."):
                next_question = engine.get_next_question(st.session_state.interview_id)
            if next_question:
                st.session_state.current_question = next_question
                st.session_state.question_number += 1
                st.session_state.question_stage = "answering"
                st.session_state.transcribed_answer = None
                st.session_state.last_feedback = None
            else:
                with st.spinner("Generating your final report..."):
                    st.session_state.final_report = engine.generate_final_report(
                        st.session_state.interview_id
                    )
                st.session_state.interview_complete = True
            st.rerun()
        except Exception as e:
            st.error(f"Could not proceed to the next step: {e}")


# --------------------------------------------------------------------------
# Final report
# --------------------------------------------------------------------------
def build_report_text(report: dict) -> str:
    lines = [
        "AI INTERVIEWER — FINAL REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 50,
        f"Overall Score: {report.get('overall_score', 0):.1f} / 10",
        "",
        "Summary:",
        report.get("summary", ""),
        "",
        "Strengths:",
    ]
    lines += [f"- {s}" for s in report.get("strengths", [])]
    lines += ["", "Areas for Improvement:"]
    lines += [f"- {a}" for a in report.get("areas_for_improvement", [])]
    lines += ["", "Question-by-Question Breakdown:", "-" * 50]
    breakdown = report.get("question_breakdown") or st.session_state.transcript
    for i, item in enumerate(breakdown, start=1):
        lines += [
            f"Q{i}: {item.get('question')}",
            f"Answer: {item.get('answer')}",
            f"Feedback: {item.get('feedback')}",
            f"Score: {item.get('score')}/10",
            "",
        ]
    return "\n".join(lines)


def render_final_report():
    st.header("📊 Final Interview Report")
    report = st.session_state.final_report
    if not report:
        st.warning("No report data is available.")
        return

    st.metric("Overall Score", f"{report.get('overall_score', 0):.1f} / 10")

    st.subheader("Summary")
    st.write(report.get("summary", ""))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Strengths")
        for s in report.get("strengths", []):
            st.write(f"- {s}")
    with col2:
        st.subheader("🔧 Areas for Improvement")
        for a in report.get("areas_for_improvement", []):
            st.write(f"- {a}")

    st.subheader("Question-by-Question Breakdown")
    breakdown = report.get("question_breakdown") or st.session_state.transcript
    for i, item in enumerate(breakdown, start=1):
        with st.expander(f"Q{i}: {str(item.get('question', ''))[:80]}"):
            st.write(f"**Question:** {item.get('question')}")
            st.write(f"**Your Answer:** {item.get('answer')}")
            st.write(f"**Feedback:** {item.get('feedback')}")
            st.write(f"**Score:** {item.get('score')}/10")

    st.download_button(
        "⬇️ Download Report (.txt)",
        data=build_report_text(report),
        file_name="interview_report.txt",
        mime="text/plain",
    )

    if st.button("Start a New Interview"):
        reset_interview()
        st.rerun()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    init_session_state()

    try:
        engine = get_interviewer_engine()
    except Exception as e:
        st.error(f"Failed to initialize the interview engine: {e}")
        st.info(
            "Ensure Ollama is running locally (`ollama serve`) and that the "
            "Llama 3 model has been pulled (`ollama pull llama3`)."
        )
        st.stop()

    render_sidebar()

    st.title("🎤 AI Interviewer")
    st.caption(
        "Resume-aware mock interviews powered by a local Llama 3 model, "
        "Whisper transcription, and offline text-to-speech."
    )

    if st.session_state.interview_complete:
        render_final_report()
    elif not st.session_state.interview_started:
        render_upload_section(engine)
    else:
        render_interview_section(engine)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        st.error("An unexpected error occurred.")
        st.exception(exc)
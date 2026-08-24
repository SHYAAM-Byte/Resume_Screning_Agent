

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import database
import Speech
from ai_engine import AIEngine
from Utils import average_score, clean_resume_text, current_iso_timestamp, is_valid_session_id

logger = logging.getLogger(__name__)

__all__ = ["Interviewer", "InterviewerError"]


class InterviewerError(Exception):
    """Raised for workflow-level problems: invalid input, missing/duplicate session, etc."""


class Interviewer:

    def __init__(
        self,
        ai_engine: Optional[AIEngine] = None,
        speech_module: Any = Speech,
        db_path: str = database.DEFAULT_DB_PATH,
        num_questions: int = 5,
        difficulty: str = "medium",
    ) -> None:
        # `ai_engine` and `speech_module` are injectable so this class can be
        # unit-tested with fakes, without a real Ollama server or Whisper model.
        self.ai_engine = ai_engine or AIEngine()
        self.speech_module = speech_module
        self.db_path = db_path
        self.num_questions = num_questions
        self.difficulty = difficulty

    # ---------------------------------------------------------------- #
    # 1. Start a new interview from resume text
    # ---------------------------------------------------------------- #
    def start_interview(
        self,
        resume_text: str,
        candidate_name: Optional[str] = None,
        interview_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a new interview session from already-extracted resume text.

        Workflow: clean the resume text -> ask ai_engine.py to analyze it ->
        persist the candidate -> open a session -> ask ai_engine.py for a
        tailored question set -> persist the questions -> return the first one.

        Args:
            resume_text: Plain-text resume content. File parsing (PDF/DOCX)
                happens upstream of this module — it only works with text.
            candidate_name: Optional known display name; used if provided,
                otherwise falls back to whatever the resume analysis extracts.
            interview_id: Optional caller-supplied session id (must be a
                valid UUID string). If omitted, one is generated here.

        Returns:
            {
                "interview_id": str,
                "candidate_id": str,
                "first_question": str,
                "total_questions": int,
            }

        Raises:
            InterviewerError: Empty resume text, an invalid/duplicate
                interview_id, or no questions could be generated.
        """
        cleaned_resume_text = clean_resume_text(resume_text)
        if not cleaned_resume_text:
            raise InterviewerError("Resume text is empty; cannot start an interview.")

        if interview_id is not None and not is_valid_session_id(interview_id):
            raise InterviewerError(f"'{interview_id}' is not a valid interview id (expected a UUID string).")
        interview_id = interview_id or str(uuid.uuid4())

        if database.get_interview_session(interview_id, db_path=self.db_path) is not None:
            raise InterviewerError(f"Interview session '{interview_id}' already exists.")

        # Step 1: understand the candidate.
        resume_analysis = self.ai_engine.analyze_resume(cleaned_resume_text)

        # Step 2: persist the candidate.
        candidate_id = str(uuid.uuid4())
        database.create_candidate(
            candidate_id=candidate_id,
            name=candidate_name or resume_analysis.get("candidate_name"),
            resume_text=cleaned_resume_text,
            seniority_level=resume_analysis.get("seniority_level"),
            primary_role=resume_analysis.get("primary_role"),
            skills=resume_analysis.get("skills"),
            years_of_experience=resume_analysis.get("years_of_experience"),
            key_projects=resume_analysis.get("key_projects"),
            summary=resume_analysis.get("summary"),
            db_path=self.db_path,
        )

        # Step 3: open the interview session (starts at question index 0).
        database.create_interview_session(interview_id, candidate_id, db_path=self.db_path)

        # Step 4: generate and persist a tailored question set.
        questions = self.ai_engine.generate_questions(
            resume_analysis, num_questions=self.num_questions, difficulty=self.difficulty
        )
        for index, question_text in enumerate(questions):
            database.create_question(interview_id, index, question_text, db_path=self.db_path)

        first_question = database.get_question_by_index(interview_id, 0, db_path=self.db_path)
        if first_question is None:
            raise InterviewerError("No interview questions were generated; cannot start the interview.")

        return {
            "interview_id": interview_id,
            "candidate_id": candidate_id,
            "first_question": first_question["question_text"],
            "total_questions": len(questions),
        }

    # ---------------------------------------------------------------- #
    # 2. Receive a speech transcript from speech.py
    # ---------------------------------------------------------------- #
    def transcribe_answer(self, audio_bytes: bytes) -> str:
        """
        Convert recorded/uploaded answer audio into plain text.

        A thin pass-through to speech.py — no transcription logic lives
        here, only the decision to call it as part of the workflow.
        """
        return self.speech_module.speech_to_text(audio_bytes)

    # ---------------------------------------------------------------- #
    # 3. Evaluate one answer (resilient to AI failures)
    # ---------------------------------------------------------------- #
    def evaluate_answer(self, interview_id: str, question: str, answer_text: str) -> Dict[str, Any]:
        """
        Evaluate one candidate answer against the current question and
        persist the result.

        The raw answer is ALWAYS saved first, before evaluation is
        attempted — losing a candidate's answer would be worse than losing
        its score. If ai_engine.py's evaluation call fails for any reason
        (transient Ollama error, repeated invalid JSON, etc.), the failure
        is logged and a graceful fallback is returned instead of raising,
        so the interview session is never aborted by a single bad call.

        Returns:
            {"feedback": str, "score": float | None, "evaluation_failed": bool}

        Raises:
            InterviewerError: Empty question/answer, unknown interview_id,
                or no stored question at the session's current position.
        """
        if not question or not question.strip():
            raise InterviewerError("No question was provided to evaluate against.")
        if not answer_text or not answer_text.strip():
            raise InterviewerError("No answer was provided to evaluate.")

        session = self._get_session_or_raise(interview_id)
        candidate = database.get_candidate(session["candidate_id"], db_path=self.db_path)
        db_question = database.get_question_by_index(
            interview_id, session["current_question_index"], db_path=self.db_path
        )
        if db_question is None:
            raise InterviewerError(
                f"No stored question found for interview '{interview_id}' at the current position."
            )

        # Persist the answer unconditionally, before anything that could fail.
        answer_id = database.create_answer(
            question_id=db_question["question_id"],
            interview_id=interview_id,
            answer_text=answer_text,
            db_path=self.db_path,
        )

        resume_analysis = _resume_analysis_from_candidate(candidate)

        # --- Fault-tolerant zone: an AI failure here must not sink the interview. ---
        try:
            evaluation = self.ai_engine.evaluate_answer(question, answer_text, resume_analysis)
            score: Optional[float] = evaluation.get("score")
            feedback = evaluation.get("feedback", "")
            strengths = evaluation.get("strengths", [])
            improvements = evaluation.get("improvements", [])
            evaluation_failed = False
        except Exception as exc:  # noqa: BLE001 - intentionally broad; see module docstring
            logger.warning(
                "Answer evaluation failed for interview '%s' (question_id=%s): %s",
                interview_id,
                db_question["question_id"],
                exc,
            )
            score = None
            feedback = (
                "This answer could not be automatically evaluated due to a technical issue. "
                "It has been saved and the interview will continue."
            )
            strengths, improvements = [], []
            evaluation_failed = True
        # --- End fault-tolerant zone. ---

        if score is not None:
            database.create_score(
                answer_id=answer_id,
                interview_id=interview_id,
                score=score,
                feedback=feedback,
                strengths=strengths,
                improvements=improvements,
                db_path=self.db_path,
            )

        return {"feedback": feedback, "score": score, "evaluation_failed": evaluation_failed}

    def get_next_question(self, interview_id: str) -> Optional[str]:
        """
        Advance the session to the next unanswered question.

        Returns:
            The next question's text, or None once every question has been
            asked — as a side effect, the session is then marked 'completed'.

        Raises:
            InterviewerError: Unknown interview_id, or the current question
                hasn't been answered yet (guards against skipping ahead).
        """
        session = self._get_session_or_raise(interview_id)

        current_question = database.get_question_by_index(
            interview_id, session["current_question_index"], db_path=self.db_path
        )
        if current_question is not None:
            answered = database.get_answer_by_question(current_question["question_id"], db_path=self.db_path)
            if answered is None:
                raise InterviewerError("Cannot advance: the current question has not been answered yet.")

        next_index = session["current_question_index"] + 1
        next_question = database.get_question_by_index(interview_id, next_index, db_path=self.db_path)

        if next_question is None:
            database.update_interview_session(
                interview_id, status="completed", completed_at=current_iso_timestamp(), db_path=self.db_path
            )
            return None

        database.update_interview_session(interview_id, current_question_index=next_index, db_path=self.db_path)
        return next_question["question_text"]

    def get_session_status(self, interview_id: str) -> Dict[str, Any]:
        """
        Return a lightweight progress snapshot for an interview session.

        Returns:
            {
                "status": "in_progress" | "completed",
                "current_question_index": int,
                "total_questions": int,
                "answered_questions": int,
            }
        """
        session = self._get_session_or_raise(interview_id)
        return {
            "status": session["status"],
            "current_question_index": session["current_question_index"],
            "total_questions": len(database.list_questions(interview_id, db_path=self.db_path)),
            "answered_questions": len(database.list_answers(interview_id, db_path=self.db_path)),
        }

    # ---------------------------------------------------------------- #
    # 5. Scoring
    # ---------------------------------------------------------------- #
    def get_current_score(self, interview_id: str) -> float:
        """Return the running average score across all successfully-evaluated answers so far."""
        scores = database.list_scores(interview_id, db_path=self.db_path)
        return average_score([row["score"] for row in scores])

    # ---------------------------------------------------------------- #
    # 6. Final structured report
    # ---------------------------------------------------------------- #
    def generate_final_report(self, interview_id: str) -> Dict[str, Any]:
        """
        Calculate the final score and synthesize a structured interview
        report, persist it on the session record, and return it.

        If ai_engine.py's summary synthesis fails, this falls back to a
        locally computed report (based purely on the stored numeric scores)
        rather than raising, so a technical hiccup at the very last step
        doesn't cost the candidate their entire report.

        Returns:
            {
                "interview_id": str,
                "candidate_name": str | None,
                "overall_score": float,
                "summary": str,
                "strengths": list[str],
                "areas_for_improvement": list[str],
                "recommendation": str,
                "total_questions": int,
                "answered_questions": int,
                "question_breakdown": [
                    {"question": str, "answer": str, "feedback": str, "score": float | None}, ...
                ],
            }

        Raises:
            InterviewerError: Unknown interview_id, or no question has been
                answered yet.
        """
        session = self._get_session_or_raise(interview_id)
        candidate = database.get_candidate(session["candidate_id"], db_path=self.db_path)
        resume_analysis = _resume_analysis_from_candidate(candidate) or {}

        qa_history = self._build_qa_history(interview_id)
        if not qa_history:
            raise InterviewerError(
                f"Interview '{interview_id}' has no answered questions; cannot generate a report."
            )

        try:
            synthesis = self.ai_engine.generate_summary(resume_analysis, qa_history)
        except Exception as exc:  # noqa: BLE001 - fall back rather than lose the report entirely
            logger.warning("Final report synthesis failed for interview '%s': %s", interview_id, exc)
            synthesis = _build_fallback_summary(qa_history)

        database.update_interview_session(
            interview_id,
            status="completed",
            overall_score=synthesis["overall_score"],
            summary=synthesis.get("summary", ""),
            strengths=synthesis.get("strengths", []),
            areas_for_improvement=synthesis.get("areas_for_improvement", []),
            recommendation=synthesis.get("recommendation", ""),
            completed_at=current_iso_timestamp(),
            db_path=self.db_path,
        )

        return {
            "interview_id": interview_id,
            "candidate_name": candidate.get("name") if candidate else None,
            "overall_score": synthesis["overall_score"],
            "summary": synthesis.get("summary", ""),
            "strengths": synthesis.get("strengths", []),
            "areas_for_improvement": synthesis.get("areas_for_improvement", []),
            "recommendation": synthesis.get("recommendation", ""),
            "total_questions": len(database.list_questions(interview_id, db_path=self.db_path)),
            "answered_questions": len(qa_history),
            "question_breakdown": qa_history,
        }

    # ---------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------- #
    def _get_session_or_raise(self, interview_id: str) -> Dict[str, Any]:
        """Fetch an interview session by id, raising a clear error if it doesn't exist."""
        session = database.get_interview_session(interview_id, db_path=self.db_path)
        if session is None:
            raise InterviewerError(f"No interview session found for id '{interview_id}'.")
        return session

    def _build_qa_history(self, interview_id: str) -> List[Dict[str, Any]]:
        """
        Assemble the ordered question/answer/feedback/score history for an
        interview by composing plain CRUD reads from database.py.

        Deciding how these three tables relate to form a reportable
        transcript is workflow logic, so it lives here rather than in
        database.py. Questions with no answer yet (e.g. an interview ended
        early) are skipped; answers with no score (a failed evaluation)
        are included with score=None so the report stays transparent.
        """
        history: List[Dict[str, Any]] = []
        for question in database.list_questions(interview_id, db_path=self.db_path):
            answer = database.get_answer_by_question(question["question_id"], db_path=self.db_path)
            if answer is None:
                continue
            score_row = database.get_score_by_answer(answer["answer_id"], db_path=self.db_path)
            history.append(
                {
                    "question": question["question_text"],
                    "answer": answer["answer_text"],
                    "feedback": score_row["feedback"] if score_row else "Not evaluated.",
                    "score": score_row["score"] if score_row else None,
                }
            )
        return history


# --------------------------------------------------------------------------
# Module-level helpers (no `self` needed, kept outside the class for clarity)
# --------------------------------------------------------------------------
def _resume_analysis_from_candidate(candidate: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Reconstruct an ai_engine-shaped resume analysis dict from a stored candidate row."""
    if not candidate:
        return None
    return {
        "candidate_name": candidate.get("name"),
        "seniority_level": candidate.get("seniority_level"),
        "primary_role": candidate.get("primary_role"),
        "skills": candidate.get("skills"),
        "years_of_experience": candidate.get("years_of_experience"),
        "key_projects": candidate.get("key_projects"),
        "summary": candidate.get("summary"),
    }


def _build_fallback_summary(qa_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    A minimal, non-AI report used only if ai_engine.generate_summary()
    itself fails, so the candidate still gets a usable result instead of
    nothing at all.
    """
    numeric_scores = [item["score"] for item in qa_history if isinstance(item.get("score"), (int, float))]
    unscored_count = sum(1 for item in qa_history if item.get("score") is None)

    note = (
        f"An automated summary could not be generated for this interview. This is a fallback "
        f"report based on {len(numeric_scores)} scored answer(s)"
    )
    note += f" ({unscored_count} answer(s) were not evaluated)." if unscored_count else "."

    return {
        "overall_score": average_score(numeric_scores),
        "summary": note,
        "strengths": [],
        "areas_for_improvement": [],
        "recommendation": "Manual review recommended",
    }
import io
import fitz  # PyMuPDF
from docx import Document


def parse_resume(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from a PDF or DOCX resume.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename (used to detect file type).

    Returns:
        Extracted resume text as a string.

    Raises:
        ValueError: If the file type is unsupported or no text could be extracted.
        RuntimeError: If parsing fails.
    """

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    filename = filename.lower()

    try:
        # ---------------- PDF ----------------
        if filename.endswith(".pdf"):
            text = ""

            with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
                for page in pdf:
                    page_text = page.get_text("text")
                    if page_text:
                        text += page_text + "\n"

        # ---------------- DOCX ----------------
        elif filename.endswith(".docx"):
            document = Document(io.BytesIO(file_bytes))

            paragraphs = [
                p.text.strip()
                for p in document.paragraphs
                if p.text.strip()
            ]

            text = "\n".join(paragraphs)

            # Also extract text from tables
            for table in document.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text += "\n" + cell.text.strip()

        else:
            raise ValueError(
                "Unsupported file type. Please upload a PDF or DOCX resume."
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "No readable text was found in the uploaded resume."
            )

        return text

    except Exception as e:
        raise RuntimeError(f"Failed to parse resume: {e}")
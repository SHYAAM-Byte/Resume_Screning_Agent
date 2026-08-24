"""
database.py — SQLite persistence layer for the AI Interviewer.

Pure data-access module: five tables, generic reusable CRUD primitives, and
one thin create/get/update/delete/list function set per table. Nothing in
here decides *when* to create a question, *how* to score an answer, or
*what* the next step in an interview is — that is interview.py's job. This
module only stores and retrieves whatever it is asked to.

Tables:
    candidates          one row per uploaded resume / candidate
    interview_sessions  one row per interview attempt (keyed by interview_id)
    questions           one row per generated question, ordered by index
    answers             one row per submitted answer (1:1 with a question)
    scores              one row per evaluated answer (1:1 with an answer)

Key design points:
    - The database file (and its parent directory, if any) is created
      automatically on first use — no manual setup step is required.
    - Table schemas are created automatically the first time a connection
      is opened for a given path in this process (and again, harmlessly,
      via `CREATE TABLE IF NOT EXISTS` if the file already has them).
    - Every query uses parameterized placeholders ("?"); no value is ever
      interpolated into SQL text. Table/column names used in dynamically
      built statements come only from an internal whitelist, never from
      caller input.
    - All SQLite errors are caught and re-raised as `DatabaseError` with a
      clear message — callers never need to catch `sqlite3.Error` directly.
    - Foreign keys are enforced (`PRAGMA foreign_keys = ON`) and every
      child table cascades on delete of its parent.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "DatabaseError",
    "DEFAULT_DB_PATH",
    "init_db",
    "get_connection",
    # Candidates
    "create_candidate",
    "get_candidate",
    "update_candidate",
    "delete_candidate",
    "list_candidates",
    # Interview sessions
    "create_interview_session",
    "get_interview_session",
    "update_interview_session",
    "delete_interview_session",
    "list_interview_sessions",
    # Questions
    "create_question",
    "get_question",
    "get_question_by_index",
    "update_question",
    "delete_question",
    "list_questions",
    # Answers
    "create_answer",
    "get_answer",
    "get_answer_by_question",
    "update_answer",
    "delete_answer",
    "list_answers",
    # Scores
    "create_score",
    "get_score",
    "get_score_by_answer",
    "update_score",
    "delete_score",
    "list_scores",
]

# Configurable via environment variable so deployments can point this at a
# persistent volume/path without touching code, e.g. INTERVIEWER_DB_PATH=/data/interviews.db
DEFAULT_DB_PATH = os.environ.get("INTERVIEWER_DB_PATH", "interviews.db")


class DatabaseError(Exception):
    """Raised when a database operation fails, wrapping the underlying SQLite error."""


# --------------------------------------------------------------------------
# Connection handling + automatic database/schema creation
# --------------------------------------------------------------------------
# Tracks which db_path values have already had their schema verified/created
# in this process, so we don't re-run `CREATE TABLE IF NOT EXISTS` on every
# single call (harmless, but wasteful).
_initialized_paths: Set[str] = set()


def _ensure_parent_directory(db_path: str) -> None:
    """Create the database file's parent directory if it doesn't exist yet."""
    directory = os.path.dirname(db_path)
    if directory and not os.path.isdir(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            raise DatabaseError(
                f"Could not create directory '{directory}' for database '{db_path}': {exc}"
            ) from exc


def _ensure_schema(conn: sqlite3.Connection, db_path: str) -> None:
    """Create all tables/indexes for `db_path` if they don't already exist."""
    if db_path in _initialized_paths:
        return
    conn.executescript(_SCHEMA)
    _initialized_paths.add(db_path)


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Explicitly ensure the database file and all tables/indexes exist.

    This is called automatically the first time `get_connection()` is used
    for a given path, so calling it directly is optional. It's provided for
    callers that want to fail fast on startup (e.g. bad permissions/path)
    rather than on the first CRUD call, or for test setup.
    """
    with get_connection(db_path):
        pass  # Connecting alone triggers automatic schema creation below.


@contextmanager
def get_connection(db_path: str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """
    Open a SQLite connection for the duration of one operation.

    The database file and its schema are created automatically on first use
    for `db_path`. Commits on success; rolls back on any error. Always
    closes the connection, even on failure.

    Raises:
        DatabaseError: If the database/directory can't be created or opened,
            or if any SQLite operation performed inside the `with` block fails.
    """
    _ensure_parent_directory(db_path)

    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not open database at '{db_path}': {exc}") from exc

    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        _ensure_schema(conn, db_path)
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        logger.exception("Database operation failed against '%s'", db_path)
        raise DatabaseError(f"Database operation failed: {exc}") from exc
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    name TEXT,
    resume_filename TEXT,
    resume_text TEXT,
    seniority_level TEXT,
    primary_role TEXT,
    skills TEXT,                 -- JSON-encoded list[str]
    years_of_experience REAL,
    key_projects TEXT,           -- JSON-encoded list[str]
    summary TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS interview_sessions (
    interview_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'in_progress',
    current_question_index INTEGER NOT NULL DEFAULT 0,
    overall_score REAL,
    summary TEXT,
    strengths TEXT,               -- JSON-encoded list[str]
    areas_for_improvement TEXT,   -- JSON-encoded list[str]
    recommendation TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY (candidate_id) REFERENCES candidates (candidate_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS questions (
    question_id INTEGER PRIMARY KEY AUTOINCREMENT,
    interview_id TEXT NOT NULL,
    question_index INTEGER NOT NULL,
    question_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (interview_id) REFERENCES interview_sessions (interview_id) ON DELETE CASCADE,
    UNIQUE (interview_id, question_index)
);

CREATE TABLE IF NOT EXISTS answers (
    answer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL,
    interview_id TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (question_id) REFERENCES questions (question_id) ON DELETE CASCADE,
    FOREIGN KEY (interview_id) REFERENCES interview_sessions (interview_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scores (
    score_id INTEGER PRIMARY KEY AUTOINCREMENT,
    answer_id INTEGER NOT NULL,
    interview_id TEXT NOT NULL,
    score REAL NOT NULL,
    feedback TEXT,
    strengths TEXT,               -- JSON-encoded list[str]
    improvements TEXT,            -- JSON-encoded list[str]
    created_at TEXT NOT NULL,
    FOREIGN KEY (answer_id) REFERENCES answers (answer_id) ON DELETE CASCADE,
    FOREIGN KEY (interview_id) REFERENCES interview_sessions (interview_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_candidate ON interview_sessions (candidate_id);
CREATE INDEX IF NOT EXISTS idx_questions_interview ON questions (interview_id);
CREATE INDEX IF NOT EXISTS idx_answers_interview ON answers (interview_id);
CREATE INDEX IF NOT EXISTS idx_answers_question ON answers (question_id);
CREATE INDEX IF NOT EXISTS idx_scores_interview ON scores (interview_id);
CREATE INDEX IF NOT EXISTS idx_scores_answer ON scores (answer_id);
"""


# --------------------------------------------------------------------------
# Generic, reusable CRUD primitives (internal use only)
# --------------------------------------------------------------------------
# Every dynamically-built query below only ever inserts table/column names
# taken from this whitelist (or from the fixed column names hardcoded in
# each public function) — never from caller-supplied strings. All actual
# *values* are always passed as parameterized placeholders.
_VALID_TABLES = {"candidates", "interview_sessions", "questions", "answers", "scores"}


def _check_table(table: str) -> None:
    """Guard against programming errors: only whitelisted table names are allowed."""
    if table not in _VALID_TABLES:
        raise ValueError(f"Unknown table: '{table}'")


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    """Convert a sqlite3.Row into a plain dict (or None through unchanged)."""
    return dict(row) if row is not None else None


def _insert(conn: sqlite3.Connection, table: str, fields: Dict[str, Any]) -> int:
    """Insert one row into `table`. Returns the new row's rowid/primary key."""
    _check_table(table)
    columns = ", ".join(fields.keys())
    placeholders = ", ".join("?" for _ in fields)
    cursor = conn.execute(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
        tuple(fields.values()),
    )
    return cursor.lastrowid


def _update(
    conn: sqlite3.Connection, table: str, id_column: str, id_value: Any, fields: Dict[str, Any]
) -> bool:
    """Update selected columns of the row in `table` matching `id_column` = `id_value`."""
    _check_table(table)
    if not fields:
        return False
    assignments = ", ".join(f"{column} = ?" for column in fields)
    values: Tuple[Any, ...] = tuple(fields.values()) + (id_value,)
    cursor = conn.execute(f"UPDATE {table} SET {assignments} WHERE {id_column} = ?", values)
    return cursor.rowcount > 0


def _delete(conn: sqlite3.Connection, table: str, id_column: str, id_value: Any) -> bool:
    """Delete the row in `table` matching `id_column` = `id_value`."""
    _check_table(table)
    cursor = conn.execute(f"DELETE FROM {table} WHERE {id_column} = ?", (id_value,))
    return cursor.rowcount > 0


def _get_by_id(
    conn: sqlite3.Connection, table: str, id_column: str, id_value: Any
) -> Optional[sqlite3.Row]:
    """Fetch a single row from `table` matching `id_column` = `id_value`."""
    _check_table(table)
    cursor = conn.execute(f"SELECT * FROM {table} WHERE {id_column} = ?", (id_value,))
    return cursor.fetchone()


def _list(
    conn: sqlite3.Connection,
    table: str,
    where_clause: str = "",
    params: Sequence[Any] = (),
    order_by: str = "",
) -> List[sqlite3.Row]:
    """Fetch rows from `table`, optionally filtered and ordered."""
    _check_table(table)
    query = f"SELECT * FROM {table}"
    if where_clause:
        query += f" WHERE {where_clause}"
    if order_by:
        query += f" ORDER BY {order_by}"
    cursor = conn.execute(query, tuple(params))
    return cursor.fetchall()


def _dumps(value: Any) -> Optional[str]:
    """JSON-encode a Python value for storage, passing None through unchanged."""
    return json.dumps(value) if value is not None else None


def _loads(value: Optional[str], default: Any = None) -> Any:
    """JSON-decode a stored value, tolerating missing/invalid data via `default`."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not decode stored JSON value; returning default.")
        return default


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------
def create_candidate(
    candidate_id: str,
    name: Optional[str] = None,
    resume_filename: Optional[str] = None,
    resume_text: Optional[str] = None,
    seniority_level: Optional[str] = None,
    primary_role: Optional[str] = None,
    skills: Optional[List[str]] = None,
    years_of_experience: Optional[float] = None,
    key_projects: Optional[List[str]] = None,
    summary: Optional[str] = None,
    created_at: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """
    Insert a new candidate record.

    Args:
        candidate_id: Caller-supplied unique id (e.g. a UUID string).
        skills, key_projects: Stored as JSON; pass plain Python lists.
        created_at: ISO-8601 timestamp; defaults to "now" (UTC) if omitted.

    Returns:
        The candidate_id, for convenience/chaining.
    """
    fields = {
        "candidate_id": candidate_id,
        "name": name,
        "resume_filename": resume_filename,
        "resume_text": resume_text,
        "seniority_level": seniority_level,
        "primary_role": primary_role,
        "skills": _dumps(skills),
        "years_of_experience": years_of_experience,
        "key_projects": _dumps(key_projects),
        "summary": summary,
        "created_at": created_at or _now(),
    }
    with get_connection(db_path) as conn:
        _insert(conn, "candidates", fields)
    return candidate_id


def _deserialize_candidate(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if data is None:
        return None
    data["skills"] = _loads(data.get("skills"), default=[])
    data["key_projects"] = _loads(data.get("key_projects"), default=[])
    return data


def get_candidate(candidate_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetch one candidate by id, or None if not found."""
    with get_connection(db_path) as conn:
        row = _get_by_id(conn, "candidates", "candidate_id", candidate_id)
    return _deserialize_candidate(_row_to_dict(row))


def update_candidate(candidate_id: str, db_path: str = DEFAULT_DB_PATH, **fields: Any) -> bool:
    """
    Update one or more fields of a candidate (e.g. update_candidate(cid, summary="...")).
    `skills`/`key_projects`, if provided, are JSON-encoded automatically.

    Returns:
        True if a row was updated, False if no candidate matched the id.
    """
    if "skills" in fields:
        fields["skills"] = _dumps(fields["skills"])
    if "key_projects" in fields:
        fields["key_projects"] = _dumps(fields["key_projects"])
    with get_connection(db_path) as conn:
        return _update(conn, "candidates", "candidate_id", candidate_id, fields)


def delete_candidate(candidate_id: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Delete a candidate. Cascades to their interview sessions, questions, answers, and scores."""
    with get_connection(db_path) as conn:
        return _delete(conn, "candidates", "candidate_id", candidate_id)


def list_candidates(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """List all candidates, most recently created first."""
    with get_connection(db_path) as conn:
        rows = _list(conn, "candidates", order_by="created_at DESC")
    return [_deserialize_candidate(_row_to_dict(row)) for row in rows]


# --------------------------------------------------------------------------
# Interview sessions
# --------------------------------------------------------------------------
def create_interview_session(
    interview_id: str,
    candidate_id: str,
    status: str = "in_progress",
    started_at: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """
    Insert a new interview session linked to an existing candidate.

    Returns:
        The interview_id, for convenience/chaining.
    """
    fields = {
        "interview_id": interview_id,
        "candidate_id": candidate_id,
        "status": status,
        "current_question_index": 0,
        "started_at": started_at or _now(),
    }
    with get_connection(db_path) as conn:
        _insert(conn, "interview_sessions", fields)
    return interview_id


def _deserialize_session(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if data is None:
        return None
    data["strengths"] = _loads(data.get("strengths"), default=[])
    data["areas_for_improvement"] = _loads(data.get("areas_for_improvement"), default=[])
    return data


def get_interview_session(
    interview_id: str, db_path: str = DEFAULT_DB_PATH
) -> Optional[Dict[str, Any]]:
    """Fetch one interview session by id, or None if not found."""
    with get_connection(db_path) as conn:
        row = _get_by_id(conn, "interview_sessions", "interview_id", interview_id)
    return _deserialize_session(_row_to_dict(row))


def update_interview_session(
    interview_id: str, db_path: str = DEFAULT_DB_PATH, **fields: Any
) -> bool:
    """
    Update one or more fields of an interview session (e.g. status, score,
    current_question_index, completed_at). `strengths`/`areas_for_improvement`,
    if provided, are JSON-encoded automatically.

    Returns:
        True if a row was updated, False if no session matched the id.
    """
    if "strengths" in fields:
        fields["strengths"] = _dumps(fields["strengths"])
    if "areas_for_improvement" in fields:
        fields["areas_for_improvement"] = _dumps(fields["areas_for_improvement"])
    with get_connection(db_path) as conn:
        return _update(conn, "interview_sessions", "interview_id", interview_id, fields)


def delete_interview_session(interview_id: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Delete an interview session. Cascades to its questions, answers, and scores."""
    with get_connection(db_path) as conn:
        return _delete(conn, "interview_sessions", "interview_id", interview_id)


def list_interview_sessions(
    candidate_id: Optional[str] = None, db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """List interview sessions, optionally filtered to one candidate, most recent first."""
    with get_connection(db_path) as conn:
        if candidate_id:
            rows = _list(
                conn,
                "interview_sessions",
                where_clause="candidate_id = ?",
                params=(candidate_id,),
                order_by="started_at DESC",
            )
        else:
            rows = _list(conn, "interview_sessions", order_by="started_at DESC")
    return [_deserialize_session(_row_to_dict(row)) for row in rows]


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------
def create_question(
    interview_id: str,
    question_index: int,
    question_text: str,
    created_at: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """
    Insert a new question for an interview session.

    Returns:
        The auto-generated question_id.
    """
    fields = {
        "interview_id": interview_id,
        "question_index": question_index,
        "question_text": question_text,
        "created_at": created_at or _now(),
    }
    with get_connection(db_path) as conn:
        return _insert(conn, "questions", fields)


def get_question(question_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetch one question by its primary key, or None if not found."""
    with get_connection(db_path) as conn:
        row = _get_by_id(conn, "questions", "question_id", question_id)
    return _row_to_dict(row)


def get_question_by_index(
    interview_id: str, question_index: int, db_path: str = DEFAULT_DB_PATH
) -> Optional[Dict[str, Any]]:
    """Fetch a question by its position (question_index) within an interview."""
    with get_connection(db_path) as conn:
        rows = _list(
            conn,
            "questions",
            where_clause="interview_id = ? AND question_index = ?",
            params=(interview_id, question_index),
        )
    return _row_to_dict(rows[0]) if rows else None


def update_question(question_id: int, db_path: str = DEFAULT_DB_PATH, **fields: Any) -> bool:
    """Update one or more fields of a question. Returns True if a row was updated."""
    with get_connection(db_path) as conn:
        return _update(conn, "questions", "question_id", question_id, fields)


def delete_question(question_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Delete a question. Cascades to its answer and that answer's score, if any."""
    with get_connection(db_path) as conn:
        return _delete(conn, "questions", "question_id", question_id)


def list_questions(interview_id: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """List all questions for an interview, in question_index order."""
    with get_connection(db_path) as conn:
        rows = _list(
            conn,
            "questions",
            where_clause="interview_id = ?",
            params=(interview_id,),
            order_by="question_index ASC",
        )
    return [_row_to_dict(row) for row in rows]


# --------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------
def create_answer(
    question_id: int,
    interview_id: str,
    answer_text: str,
    created_at: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """
    Insert a new answer for a question.

    Returns:
        The auto-generated answer_id.
    """
    fields = {
        "question_id": question_id,
        "interview_id": interview_id,
        "answer_text": answer_text,
        "created_at": created_at or _now(),
    }
    with get_connection(db_path) as conn:
        return _insert(conn, "answers", fields)


def get_answer(answer_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetch one answer by its primary key, or None if not found."""
    with get_connection(db_path) as conn:
        row = _get_by_id(conn, "answers", "answer_id", answer_id)
    return _row_to_dict(row)


def get_answer_by_question(
    question_id: int, db_path: str = DEFAULT_DB_PATH
) -> Optional[Dict[str, Any]]:
    """Fetch the answer submitted for a given question, or None if unanswered."""
    with get_connection(db_path) as conn:
        rows = _list(conn, "answers", where_clause="question_id = ?", params=(question_id,))
    return _row_to_dict(rows[0]) if rows else None


def update_answer(answer_id: int, db_path: str = DEFAULT_DB_PATH, **fields: Any) -> bool:
    """Update one or more fields of an answer. Returns True if a row was updated."""
    with get_connection(db_path) as conn:
        return _update(conn, "answers", "answer_id", answer_id, fields)


def delete_answer(answer_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Delete an answer. Cascades to its score, if any."""
    with get_connection(db_path) as conn:
        return _delete(conn, "answers", "answer_id", answer_id)


def list_answers(interview_id: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """List all answers submitted for an interview, oldest first."""
    with get_connection(db_path) as conn:
        rows = _list(
            conn,
            "answers",
            where_clause="interview_id = ?",
            params=(interview_id,),
            order_by="created_at ASC",
        )
    return [_row_to_dict(row) for row in rows]


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------
def create_score(
    answer_id: int,
    interview_id: str,
    score: float,
    feedback: Optional[str] = None,
    strengths: Optional[List[str]] = None,
    improvements: Optional[List[str]] = None,
    created_at: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """
    Insert a new score for an answer.

    Returns:
        The auto-generated score_id.
    """
    fields = {
        "answer_id": answer_id,
        "interview_id": interview_id,
        "score": score,
        "feedback": feedback,
        "strengths": _dumps(strengths),
        "improvements": _dumps(improvements),
        "created_at": created_at or _now(),
    }
    with get_connection(db_path) as conn:
        return _insert(conn, "scores", fields)


def _deserialize_score(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if data is None:
        return None
    data["strengths"] = _loads(data.get("strengths"), default=[])
    data["improvements"] = _loads(data.get("improvements"), default=[])
    return data


def get_score(score_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetch one score by its primary key, or None if not found."""
    with get_connection(db_path) as conn:
        row = _get_by_id(conn, "scores", "score_id", score_id)
    return _deserialize_score(_row_to_dict(row))


def get_score_by_answer(answer_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetch the score recorded for a given answer, or None if not yet scored."""
    with get_connection(db_path) as conn:
        rows = _list(conn, "scores", where_clause="answer_id = ?", params=(answer_id,))
    return _deserialize_score(_row_to_dict(rows[0])) if rows else None


def update_score(score_id: int, db_path: str = DEFAULT_DB_PATH, **fields: Any) -> bool:
    """
    Update one or more fields of a score. `strengths`/`improvements`, if
    provided, are JSON-encoded automatically. Returns True if a row was updated.
    """
    if "strengths" in fields:
        fields["strengths"] = _dumps(fields["strengths"])
    if "improvements" in fields:
        fields["improvements"] = _dumps(fields["improvements"])
    with get_connection(db_path) as conn:
        return _update(conn, "scores", "score_id", score_id, fields)


def delete_score(score_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Delete a score."""
    with get_connection(db_path) as conn:
        return _delete(conn, "scores", "score_id", score_id)


def list_scores(interview_id: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """List all scores recorded for an interview, oldest first."""
    with get_connection(db_path) as conn:
        rows = _list(
            conn,
            "scores",
            where_clause="interview_id = ?",
            params=(interview_id,),
            order_by="created_at ASC",
        )
    return [_deserialize_score(_row_to_dict(row)) for row in rows]


# --------------------------------------------------------------------------
# Small internal utility (kept local so this module has zero sibling
# imports and can be dropped into any project as-is)
# --------------------------------------------------------------------------
def _now() -> str:
    """Current UTC time as an ISO-8601 string, used for default timestamp columns."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
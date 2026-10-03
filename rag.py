"""
rag.py

Enterprise AI Research & Decision Intelligence Agent
------------------------------------------------------
Lightweight, MVP-scoped RAG (retrieval-augmented generation) module.

Responsibilities (and ONLY these):
    - Extract text from uploaded PDF, DOCX, and TXT documents.
    - Split extracted text into meaningful, sentence-aware chunks.
    - Represent chunks as lightweight embeddings and store them in a
      simple in-memory vector store.
    - Retrieve the chunks most relevant to a research question, with
      source (filename / page) metadata attached.
    - Fail gracefully on missing, empty, unsupported, or corrupted
      documents -- one bad file never breaks ingestion of the rest.

This module does NOT implement:
    - The LLM client / Ollama calls      (see ai_engine.py)
    - Web search                          (see web_search.py)
    - Workflow orchestration              (see research_agent.py)
    - Any UI                              (see app.py)

Design notes (why this stays simple)
--------------------------------------
Per the MVP scope, this module deliberately avoids:
    - External vector databases or servers (Chroma, Pinecone,
      pgvector, etc.) -- everything lives in plain Python objects for
      the lifetime of a DocumentRetriever instance.
    - Downloaded neural embedding models (sentence-transformers,
      torch, etc.) or embedding API calls -- "embeddings" here are a
      hand-rolled TF-IDF vector representation (stdlib `math` +
      `collections.Counter` only), which is a well-understood,
      dependency-free stand-in appropriate for MVP-scale document
      sets. The public interface below does not leak this detail, so
      it can be swapped for a real embedding model later without
      touching research_agent.py.

Each `ingest()` call rebuilds the index from scratch from the given
document paths; incremental/append indexing is out of scope for this
MVP and would complicate the TF-IDF vocabulary in ways not worth the
complexity here.

Public interface (what research_agent.py calls)
--------------------------------------------------
    class DocumentRetriever:
        def ingest(self, document_paths: list[str]) -> None
        def retrieve(self, query: str, top_k: int = 6) -> list[dict]

    Each dict returned by `retrieve()` has the keys:
        "text"   : str            -- the retrieved chunk's text
        "source" : str            -- originating filename
        "page"   : int | None     -- 1-based page number (PDF only)
        "score"  : float          -- cosine similarity, 0.0-1.0

Lower-level, independently reusable building blocks (`extract_pages`,
`chunk_text`) are also exposed for callers that want extraction or
chunking without the full retrieval pipeline.
"""

import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from docx import Document as DocxDocument
from pypdf import PdfReader

logger = logging.getLogger(__name__)

__all__ = [
    "DocumentChunk",
    "DocumentExtractionError",
    "extract_pages",
    "chunk_text",
    "LocalVectorStore",
    "DocumentRetriever",
]

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".pdf", ".docx", ".txt")

DEFAULT_CHUNK_SIZE = 1000     # target characters per chunk (approximate)
DEFAULT_CHUNK_OVERLAP = 150   # characters of trailing context carried forward
DEFAULT_TOP_K = 6

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_TXT_ENCODINGS: Tuple[str, ...] = ("utf-8", "utf-16", "latin-1")


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
class DocumentExtractionError(Exception):
    """Raised when a single document cannot be opened or parsed.

    Callers within this module (DocumentRetriever.ingest) catch this
    and skip the offending file with a logged warning rather than
    letting it abort the whole ingest.
    """


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class DocumentChunk:
    """A single chunk of text extracted from an uploaded document."""

    id: str
    text: str
    source: str            # filename the chunk came from
    page: Optional[int]    # 1-based page number, or None if not applicable
    chunk_index: int        # position of this chunk within its source document


# ----------------------------------------------------------------------
# Text extraction
# ----------------------------------------------------------------------
def extract_pages(path: str) -> List[Tuple[Optional[int], str]]:
    """Extract text from a PDF, DOCX, or TXT file.

    Returns a list of (page_number, text) tuples. `page_number` is a
    1-based page index for PDFs and None for formats without a native
    page concept (DOCX, TXT).

    Raises:
        DocumentExtractionError: if the file is missing, of an
            unsupported type, or cannot be parsed.
    """
    if not os.path.isfile(path):
        raise DocumentExtractionError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentExtractionError(
            f"Unsupported file type '{ext}' for '{path}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    if ext == ".pdf":
        return _extract_pdf_pages(path)
    if ext == ".docx":
        return _extract_docx_pages(path)
    return _extract_txt_pages(path)  # ext == ".txt"


def _extract_pdf_pages(path: str) -> List[Tuple[Optional[int], str]]:
    try:
        reader = PdfReader(path)
    except Exception as exc:
        raise DocumentExtractionError(f"Could not open PDF '{path}': {exc}") from exc

    pages: List[Tuple[Optional[int], str]] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            # A single unreadable page shouldn't sink the whole PDF.
            logger.warning("Could not extract text from page %d of '%s': %s", index, path, exc)
            text = ""
        pages.append((index, text))

    if not pages:
        raise DocumentExtractionError(f"PDF '{path}' has no pages.")
    return pages


def _extract_docx_pages(path: str) -> List[Tuple[Optional[int], str]]:
    try:
        document = DocxDocument(path)
    except Exception as exc:
        raise DocumentExtractionError(f"Could not open DOCX '{path}': {exc}") from exc

    paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    # DOCX has no reliable page concept without rendering the layout,
    # so the whole document is treated as one page-less unit.
    return [(None, "\n\n".join(paragraphs))]


def _extract_txt_pages(path: str) -> List[Tuple[Optional[int], str]]:
    last_error: Optional[Exception] = None
    for encoding in _TXT_ENCODINGS:
        try:
            with open(path, "r", encoding=encoding) as handle:
                return [(None, handle.read())]
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except OSError as exc:
            raise DocumentExtractionError(f"Could not read TXT '{path}': {exc}") from exc

    raise DocumentExtractionError(
        f"Could not decode text file '{path}' with any of {_TXT_ENCODINGS}: {last_error}"
    )


# ----------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------
def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split `text` into meaningful, sentence-respecting chunks of
    roughly `chunk_size` characters, carrying up to `overlap`
    characters of trailing context into the next chunk for continuity.

    `chunk_size`/`overlap` are approximate targets rather than hard
    guarantees -- appropriate for this module's MVP scope. A single
    sentence longer than `chunk_size` is hard-split on its own so it
    can never produce an unbounded chunk.
    """
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return []

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(normalized) if s.strip()]
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        chunks.append(" ".join(current))

        if overlap <= 0:
            current, current_len = [], 0
            return

        carry: List[str] = []
        carry_len = 0
        for sentence in reversed(current):
            extra = len(sentence) + (1 if carry else 0)
            if carry_len + extra > overlap:
                break
            carry.insert(0, sentence)
            carry_len += extra

        current, current_len = carry, carry_len

    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                flush()
            for start in range(0, len(sentence), chunk_size):
                piece = sentence[start : start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
            current, current_len = [], 0
            continue

        added_len = len(sentence) + (1 if current else 0)
        if current and current_len + added_len > chunk_size:
            flush()
            added_len = len(sentence) + (1 if current else 0)

        current.append(sentence)
        current_len += added_len

    if current:
        chunks.append(" ".join(current))

    return chunks


# ----------------------------------------------------------------------
# Minimal, dependency-free TF-IDF "embedding" + cosine similarity
# ----------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class _TfidfVectorizer:
    """A minimal, dependency-free TF-IDF vectorizer used as a
    lightweight stand-in for a full embedding model. Fit once per
    `LocalVectorStore.index()` call on the freshly extracted chunk
    texts, then reused (without refitting) to vectorize queries.
    """

    def __init__(self) -> None:
        self._idf: Dict[str, float] = {}

    def fit(self, documents: List[str]) -> None:
        doc_count = len(documents)
        doc_frequency: Counter = Counter()
        for document in documents:
            for token in set(_tokenize(document)):
                doc_frequency[token] += 1

        # Smoothed IDF (always > 0); unseen-at-fit-time query terms are
        # simply absent from the resulting vector rather than erroring.
        self._idf = {
            token: math.log((1 + doc_count) / (1 + df)) + 1.0
            for token, df in doc_frequency.items()
        }

    def transform(self, text: str) -> Dict[str, float]:
        """Return a sparse TF-IDF vector as a {token: weight} dict,
        L2-normalized so cosine similarity reduces to a dot product.
        """
        tokens = _tokenize(text)
        if not tokens:
            return {}

        term_frequency = Counter(tokens)
        weights = {
            token: (count / len(tokens)) * self._idf[token]
            for token, count in term_frequency.items()
            if token in self._idf
        }

        norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        if norm == 0:
            return {}
        return {token: weight / norm for token, weight in weights.items()}


def _cosine_similarity(vector_a: Dict[str, float], vector_b: Dict[str, float]) -> float:
    """Dot product of two already L2-normalized sparse vectors."""
    if not vector_a or not vector_b:
        return 0.0
    if len(vector_a) > len(vector_b):
        vector_a, vector_b = vector_b, vector_a
    return sum(weight * vector_b.get(token, 0.0) for token, weight in vector_a.items())


# ----------------------------------------------------------------------
# Local vector store
# ----------------------------------------------------------------------
class LocalVectorStore:
    """A minimal, in-memory vector store: TF-IDF vectors kept in plain
    Python objects, no external database or ANN index. Appropriate for
    the modest chunk counts typical of an MVP document-upload feature;
    not intended to scale to large corpora.
    """

    def __init__(self) -> None:
        self._vectorizer = _TfidfVectorizer()
        self._chunks: List[DocumentChunk] = []
        self._vectors: List[Dict[str, float]] = []

    @property
    def is_empty(self) -> bool:
        return not self._chunks

    def index(self, chunks: List[DocumentChunk]) -> None:
        """(Re)build the store from `chunks`, replacing any previous
        contents (see module docstring on why indexing is a full
        rebuild rather than incremental).
        """
        self._chunks = list(chunks)
        self._vectorizer = _TfidfVectorizer()
        self._vectors = []

        if not self._chunks:
            return

        texts = [chunk.text for chunk in self._chunks]
        self._vectorizer.fit(texts)
        self._vectors = [self._vectorizer.transform(text) for text in texts]

    def search(self, query: str, top_k: int) -> List[Tuple[DocumentChunk, float]]:
        """Return up to `top_k` (chunk, score) pairs most similar to
        `query`, best first. Returns [] if the store is empty, the
        query has no in-vocabulary terms, or top_k <= 0.
        """
        if self.is_empty or top_k <= 0:
            return []

        query_vector = self._vectorizer.transform(query)
        if not query_vector:
            return []

        scored = (
            (chunk, _cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self._chunks, self._vectors)
        )
        relevant = [(chunk, score) for chunk, score in scored if score > 0.0]
        relevant.sort(key=lambda pair: pair[1], reverse=True)
        return relevant[:top_k]


# ----------------------------------------------------------------------
# Public retriever (what research_agent.py uses)
# ----------------------------------------------------------------------
class DocumentRetriever:
    """Ingests uploaded PDF/DOCX/TXT documents, chunks and indexes them
    in a lightweight in-memory vector store, and retrieves the chunks
    most relevant to a research question, with source metadata.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._store = LocalVectorStore()

    def ingest(self, document_paths: List[str]) -> None:
        """Extract, chunk, and index `document_paths`.

        Replaces any previously ingested documents. Files that are
        missing, unsupported, empty, or corrupted are skipped with a
        logged warning; this method itself never raises for those
        cases, so one bad upload can't prevent the others from being
        used.
        """
        chunks: List[DocumentChunk] = []

        for path in document_paths or []:
            source = os.path.basename(path)
            try:
                pages = extract_pages(path)
            except DocumentExtractionError as exc:
                logger.warning("Skipping document '%s': %s", source, exc)
                continue

            doc_chunk_count = 0
            for page_number, page_text in pages:
                for piece in chunk_text(page_text, self.chunk_size, self.chunk_overlap):
                    chunks.append(
                        DocumentChunk(
                            id=f"{source}::{len(chunks)}",
                            text=piece,
                            source=source,
                            page=page_number,
                            chunk_index=doc_chunk_count,
                        )
                    )
                    doc_chunk_count += 1

            if doc_chunk_count == 0:
                logger.warning("No extractable text found in '%s'; it will be ignored.", source)

        self._store.index(chunks)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Return up to `top_k` chunks most relevant to `query`, each
        with its source metadata, as plain dicts:

            {"text": str, "source": str, "page": int | None, "score": float}

        Returns an empty list -- never raises -- if no documents have
        been ingested, the query is empty, or nothing matches;
        retrieval degrades gracefully rather than breaking the wider
        research workflow.
        """
        query = (query or "").strip()
        if not query:
            return []

        results = self._store.search(query, top_k=top_k)
        return [
            {
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "score": round(score, 4),
            }
            for chunk, score in results
        ]
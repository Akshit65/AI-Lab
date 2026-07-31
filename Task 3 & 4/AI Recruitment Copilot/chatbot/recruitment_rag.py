"""
recruitment_rag.py

RAG pipeline for the AI Recruitment Copilot.
Handles: PDF resume loading, chunking, embedding, FAISS indexing,
top-k retrieval, and Groq-based match report generation.

Drop this file into: Enterprise Bot/chatbot/recruitment_rag.py
"""

from __future__ import annotations

import os
import json
import re
from collections import defaultdict

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from config import GROQ_API_KEY, GROQ_MODEL

# ---------------------------------------------------------------------------
# Setup — embeddings + LLM are created once and reused
# ---------------------------------------------------------------------------

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

_index = None  # in-memory FAISS index cache, built once at startup

RESUME_FOLDER = os.path.join("data", "resumes")


# ---------------------------------------------------------------------------
# 1. PDF loading
# ---------------------------------------------------------------------------

def extract_text_from_pdf(filepath: str) -> str:
    """Extract raw text from a single PDF file.

    Tries normal text extraction first. If that returns nothing (common
    for scanned resumes or flattened/graphic PDF templates with no real
    text layer), falls back to OCR via Tesseract.
    """
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(layout=True) or page.extract_text()
            if page_text:
                text += page_text + "\n"

    if text.strip():
        return text

    # Fallback: OCR the PDF page images (requires: brew install tesseract poppler,
    # pip install pytesseract pdf2image)
    print(f"No text layer found in {filepath}, falling back to OCR...")
    ocr_text = ""
    try:
        pages = convert_from_path(filepath)
        for page_image in pages:
            ocr_text += pytesseract.image_to_string(page_image) + "\n"
    except Exception as e:
        print(f"OCR failed for {filepath}: {e}")
        return ""

    return ocr_text


def load_resumes(folder: str = RESUME_FOLDER) -> list[dict]:
    """Load every PDF in the folder. Returns [{'id': candidate_id, 'text': ...}, ...]."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Resume folder not found: {folder}")

    resumes = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(folder, fname)
            text = extract_text_from_pdf(path)
            if not text.strip():
                continue  # skip unreadable/empty PDFs
            candidate_id = os.path.splitext(fname)[0]
            resumes.append({"id": candidate_id, "text": text})

    if not resumes:
        raise ValueError(f"No readable PDFs found in {folder}")

    return resumes


# ---------------------------------------------------------------------------
# 2. Index building
# ---------------------------------------------------------------------------

def build_index(resumes: list[dict] | None = None) -> FAISS:
    """Chunk + embed all resumes and build the FAISS index. Caches result in memory."""
    global _index

    if resumes is None:
        resumes = load_resumes()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,  # slightly higher overlap to soften multi-column section cuts
    )

    docs = []
    for r in resumes:
        chunks = splitter.split_text(r["text"])
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"candidate_id": r["id"]}
                )
            )

    _index = FAISS.from_documents(docs, embeddings)
    return _index


def get_index() -> FAISS:
    """Return the cached index, building it on first call."""
    global _index
    if _index is None:
        build_index()
    return _index


# ---------------------------------------------------------------------------
# 3. Retrieval (top-k candidates, not just top-k chunks)
# ---------------------------------------------------------------------------

def retrieve_top_candidates(job_description: str, top_k: int = 3, chunks_per_search: int = 15):
    """
    Retrieve the top_k candidates most relevant to the JD.
    Pulls a wider chunk pool first, then groups by candidate and ranks
    by best (lowest-distance) chunk score, so strong candidates aren't
    crowded out by one candidate hogging multiple top slots.
    """
    index = get_index()

    # similarity_search_with_score returns (Document, distance) — lower distance = better match
    results = index.similarity_search_with_score(job_description, k=chunks_per_search)

    grouped = defaultdict(list)
    for doc, score in results:
        grouped[doc.metadata["candidate_id"]].append((doc, score))

    # rank candidates by their best (lowest) chunk distance
    ranked_candidates = sorted(
        grouped.items(),
        key=lambda item: min(score for _, score in item[1])
    )

    top_candidates = ranked_candidates[:top_k]

    # return {candidate_id: [chunk_text, ...]}
    return {
        cid: [doc.page_content for doc, _ in chunks]
        for cid, chunks in top_candidates
    }


# ---------------------------------------------------------------------------
# 4. Report generation (Groq)
# ---------------------------------------------------------------------------

REPORT_PROMPT = """You are a recruitment assistant. Compare the job description below to the candidate's resume excerpts.

Job Description:
{job_description}

Resume Excerpts (Candidate: {candidate_id}):
{context}

Return ONLY valid JSON, no markdown formatting, no extra text, in exactly this format:
{{"candidate_id": "{candidate_id}", "match_score": <integer 0-100>, "matching_skills": [<strings>], "missing_skills": [<strings>], "reasoning": "<one short paragraph>"}}
"""


def _clean_json_response(raw: str) -> dict:
    """Strip markdown fences etc. and parse the model's JSON output safely."""
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "candidate_id": "unknown",
            "match_score": 0,
            "matching_skills": [],
            "missing_skills": [],
            "reasoning": "Could not parse model output for this candidate."
        }


def generate_report_for_candidate(job_description: str, candidate_id: str, chunks: list[str]) -> dict:
    context = "\n---\n".join(chunks)
    prompt = REPORT_PROMPT.format(
        job_description=job_description,
        candidate_id=candidate_id,
        context=context
    )
    response = llm.invoke(prompt)
    return _clean_json_response(response.content)


# ---------------------------------------------------------------------------
# 5. Full pipeline entry point — used by the chat route
# ---------------------------------------------------------------------------

def handle_recruitment_query(message: str, top_k: int = 3) -> str:
    """
    Takes a natural-language chat message containing a job description,
    retrieves the top_k best-matching candidates, generates a match report
    for each, and returns a formatted plain-text reply for the chat UI.
    """
    top_candidates = retrieve_top_candidates(message, top_k=top_k)

    if not top_candidates:
        return "I couldn't find any matching candidates in the resume database."

    reports = []
    for candidate_id, chunks in top_candidates.items():
        report = generate_report_for_candidate(message, candidate_id, chunks)
        reports.append(report)

    # sort by match score, highest first, in case the LLM's scoring reorders things
    reports.sort(key=lambda r: r.get("match_score", 0), reverse=True)

    return format_reports_as_text(reports)


def format_reports_as_text(reports: list[dict]) -> str:
    lines = [f"Here are the top {len(reports)} matching candidates:\n"]
    for r in reports:
        lines.append(f"Candidate: {r.get('candidate_id', 'unknown')}")
        lines.append(f"Match Score: {r.get('match_score', 0)}%")
        matching = ", ".join(r.get("matching_skills", [])) or "None identified"
        missing = ", ".join(r.get("missing_skills", [])) or "None identified"
        lines.append(f"✅ Matching Skills: {matching}")
        lines.append(f"❌ Missing Skills: {missing}")
        lines.append(f"Reasoning: {r.get('reasoning', '')}")
        lines.append("")  # blank line between candidates
    return "\n".join(lines)
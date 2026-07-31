"""
tools.py — Task 4 Agentic AI tool library

Three independently testable tools the agent can call:
  1. retrieve_candidates   — wraps Task 3's FAISS retrieval
  2. score_candidates      — wraps Task 3's Groq match-report generation
  3. draft_outreach        — new: drafts a personalized outreach email per candidate

Each tool is a plain Python function (no framework dependency) so it can be
unit tested in isolation, then wired into the LangGraph nodes in agent_graph.py.
"""

from __future__ import annotations

import sys
import os

# Ensure the project root (parent of task4_agent/) is on the path so
# `config` and `chatbot` resolve regardless of where this script is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL

# Reuse Task 3's pipeline directly — no reimplementation
from chatbot.recruitment_rag import retrieve_top_candidates, generate_report_for_candidate

client = Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Tool 1 — Retrieve candidates
# ---------------------------------------------------------------------------

def retrieve_candidates(job_description: str, top_k: int = 3) -> dict:
    """
    Returns {candidate_id: [chunk_text, ...]} for the top_k best-matching
    candidates against the job description. Raises no exception on empty
    results — returns {} so the caller can branch on it.
    """
    try:
        return retrieve_top_candidates(job_description, top_k=top_k)
    except Exception as e:
        print(f"[retrieve_candidates] failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Tool 2 — Score candidates
# ---------------------------------------------------------------------------

def score_candidates(job_description: str, candidates: dict) -> list[dict]:
    """
    Takes {candidate_id: [chunks]} and returns a list of match report dicts:
    [{"candidate_id", "match_score", "matching_skills", "missing_skills", "reasoning"}, ...]
    """
    reports = []
    for candidate_id, chunks in candidates.items():
        try:
            report = generate_report_for_candidate(job_description, candidate_id, chunks)
            reports.append(report)
        except Exception as e:
            print(f"[score_candidates] failed for {candidate_id}: {e}")
    return reports


# ---------------------------------------------------------------------------
# Tool 3 — Draft outreach email
# ---------------------------------------------------------------------------

OUTREACH_PROMPT = """You are a recruiter's assistant drafting a short, professional outreach email
to a candidate who matched a job opening.

Job Description:
{job_description}

Candidate ID: {candidate_id}
Match Score: {match_score}%
Matching Skills: {matching_skills}
Reasoning: {reasoning}

Write a short (under 120 words), warm, professional outreach email inviting this candidate
to a first conversation about the role. Reference 1-2 of their matching skills specifically.
Do not invent a company name — refer to it generically as "our team".

Return ONLY valid JSON, no markdown, in this format:
{{"candidate_id": "{candidate_id}", "subject": "<email subject>", "body": "<email body>"}}
"""


def _clean_json(raw: str) -> dict:
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"candidate_id": "unknown", "subject": "", "body": "Could not draft email — parse error."}


def draft_outreach(job_description: str, report: dict) -> dict:
    """Drafts one outreach email for a single scored candidate report."""
    prompt = OUTREACH_PROMPT.format(
        job_description=job_description,
        candidate_id=report.get("candidate_id", "unknown"),
        match_score=report.get("match_score", 0),
        matching_skills=", ".join(report.get("matching_skills", [])),
        reasoning=report.get("reasoning", "")
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=300
    )
    return _clean_json(response.choices[0].message.content)
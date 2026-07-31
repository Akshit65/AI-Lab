"""
agent_graph.py — Task 4 Agentic AI: Candidate Shortlisting & Outreach Agent

Graph flow:

    START
      -> retrieve_node        (Tool 1: FAISS retrieval, reused from Task 3)
      -> score_node            (Tool 2: Groq match scoring, reused from Task 3)
      -> quality_check_node    (CONDITIONAL BRANCH on match quality)
            |-- strong matches  -> draft_node   (Tool 3: draft outreach emails)
            |-- no strong match -> no_match_node (graceful exit, no outreach)
      -> human_approval_node   (HITL GATE: pauses for approve/reject)
            |-- approved -> finalize_node -> END
            |-- rejected -> rejected_node  -> END
      -> END

Every node writes to the audit trail (audit_log.jsonl) before returning.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import os
from datetime import datetime, timezone
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from tools import retrieve_candidates, score_candidates, draft_outreach

MATCH_SCORE_THRESHOLD = 50  # below this, no candidate counts as a "strong match"
AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "audit_log.jsonl")


# ---------------------------------------------------------------------------
# Audit trail — component 8 (required)
# ---------------------------------------------------------------------------

def log_audit(node: str, input_summary: str, output_summary: str, decision: Optional[str] = None):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": node,
        "input": input_summary,
        "output": output_summary,
        "decision": decision,
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# State — the shared object every node reads from and writes to
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    job_description: str
    candidates: dict            # {candidate_id: [chunks]}
    reports: list               # [{candidate_id, match_score, matching_skills, missing_skills, reasoning}]
    strong_matches: list        # subset of reports above threshold
    outreach_emails: list       # [{candidate_id, subject, body}]
    branch_taken: str           # "strong_match" | "no_match"
    human_decision: str         # "approved" | "rejected" | "pending"
    final_status: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve_node(state: AgentState) -> AgentState:
    jd = state["job_description"]
    candidates = retrieve_candidates(jd, top_k=3)
    log_audit(
        "retrieve_node",
        input_summary=f"job_description='{jd[:80]}...'",
        output_summary=f"retrieved {len(candidates)} candidates: {list(candidates.keys())}",
    )
    state["candidates"] = candidates
    return state


def score_node(state: AgentState) -> AgentState:
    jd = state["job_description"]
    candidates = state["candidates"]

    if not candidates:
        # Failure handling: nothing retrieved, don't call the LLM for nothing
        log_audit("score_node", input_summary="no candidates to score", output_summary="skipped scoring")
        state["reports"] = []
        return state

    reports = score_candidates(jd, candidates)
    log_audit(
        "score_node",
        input_summary=f"scoring {len(candidates)} candidates",
        output_summary=f"produced {len(reports)} reports: "
                        f"{[(r.get('candidate_id'), r.get('match_score')) for r in reports]}",
    )
    state["reports"] = reports
    return state


def quality_check_node(state: AgentState) -> AgentState:
    """Observation handling + conditional branching decision point."""
    reports = state["reports"]
    strong = [r for r in reports if r.get("match_score", 0) >= MATCH_SCORE_THRESHOLD]
    state["strong_matches"] = strong

    branch = "strong_match" if strong else "no_match"
    state["branch_taken"] = branch

    log_audit(
        "quality_check_node",
        input_summary=f"{len(reports)} reports evaluated against threshold {MATCH_SCORE_THRESHOLD}",
        output_summary=f"{len(strong)} strong matches found",
        decision=branch,
    )
    return state


def route_after_quality_check(state: AgentState) -> str:
    """LangGraph conditional edge function — routes based on state."""
    return state["branch_taken"]


def draft_node(state: AgentState) -> AgentState:
    jd = state["job_description"]
    emails = []
    for report in state["strong_matches"]:
        try:
            email = draft_outreach(jd, report)
            emails.append(email)
        except Exception as e:
            # Failure handling: one bad draft shouldn't kill the whole batch
            log_audit("draft_node", input_summary=f"drafting for {report.get('candidate_id')}",
                       output_summary=f"FAILED: {e}")
    state["outreach_emails"] = emails
    log_audit(
        "draft_node",
        input_summary=f"drafting outreach for {len(state['strong_matches'])} strong matches",
        output_summary=f"{len(emails)} emails drafted",
    )
    return state


def no_match_node(state: AgentState) -> AgentState:
    state["outreach_emails"] = []
    state["final_status"] = "no_strong_match"
    log_audit(
        "no_match_node",
        input_summary="no candidates cleared the match threshold",
        output_summary="agent stopped gracefully, no outreach drafted",
    )
    return state


def human_approval_node(state: AgentState) -> AgentState:
    """
    HITL gate (required component 6). In this local/CLI version, approval is
    collected via input(); in the Flask/UI version this would pause the graph
    and wait for a POST from the approval button before resuming.
    """
    print("\n--- HUMAN APPROVAL REQUIRED ---")
    for email in state["outreach_emails"]:
        print(f"\nCandidate: {email.get('candidate_id')}")
        print(f"Subject: {email.get('subject')}")
        print(f"Body: {email.get('body')}")

    decision = input("\nApprove sending these outreach emails? (yes/no): ").strip().lower()
    state["human_decision"] = "approved" if decision == "yes" else "rejected"

    log_audit(
        "human_approval_node",
        input_summary=f"{len(state['outreach_emails'])} drafted emails presented for approval",
        output_summary=f"human responded: {state['human_decision']}",
        decision=state["human_decision"],
    )
    return state


def route_after_approval(state: AgentState) -> str:
    return state["human_decision"]


def finalize_node(state: AgentState) -> AgentState:
    state["final_status"] = "approved_and_logged"
    log_audit(
        "finalize_node",
        input_summary=f"{len(state['outreach_emails'])} approved emails",
        output_summary="finalized (sandboxed — no real email sent)",
    )
    return state


def rejected_node(state: AgentState) -> AgentState:
    state["final_status"] = "rejected_by_human"
    log_audit(
        "rejected_node",
        input_summary=f"{len(state['outreach_emails'])} drafted emails",
        output_summary="human rejected — no action taken",
    )
    return state


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("score", score_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("draft", draft_node)
    graph.add_node("no_match", no_match_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("rejected", rejected_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "score")
    graph.add_edge("score", "quality_check")

    graph.add_conditional_edges(
        "quality_check",
        route_after_quality_check,
        {"strong_match": "draft", "no_match": "no_match"},
    )

    graph.add_edge("draft", "human_approval")
    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"approved": "finalize", "rejected": "rejected"},
    )

    graph.add_edge("no_match", END)
    graph.add_edge("finalize", END)
    graph.add_edge("rejected", END)

    return graph.compile()


if __name__ == "__main__":
    agent = build_agent()
    job_description = input("Enter a job description: ").strip()

    initial_state: AgentState = {
        "job_description": job_description,
        "candidates": {},
        "reports": [],
        "strong_matches": [],
        "outreach_emails": [],
        "branch_taken": "",
        "human_decision": "pending",
        "final_status": "",
    }

    final_state = agent.invoke(initial_state)
    print(f"\n--- AGENT FINISHED: {final_state['final_status']} ---")
    print(f"Full audit trail: {AUDIT_LOG_PATH}")
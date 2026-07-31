"""
test_tools.py — unit tests for Task 4's tool library

Run from project root: python -m pytest task4_agent/test_tools.py -v
(or: python task4_agent/test_tools.py to run without pytest)
"""

from tools import retrieve_candidates, score_candidates, draft_outreach


def test_retrieve_candidates_returns_dict():
    result = retrieve_candidates("Looking for accounting experience and QuickBooks knowledge", top_k=3)
    assert isinstance(result, dict)
    print(f"retrieve_candidates: got {len(result)} candidates -> PASS")


def test_retrieve_candidates_handles_nonsense_query():
    # Should not crash on an odd/empty-ish query, just return best-effort or {}
    result = retrieve_candidates("asdkjhaskjdh", top_k=3)
    assert isinstance(result, dict)
    print("retrieve_candidates handles nonsense query without crashing -> PASS")


def test_score_candidates_structure():
    fake_candidates = {"id_1": ["Experienced accountant with QuickBooks and financial reporting skills."]}
    reports = score_candidates("Looking for accounting and QuickBooks experience", fake_candidates)
    assert isinstance(reports, list)
    if reports:
        r = reports[0]
        assert "candidate_id" in r and "match_score" in r
    print(f"score_candidates: got {len(reports)} reports -> PASS")


def test_score_candidates_empty_input():
    reports = score_candidates("any JD", {})
    assert reports == []
    print("score_candidates handles empty candidate dict -> PASS")


def test_draft_outreach_structure():
    fake_report = {
        "candidate_id": "id_1",
        "match_score": 85,
        "matching_skills": ["QuickBooks", "financial reporting"],
        "reasoning": "Strong overlap with accounting requirements."
    }
    email = draft_outreach("Looking for accounting experience", fake_report)
    assert "subject" in email and "body" in email
    print(f"draft_outreach: subject='{email.get('subject')}' -> PASS")


if __name__ == "__main__":
    test_retrieve_candidates_returns_dict()
    test_retrieve_candidates_handles_nonsense_query()
    test_score_candidates_structure()
    test_score_candidates_empty_input()
    test_draft_outreach_structure()
    print("\nAll tool tests passed.")
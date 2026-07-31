AI Labs Report

Tasks 1–4: Prompt-Driven Resume Classification, Job Market Analytics Dashboard, AI Recruitment Copilot (RAG) and Candidate Shortlisting & Outreach Agent (LangGraph)

Task 1 — Resume & Job Posting Classification via Prompt Engineering
Context
Recruiters and job platforms deal with a huge volume of unstructured resume and job posting text, and manually tagging it by category, seniority, or sentiment is slow and honestly pretty inconsistent from person to person.
Problem
There wasn't a quick, training-free way to sort resume or job-posting text into structured categories (skills domain, seniority level, sentiment toward the role) without building or fine-tuning a custom model.
Approach
I used a labelled dataset of 60 resumes and ran a local Llama model through Ollama (llama3.2:latest) to do three prompt-driven NLP tasks: domain classification, entity extraction (skills / years of experience), and match-suitability summarization. For classification I compared zero-shot vs few-shot prompting directly against each other, and for entity extraction I compared zero-shot vs chain-of-thought.

Task 2 — Job Market Analytics Dashboard
Context
Job seekers and HR teams often work off gut-feel impressions of which skills are 'hot,' without a structured, data-backed view of which skills are actually rising or declining across roles or industries.
Problem
There wasn't an easy way to see which skills are trending, where the gaps are between what job postings ask for and what candidates actually have, and how that varies by role or domain — without manually scanning postings one by one.
Approach
I built an ETL pipeline from a resume/job posting dataset into a structured data model (four CSVs: KPI summaries, domain gaps, role breakdowns, and skill gaps), then built an interactive Tableau dashboard with KPI cards, drill-downs, and an anomaly table for skill demand vs supply.

Task 3 — AI Recruitment Copilot using Retrieval-Augmented Generation (RAG)
Context
Most companies get hundreds of resumes per opening, and the ATS tools they use mostly just scan for keywords instead of actually understanding what's in the resume.
Problem
Because of this keyword-only matching, recruiters can miss good candidates whose resumes use different words than the job description, and the system doesn't explain why it picked or rejected someone.
Approach
I used resume data along with semantic embeddings and RAG so the system compares resumes and job descriptions based on meaning, not just keywords, and pulls out the relevant resume sections to support its decision. Rather than build this as a standalone script, I integrated it directly into EnterpriseBot — an existing Flask chatbot I'd already built for a different task — so the recruitment copilot works through the same chat interface as general conversation, with an intent classifier routing messages to either the general chatbot or the recruitment RAG pipeline.

Task 4 — Candidate Shortlisting & Outreach Agent (LangGraph)
Context
Recruiters manually orchestrate a multi-step workflow for every open role — screening resumes against a job description, shortlisting the strongest candidates, and drafting personalized outreach to each one — and currently do this by hand for every posting.
Problem
This orchestration is repetitive, slow, and inconsistent between recruiters; there is no system that plans and executes the full screen-to-outreach sequence autonomously while still keeping a human in control of what actually gets sent.
Approach
I will build a LangGraph agent that takes a job description as its goal, retrieves and scores candidates using the RAG pipeline built in Task 3, branches based on whether strong matches exist, drafts personalized outreach emails for shortlisted candidates via Groq, and pauses for human approval before finalizing — with every step logged to an audit trail.


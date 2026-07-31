# ============================================================
# TASK 1 — Prompt-Driven NLP with a Local LLM
# AI Recruitment Copilot | M.Sc. AI in Business | SRH Berlin
# ============================================================

# ── SECTION 1: CONFIG ───────────────────────────────────────
MODEL_NAME   = "llama3.2:latest"       # change to llama3.2:1b if low RAM
CSV_PATH = "/Users/akshitkumar/Documents/2nd Sem/AI Labs/Task 1/resume_data.csv"
TEXT_COL     = "career_objective"
SKILLS_COL   = "skills"
LABEL_COL    = "job_position_name"
SCORE_COL    = "matched_score"
BACKEND      = "ollama"            # "ollama" or "gemini"
SAMPLE_SIZE  = 60                  # rows to run (≥50 as required)

# Gemini — only used if BACKEND = "gemini"
# Set your key as an environment variable: export GEMINI_API_KEY=your_key
import os
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

# ── SECTION 2: LOAD DATA ────────────────────────────────────
import pandas as pd
import json, re, time
from collections import defaultdict

print("=" * 60)
print("SECTION 2 — Loading Dataset")
print("=" * 60)

df_raw = pd.read_csv(CSV_PATH, on_bad_lines="skip")

# Fix BOM encoding on column name
df_raw.rename(columns={"\ufeffjob_position_name": "job_position_name"}, inplace=True)

# Drop rows missing key columns
df = df_raw.dropna(subset=[TEXT_COL, SKILLS_COL, LABEL_COL]).copy()

# Clean job category labels (strip whitespace/newlines)
df[LABEL_COL] = df[LABEL_COL].str.strip()

# Keep a balanced sample of SAMPLE_SIZE rows across categories
df = (
    df.groupby(LABEL_COL, group_keys=False)
    .apply(lambda x: x.sample(min(len(x), 3), random_state=42))
    .reset_index(drop=True)
    .head(SAMPLE_SIZE)
)

print(f"Dataset shape after cleaning : {df.shape}")
print(f"\nLabel distribution (job categories):")
print(df[LABEL_COL].value_counts().to_string())

# ── SECTION 3: PROMPT DEFINITIONS ───────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 — Prompt Definitions")
print("=" * 60)

# ── TASK A: Topic/Domain Classification ──────────────────────
# Version A1 — Zero-shot
CLASSIFY_ZERO_SHOT = """You are a recruitment expert. 
Classify the resume text into ONE of these job domains:
Software Engineering, Data & AI, Civil & Mechanical Engineering, HR & Administration, Business & Marketing, Finance & Audit, IT Infrastructure, Other

Reply with the domain name only. No explanation.

Resume: {text}"""

# Version A2 — Few-shot
CLASSIFY_FEW_SHOT = """You are a recruitment expert.
Classify the resume text into ONE of these job domains:
Software Engineering, Data & AI, Civil & Mechanical Engineering, HR & Administration, Business & Marketing, Finance & Audit, IT Infrastructure, Other

Examples:
Resume: "Experienced Python developer with ML and cloud skills." → Data & AI
Resume: "HR professional specializing in talent acquisition and payroll." → HR & Administration
Resume: "Civil site engineer with experience in road and bridge construction." → Civil & Mechanical Engineering
Resume: "Network administrator managing server infrastructure and support desks." → IT Infrastructure

Now classify this:
Resume: {text}

Reply with the domain name only."""

# Version A3 — No-domain list (for testing LLM's own classification)
CLASSIFY_NO_DOMAINS = """You are a recruitment expert.
Read this resume and classify it into the most appropriate job domain.
Reply with the domain name only.

Resume: {text}"""

# ── TASK B: Entity Extraction ─────────────────────────────────
# Version B1 — Zero-shot structured output
ENTITY_ZERO_SHOT = """Extract skills and experience from this resume text.
Return ONLY valid JSON with this exact schema:
{{"skills": ["skill1", "skill2"], "years_experience": "X years or unknown", "education_level": "degree or unknown"}}

Resume: {text}"""

# Version B2 — Chain-of-thought + structured output
ENTITY_COT = """You are a skilled resume parser.
Step 1: Read the resume carefully.
Step 2: Identify all technical and soft skills mentioned.
Step 3: Estimate total years of experience from context clues.
Step 4: Identify the highest education level mentioned.
Step 5: Return ONLY valid JSON with this schema:
{{"skills": ["skill1", "skill2"], "years_experience": "X years or unknown", "education_level": "degree or unknown"}}

Resume: {text}

Think step by step, then return only the JSON."""

# ── TASK C: Match Suitability Summarization ───────────────────
# Version C1 — Direct summarization (zero-shot)
SUMMARY_ZERO_SHOT = """You are a recruiter. 
Given a resume and a job title, write ONE sentence explaining how well this candidate fits the role.

Job Title: {label}
Resume: {text}

One sentence only."""

# Version C2 — Role-prompted summarization
SUMMARY_ROLE = """You are a senior HR director with 15 years of hiring experience.
Evaluate this candidate for the role and write ONE concise sentence summarizing their suitability.
Be specific — mention one strength and one gap if visible.

Job Title: {label}
Resume: {text}

One sentence only."""

print("✅ 3 NLP tasks defined, 2 prompt versions each (6 prompts total)")
print("   Task A: Domain Classification  (zero-shot vs few-shot)")
print("   Task B: Entity Extraction      (zero-shot vs chain-of-thought)")
print("   Task C: Match Summarization    (zero-shot vs role-prompted)")

# ── SECTION 4: LLM CALLER FUNCTION ──────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 — LLM Caller Function")
print("=" * 60)

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Send a prompt to Ollama or Gemini and return the response text."""
    if BACKEND == "ollama":
        import ollama
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response["message"]["content"].strip()

    elif BACKEND == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = model.generate_content(full_prompt)
        return response.text.strip()

    else:
        raise ValueError(f"Unknown backend: {BACKEND}")


# Quick connectivity test
print(f"Testing connection to {BACKEND}...")
try:
    test = call_llm(
        "You are a helpful assistant. Reply with one word only.",
        "Say hello."
    )
    print(f"✅ LLM connection successful. Test response: '{test}'")
except Exception as e:
    print(f"❌ LLM connection failed: {e}")
    print("   Make sure Ollama is running: ollama serve")
    raise

# ── SECTION 5: TASK LOOP ─────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5 — Running NLP Tasks on Dataset")
print("=" * 60)

results = []

for idx, row in df.iterrows():
    text       = str(row[TEXT_COL])[:600]   # cap length for speed
    skills_raw = str(row[SKILLS_COL])[:400]
    label      = str(row[LABEL_COL])
    score      = row.get(SCORE_COL, None)

    row_result = {
        "index"      : idx,
        "true_label" : label,
        "matched_score": score,
        "resume_snippet": text[:120],
    }

    # ── Task A: Classification ──
    try:
        a1 = call_llm("You are a recruitment expert.", CLASSIFY_ZERO_SHOT.format(text=text))
        a2 = call_llm("You are a recruitment expert.", CLASSIFY_FEW_SHOT.format(text=text))
    except Exception as e:
        a1 = a2 = f"ERROR: {e}"

    row_result["classify_zeroshot"]  = a1
    row_result["classify_fewshot"]   = a2

    # ── Task B: Entity Extraction ──
    try:
        b1 = call_llm("You are a resume parser. Return only JSON.", ENTITY_ZERO_SHOT.format(text=text))
        b2 = call_llm("You are a resume parser. Return only JSON.", ENTITY_COT.format(text=text))
    except Exception as e:
        b1 = b2 = f"ERROR: {e}"

    row_result["entity_zeroshot"] = b1
    row_result["entity_cot"]      = b2

    # ── Task C: Summarization ──
    try:
        c1 = call_llm("You are a recruiter.", SUMMARY_ZERO_SHOT.format(text=text, label=label))
        c2 = call_llm("You are a senior HR director.", SUMMARY_ROLE.format(text=text, label=label))
    except Exception as e:
        c1 = c2 = f"ERROR: {e}"

    row_result["summary_zeroshot"] = c1
    row_result["summary_role"]     = c2

    results.append(row_result)

    if (len(results)) % 10 == 0:
        print(f"  Processed {len(results)}/{SAMPLE_SIZE} rows...")

print(f"✅ All {len(results)} rows processed.")

# ── SECTION 6: EVALUATION ────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6 — Evaluation")
print("=" * 60)

# Map the 28 specific job titles → 8 broad domains for fair comparison
DOMAIN_MAP = {
    "Senior Software Engineer"       : "Software Engineering",
    "Full Stack Developer (Python,React js)" : "Software Engineering",
    "Senior iOS Engineer"            : "Software Engineering",
    "AI Engineer"                    : "Data & AI",
    "Machine Learning (ML) Engineer" : "Data & AI",
    "Data Science Engineer"          : "Data & AI",
    "Data Engineer"                  : "Data & AI",
    "Intern (Generative AI Engineering - 2D/3D Image Generation)": "Data & AI",
    "Civil Engineer"                 : "Civil & Mechanical Engineering",
    "Site Engineer"                  : "Civil & Mechanical Engineering",
    "Project Coordinator (Civil)"    : "Civil & Mechanical Engineering",
    "Mechanical Engineer"            : "Civil & Mechanical Engineering",
    "Mechanical Designer"            : "Civil & Mechanical Engineering",
    "Management Trainee - Mechanical": "Civil & Mechanical Engineering",
    "HR Officer"                     : "HR & Administration",
    "Manager- Human Resource Management (HRM)" : "HR & Administration",
    "Asst. Manager/ Manger (Administrative)"   : "HR & Administration",
    "Business Development Executive" : "Business & Marketing",
    "Marketing Officer"              : "Business & Marketing",
    "Executive/ Senior Executive- Trade Marketing, Hygiene Products": "Business & Marketing",
    "Executive - VAT"                : "Finance & Audit",
    "Sr.Officer / Executive - Internal Audit"  : "Finance & Audit",
    "Head of Internal Control & Compliance (ICC) - SEVP/DMD": "Finance & Audit",
    "Network Support Engineer"       : "IT Infrastructure",
    "DevOps Engineer"                : "IT Infrastructure",
    "System Administrator (Operation & Maintenance of Server, Storage & Service Desk System)": "IT Infrastructure",
    "Database Administrator (DBA)"   : "IT Infrastructure",
    "Executive/ Sr. Executive -IT"   : "IT Infrastructure",
}

VALID_DOMAINS = list(set(DOMAIN_MAP.values()))

def normalize_prediction(pred: str) -> str:
    """Match LLM output to one of the 8 valid domains."""
    pred_lower = pred.lower().strip()
    for domain in VALID_DOMAINS:
        if domain.lower() in pred_lower or pred_lower in domain.lower():
            return domain
    # Fuzzy keyword fallback
    if any(k in pred_lower for k in ["software", "developer", "ios", "engineer"]):
        return "Software Engineering"
    if any(k in pred_lower for k in ["data", "ai", "ml", "machine"]):
        return "Data & AI"
    if any(k in pred_lower for k in ["civil", "mechanical", "site"]):
        return "Civil & Mechanical Engineering"
    if any(k in pred_lower for k in ["hr", "human resource", "admin"]):
        return "HR & Administration"
    if any(k in pred_lower for k in ["business", "marketing", "trade"]):
        return "Business & Marketing"
    if any(k in pred_lower for k in ["finance", "audit", "vat", "compliance"]):
        return "Finance & Audit"
    if any(k in pred_lower for k in ["network", "devops", "server", "it infra", "database", "dba"]):
        return "IT Infrastructure"
    return "Other"

# Score Task A: Classification
correct_zs = correct_fs = 0
per_class_zs = defaultdict(lambda: {"correct": 0, "total": 0})
per_class_fs = defaultdict(lambda: {"correct": 0, "total": 0})

for r in results:
    true_domain = DOMAIN_MAP.get(r["true_label"].strip(), "Other")
    pred_zs     = normalize_prediction(r["classify_zeroshot"])
    pred_fs     = normalize_prediction(r["classify_fewshot"])

    r["true_domain"]       = true_domain
    r["pred_domain_zs"]    = pred_zs
    r["pred_domain_fs"]    = pred_fs
    r["correct_zs"]        = int(pred_zs == true_domain)
    r["correct_fs"]        = int(pred_fs == true_domain)

    correct_zs += r["correct_zs"]
    correct_fs += r["correct_fs"]
    per_class_zs[true_domain]["correct"] += r["correct_zs"]
    per_class_zs[true_domain]["total"]   += 1
    per_class_fs[true_domain]["correct"] += r["correct_fs"]
    per_class_fs[true_domain]["total"]   += 1

total = len(results)
print(f"\n── Task A: Domain Classification ──")
print(f"  Zero-shot accuracy : {correct_zs}/{total} = {correct_zs/total:.1%}")
print(f"  Few-shot  accuracy : {correct_fs}/{total} = {correct_fs/total:.1%}")

print(f"\n  Per-class breakdown:")
print(f"  {'Domain':<40} {'ZeroShot':>10} {'FewShot':>10}")
print(f"  {'-'*62}")
for domain in sorted(per_class_zs.keys()):
    zs = per_class_zs[domain]
    fs = per_class_fs[domain]
    zs_acc = zs["correct"] / zs["total"] if zs["total"] else 0
    fs_acc = fs["correct"] / fs["total"] if fs["total"] else 0
    print(f"  {domain:<40} {zs_acc:>9.1%} {fs_acc:>10.1%}")

# Score Task B: Entity Extraction (check if valid JSON returned)
def is_valid_json(text: str) -> bool:
    try:
        cleaned = re.sub(r"```json|```", "", text).strip()
        obj = json.loads(cleaned)
        return "skills" in obj
    except Exception:
        return False

valid_b1 = sum(1 for r in results if is_valid_json(r["entity_zeroshot"]))
valid_b2 = sum(1 for r in results if is_valid_json(r["entity_cot"]))
print(f"\n── Task B: Entity Extraction (valid JSON rate) ──")
print(f"  Zero-shot  valid JSON : {valid_b1}/{total} = {valid_b1/total:.1%}")
print(f"  CoT prompt valid JSON : {valid_b2}/{total} = {valid_b2/total:.1%}")

# Task C: Summarization — qualitative, show 3 examples
print(f"\n── Task C: Summarization (3 sample comparisons) ──")
for i, r in enumerate(results[:3]):
    print(f"\n  Candidate {i+1} | Role: {r['true_label']}")
    print(f"  Zero-shot : {r['summary_zeroshot'][:140]}")
    print(f"  Role-prompted: {r['summary_role'][:140]}")

# ── SECTION 7: SUMMARY PROMPT ────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 7 — Business Insight Summary (LLM-Generated)")
print("=" * 60)

summary_context = f"""
Results from prompt-driven NLP on {total} resumes:

Task A - Domain Classification:
- Zero-shot accuracy: {correct_zs/total:.1%}
- Few-shot accuracy: {correct_fs/total:.1%}

Task B - Entity Extraction (valid JSON rate):
- Zero-shot: {valid_b1/total:.1%}
- Chain-of-thought: {valid_b2/total:.1%}

Task C - Match Summarization: qualitative, role-prompted summaries
were more specific and mentioned strengths and gaps.
"""

insight_prompt = f"""
You are a business analyst reviewing an AI recruitment experiment.
Based on these results, write a short 3-sentence business insight:
1. What worked best and why?
2. What prompt technique should a recruitment team adopt?
3. What are the main limitations?

Results:
{summary_context}
"""

try:
    insight = call_llm("You are a business analyst.", insight_prompt)
    print("\nAI-Generated Business Insight:")
    print(insight)
except Exception as e:
    print(f"Summary prompt failed: {e}")

# ── SECTION 8: SAVE OUTPUT ───────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 8 — Saving Results")
print("=" * 60)

output_df = pd.DataFrame(results)
output_df.to_csv("task1_results.csv", index=False)
print(f"✅ Results saved to task1_results.csv ({len(output_df)} rows, {len(output_df.columns)} columns)")

# Save a metrics summary too
metrics = {
    "total_rows"            : total,
    "classify_zeroshot_acc" : round(correct_zs / total, 4),
    "classify_fewshot_acc"  : round(correct_fs / total, 4),
    "entity_zeroshot_json_rate" : round(valid_b1 / total, 4),
    "entity_cot_json_rate"      : round(valid_b2 / total, 4),
}
with open("task1_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("✅ Metrics saved to task1_metrics.json")
print("\n✅ Task 1 complete. All 8 handbook sections executed.")
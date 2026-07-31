# ============================================================
# TASK 2 — ETL Pipeline
# Job Market Skill Gap Analytics | SRH Berlin
# ============================================================

import pandas as pd
import ast, re
from collections import Counter

# ── CONFIG ───────────────────────────────────────────────────
CSV_PATH = "resume_data.csv"

# ── LOAD ─────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
df.rename(columns={"\ufeffjob_position_name": "job_position_name"}, inplace=True)
df = df.dropna(subset=["job_position_name", "skills", "skills_required"])
df["job_position_name"] = df["job_position_name"].str.strip()
print(f"Clean rows: {df.shape[0]}")

# ── DOMAIN MAP ───────────────────────────────────────────────
DOMAIN_MAP = {
    "Senior Software Engineer"                        : "Software Engineering",
    "Full Stack Developer (Python,React js)"          : "Software Engineering",
    "Senior iOS Engineer"                             : "Software Engineering",
    "AI Engineer"                                     : "Data & AI",
    "Machine Learning (ML) Engineer"                  : "Data & AI",
    "Data Science Engineer"                           : "Data & AI",
    "Data Engineer"                                   : "Data & AI",
    "Intern (Generative AI Engineering - 2D/3D Image Generation)": "Data & AI",
    "Civil Engineer"                                  : "Civil & Mechanical Engineering",
    "Site Engineer"                                   : "Civil & Mechanical Engineering",
    "Project Coordinator (Civil)"                     : "Civil & Mechanical Engineering",
    "Mechanical Engineer"                             : "Civil & Mechanical Engineering",
    "Mechanical Designer"                             : "Civil & Mechanical Engineering",
    "Management Trainee - Mechanical"                 : "Civil & Mechanical Engineering",
    "HR Officer"                                      : "HR & Administration",
    "Manager- Human Resource Management (HRM)"        : "HR & Administration",
    "Asst. Manager/ Manger (Administrative)"          : "HR & Administration",
    "Business Development Executive"                  : "Business & Marketing",
    "Marketing Officer"                               : "Business & Marketing",
    "Executive/ Senior Executive- Trade Marketing, Hygiene Products": "Business & Marketing",
    "Executive - VAT"                                 : "Finance & Audit",
    "Sr.Officer / Executive - Internal Audit"         : "Finance & Audit",
    "Head of Internal Control & Compliance (ICC) - SEVP/DMD": "Finance & Audit",
    "Network Support Engineer"                        : "IT Infrastructure",
    "DevOps Engineer"                                 : "IT Infrastructure",
    "System Administrator (Operation & Maintenance of Server, Storage & Service Desk System)": "IT Infrastructure",
    "Database Administrator (DBA)"                    : "IT Infrastructure",
    "Executive/ Sr. Executive -IT"                    : "IT Infrastructure",
}
df["domain"] = df["job_position_name"].map(DOMAIN_MAP).fillna("Other")

# ── SKILL PARSER ─────────────────────────────────────────────
def parse_skills(skill_str):
    """Extract individual skills from a comma/semicolon separated string."""
    if pd.isna(skill_str):
        return []
    # Remove brackets, quotes, extra spaces
    cleaned = re.sub(r"[\[\]\"']", "", str(skill_str))
    skills = [s.strip().lower() for s in re.split(r"[,;|\n]", cleaned) if s.strip()]
    return skills

df["skills_list"]    = df["skills"].apply(parse_skills)
df["required_list"]  = df["skills_required"].apply(parse_skills)

# ── MATCH RATE ───────────────────────────────────────────────
def match_rate(candidate_skills, required_skills):
    """% of required skills the candidate has."""
    if not required_skills:
        return 0
    matches = set(candidate_skills) & set(required_skills)
    return round(len(matches) / len(required_skills) * 100, 1)

df["match_rate"] = df.apply(
    lambda r: match_rate(r["skills_list"], r["required_list"]), axis=1
)

# ── OUTPUT 1: KPI SUMMARY (global numbers) ───────────────────
all_demand = [s for skills in df["required_list"] for s in skills]
all_supply = [s for skills in df["skills_list"]   for s in skills]

demand_counts = Counter(all_demand)
supply_counts = Counter(all_supply)

top_demanded  = demand_counts.most_common(1)[0][0] if demand_counts else "N/A"
avg_match     = round(df["match_rate"].mean(), 1)
total_skills  = len(set(all_demand))

# Biggest gap = skill with highest (demand - supply)
all_skills = set(all_demand) | set(all_supply)
gaps = {s: demand_counts.get(s, 0) - supply_counts.get(s, 0) for s in all_skills}
biggest_gap_skill = max(gaps, key=gaps.get)

worst_domain = df.groupby("domain")["match_rate"].mean().idxmin()

kpi_df = pd.DataFrame([{
    "total_unique_skills_demanded" : total_skills,
    "top_demanded_skill"           : top_demanded,
    "avg_candidate_match_rate_pct" : avg_match,
    "biggest_skill_gap"            : biggest_gap_skill,
    "worst_domain_match_rate"      : worst_domain,
}])
kpi_df.to_csv("kpi_summary.csv", index=False)
print("✅ kpi_summary.csv saved")

# ── OUTPUT 2: DOMAIN SUMMARY ─────────────────────────────────
domain_rows = []
for domain, group in df.groupby("domain"):
    d_skills = [s for skills in group["required_list"] for s in skills]
    s_skills = [s for skills in group["skills_list"]   for s in skills]
    d_counts = Counter(d_skills)
    s_counts = Counter(s_skills)
    gap_score = sum(
        max(d_counts.get(sk, 0) - s_counts.get(sk, 0), 0)
        for sk in set(d_counts) | set(s_counts)
    )
    domain_rows.append({
        "domain"               : domain,
        "total_candidates"     : len(group),
        "avg_match_rate_pct"   : round(group["match_rate"].mean(), 1),
        "total_skills_demanded": len(set(d_skills)),
        "total_skills_supplied": len(set(s_skills)),
        "gap_score"            : gap_score,
        "top_demanded_skill"   : Counter(d_skills).most_common(1)[0][0] if d_skills else "N/A",
    })

domain_df = pd.DataFrame(domain_rows)
domain_df.to_csv("domain_summary.csv", index=False)
print("✅ domain_summary.csv saved")

# ── OUTPUT 3: ROLE SUMMARY ───────────────────────────────────
role_rows = []
for (domain, role), group in df.groupby(["domain", "job_position_name"]):
    d_skills = [s for skills in group["required_list"] for s in skills]
    s_skills = [s for skills in group["skills_list"]   for s in skills]
    role_rows.append({
        "domain"             : domain,
        "job_role"           : role,
        "total_candidates"   : len(group),
        "avg_match_rate_pct" : round(group["match_rate"].mean(), 1),
        "skills_demanded"    : len(set(d_skills)),
        "skills_supplied"    : len(set(s_skills)),
        "top_skill"          : Counter(d_skills).most_common(1)[0][0] if d_skills else "N/A",
    })

role_df = pd.DataFrame(role_rows)
role_df.to_csv("role_summary.csv", index=False)
print("✅ role_summary.csv saved")

# ── OUTPUT 4: SKILL GAP (per skill per domain) ───────────────
skill_rows = []
for domain, group in df.groupby("domain"):
    d_skills = [s for skills in group["required_list"] for s in skills]
    s_skills = [s for skills in group["skills_list"]   for s in skills]
    d_counts = Counter(d_skills)
    s_counts = Counter(s_skills)
    all_domain_skills = set(d_counts) | set(s_counts)
    for skill in all_domain_skills:
        demand  = d_counts.get(skill, 0)
        supply  = s_counts.get(skill, 0)
        gap     = demand - supply
        anomaly = "YES" if demand > 0 and supply == 0 else (
                  "YES" if demand > 0 and demand / max(supply, 1) >= 2 else "NO")
        skill_rows.append({
            "domain"      : domain,
            "skill"       : skill,
            "demand_count": demand,
            "supply_count": supply,
            "gap"         : gap,
            "anomaly_flag": anomaly,
        })

skill_df = pd.DataFrame(skill_rows).sort_values(
    ["domain", "gap"], ascending=[True, False]
)
skill_df.to_csv("skill_gap.csv", index=False)
print("✅ skill_gap.csv saved")
print(f"\n✅ ETL complete — 4 files ready for Tableau")
print(f"   Rows in skill_gap.csv : {len(skill_df)}")
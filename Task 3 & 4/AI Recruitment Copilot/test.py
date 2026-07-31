from chatbot.recruitment_rag import load_resumes

resumes = load_resumes()
print(f"Loaded {len(resumes)} resumes")
for r in resumes:
    print(f"{r['id']}: {len(r['text'])} characters")
"""
intent_router.py

Classifies incoming chat messages as either "recruitment" (route to RAG
pipeline) or "general" (route to the existing chatbot).

Drop this file into: Enterprise Bot/chatbot/intent_router.py
"""

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)

INTENT_PROMPT = """Classify the user's message into exactly one category. Respond with ONLY the category word, nothing else.

Categories:
- recruitment: the message asks to match, screen, or compare candidates/resumes against a job description, job requirements, or skills needed for a role
- general: anything else (greetings, general questions, unrelated topics)

Message: "{message}"
Category:"""


def classify_intent(message: str) -> str:
    prompt = INTENT_PROMPT.format(message=message)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5
    )
    result = response.choices[0].message.content.strip().lower()
    return "recruitment" if "recruitment" in result else "general"
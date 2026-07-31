from ollama import chat
from config import OLLAMA_MODEL


def ask_llm(prompt):
    response = chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response["message"]["content"]

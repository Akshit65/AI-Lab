import os
from dotenv import load_dotenv

load_dotenv()

# Existing general-chat model (Ollama)
OLLAMA_MODEL = "llama3.2:latest"

# Recruitment Copilot (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
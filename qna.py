import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def answer_question_with_gemini(question: str) -> str:
    """Answer a general knowledge / academic question using Gemini 1.5 Pro."""
    try:
        model = genai.GenerativeModel(model_name="models/gemini-3.8-flash")
        response = model.generate_content(question)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error in QnA: {e}"
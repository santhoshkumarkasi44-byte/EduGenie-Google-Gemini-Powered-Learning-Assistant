import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def summarize_text(text: str) -> str:
    """Summarize a long educational passage into a concise, easy-to-read version."""
    try:
        model = genai.GenerativeModel(model_name="models/gemini-3.8-flash")
        prompt = f"Summarize the following text in simple language:\n\n{text}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error in Summary: {e}"
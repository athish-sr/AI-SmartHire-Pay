# resume_parser.py
import fitz  # type: ignore # PyMuPDF
import requests
import json
import hashlib

OLLAMA_API = "http://localhost:11434"
SEED = 42
TEMPERATURE = 0

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def generate_structured_response(model_name, resume_text):
    prompt = f"""Extract resume information as JSON with this structure:
{{
  "candidate_overview": {{
    "name": "Full Name",
    "email": "email@example.com",
    "phone": "+1234567890",
    "education": [
      {{
        "institution": "Institution Name",
        "degree": "Degree Name",
        "details": "Details",
        "date_range": "Date Range"
      }}
    ]
  }},
  "skills": {{
    "frontend": ["HTML", "CSS"],
    "backend": ["Python"],
    "frameworks": ["Django"],
    "databases": ["MySQL"],
    "tools": ["Git"],
    "programming_languages": ["C++"]
  }}
}}

If any information is missing, use empty arrays/strings. Always maintain the structure.

Resume Text:
{resume_text[:3000]}"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED
        }
    }

    try:
        response = requests.post(f"{OLLAMA_API}/api/generate", json=payload)
        response.raise_for_status()
        return json.loads(response.json().get("response", "{}"))
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def process_resume(pdf_path, model_name="llama3.1"):
    extracted_text = extract_text_from_pdf(pdf_path)
    structured_data = generate_structured_response(model_name, extracted_text)
    if structured_data:
        return structured_data
    return None

# path=r"C:\Users\athis\OneDrive\Documents\PSG\1.pdf"
# print(process_resume(path))
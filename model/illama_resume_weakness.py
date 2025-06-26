import fitz  # type: ignore # PyMuPDF
import requests
import json
import re
import os
from datetime import datetime

OLLAMA_API = "http://localhost:11434"
SEED = 42
TEMPERATURE = 0

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF resume"""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def generate_structured_response(model_name, resume_text):
    prompt = f"""Extract resume information and predict suitable IT roles as JSON with this structure:
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
  }},
  "Projects": [
    "Experience point 1",
    "Experience point 2"
  ],
  "predicted_roles": ["Role1", "Role2", "Role3"]
}}

If any information is missing, use empty arrays/strings. For predicted_roles, suggest 3 specific IT job roles based on skills and experience. Maintain the structure strictly.

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

def analyze_weaknesses(data, roles):
    """Analyze weaknesses for predicted roles"""
    weaknesses = {}
    for role in roles[:3]:  # Analyze first role only
        prompt = f"""Identify weaknesses for {role} role considering:
- Technical skill gaps vs job requirements
- Experience depth and relevance
- Certification deficiencies
- Project complexity limitations
- Collaboration/team experience

Resume Data:
- Skills: {json.dumps(data.get("skills", {}))}
- Experience: {json.dumps(data.get("experience", []))}
- Education: {json.dumps(data.get("candidate_overview", {}).get("education", []))}

Return JSON with this exact structure:
{{
  "technical_gaps": ["specific missing skills", ...],
  "experience_gaps": ["specific experience shortcomings", ...]
}}"""

        raw_response_text = ""
        try:
            response = requests.post(
                f"{OLLAMA_API}/api/generate",
                json={
                    "model": "llama3.1",
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.3, "seed": SEED}
                },
            )
            response.raise_for_status()
            response_data = response.json()
            raw_response = response_data.get("response", "{}")
            raw_response_text = response.text

            # Clean JSON response
            cleaned = re.sub(r'[\s\S]*?({.*?})[\s\S]*', r'\1', raw_response, flags=re.DOTALL)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            cleaned = re.sub(r',\s*}', '}', cleaned)

            weak_data = json.loads(cleaned)
            if not all(key in weak_data for key in ["technical_gaps", "experience_gaps"]):
                raise ValueError("Invalid weakness structure")
                
            weaknesses[role] = weak_data

        except Exception as e:
            print(f"Error analyzing {role}: {str(e)}")
            if raw_response_text:
                print(f"Raw response: {raw_response_text[:500]}...")
            weaknesses[role] = {"technical_gaps": [], "experience_gaps": []}

    return weaknesses

def generate_hr_strategies(weaknesses):
    """Generate HR negotiation strategies based on weaknesses"""
    prompt = f"""Create HR strategies considering these weaknesses:
{json.dumps(weaknesses, indent=2)}

Return JSON format with concrete examples:
{{
  "salary_leverage": [
    {{
      "reason": "Missing Java experience",
      "impact": "15% salary adjustment potential"
    }}
  ],
  "benefit_adjustments": [
    {{
      "type": "Training Budget",
      "recommendation": "Offer 2000 INR annual upskilling fund"
    }}
  ]
}}"""

    try:
        response = requests.post(
            f"{OLLAMA_API}/api/generate",
            json={
                "model": "llama3.1",
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.4, "seed": SEED}
            },
        )
        response.raise_for_status()
        return json.loads(response.json().get("response", "{}"))
    except Exception as e:
        print(f"HR strategies error: {e}")
        return {}


def process_weakness_resume(pdf_path):
    print(f"Processing resume: {pdf_path}")
    
    # Create output directories
    os.makedirs("media", exist_ok=True)
    
    text = extract_text_from_pdf(pdf_path)
    structured_data = generate_structured_response("llama3.1", text)
    
    if not structured_data:
        print("Failed to structure resume data")
        return

    weaknesses = analyze_weaknesses(structured_data, structured_data.get("predicted_roles", []))
    hr_tips = generate_hr_strategies(weaknesses)
    
    base_name = re.sub(r'[^a-zA-Z0-9_]', '_', 
                      structured_data.get("candidate_overview", {})
                      .get("name", "unknown_candidate").strip())
    
    try:
        # Save structured data
        with open(f"media/resume.json", "w") as f:
            json.dump({
                "structured_data": structured_data,
                "weakness_analysis": weaknesses,
                "hr_strategies": hr_tips
            }, f, indent=2)
            
        print(f"\nAnalysis saved successfully:")
        print(f"- JSON data: media/{base_name}.json")
        
    except Exception as e:
        print(f"Failed to save results: {str(e)}")

# if __name__ == "__main__":
#     pdf_path = r"C:\Users\athis\OneDrive\Documents\PSG\1.pdf"
    
#     if not os.path.exists(pdf_path):
#         print("Error: File not found")
#         exit()
        
#     process_weakness_resume(pdf_path)
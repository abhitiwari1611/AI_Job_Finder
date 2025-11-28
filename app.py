import os
from io import BytesIO

from flask import Flask, render_template, request
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import requests

# --------------------------
# 1) Load ENV + clients
# --------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")

if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in .env")
if not RAPIDAPI_KEY or not RAPIDAPI_HOST:
    raise ValueError("Please set RAPIDAPI_KEY and RAPIDAPI_HOST in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)


# --------------------------
# 2) Resume text extraction
# --------------------------
def extract_text_from_pdf(file_storage) -> str:
    """
    file_storage: Flask uploaded file object
    """
    pdf_bytes = file_storage.read()
    file_storage.seek(0)

    reader = PdfReader(BytesIO(pdf_bytes))
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)

    full_text = "\n".join(texts)
    return " ".join(full_text.split())


# --------------------------
# 3) JSearch: live jobs
# --------------------------
def search_jobs(query: str, location: str, num_pages: int = 1):
    url = "https://jsearch.p.rapidapi.com/search"

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }

    params = {
        "query": f"{query} in {location}",
        "page": "1",
        "num_pages": str(num_pages),
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


# --------------------------
# 4) GPT: score each job vs resume
# --------------------------
def score_job_against_resume(resume_text: str, job: dict) -> dict:
    """
    GPT ko bolo: is candidate vs is job ka score + message do.
    Output:
      {
        "score": int,
        "analysis_text": "full text with score/reason/message"
      }
    """
    job_title = job.get("job_title", "")
    company = job.get("employer_name", "")
    description = job.get("job_description", "")
    location = job.get("job_city") or job.get("job_country") or ""

    user_prompt = f"""
Candidate resume:

{resume_text}

Job details:
Title: {job_title}
Company: {company}
Location: {location}
Description: {description}

You are a job match AI. Compare the resume and job.

Respond in this exact format (no extra text):

Score: <number between 0 and 100>
Reason: <2-3 lines why this is or isn't a good fit>
Message: <a short LinkedIn DM the candidate can send to a recruiter about this job>
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # change to gpt-4.1 or gpt-3.5-turbo if needed
        messages=[
            {"role": "system", "content": "You are a strict job match evaluator."},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content.strip()

    # Try to extract score from first line
    score = 0
    try:
        first_line = content.splitlines()[0]
        # e.g. "Score: 82"
        digits = "".join(ch for ch in first_line if ch.isdigit())
        if digits:
            score = int(digits)
    except Exception:
        score = 0

    return {
        "score": score,
        "analysis_text": content,
    }


# --------------------------
# 5) Flask route
# --------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    jobs_result = []

    if request.method == "POST":
        uploaded_file = request.files.get("resume")
        target_role = request.form.get("target_role", "").strip()
        location = request.form.get("location", "").strip() or "India"
        max_jobs = int(request.form.get("max_jobs", "10"))

        if not uploaded_file or uploaded_file.filename == "":
            error = "Please upload a PDF resume."
        elif not uploaded_file.filename.lower().endswith(".pdf"):
            error = "Only PDF files are supported."
        elif not target_role:
            error = "Please enter a target role."
        else:
            try:
                resume_text = extract_text_from_pdf(uploaded_file)
                if not resume_text.strip():
                    error = "Could not extract text from this PDF. Try a text-based resume PDF."
                else:
                    # 1) Live jobs fetch
                    query = f"{target_role}"
                    raw_jobs = search_jobs(query=query, location=location, num_pages=1)

                    if not raw_jobs:
                        error = "No jobs found from the API. Try changing role or location."
                    else:
                        # 2) Score each job (limit to first N)
                        scored_jobs = []
                        for job in raw_jobs[:max_jobs]:
                            match = score_job_against_resume(resume_text, job)
                            scored_jobs.append({
                                "title": job.get("job_title", ""),
                                "company": job.get("employer_name", ""),
                                "location": job.get("job_city") or job.get("job_country") or "",
                                "apply_link": job.get("job_apply_link") or job.get("job_google_link") or "",
                                "score": match["score"],
                                "analysis_text": match["analysis_text"],
                            })

                        # 3) sort high-score first
                        scored_jobs.sort(key=lambda x: x["score"], reverse=True)
                        jobs_result = scored_jobs

            except Exception as e:
                error = f"Something went wrong: {e}"

    return render_template("index.html", error=error, jobs=jobs_result)


if __name__ == "__main__":
    app.run(debug=True)

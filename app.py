import os
from io import BytesIO

from flask import Flask, render_template, request
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import time
import requests
from requests.exceptions import ReadTimeout, RequestException

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
# Utility: Reduce token size
# --------------------------
def compress_text(text, max_chars=1500):
    """
    Zyada long text ko truncate karta hai taaki GPT tokens kam use kare
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


# --------------------------
# 3) JSearch: live jobs (with retry + timeout)
# --------------------------
def search_jobs(
    query: str,
    location: str,
    page: int = 1,
    num_pages: int = 1,
    timeout: int = 60,
    max_retries: int = 3,
):
    """
    JSearch se jobs lane ka wrapper.
    - Retry logic
    - Bada timeout
    - Error pe empty list return
    """
    url = "https://jsearch.p.rapidapi.com/search"

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }

    params = {
        "query": f"{query} in {location}",
        "page": str(page),
        "num_pages": str(num_pages),
        
        "date_posted": "week",  # last 7 days
    }

    last_exception = None

    for attempt in range(max_retries):
        try:
            resp = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=timeout,  # 30 -> 60 seconds
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])

        except ReadTimeout as e:
            print(f"[search_jobs] Read timeout (attempt {attempt + 1}/{max_retries})")
            last_exception = e

            if attempt < max_retries - 1:
                
                time.sleep(3)
            else:
                break

        except RequestException as e:
            # Network / RapidAPI related koi aur error
            print(f"[search_jobs] Request failed: {e}")
            last_exception = e
            break

    print("[search_jobs] Returning empty list due to error:", last_exception)
    return []


# --------------------------
# 4) GPT: score ALL jobs vs resume (single API call + rate-limit safe)
# --------------------------
def score_all_jobs_against_resume(resume_text: str, jobs: list) -> list:
    """
    Sab jobs ko ek hi GPT call me score karo.
    Rate limit (429) aane par:
      - Crash nahi karega
      - Har job ke liye score=0 + friendly message dega
    Token usage kam rakhne ke liye:
      - Resume + job description truncate kar rahe hain.
    """
    if not jobs:
        return []

    # ---------- 1) Prompt build ----------
    job_blocks = []
    for i, job in enumerate(jobs, start=1):
        job_blocks.append(f"""
Job {i}:
Title: {job.get("job_title","")}
Company: {job.get("employer_name","")}
Location: {job.get("job_city") or job.get("job_country") or ""}
Description: {compress_text(job.get("job_description",""), 800)}
""")

    all_jobs_text = "\n".join(job_blocks)

    user_prompt = f"""
Candidate Resume:
{compress_text(resume_text, 2000)}

Below are multiple jobs. Evaluate each independently.

For EACH job, return in EXACT format (no extra commentary outside this):

Job <number>
Score: <0-100>
Reason: <2 lines>
Message: <LinkedIn DM>

Now here are the jobs:
{all_jobs_text}
"""

    # ---------- 2) GPT call with retry + rate limit handling ----------
    max_gpt_retries = 2
    last_error = None
    response = None

    for attempt in range(max_gpt_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a strict job match evaluator."},
                    {"role": "user", "content": user_prompt},
                ],
            )
            # success -> loop se bahar
            break

        except Exception as e:
            last_error = e
            error_text = str(e).lower()
            print(f"[score_all_jobs] GPT error on attempt {attempt+1}: {e}")

            # Agar rate limit hai
            if "rate limit" in error_text or "rate_limit_exceeded" in error_text:
                # last attempt hai -> fallback result
                if attempt == max_gpt_retries - 1:
                    print("[score_all_jobs] Rate limit hit repeatedly, returning fallback results.")
                    return [
                        {
                            "score": 0,
                            "analysis_text": (
                                "Rate limit reached while scoring this job. "
                                "Please try running the job search again after some time."
                            ),
                        }
                        for _ in jobs
                    ]
                
                time.sleep(5)
            else:
                
                raise

    if response is None:
        
        print("[score_all_jobs] No response from GPT, returning generic fallback.")
        return [
            {
                "score": 0,
                "analysis_text": "Could not get AI-based scoring due to an internal error.",
            }
            for _ in jobs
        ]

    # ---------- 3) Parse GPT response ----------
    content = response.choices[0].message.content.strip()

    raw_blocks = content.split("Job ")
    results = []

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.splitlines()
        score = 0

        for line in lines:
            if line.lower().startswith("score"):
                digits = "".join(c for c in line if c.isdigit())
                if digits:
                    try:
                        score = int(digits)
                    except ValueError:
                        score = 0
                break

        results.append(
            {
                "score": score,
                "analysis_text": "Job " + block,
            }
        )

    return results


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

       
        try:
            max_jobs = int(request.form.get("max_jobs", "5"))
        except ValueError:
            max_jobs = 5
        
        max_jobs = max(1, min(max_jobs, 3))

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
                    raw_jobs = search_jobs(
                        query=query,
                        location=location,
                        page=1,
                        num_pages=1,  # safe
                    )

                    if not raw_jobs:
                        error = "No jobs found from the API (or the job API timed out). Try changing role or location and try again."
                    else:
                        # Limit jobs
                        limited_jobs = raw_jobs[:max_jobs]

                        # 2) Single GPT call to score ALL jobs
                        ai_results = score_all_jobs_against_resume(resume_text, limited_jobs)

                        # Safety: length mismatch handle
                        if not ai_results or len(ai_results) != len(limited_jobs):
                            # Fallback: mark score 0 but still show jobs
                            scored_jobs = []
                            for job in limited_jobs:
                                scored_jobs.append(
                                    {
                                        "title": job.get("job_title", ""),
                                        "company": job.get("employer_name", ""),
                                        "location": job.get("job_city")
                                        or job.get("job_country")
                                        or "",
                                        "apply_link": job.get("job_apply_link")
                                        or job.get("job_google_link")
                                        or "",
                                        "score": 0,
                                        "analysis_text": "Could not get AI-based scoring for this job due to a parsing issue.",
                                    }
                                )
                        else:
                            # 3) Merge jobs + AI scores
                            scored_jobs = []
                            for job, match in zip(limited_jobs, ai_results):
                                scored_jobs.append(
                                    {
                                        "title": job.get("job_title", ""),
                                        "company": job.get("employer_name", ""),
                                        "location": job.get("job_city")
                                        or job.get("job_country")
                                        or "",
                                        "apply_link": job.get("job_apply_link")
                                        or job.get("job_google_link")
                                        or "",
                                        "score": match.get("score", 0),
                                        "analysis_text": match.get("analysis_text", ""),
                                    }
                                )

                        # 4) sort high-score first
                        scored_jobs.sort(key=lambda x: x["score"], reverse=True)
                        jobs_result = scored_jobs

            except Exception as e:
                error = f"Something went wrong: {e}"

    return render_template("index.html", error=error, jobs=jobs_result)


if __name__ == "__main__":
    app.run(debug=True)

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import pandas as pd
import tempfile
import os
from docx import Document
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import spacy
from spacy.matcher import PhraseMatcher
import re

app = FastAPI(title="ATS Resume + Job Recommendation API")

# ------------------ Setup Templates & Static ------------------
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------ Load Models & Data ------------------
nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
jobs_df = pd.read_csv("data/jobs.csv")  # Ensure this file exists

# ------------------ Utility Functions ------------------
def extract_text_from_pdf(file_path):
    text = ""
    pdf = PdfReader(file_path)
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ------------------ Resume Parsing (Regex + SpaCy) ------------------
def parse_resume(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")

    text = clean_text(text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Initialize section structure
    sections = {
        "education": [],
        "projects": [],
        "internships": [],
        "certifications": [],
        "skills": [],
        "achievements": []
    }
    current_section = None

    # Define resume headings (your format)
    section_keywords = {
        "education": ["education"],
        "projects": ["projects"],
        "internships": ["internships", "experience", "work experience"],
        "certifications": ["certifications", "publications"],
        "skills": ["core skills", "skills", "technical skills"],
        "achievements": ["achievements", "positions of responsibilities"]
    }

    # Match lines to sections
    for line in lines:
        lower_line = line.lower()
        matched_section = None
        for sec, keywords in section_keywords.items():
            if any(k in lower_line for k in keywords):
                matched_section = sec
                break
        if matched_section:
            current_section = matched_section
            continue
        if current_section:
            sections[current_section].append(line)

    # ---------- Regex extraction ----------
    degree_regex = r"(B\.?Tech|M\.?Tech|B\.?Sc|M\.?Sc|MBA|Ph\.?D|Bachelor|Master)"
    year_regex = r"\b(19|20)\d{2}\b"
    certifications_regex = r"(certified|certification|course|diploma|training|publication)"

    education_matches = re.findall(degree_regex, text, re.I)
    year_matches = re.findall(year_regex, text)
    cert_matches = re.findall(certifications_regex, text, re.I)

    # ---------- SpaCy + PhraseMatcher for skills ----------
    skills_list = [
        "python", "java", "sql", "excel", "power bi", "tensorflow", "keras", "pandas",
        "numpy", "scikit-learn", "react", "node.js", "flask", "django", "c++", "git",
        "mongodb", "mysql", "docker", "kubernetes", "nlp", "ai", "ml"
    ]
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skills_list]
    matcher.add("SKILLS", patterns)

    doc = nlp(text)
    matches = matcher(doc)
    spacy_skills = list(set([doc[start:end].text for match_id, start, end in matches]))

    # Combine skills
    sections["skills"] = list(dict.fromkeys(sections.get("skills", []) + spacy_skills))

    # Add regex-based education & certifications
    if education_matches:
        sections["education"].extend(list(set(education_matches)))
    if year_matches:
        sections["education"].extend(list(set(year_matches)))
    if cert_matches:
        sections["certifications"].extend(list(set(cert_matches)))

    # Deduplicate
    for sec in sections:
        sections[sec] = list(dict.fromkeys(sections[sec]))

    return {
        "text": text,
        "education": sections.get("education", []),
        "projects": sections.get("projects", []),
        "internships": sections.get("internships", []),
        "certifications": sections.get("certifications", []),
        "skills": sections.get("skills", []),
        "achievements": sections.get("achievements", [])
    }

# ------------------ Job Recommendation ------------------
def recommend_jobs(resume_text, top_n=5):
    resume_emb = embed_model.encode(resume_text, convert_to_tensor=True)
    job_descs = jobs_df['description'].tolist()
    job_embs = embed_model.encode(job_descs, convert_to_tensor=True)
    cosine_scores = util.cos_sim(resume_emb, job_embs)[0]

    jobs_df['score'] = cosine_scores.cpu().numpy()
    top_jobs = jobs_df.sort_values(by='score', ascending=False).head(top_n)

    top_jobs['score'] = (top_jobs['score'] * 100).round(2)
    return top_jobs[['title', 'description', 'score']].to_dict(orient='records')

# ------------------ API Endpoints ------------------
@app.get("/api/health")
def health_check():
    return {"message": "ATS Resume + Job Recommendation API is running."}

@app.post("/api/upload_resume")
async def api_upload_resume(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        resume_data = parse_resume(tmp_path)
        recommended_jobs = recommend_jobs(resume_data["text"])
        os.remove(tmp_path)

        return JSONResponse({
            "education": resume_data["education"],
            "projects": resume_data["projects"],
            "internships": resume_data["internships"],
            "certifications": resume_data["certifications"],
            "skills": resume_data["skills"],
            "achievements": resume_data["achievements"],
            "recommended_jobs": recommended_jobs
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/analyze_resume")
async def analyze_resume(request: Request, file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    resume_data = parse_resume(tmp_path)
    recommended_jobs = recommend_jobs(resume_data["text"])
    os.remove(tmp_path)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "data": resume_data,
        "jobs": recommended_jobs
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

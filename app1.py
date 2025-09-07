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
import re

app = FastAPI(title="ATS Resume + Job Recommendation API")

# ------------------ Setup Templates & Static ------------------
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------ Load Models & Data ------------------
nlp = spacy.load("en_core_web_sm")  # SpaCy model

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

jobs_df = pd.read_csv("data/jobs.csv")  # Ensure this file exists

# ------------------ Utility Functions ------------------
def extract_text_from_pdf(file_path):
    text = ""
    pdf = PdfReader(file_path)
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ------------------ Resume Parsing with SpaCy ------------------
def parse_resume(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")
    
    text = clean_text(text)
    doc = nlp(text)
    
    # Extract education (look for ORG + EDUCATION keywords)
    education_keywords = ["university", "college", "bachelor", "master", "btech", "mba", "degree"]
    education = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "EDUCATION"] or any(word in ent.text.lower() for word in education_keywords):
            education.append(ent.text)
    
    # Extract experience (look for WORK_OF_ART, DATE, ROLE, ORG)
    experience_keywords = ["internship", "experience", "worked", "role", "project"]
    experience = []
    for sent in doc.sents:
        if any(word in sent.text.lower() for word in experience_keywords):
            experience.append(sent.text.strip())
    
    # Extract skills (from pre-defined skill list)
    skills_keywords = ["python", "java", "sql", "excel", "power bi", "tensorflow", "pandas", 
                       "hadoop", "css", "react", "ml", "ai", "flask", "django", "javascript"]
    skills = [token.text for token in doc if token.text.lower() in skills_keywords]
    
    # Extract projects
    projects = []
    for sent in doc.sents:
        if "project" in sent.text.lower():
            projects.append(sent.text.strip())
    
    # Deduplicate
    education = list(dict.fromkeys(education))
    experience = list(dict.fromkeys(experience))
    skills = list(dict.fromkeys(skills))
    projects = list(dict.fromkeys(projects))
    
    return {
        "text": text,
        "education": education,
        "experience": experience,
        "skills": skills,
        "projects": projects
    }

# ------------------ Job Recommendation ------------------
def recommend_jobs(resume_text, top_n=5):
    resume_emb = embed_model.encode(resume_text, convert_to_tensor=True)
    job_descs = jobs_df['description'].tolist()
    job_embs = embed_model.encode(job_descs, convert_to_tensor=True)
    cosine_scores = util.cos_sim(resume_emb, job_embs)[0]
    
    jobs_df['score'] = cosine_scores.cpu().numpy()
    top_jobs = jobs_df.sort_values(by='score', ascending=False).head(top_n)
    
    return top_jobs[['title', 'description', 'score']].to_dict(orient='records')

# ------------------ API Endpoints ------------------

# JSON Health Check
@app.get("/api/health")
def health_check():
    return {"message": "ATS Resume + Job Recommendation API is running."}

# JSON Resume Upload (API)
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
            "experience": resume_data["experience"],
            "skills": resume_data["skills"],
            "projects": resume_data["projects"],
            "recommended_jobs": recommended_jobs
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# Upload Page (HTML)
@app.get("/", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# Upload Page Alias
@app.get("/upload_resume/", response_class=HTMLResponse)
def upload_page_alias(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# Analyze Resume (Form POST → HTML results)
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

# ------------------ Run Server ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

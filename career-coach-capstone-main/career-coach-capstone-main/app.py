from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import spacy
import pandas as pd
import tempfile
import os
from docx import Document
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="ATS Resume + Job Recommendation API")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load job dataset
jobs_df = pd.read_csv("data/jobs.csv")  # columns: 'title', 'description'

def extract_text_from_pdf(file_path):
    text = ""
    pdf = PdfReader(file_path)
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def parse_resume(file_path):
    ext = os.path.splitext(file_path)[1]
    if ext.lower() == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext.lower() == ".docx":
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")
    doc = nlp(text)
    # Example: extract simple info (you can extend this)
    skills = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT", "SKILL")]
    return {"text": text, "skills": skills}

def recommend_jobs(resume_text, top_n=5):
    # Combine resume with job descriptions
    corpus = jobs_df['description'].tolist()
    corpus.append(resume_text)
    
    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Cosine similarity between resume and jobs
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    jobs_df['score'] = similarity_scores
    top_jobs = jobs_df.sort_values(by='score', ascending=False).head(top_n)
    
    return top_jobs[['title', 'description', 'score']].to_dict(orient='records')

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        # Parse resume
        resume_data = parse_resume(tmp_path)
        
        # Recommend jobs
        recommended_jobs = recommend_jobs(resume_data["text"])
        
        # Cleanup temp file
        os.remove(tmp_path)
        
        return JSONResponse({"resume_skills": resume_data["skills"], "recommended_jobs": recommended_jobs})
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/")
def home():
    return {"message": "ATS Resume + Job Recommendation API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
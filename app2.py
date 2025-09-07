import os
import re
from flask import Flask, request, render_template
import pdfplumber
from werkzeug.utils import secure_filename

# ---------------------------
# App Initialization
# ---------------------------
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ---------------------------
# Job Role Keywords
# ---------------------------
JOB_KEYWORDS = {
    "Frontend Developer": ["HTML", "CSS", "JavaScript", "React", "Tailwind", "Node.js"],
    "Backend Developer": ["Python", "Flask", "Django", "MySQL", "PostgreSQL", "MongoDB"],
    "Data Analyst": ["Python", "SQL", "Excel", "Pandas", "Tableau", "Power BI"],
    "Machine Learning Engineer": ["Python", "TensorFlow", "PyTorch", "Scikit-learn", "ML", "Data Science"]
}

# ---------------------------
# Helper Functions
# ---------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def parse_resume(text):
    """
    Parse resume text based on headings and optional DOB detection.
    """
    data = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    data['Name'] = lines[0] if lines else "N/A"

    # DOB detection: looks for 'Date of Birth' or variations
    dob_match = re.search(r'Date of Birth[:\s]*([0-9]{2}/[0-9]{2}/[0-9]{4})', text, re.IGNORECASE)
    data['DOB'] = dob_match.group(1) if dob_match else "Not Found"

    # Headings regex patterns (case insensitive, cover variations)
    headings = {
        'Education': r'(Education.*?)(Projects|Internships|Core Skills|Certifications|Achievements|$)',
        'Projects': r'(Projects.*?)(Internships|Core Skills|Certifications|Achievements|$)',
        'Internships': r'(Internships.*?)(Core Skills|Certifications|Achievements|$)',
        'Core Skills': r'(Core Skills.*?)(Certifications|Achievements|$)',
        'Achievements': r'(Achievements.*|Achievements and Positions of Responsibility.*)',
        'Certificates': r'(Certifications.*|Certifications and Publications.*)'
    }

    for key, pattern in headings.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        data[key] = match.group(1).strip() if match else "Not Found"

    return data

# ---------------------------
# ATS Scoring
# ---------------------------
def generate_ats_score(resume_data, text, keywords):
    score = 0
    feedback = []

    # Check essential sections
    sections = {
        'Education': 20,
        'Projects': 20,
        'Internships': 15,
        'Core Skills': 20,
        'Achievements': 10,
        'Certificates': 5
    }

    for sec, points in sections.items():
        content = resume_data.get(sec) or ""
        if content and content.strip() != "Not Found":
            score += points
        else:
            feedback.append(f"{sec} section is missing or incomplete.")

    # Job role keyword matching
    text_lower = text.lower()
    keyword_hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    score += min(keyword_hits * 2, 10)  # max 10 points bonus

    if keyword_hits == 0:
        feedback.append("No relevant keywords found for the selected job role.")

    return min(score, 100), feedback

# ---------------------------
# Routes
# ---------------------------
@app.route('/', methods=['GET'])
def home():
    return render_template('upload.html')

@app.route('/resume_view', methods=['POST'])
def resume_view():
    if 'file' not in request.files:
        return "No file part in request"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        text = extract_text_from_pdf(filepath)
        resume_data = parse_resume(text)
        return render_template('resume_view.html', data=resume_data)
    else:
        return "File type not allowed. Please upload a PDF."

@app.route('/ats_score', methods=['POST'])
def ats_score():
    if 'file' not in request.files:
        return "No file part in request"
    file = request.files['file']
    job_role = request.form.get('job_role')
    if file.filename == '':
        return "No selected file"
    if not job_role or job_role not in JOB_KEYWORDS:
        return "Please select a valid job role"
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        text = extract_text_from_pdf(filepath)
        resume_data = parse_resume(text)
        score, feedback = generate_ats_score(resume_data, text, JOB_KEYWORDS[job_role])
        return render_template('ats_score.html', data=resume_data, score=score, feedback=feedback, job_role=job_role)
    else:
        return "File type not allowed. Please upload a PDF."

# ---------------------------
# Run the App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)

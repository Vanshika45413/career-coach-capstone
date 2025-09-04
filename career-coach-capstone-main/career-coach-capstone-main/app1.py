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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------
# PDF Parsing Functions
# ---------------------------
def extract_text_from_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_dob(text):
    dob = "Not Found"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if re.search(r'(Date of Birth|DOB)', line, re.IGNORECASE):
            # Check same line
            match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{2,4})', line)
            if match:
                dob = match.group(1)
                break
            # Check next line
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                match_next = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{2,4})', next_line)
                if match_next:
                    dob = match_next.group(1)
                    break
    return dob

def parse_resume(text):
    data = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    data['Name'] = lines[0] if lines else "N/A"
    data['Date of Birth'] = extract_dob(text)

    # Headings to detect
    headings = {
        'Education': r'Education',
        'Projects': r'Projects',
        'Internships': r'Internships',
        'Certifications': r'(Certifications|Certifications and Publications)',
        'Core Skills': r'(Core Skills|CORE SKILLS)',
        'Achievements': r'(Achievements|Achievements and Positions of Responsibility)'
    }

    # Initialize
    content = {k: "" for k in headings.keys()}
    current_heading = None

    for line in lines[1:]:  # skip name
        # Detect heading
        matched = False
        for key, pattern in headings.items():
            if re.match(pattern, line, re.IGNORECASE):
                current_heading = key
                matched = True
                break
        if not matched and current_heading:
            content[current_heading] += line + "\n"

    # Assign to data
    for key in headings.keys():
        data[key] = content[key].strip() if content[key].strip() else "Not Found"

    # Optional fields
    if 'Certifications' not in data:
        data['Certifications'] = "Not Found"

    return data

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
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

# ---------------------------
# Run the App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)

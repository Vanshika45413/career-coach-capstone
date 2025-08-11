from flask import Flask, render_template, request
import pdfplumber
import os
import re

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def extract_resume_text(pdf_path):
    """Extracts all text from a PDF file using pdfplumber."""
    text_content = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
    return "\n".join(text_content)

def parse_resume(text):
    """Parses key sections from the resume text."""
    sections = {
        "Name": "",
        "DOB": "",
        "Education": "",
        "Projects": "",
        "Internships": "",
        "Core Skills": "",
        "Achievements": ""
    }

    # Clean up text: remove extra spaces and normalize
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n", text)  # remove extra blank lines

    # Split into lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Improved Name Extraction
    if lines:
        # Skip generic titles like "Resume" or "Curriculum Vitae"
        first_line = lines[0]
        if re.match(r"(resume|curriculum vitae)", first_line, re.IGNORECASE):
            if len(lines) > 1:
                sections["Name"] = lines[1]
        else:
            sections["Name"] = first_line

    # Extract DOB
    dob_match = re.search(r"(Date of Birth|DOB)[:\-]?\s*(.*)", text, re.IGNORECASE)
    if dob_match:
        sections["DOB"] = dob_match.group(2).strip()

    # More flexible headings
    headers = {
        "Education": r"education[\s:_-]*",
        "Projects": r"projects[\s:_-]*",
        "Internships": r"internships[\s:_-]*",
        "Core Skills": r"core\s*skills[\s:_-]*",
        "Achievements": r"achievements(?:\s+and\s+positions\s+of\s+responsibility)?[\s:_-]*"
    }

    # Build combined pattern for all headers
    all_headers_pattern = r"(Education|Projects|Internships|Core Skills|Achievements(?:\s+and\s+Positions\s+of\s+Responsibility)?)"

    for key, header in headers.items():
        # Find text from this header until the next header or end of text
        pattern = rf"{header}(.*?)(?={all_headers_pattern}|$)"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.S)
        if match:
            section_text = match.group(1).strip()
            sections[key] = section_text

    return sections

@app.route("/", methods=["GET", "POST"])
def upload_resume():
    if request.method == "POST":
        if "resume" not in request.files:
            return "No file part"
        file = request.files["resume"]
        if file.filename == "":
            return "No selected file"
        if file and file.filename.lower().endswith(".pdf"):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            text_data = extract_resume_text(file_path)
            parsed_data = parse_resume(text_data)

            return render_template("resume_view.html", data=parsed_data)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)

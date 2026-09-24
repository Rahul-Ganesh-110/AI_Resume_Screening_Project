# AI-Based Resume Screening & Job Fit Analyzer

## Project Type
AI/ML Minor Project — Natural Language Processing

## Objective
Build a Python application that compares a candidate resume with a job description and estimates how closely the resume matches the role.

## Main Techniques
- Text preprocessing
- TF-IDF (Term Frequency–Inverse Document Frequency)
- Cosine similarity
- Skill extraction using pattern matching
- Feature-based score combination
- Streamlit web application

## Architecture

Resume PDF/DOCX/TXT
        |
        v
Text Extraction
        |
        v
Text Cleaning & Normalization
        |
        +----------------------+
        |                      |
        v                      v
TF-IDF Vectorization     Skill Extraction
        |                      |
        v                      v
Cosine Similarity        Skill Overlap
        |                      |
        +----------+-----------+
                   |
                   v
             Final Score
                   |
                   v
         Match + Missing Skills

## Run Locally

### 1. Create environment
```bash
python -m venv .venv
```

### 2. Activate on Windows
```bash
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
streamlit run app.py
```

The browser will open the application.

## Suggested Demo
Use `sample_resume.txt` and `sample_job.txt`, then upload the resume and paste the job description.

## Important Academic Note
The project is a prototype for educational purposes. A real recruitment system would require larger datasets, validated models, bias testing, privacy controls, and human review.

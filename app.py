import re
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

SKILLS = [
    "python", "java", "c++", "c", "javascript", "typescript", "html", "css",
    "react", "node.js", "node", "express", "mongodb", "mysql", "sql",
    "postgresql", "git", "github", "docker", "aws", "azure", "gcp",
    "machine learning", "deep learning", "artificial intelligence", "ai",
    "data science", "data analysis", "pandas", "numpy", "scikit-learn",
    "tensorflow", "keras", "pytorch", "opencv", "nlp", "natural language processing",
    "computer vision", "streamlit", "flask", "django", "fastapi",
    "rest api", "api", "linux", "cybersecurity", "networking", "computer networks",
    "operating systems", "dsa", "data structures", "algorithms",
    "communication", "problem solving", "excel", "power bi", "tableau"
]

SKILL_ALIASES = {
    "scikit learn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "machine-learning": "machine learning",
    "deep-learning": "deep learning",
    "natural-language-processing": "natural language processing",
    "computer-vision": "computer vision",
    "node js": "node.js",
    "restful api": "rest api",
    "rest apis": "rest api",
}

STOPWORDS = {
    "the","and","for","with","that","this","from","have","has","are","was",
    "will","your","you","our","their","into","using","used","such","about",
    "should","would","could","can","job","role","work","years","year","skills",
    "experience","candidate","team","looking","required","requirements"
}

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\-/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    for old, new in SKILL_ALIASES.items():
        text = text.replace(old, new)
    return text.strip()

def extract_pdf(file_bytes):
    if PdfReader is None:
        return ""
    reader = PdfReader(BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_docx(file_bytes):
    if Document is None:
        return ""
    doc = Document(BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def extract_uploaded_text(uploaded_file):
    if uploaded_file is None:
        return ""
    data = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_pdf(data)
    if name.endswith(".docx"):
        return extract_docx(data)
    return data.decode("utf-8", errors="ignore")

def extract_skills(text):
    t = normalize_text(text)
    found = []
    for skill in SKILLS:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, t):
            found.append(skill)
    return sorted(set(found))

def important_terms(text, top_n=12):
    t = normalize_text(text)
    words = re.findall(r"[a-z][a-z0-9+#.-]{2,}", t)
    words = [w for w in words if w not in STOPWORDS]
    counts = pd.Series(words).value_counts()
    return list(counts.head(top_n).index)

def calculate_match(resume, job_description):
    corpus = [normalize_text(resume), normalize_text(job_description)]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )
    matrix = vectorizer.fit_transform(corpus)
    score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100)

    resume_skills = set(extract_skills(resume))
    job_skills = set(extract_skills(job_description))
    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)

    if job_skills:
        skill_score = len(matched) / len(job_skills) * 100
        final_score = 0.65 * score + 0.35 * skill_score
    else:
        skill_score = 0
        final_score = score

    return {
        "text_similarity": round(score, 2),
        "skill_match": round(skill_score, 2),
        "final_score": round(min(final_score, 100), 2),
        "matched": matched,
        "missing": missing,
    }

def recommendation(score):
    if score >= 80:
        return "Strong Match", "The resume has high textual similarity and/or strong overlap with the required skills."
    if score >= 60:
        return "Good Match", "The candidate matches a substantial portion of the job requirements."
    if score >= 40:
        return "Partial Match", "There is some relevant overlap, but several requirements may be missing."
    return "Low Match", "The resume has limited overlap with the supplied job description."

st.title("📄 AI-Based Resume Screening & Job Fit Analyzer")
st.caption("NLP + TF-IDF + Cosine Similarity + Skill Extraction")

with st.sidebar:
    st.header("About the Project")
    st.write(
        "This system compares a candidate resume with a job description using "
        "TF-IDF vectorization and cosine similarity, then combines the similarity "
        "score with skill overlap."
    )
    st.markdown("**Technologies**")
    st.write("Python • Scikit-learn • Pandas • Streamlit • NLP")
    st.divider()
    st.info("For academic demonstration only. It should not be used as the sole basis for real hiring decisions.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Candidate Resume")
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"]
    )
    resume_text = st.text_area(
        "Or paste resume text",
        height=300,
        placeholder="Paste the candidate's resume here..."
    )

with col2:
    st.subheader("2. Job Description")
    job_text = st.text_area(
        "Paste the job description",
        height=300,
        placeholder="Example: We are looking for a Python developer with SQL, Pandas, Scikit-learn, REST API and Git experience..."
    )

sample_job = """We are looking for an entry-level AI/ML Python developer.
Required skills: Python, Machine Learning, Pandas, NumPy, Scikit-learn, SQL,
Git/GitHub, NLP and problem solving. Knowledge of Streamlit and REST APIs is a plus."""

if st.button("Load Sample Job Description"):
    st.session_state["job_sample"] = sample_job

if "job_sample" in st.session_state and not job_text:
    job_text = st.session_state["job_sample"]

if uploaded:
    extracted = extract_uploaded_text(uploaded)
    if extracted.strip():
        resume_text = extracted

if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
    if not resume_text.strip():
        st.error("Please upload a resume or paste resume text.")
        st.stop()
    if not job_text.strip():
        st.error("Please enter a job description.")
        st.stop()

    result = calculate_match(resume_text, job_text)
    label, explanation = recommendation(result["final_score"])

    st.divider()
    st.subheader("📊 Screening Result")

    a, b, c = st.columns(3)
    a.metric("Final Match Score", f'{result["final_score"]}%')
    b.metric("Text Similarity", f'{result["text_similarity"]}%')
    c.metric("Skill Match", f'{result["skill_match"]}%')

    st.progress(result["final_score"] / 100)
    st.success(f"**{label}** — {explanation}")

    left, right = st.columns(2)
    with left:
        st.markdown("### ✅ Matched Skills")
        if result["matched"]:
            for skill in result["matched"]:
                st.write(f"• {skill}")
        else:
            st.write("No matching skills detected.")

    with right:
        st.markdown("### ⚠️ Skills to Improve")
        if result["missing"]:
            for skill in result["missing"]:
                st.write(f"• {skill}")
        else:
            st.write("No missing skills detected from the configured skill list.")

    st.markdown("### 🔑 Important Terms in Resume")
    terms = important_terms(resume_text)
    if terms:
        st.write(", ".join(terms))
    else:
        st.write("No terms extracted.")

    st.markdown("### 🧠 How the score is calculated")
    st.code(
        "TF-IDF vectors → Cosine Similarity\n"
        "                 ↓\n"
        "Skill overlap → Skill Match Score\n"
        "                 ↓\n"
        "Final = 65% Text Similarity + 35% Skill Match",
        language="text"
    )

st.divider()
st.caption("Academic minor project — NLP-based resume screening prototype")

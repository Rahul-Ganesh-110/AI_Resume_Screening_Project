# AI-Based Resume Screening and Job Fit Analyzer

## 1. Abstract
Recruitment teams often receive a large number of resumes for a single job opening. Manually comparing every resume with a job description can be time-consuming. This project presents an NLP-based resume screening prototype that compares resume text with a job description and produces a job-fit score. The system uses TF-IDF vectorization and cosine similarity to measure textual similarity and a rule-based skill extraction component to identify overlapping and missing skills. A Streamlit interface allows the system to be demonstrated as a web application.

## 2. Problem Statement
Manual resume screening is repetitive and can make it difficult to consistently identify relevant skills from large amounts of text. The project aims to create an automated academic prototype that assists in comparing resumes with job descriptions.

## 3. Objectives
1. Extract text from PDF, DOCX and TXT resumes.
2. Preprocess and normalize resume and job-description text.
3. Convert text into numerical TF-IDF vectors.
4. Calculate cosine similarity between the resume and job description.
5. Extract technical skills from both documents.
6. Calculate skill overlap.
7. Combine textual similarity and skill overlap into a final match score.
8. Provide a simple interactive Streamlit interface.

## 4. Technologies
- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- pypdf
- python-docx

## 5. Methodology

### 5.1 Text Extraction
The system reads TXT directly and extracts text from PDF and DOCX files.

### 5.2 Preprocessing
Text is converted to lowercase, whitespace is normalized, punctuation noise is reduced, and common irrelevant words are removed by the TF-IDF vectorizer.

### 5.3 TF-IDF
TF-IDF assigns importance to words based on how frequently they occur in a document and how uncommon they are across documents.

### 5.4 Cosine Similarity
The resume and job description are represented as vectors. Cosine similarity measures the angle between these vectors.

Formula:

cos(theta) = (A · B) / (||A|| ||B||)

A value closer to 1 indicates greater textual similarity.

### 5.5 Skill Matching
A predefined technical-skill dictionary is searched in both texts. Skills present in both documents are treated as matched skills.

Skill Match = Matched Required Skills / Total Required Skills × 100

### 5.6 Final Score
The prototype uses:

Final Score = 0.65 × Text Similarity + 0.35 × Skill Match

This weighting is an application-design choice for the academic prototype, not a validated hiring metric.

## 6. System Architecture
Input Resume + Job Description
→ Text Extraction
→ Preprocessing
→ TF-IDF
→ Cosine Similarity
→ Skill Extraction
→ Skill Overlap
→ Score Combination
→ Result Dashboard

## 7. Expected Output
The application displays:
- Final match percentage
- Text similarity
- Skill match percentage
- Matched skills
- Missing skills
- Important resume terms
- Match category

## 8. Advantages
- Simple and explainable
- Fast execution
- Works without an external AI API
- Supports PDF/DOCX/TXT input
- Easy to demonstrate
- Uses standard NLP/ML techniques

## 9. Limitations
- Skill dictionary is predefined.
- Semantic meaning is not fully understood.
- Synonyms and context may be missed.
- The score is not a validated employment decision metric.
- Real recruitment systems require larger datasets and fairness/privacy evaluation.

## 10. Future Scope
- Use sentence-transformer embeddings.
- Add a trained resume-category classifier.
- Add semantic skill matching.
- Support multiple resumes at once.
- Add a database.
- Add explainable ranking features.
- Add multilingual resume support.
- Evaluate on a labeled benchmark dataset.

## 11. Conclusion
The project demonstrates how NLP and machine-learning techniques can be combined to build a practical resume screening prototype. TF-IDF provides an interpretable representation of text, while cosine similarity estimates textual overlap. Skill extraction adds a domain-specific component, and Streamlit turns the model into an interactive application.

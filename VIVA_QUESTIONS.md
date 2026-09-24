# Viva Questions and Short Answers

## 1. What is the project?
It is an NLP-based resume screening prototype that compares a resume with a job description and calculates a job-fit score.

## 2. Why did you use NLP?
Resumes and job descriptions are mostly unstructured natural language text. NLP converts that text into features that a computer can analyze.

## 3. What is TF-IDF?
TF-IDF stands for Term Frequency–Inverse Document Frequency. It gives higher importance to terms that are frequent in a document but relatively uncommon across the collection.

## 4. What is cosine similarity?
It measures similarity between two vectors using the cosine of the angle between them.

## 5. Why cosine similarity?
It is simple, fast and works well for comparing sparse text vectors such as TF-IDF representations.

## 6. Is this supervised or unsupervised machine learning?
The core similarity component is an unsupervised/text-matching approach because it does not require labeled training examples.

## 7. What is the role of skill extraction?
It identifies important technical skills and calculates how many job-required skills also appear in the resume.

## 8. How is the final score calculated?
The prototype combines 65% textual similarity and 35% skill overlap.

## 9. Why Streamlit?
Streamlit allows a Python ML project to be converted into an interactive web application quickly.

## 10. What are the limitations?
It depends on a predefined skill dictionary and does not fully understand semantic meaning, context, or candidate quality.

## 11. How could you improve it?
Use transformer embeddings such as sentence-transformers, a labeled dataset, semantic skill matching, and proper model evaluation.

## 12. What evaluation metrics could be used for a trained classifier?
Accuracy, precision, recall, F1-score and confusion matrix.

## 13. What is overfitting?
Overfitting occurs when a model learns training data too closely and performs poorly on unseen data.

## 14. What is feature extraction?
It is the process of converting raw information such as text into numerical representations that an algorithm can process.

## 15. Is the final score a real hiring decision?
No. It is an educational prototype and should only assist human review.

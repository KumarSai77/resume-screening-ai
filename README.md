# 📄 Resume Screening AI

An AI-powered Resume Screening System that matches and ranks resumes based on job descriptions using NLP techniques and semantic similarity.

---

## 🚀 Features

- 📌 Upload multiple resumes (PDF)
- 🧠 Semantic matching using BERT (local) / TF-IDF (deployment fallback)
- 📊 Resume ranking based on similarity scores
- 🧾 Skill gap analysis (identifies missing skills)
- 🌐 Interactive web app using Streamlit

---

## 🛠️ Tech Stack

- Python
- NLP (TF-IDF, BERT - Sentence Transformers)
- Scikit-learn
- Streamlit
- PyPDF2
- Pandas

---

## ⚙️ How It Works

1. Extracts text from resumes (PDF)
2. Converts job description and resumes into vectors:
   - BERT (semantic embeddings) locally
   - TF-IDF (fallback for deployment)
3. Computes similarity using cosine similarity
4. Ranks resumes based on match score
5. Identifies missing skills from predefined skill set

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/resume-screening-ai.git
cd resume-screening-ai
pip install -r requirements.txt
streamlit run app.py

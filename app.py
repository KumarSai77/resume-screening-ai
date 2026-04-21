import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')

SKILLS_DB = [
    "python", "java", "c++", "machine learning", "deep learning",
    "nlp", "data analysis", "pandas", "numpy", "tensorflow",
    "pytorch", "sql", "aws", "docker", "kubernetes",
    "excel", "power bi", "tableau", "git", "linux"
]
# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def extract_skills(text):
    text = text.lower()
    found_skills = []

    for skill in SKILLS_DB:
        if skill in text:
            found_skills.append(skill)

    return found_skills        


# Rank resumes
def rank_resumes(job_description, resumes, file_names):
    documents = [job_description] + resumes

    embeddings = model.encode(documents)
    
    job_embeddings = embeddings[0]
    resume_embeddings = embeddings[1:]

    #compute similaritu
    similarity = cosine_similarity([job_embeddings], resume_embeddings)[0]


    ranked = sorted(zip(file_names, similarity), key=lambda x: x[1], reverse=True)

    return ranked

# Streamlit UI
st.title("📄 Resume Screening AI")

job_description = st.text_area("📌 Paste Job Description")

uploaded_files = st.file_uploader(
    "📂 Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True
)

if st.button("🚀 Rank Resumes"):
    if not job_description or not uploaded_files:
        st.warning("Please provide both Job Description and Resumes")
    else:
        resumes = []
        file_names = []

        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes.append(text)
            file_names.append(file.name)

        ranked_resumes = rank_resumes(job_description, resumes, file_names)

        st.subheader("📊 Ranking Results")

        # ✅ Extract job skills
        job_skills = extract_skills(job_description)

        results = []

        for file, score in ranked_resumes:
            resume_index = file_names.index(file)
            resume_text = resumes[resume_index]

            resume_skills = extract_skills(resume_text)

            missing_skills = list(set(job_skills) - set(resume_skills))

            results.append({
                "Resume": file,
                "Score": round(score, 4),
                "Missing Skills": ", ".join(missing_skills) if missing_skills else "None"
            })

        df = pd.DataFrame(results)

        # ✅ Show table
        st.dataframe(df)

        # ✅ Show chart
        st.bar_chart(df.set_index("Resume")[["Score"]])



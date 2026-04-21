import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ✅ Try loading BERT (will fail on Streamlit Cloud)
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    USE_BERT = True
except:
    USE_BERT = False

# ✅ Skills database
SKILLS_DB = [
    "python", "java", "c++", "machine learning", "deep learning",
    "nlp", "data analysis", "pandas", "numpy", "tensorflow",
    "pytorch", "sql", "aws", "docker", "kubernetes",
    "excel", "power bi", "tableau", "git", "linux"
]

# ✅ Extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ✅ Extract skills
def extract_skills(text):
    text = text.lower()
    return [skill for skill in SKILLS_DB if skill in text]

# ✅ Ranking function (BERT + fallback)
def rank_resumes(job_description, resumes, file_names):
    documents = [job_description] + resumes

    if USE_BERT:
        embeddings = model.encode(documents)
        job_embedding = embeddings[0]
        resume_embeddings = embeddings[1:]

        similarity = cosine_similarity([job_embedding], resume_embeddings)[0]
    else:
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    ranked = sorted(zip(file_names, similarity), key=lambda x: x[1], reverse=True)
    return ranked

# ✅ UI
st.title("📄 Resume Screening AI")

# Show mode
if USE_BERT:
    st.success("Using BERT (Semantic Matching)")
else:
    st.warning("Using TF-IDF (Deployment Mode)")

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

        job_skills = extract_skills(job_description)

        results = []

        for file, score in ranked_resumes:
            idx = file_names.index(file)
            resume_text = resumes[idx]

            resume_skills = extract_skills(resume_text)
            missing_skills = list(set(job_skills) - set(resume_skills))

            results.append({
                "Resume": file,
                "Score": round(score, 4),
                "Missing Skills": ", ".join(missing_skills) if missing_skills else "None"
            })

        df = pd.DataFrame(results)

        st.dataframe(df)
        st.bar_chart(df.set_index("Resume")[["Score"]])
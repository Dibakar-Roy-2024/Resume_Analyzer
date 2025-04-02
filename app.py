import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time  # For loading animation

# Load trained model & vectorizer
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 🎨 Custom Styling
st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Features")
st.sidebar.markdown("✔ AI-Powered Resume Analysis")
st.sidebar.markdown("✔ Predicts Job Category")
st.sidebar.markdown("✔ Supports Text & File Upload")
st.sidebar.markdown("✔ Provides Downloadable Report")
st.sidebar.markdown("✔ User-Friendly & Fast")

# 🎯 Main Page
st.title("📄 AI-Powered Resume Analyzer")
st.markdown("🔍 **Upload or Paste your resume to get instant job category predictions!**")

# 📝 File Upload Option
uploaded_file = st.file_uploader("📤 Upload Resume (TXT, PDF)", type=["txt", "pdf"])
resume_text = ""

if uploaded_file:
    try:
        import PyPDF2  # For PDF reading

        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    except:
        resume_text = uploaded_file.read().decode("utf-8")  # For TXT files

# 📌 Text Area for Manual Input
else:
    resume_text = st.text_area("📄 Or Paste Your Resume Here!")

# 🔍 Analyze Button
if st.button("🔎 Analyse Resume"):
    if resume_text.strip():
        with st.spinner("🔄 Analyzing Resume... Please Wait..."):
            time.sleep(2)  # Simulating processing delay
            resume_vector = vectorizer.transform([resume_text]).toarray()
            category = model.predict(resume_vector)[0]

        st.success(f"**Predicted Job Category:** 🎯 {category}")

        # 📥 Download Report
        st.download_button("📥 Download Report", resume_text, file_name="Resume_Analysis.txt")

    else:
        st.warning("⚠ Please provide resume text for analysis!")

# 🔗 Footer
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Created by Dibakar Roy**")
st.sidebar.markdown("📧 Contact: [Your Email](mailto:youremail@example.com)")

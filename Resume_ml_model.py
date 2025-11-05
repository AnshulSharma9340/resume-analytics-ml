# ---------------- Resume_ml_model.py ----------------
import streamlit as st
import joblib

# Must be first Streamlit command
st.set_page_config(page_title="Veridia Resume Analytics", layout="wide")

# ---------------------------------------------------------
# 1️⃣ Load models (cached)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    pipe_ready = joblib.load("job_readiness_pipeline.pkl")
    pipe_cat = joblib.load("candidate_category_pipeline.pkl")
    return pipe_ready, pipe_cat

pipe_ready, pipe_cat = load_models()
st.sidebar.success("✅ Models loaded successfully!")

# ---------------------------------------------------------
# 2️⃣ Page UI
# ---------------------------------------------------------
st.title("📊 Veridia Resume Intelligence System")
st.write("Analyze resumes using ML models to predict **Job Readiness** and **Category**.")

resume_text = st.text_area("📝 Paste Resume Text Here", height=300, placeholder="Paste your resume text...")

if st.button("🔍 Analyze Resume"):
    if resume_text.strip() == "":
        st.warning("Please paste some resume text first.")
    else:
        try:
            # Make predictions
            job_pred = pipe_ready.predict([resume_text])[0]
            cat_pred = pipe_cat.predict([resume_text])[0]

            # Display results
            st.subheader("🎯 Prediction Results")
            st.markdown(f"**Job Readiness:** {'✅ Ready' if job_pred == 1 else '❌ Not Ready'}")
            st.markdown(f"**Predicted Category:** `{cat_pred}`")

        except Exception as e:
            st.error(f"Error making prediction: {e}")

# ---------------------------------------------------------
# 3️⃣ Sidebar info
# ---------------------------------------------------------
st.sidebar.header("About the Project")
st.sidebar.markdown("""
**Veridia Resume Analytics**
- Predicts candidate job readiness
- Classifies resumes into professional categories
- Built using TF-IDF + Logistic Regression
""")

st.sidebar.markdown("---")
st.sidebar.info("Developed by **Anshul** 🚀")
#resume Project for Veredia Internship

# ---------------- train_resume_models.py ----------------
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------
# 1️⃣ Load dataset
# ---------------------------------------------------------
df = pd.read_csv("resume_dataset.csv")  # change to your dataset filename

# Inspect columns
print("Columns in dataset:", df.columns)

# Expected columns:
# 'Resume_str'  - cleaned resume text
# 'Job Readiness' - binary or numeric label
# 'Category' - multi-class label (e.g. 'Data Science', 'HR', etc.)

# ---------------------------------------------------------
# 2️⃣ Prepare data
# ---------------------------------------------------------
X = df["Resume_str"]

# Replace with your column names if different
y_ready = df["Job Readiness"]
y_cat = df["Category"]

# Split into training and test (optional, just to validate)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_ready, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 3️⃣ Build pipelines
# ---------------------------------------------------------
pipe_ready = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words='english', max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe_cat = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words='english', max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# ---------------------------------------------------------
# 4️⃣ Train models
# ---------------------------------------------------------
print("Training Job Readiness model...")
pipe_ready.fit(X_train_r, y_train_r)

print("Training Candidate Category model...")
pipe_cat.fit(X_train_c, y_train_c)

# ---------------------------------------------------------
# 5️⃣ Save fitted models
# ---------------------------------------------------------
joblib.dump(pipe_ready, "job_readiness_pipeline.pkl")
joblib.dump(pipe_cat, "candidate_category_pipeline.pkl")

print("✅ Trained and saved fitted models successfully!")

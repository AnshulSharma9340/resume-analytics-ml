📊 Veridia Resume Intelligence System

🌟 Project Overview

Veridia Resume Intelligence System is an AI-powered recruitment assistant built with Streamlit. It intelligently parses resumes, predicts job readiness scores, categorizes candidates into skill domains, and visualizes data in a clean interactive dashboard.

💡 Empowering recruiters with data-driven insights to save time, prioritize candidates, and make smarter hiring decisions.

✨ Key Features
Feature	Description
✅ Job Readiness Score	Quantitative prediction of a candidate's fit for generic roles. Prioritize screening efficiently.
🧠 Candidate Category Prediction	Classifies candidates into skill domains like Data Science, HR, Web Development, etc.
⬆️ Flexible Data Ingestion	Supports PDF/DOCX uploads and text pasting.
📊 Interactive Dashboard	Streamlit-powered interactive dashboard for analytics & visualization.
🤖 Robust ML Pipelines	Pre-trained ML models for classification & regression ensure accurate predictions.
⚡ Easy Deployment	Lightweight Python system, easily deployable locally or on the cloud.
🛠️ Tech Stack

Core: Python

Web App: Streamlit

ML Models: scikit-learn pipelines

Data Processing: pandas & numpy

Resume Parsing: python-docx, PyPDF2

Visualization: plotly, seaborn, matplotlib

🚀 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/AnshulSharma9340/resume-analytics-ml.git
cd resume-analytics-ml

2️⃣ Create a Virtual Environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt


⚠️ If requirements.txt is missing, include these:

streamlit
scikit-learn
pandas
numpy
python-docx
PyPDF2
plotly
matplotlib
seaborn

4️⃣ Run the Application
streamlit run app.py


Opens automatically at: http://localhost:8501

🧩 How It Works

Upload Resume – Upload PDF/DOCX or paste plain text.

Predict Readiness & Category – ML models evaluate candidate.

Visualize Insights – View interactive charts and metrics in the dashboard.

Example:


🤝 Contribution

Contributions are welcome! You can:

Fix bugs & issues

Add new ML models or features

Improve documentation & visuals

Optimize code & performance

Steps to contribute:

# Fork repository
git checkout -b feature/awesome-feature
# Make changes
git commit -m "Add new feature"
git push origin feature/awesome-feature
# Open a Pull Request

📧 Contact & Support

Author: Anshul Sharma

LinkedIn

Email: anshulsharma7162@gmail.com

GitHub Project: Resume Analytics ML

📜 License

This project is licensed under the MIT License – see LICENSE
 for details.


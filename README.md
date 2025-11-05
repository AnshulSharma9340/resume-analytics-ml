📊 Veridia Resume Intelligence System

🌟 Overview

The Veridia Resume Intelligence System is an AI-powered Streamlit web application designed to streamline the recruitment process. It analyzes unstructured text data from resumes to provide objective metrics and classifications, helping recruiters quickly filter and categorize candidates.

The core functionality includes machine learning models that predict candidate suitability and specialized domain.

✨ Core Features

Icon

Feature

Description

✅

Job Readiness Score

A quantitative prediction of a candidate's fit for a generic role, helping to prioritize the screening queue.

🧠

Candidate Category

Classifies the candidate's primary skill domain (e.g., Data Science, HR, Web Developer) for easy pipeline routing.

⬆️

Direct Data Ingestion

Supports flexible input via direct resume file upload (PDF/DOCX) or text pasting.

📊

Interactive Dashboard

Results are displayed in a clean, interactive dashboard built using Streamlit.

🤖

ML Pipelines

Uses robust Machine Learning–based classification and regression pipelines for predictive tasks.

⚙️ Tech Stack

This project is built using the following technologies:

Python 🐍: The primary development language.

Streamlit 🌐: Used for creating the interactive web application interface.

scikit-learn 🤖: The core library for building, training, and deploying the ML models.

pandas / numpy 📊: Essential libraries for data cleaning, manipulation, and numerical processing.

🚀 Installation and Setup

Follow these steps to set up the project locally.

1. Clone the Repository

git clone [https://github.com/AnshulSharma9340/resume-analytics-ml.git)
cd veridia-resume-ai


2. Create a Virtual Environment (Recommended)

Using a virtual environment prevents dependency conflicts with other Python projects.

# Create the environment
python -m venv venv

# Activate the environment (Windows)
.\venv\Scripts\activate

# Activate the environment (macOS/Linux)
source venv/bin/activate


3. Install Dependencies

You need to install all necessary Python packages. This assumes you have a requirements.txt file in your repository.

pip install -r requirements.txt


(If you do not have a requirements.txt file yet, you will need to create one listing all packages like streamlit, scikit-learn, pandas, etc.)

4. Run the Application

Execute the main application file using Streamlit:

streamlit run "file_name".py 
# Note: Ensure your main Streamlit script is named 'file-name' or adjust the command.


The application will start and automatically open in your default web browser at http://localhost:8501.

🤝 Contribution

Feel free to open issues or submit pull requests for any bug fixes, feature additions, or improvements.

✉️ Contact

Anshul Sharma 
linkedIn - https://www.linkedin.com/in/anshul-sharma-9856882a4/
Email - anshulsharma7162@gmail.com

Project Link: https://github.com/AnshulSharma9340/resume-analytics-ml/

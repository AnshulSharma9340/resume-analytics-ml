# Streamlit Resume Dashboard
# Run: streamlit run streamlit_resume_dashboard.py
# Requirements: pip install streamlit pandas plotly sklearn wordcloud matplotlib

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import base64

st.set_page_config(page_title="Resume Analytics Dashboard", layout="wide")

# ---------------------- Helpers ----------------------
EMAIL = None

@st.cache_data
def load_data(path=None):
    if path:
        df = pd.read_csv(path)
    else:
        # try default filename
        df = pd.read_csv("resume_dataset_cleaned.csv")
    # Basic safety: ensure expected columns exist
    for col in ['resume_clean','resume_len_words','resume_len_chars','skills_found','num_skills','degrees','has_degree','exp_years']:
        if col not in df.columns:
            df[col] = np.nan
    # Normalize types
    df['skills_found'] = df['skills_found'].apply(lambda x: x if isinstance(x, list) else (eval(x) if isinstance(x, str) and x.startswith('[') else ([] if pd.isna(x) else [str(x)])))
    df['degrees'] = df['degrees'].apply(lambda x: x if isinstance(x, list) else (eval(x) if isinstance(x, str) and x.startswith('[') else ([] if pd.isna(x) else [str(x)])))
    df['exp_years'] = pd.to_numeric(df['exp_years'], errors='coerce')
    df['num_skills'] = pd.to_numeric(df['num_skills'], errors='coerce').fillna(0).astype(int)
    return df

@st.cache_data
def top_skills(df, top_n=20):
    skills_exploded = df.explode('skills_found')
    skill_counts = skills_exploded['skills_found'].value_counts().reset_index()
    skill_counts.columns = ['skill','count']
    return skill_counts.head(top_n)

@st.cache_data
def skill_cooccurrence(df, top_skills):
    top_sk = top_skills['skill'].tolist()
    skill_bin = pd.DataFrame(0, index=df.index, columns=top_sk)
    for s in top_sk:
        skill_bin[s] = df['skills_found'].apply(lambda x: 1 if s in (x if isinstance(x,list) else []) else 0)
    cooc = skill_bin.T.dot(skill_bin)
    return cooc

def make_wordcloud(text, width=800, height=400, max_words=150):
    wc = WordCloud(width=width, height=height, background_color='white', collocations=False, max_words=max_words).generate(text)
    return wc

# ---------------------- Sidebar ----------------------
st.sidebar.title("Data & Filters")
uploaded_file = st.sidebar.file_uploader("Upload cleaned CSV (optional)", type=['csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    try:
        df = load_data()
    except Exception as e:
        st.sidebar.error("Could not load default file 'resume_dataset_cleaned.csv'. Please upload the cleaned CSV.")
        st.stop()

st.sidebar.markdown(f"**Total resumes:** {len(df)}")

# Filters
all_skills = sorted(list({s for row in df['skills_found'] for s in (row if isinstance(row,list) else [])}))
selected_skills = st.sidebar.multiselect("Filter by skills (match any)", options=all_skills, default=None)
exp_min, exp_max = int(df[df['exp_years']>=0]['exp_years'].min() if (df['exp_years']>=0).any() else 0), int(df[df['exp_years']>=0]['exp_years'].max() if (df['exp_years']>=0).any() else 20)
exp_range = st.sidebar.slider("Experience (years)", min_value=exp_min, max_value=max(exp_max,exp_min+1), value=(exp_min, max(exp_min+2, exp_max)))
degree_options = sorted(list({d for row in df['degrees'] for d in (row if isinstance(row,list) else []) if d}))
selected_degrees = st.sidebar.multiselect("Filter by detected degree", options=degree_options, default=None)

# Apply filters
filtered = df.copy()
if selected_skills:
    filtered = filtered[filtered['skills_found'].apply(lambda x: any(s in (x if isinstance(x,list) else []) for s in selected_skills))]
if selected_degrees:
    filtered = filtered[filtered['degrees'].apply(lambda x: any(d in (x if isinstance(x,list) else []) for d in selected_degrees))]
filtered = filtered[ (filtered['exp_years']>=exp_range[0]) & (filtered['exp_years']<=exp_range[1]) | (filtered['exp_years'].isna()) ]

# ---------------------- Layout ----------------------
st.title("Veridia — Resume Analytics Dashboard")
st.markdown("Upload your cleaned dataset or use the default file. Use filters on the left to slice the candidate pool.")

# Top row: key metrics
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Candidates", len(filtered))
with k2:
    st.metric("Avg. Skills", round(filtered['num_skills'].mean(),2))
with k3:
    med_exp = round(filtered[filtered['exp_years']>=0]['exp_years'].median() if (filtered['exp_years']>=0).any() else np.nan,2)
    st.metric("Median Experience (yrs)", med_exp)
with k4:
    pct_deg = round(filtered['has_degree'].mean()*100,2)
    st.metric("% With Degree Detected", f"{pct_deg}%")

st.markdown("---")

# Main: Charts
col1, col2 = st.columns((2,3))

with col1:
    st.subheader("Top Skills")
    ts = top_skills(filtered, top_n=30)
    fig1 = px.bar(ts, x='count', y='skill', orientation='h', title='Top Skills (filtered)')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Experience Distribution")
    exp_data = filtered[filtered['exp_years']>=0]
    if not exp_data.empty:
        fig2 = px.histogram(exp_data, x='exp_years', nbins=20, title='Experience (years)')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No experience data available in this filtered set")

with col2:
    st.subheader("Degree Distribution")
    degs = filtered.explode('degrees')
    deg_counts = degs['degrees'].value_counts().reset_index()
    deg_counts.columns = ['degree','count']
    if not deg_counts.empty:
        fig3 = px.pie(deg_counts, names='degree', values='count', title='Detected Degrees')
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No degree info in this filtered set")

st.markdown("---")

# Skill co-occurrence heatmap
st.subheader("Skill Co-occurrence (Top 25)")
if len(filtered) > 0:
    ts25 = top_skills(filtered, top_n=25)
    cooc = skill_cooccurrence(filtered, ts25)
    # plotly heatmap
    fig_co = go.Figure(data=go.Heatmap(z=cooc.values, x=cooc.columns, y=cooc.index, colorscale='Blues'))
    fig_co.update_layout(height=600, width=900)
    st.plotly_chart(fig_co, use_container_width=True)
else:
    st.info("No data for co-occurrence")

st.markdown("---")

# Wordcloud
st.subheader("Wordcloud of Resumes")
corpus = " ".join(filtered['resume_clean'].dropna().astype(str).values)
if corpus.strip():
    wc = make_wordcloud(corpus)
    fig_wc = plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig_wc)
else:
    st.info("No text available to create wordcloud")

st.markdown("---")

# Table view & download
st.subheader("Candidate Table (sample)")
st.dataframe(filtered[['resume_len_words','num_skills','degrees','exp_years']].head(200))

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(filtered)
st.download_button("Download filtered CSV", data=csv, file_name='filtered_resumes.csv', mime='text/csv')

# ---------------------- Footer / Tips ----------------------
st.sidebar.markdown("---")
st.sidebar.title("Tips & Next Steps")
st.sidebar.markdown(" - Improve skills extraction by expanding the skill lexicon or using NER models.\n - Standardize application form to include structured fields for experience and degree.\n - Use this filtered export to create candidate shortlists for specific roles.")

st.sidebar.markdown("\n---\nMade for Veridia — Resume analytics. Modify thresholds and skill lists in the source code to tune results.")

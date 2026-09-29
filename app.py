"""Streamlit web app.  Run locally:  streamlit run app.py"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


bundle = load_model()
pipe, FEATURES = bundle["pipeline"], bundle["features"]

st.title("🎓 Student Performance Predictor")
st.write("Enter the student's details to predict their Performance Index (10–100).")

col1, col2 = st.columns(2)
with col1:
    hours = st.slider("Hours Studied", 1, 9, 5)
    prev = st.slider("Previous Scores", 40, 99, 70)
    extra = st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True)
with col2:
    sleep = st.slider("Sleep Hours", 4, 9, 7)
    papers = st.slider("Sample Question Papers Practiced", 0, 9, 5)

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{
        "Hours Studied": hours,
        "Previous Scores": prev,
        "Sleep Hours": sleep,
        "Sample Question Papers Practiced": papers,
        "Extracurricular Activities_Yes": 1 if extra == "Yes" else 0,
    }])[FEATURES]  # enforce training column order
    pred = float(pipe.predict(row)[0])
    pred = max(10.0, min(100.0, pred))  # keep inside the dataset's range
    st.metric("Predicted Performance Index", f"{pred:.1f}")

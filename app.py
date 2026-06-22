import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("heart_model.pkl")

st.title("Heart Disease Prediction App")
st.write("Enter patient details below:")

age = st.number_input("Age", min_value=1, max_value=100)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate")
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2, 3])

if st.button("Predict"):
    data = np.array([[age, sex, cp, trestbps, chol,
                      fbs, restecg, thalach, exang,
                      oldpeak, slope, ca, thal]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")
        # Upload and display dataset




st.sidebar.header("Upload Heart Disease Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)
import pandas as pd
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset Uploaded Successfully!")
    st.subheader("Heart Disease Dataset")
    st.dataframe(df)
    st.write("Dataset Shape:", df.shape)
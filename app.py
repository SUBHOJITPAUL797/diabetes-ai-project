import streamlit as st
import requests

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details below:")

# Inputs
preg = st.number_input("Pregnancies", min_value=0, max_value=17, step=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200)
bp = st.number_input("Blood Pressure", min_value=0, max_value=122)
insulin = st.number_input("Insulin", min_value=0, max_value=846)
bmi = st.number_input("BMI", min_value=0.0, max_value=67.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
age = st.number_input("Age", min_value=21, max_value=100)

# Button
if st.button("Predict"):
    url = "http://127.0.0.1:8000/api/predict"

    data = {
        "Pregnancies": preg,
        "Glucose": glucose,
        "BloodPressure": bp,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    try:
        response = requests.post(url, json=data)
        result = response.json()

        if response.status_code == 200:
            st.success("Prediction Done ✅")

            risk = result["risk_level"]
            if risk == "High":
                st.error("🔴 High Risk of Diabetes")
            elif risk == "Medium":
                st.warning("🟡 Medium Risk")
            else:
                st.success("🟢 Low Risk")

            st.write("### Details:")
            st.write(f"Probability: {result['probability']}")
            st.write(f"Risk Level: {result['risk_level']}")
            st.write(f"Confidence: {result['confidence']}%")

        else:
            st.error("Error from API")

    except Exception as e:
        st.error(f"Connection error: {e}")
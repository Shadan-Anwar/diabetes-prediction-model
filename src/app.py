
# # app.py
# import pickle
# import streamlit as st
# import numpy as np
# import numpy
# print(numpy.__version__)


# # load the model

# model = pickle.load(open("logistic_model.pkl", "rb"))

# st.title("Diabetes Prediction App")
# st.header("This is tb predictor pltform please select your input carefully")

# preg = st.number_input("Pregnancies", 0, 20)
# glucose = st.number_input("Glucose", 0, 500)
# bp = st.number_input("BloodPressure", 0, 200)
# skinthick = st.number_input("SkinThickness", 0, 200)
# insulin = st.number_input("Insulin", 0, 1000)
# bmi = st.number_input("BMI", 0, 500)
# diabetes_pred_fun = st.number_input("DiabetesPedigreeFunction", 0, 100)
# age = st.number_input("Age", 0, 100)

# if st.button("Predict"):
#     data = np.array(
#         [[preg, glucose, bp, skinthick, insulin, bmi, diabetes_pred_fun, age]])
#     prediction = model.predict(data)[0]
#     st.success("Diabetic" if prediction == 1 else "Non Diabetic")


# app.py
import joblib
import streamlit as st
import numpy as np
import os

# Path to model from src/app.py
# MODEL_PATH = os.path.join(os.path.dirname(
#     __file__), "..", "model", "logistic_model.joblib")

# Load model safely
# model = joblib.load(MODEL_PATH)

# Load the model (use joblib, not pickle)
model = joblib.load("model/logistic_model.joblib")

st.title("Diabetes Prediction App")
st.header(
    "This is a diabetes predictor platform. Please select your input carefully.")

# Input fields
preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 500)
bp = st.number_input("BloodPressure", 0, 200)
skinthick = st.number_input("SkinThickness", 0, 200)
insulin = st.number_input("Insulin", 0, 1000)
bmi = st.number_input("BMI", 0.0, 100.0)
diabetes_pred_fun = st.number_input("DiabetesPedigreeFunction", 0.0, 3.0)
age = st.number_input("Age", 0, 120)

if st.button("Predict"):
    data = np.array(
        [[preg, glucose, bp, skinthick, insulin, bmi, diabetes_pred_fun, age]])
    prediction = model.predict(data)[0]
    st.success("Diabetic" if prediction == 1 else "Non Diabetic")

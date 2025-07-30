# 🩺 Diabetes Prediction App

A machine learning web application to predict the likelihood of diabetes using logistic regression. Built with **Python**, **scikit-learn**, and **Streamlit**.

---

## 📌 Overview

This project uses the Pima Indians Diabetes Dataset to train a Logistic Regression model that predicts whether a person is diabetic or not based on certain health parameters.

The app provides a user-friendly interface to input values and get instant predictions.

---

## 🚀 Features

- Predicts diabetes based on:
  - Number of pregnancies
  - Glucose level
  - Blood pressure
  - Skin thickness
  - Insulin level
  - BMI
  - Diabetes pedigree function
  - Age
- Model trained using Logistic Regression
- Simple and interactive UI with Streamlit

---

## 🗂️ Project Structure

diabetes_prediction_project/
├── data/
│ └── diabetes.csv
├── model/
│ └── logistic_model.joblib
├── src/
│ ├── app.py
│ └── train_model.py
├── requirements.txt
└── README.md

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash

git clone https://github.com/your-username/diabetes_prediction_project.git
cd diabetes_prediction_project

2. Create a virtual environment (optional but recommended)
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate       # On Linux/macOS
# OR
venv\Scripts\activate  

3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt

4. Train the model
bash
Copy
Edit
cd src
python train_model.py

5. Run the Streamlit app
bash
Copy
Edit
streamlit run app.py

Then open your browser and go to http://localhost:8501.


📊 Dataset Used
Name: Pima Indians Diabetes Database

Source: UCI Machine Learning Repository

📈 Model Performance
Algorithm: Logistic Regression

Accuracy: ~79% (on testing set)

👨‍💻 Author
Shadan Anwar
AI & IoT Developer | MBA Student
📧 Email: your.email@example.com
🔗 GitHub: github.com/your-username

📝 License
This project is licensed under the MIT License.

yaml
Copy
Edit

---

Let me know if you want me to customize this with your **actual email**, **GitHub username**, or plan to **deploy it on Streamlit Cloud** – I’ll help you make it more polished.

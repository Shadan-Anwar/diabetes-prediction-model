

# train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset
# df = pd.read_csv("data/diabetes.csv")
# df = pd.read_csv("/home/shaddy/Desktop/AI/Projects/diabetes_prediction_project/data")
df = pd.read_csv(
    "/home/shaddy/Desktop/AI/Projects/diabetes_prediction_project/data/diabetes.csv")


# Define X and y
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model locally
# joblib.dump(model, "logistic_model.joblib")
joblib.dump(model, "../model/logistic_model.joblib")

print("Model saved as logistic_model.joblib")


# requirement.txt with version

# altair==5.4.1
# attrs==25.3.0
# blinker==1.8.2
# cachetools==5.5.2
# certifi==2025.7.14
# charset-normalizer==3.4.2
# click==8.1.8
# gitdb==4.0.12
# gitpython==3.1.45
# idna==3.10
# importlib-resources==6.4.5
# jinja2==3.1.6
# joblib==1.4.2
# jsonschema==4.23.0
# jsonschema-specifications==2023.12.1
# markdown-it-py==3.0.0
# MarkupSafe==2.1.5
# mdurl==0.1.2
# narwhals==1.42.1
# numpy==1.24.4
# packaging==24.2
# pandas==2.0.3
# pillow==10.4.0
# pkgutil-resolve-name==1.3.10
# protobuf==5.29.5
# pyarrow==17.0.0
# pydeck==0.9.1
# pygments==2.19.2
# python-dateutil==2.9.0.post0
# pytz==2025.2
# referencing==0.35.1
# requests==2.32.4
# rich==13.9.4
# rpds-py==0.20.1
# scikit-learn==1.3.2
# scipy==1.10.1
# six==1.17.0
# smmap==5.0.2
# streamlit==1.40.1
# tenacity==9.0.0
# threadpoolctl==3.5.0
# toml==0.10.2
# tornado==6.4.2
# typing-extensions==4.13.2
# tzdata==2025.2
# urllib3==2.2.3
# watchdog==4.0.2
# zipp==3.20.2

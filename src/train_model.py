

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

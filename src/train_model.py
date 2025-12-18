import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
data = pd.read_csv("data/creditcard.csv")

# Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Save feature names
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "models/model.pkl")

print("Model trained and saved successfully!")

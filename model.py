import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("creditcard.csv")

# Selecting relevant features
X = df[['Time', 'Amount']]
y = df['Class']  # Fraudulent or Not

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features correctly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)  # Transform test data

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save model and scaler properly
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)  # Ensure we're saving the full StandardScaler object

print("Model and Scaler saved successfully!")

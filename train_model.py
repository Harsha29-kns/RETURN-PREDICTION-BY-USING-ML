import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("synthetic_returns_data_updated.csv")

# Feature Engineering
df["total_order_value"] = df["product_price"] * df["order_quantity"]
df["discount_percentage"] = (df["discount_applied"] / df["product_price"]) * 100
df["high_discount"] = (df["discount_percentage"] > 30).astype(int)
df["fast_shipping"] = (df["shipping_time"] <= 3).astype(int)

# Select Features
feature_columns = ["product_price", "discount_applied", "shipping_time", "order_quantity", 
                   "total_order_value", "discount_percentage", "high_discount", "fast_shipping"]
X = df[feature_columns]  # Select only relevant features
y = df["Returned"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🔥 Updated Model Accuracy: {accuracy:.2f}")

# Save Model, Scaler, and Feature Columns
with open("return_prediction_model.pkl", "wb") as file:
    pickle.dump(model, file)
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)
with open("feature_columns.pkl", "wb") as file:
    pickle.dump(feature_columns, file)

print("✅ Updated Model, Scaler, and Feature List Saved")

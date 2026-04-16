import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("house_cleaned.csv")

# -------------------------
# Feature Engineering
# -------------------------

# Clean area
df['area'] = df['area'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

# Convert numeric columns
df['bedRoom'] = pd.to_numeric(df['bedRoom'], errors='coerce')
df['bathroom'] = pd.to_numeric(df['bathroom'], errors='coerce')
df['floorNum'] = pd.to_numeric(df['floorNum'], errors='coerce')

# Balcony cleanup
df['balcony'] = df['balcony'].astype(str).str.replace('+', '', regex=False)
df['balcony'] = pd.to_numeric(df['balcony'], errors='coerce')

# Simple NLP feature
df['luxury'] = df['description'].str.contains(
    'luxury|premium|spacious|modern', case=False, na=False
).astype(int)

# Use price_per_sqft
df['price_per_sqft'] = pd.to_numeric(df['price_per_sqft'], errors='coerce')

# Features
features = [
    'area',
    'bedRoom',
    'bathroom',
    'floorNum',
    'price_per_sqft',
    'balcony',
    'luxury'
]

target = 'price'

df = df[features + [target]].dropna()

# -------------------------
# Train-test split
# -------------------------
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Train model
# -------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# Predictions
# -------------------------
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# -------------------------
# Metrics
# -------------------------
train_mae = mean_absolute_error(y_train, train_preds)
test_mae = mean_absolute_error(y_test, test_preds)

print(f"Train MAE: {train_mae}")
print(f"Test MAE: {test_mae}")

# -------------------------
# Feature Importance Plot
# -------------------------
importance = model.feature_importances_

plt.figure()
plt.barh(features, importance)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.savefig("feature_importance.png")

# -------------------------
# Save Markdown Report (CML)
# -------------------------
with open("report.md", "w") as f:
    f.write(f"""
# House Price Prediction Results

## Metrics
- Train MAE: {train_mae:.2f}
- Test MAE: {test_mae:.2f}

## Feature Importance
![Feature Importance](feature_importance.png)
""")

print("✅ Training complete. Report + plot saved.")

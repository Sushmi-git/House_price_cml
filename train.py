import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("house_cleaned.csv")

# Select features
features = ['area', 'bedRoom', 'bathroom']
target = 'price'

# Clean data
df = df[features + [target]].dropna()

# Convert to numeric
df['area'] = df['area'].astype(str).str.extract('(\d+)').astype(float)
df['bedRoom'] = pd.to_numeric(df['bedRoom'], errors='coerce')
df['bathroom'] = pd.to_numeric(df['bathroom'], errors='coerce')

df = df.dropna()

# Split data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, preds)

print(f"MAE: {mae}")

# Save metrics
with open("report.txt", "w") as f:
    f.write(f"MAE: {mae}\n")

# Feature importance
importance = model.feature_importances_

plt.figure()
plt.barh(features, importance)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.savefig("feature_importance.png")

print("Training complete. Report + plot saved.")
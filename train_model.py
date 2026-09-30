import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. Load Data
df = pd.read_csv("Crop_recommendation.csv")
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

# 2. Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Compare Models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
}

print("--- Training Models ---")
for name, clf in models.items():
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"{name} Accuracy: {acc * 100:.2f}%")

# 4. Save Best Model (Random Forest)
best_model = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)
best_model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
with open("models/crop_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(
    "\nTrained Random Forest model successfully saved to 'models/crop_model.pkl'"
)
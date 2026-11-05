import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def _synthetic_training_data(seed=42, n=1200):
    rng = np.random.default_rng(seed)
    age = rng.integers(0, 60, n)
    height = 48 + age * 0.58 + rng.normal(0, 5, n)
    weight = 3.5 + age * 0.17 + rng.normal(0, 1.8, n)
    diversity = rng.integers(0, 9, n)
    illness = rng.integers(0, 2, n)
    food = rng.integers(0, 3, n)
    # Artificial rule purely for UI demonstration; not a medical label.
    score = (weight < (4.5 + age * .14)).astype(int) + (diversity < 3).astype(int) + illness + (food == 2).astype(int)
    y = (score >= 2).astype(int)
    X = np.column_stack([age, height, weight, diversity, illness, food])
    return X, y

def train_demo_model():
    X, y = _synthetic_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model, accuracy_score(y_test, model.predict(X_test))

def predict_demo_risk(model, age, height, weight, diversity, illness, food_security):
    food_map = {"Secure": 0, "Moderate concern": 1, "High concern": 2}
    row = np.array([[age, height, weight, diversity, int(illness), food_map[food_security]]])
    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0][1]
    label = "Higher demo risk pattern" if prediction else "Lower demo risk pattern"
    return f"{label} (synthetic-model score: {probability:.0%})"

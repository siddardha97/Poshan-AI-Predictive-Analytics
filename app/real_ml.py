
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_real_model():

    # 1. Load actual NFHS-5 dataset
    df = pd.read_csv("data/NFHS5_district_nutrition_cleaned.csv")

    # 2. Select input features
    features = [
        "wasted_under5_pct",
        "severely_wasted_under5_pct",
        "underweight_under5_pct",
        "improved_drinking_water_pct",
        "improved_sanitation_pct",
        "clean_cooking_fuel_pct",
        "households_surveyed"
    ]

    target = "stunted_under5_pct"

    # 3. Separate input and target
    X = df[features]
    y = df[target]

    # Remove rows with missing target values
    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    # 4. Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 5. Create Random Forest pipeline
    model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", LinearRegression())
])

    # 6. Perform 5-fold cross-validation on training data
    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kfold,
        scoring="r2"
    )

    # 7. Train final model
    model.fit(X_train, y_train)
    # 8. Predict test data
    predictions = model.predict(X_test)

    # 8.1 Create validation results
    validation_df = pd.DataFrame({
        "Actual Stunting (%)": y_test.values,
        "Predicted Stunting (%)": predictions
    })

    validation_df["Prediction Error"] = (
        validation_df["Actual Stunting (%)"]
        - validation_df["Predicted Stunting (%)"]
    )
    # 9. Evaluate test performance
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # 10. Store evaluation results
    results = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4),
        "CV R2 Mean": round(float(cv_scores.mean()), 4),
        "CV R2 Std": round(float(cv_scores.std()), 4),
        "Training Records": len(X_train),
        "Testing Records": len(X_test)
    }

    return model, results, validation_df

def compare_models():

    # Load NFHS-5 dataset
    df = pd.read_csv("data/NFHS5_district_nutrition_cleaned.csv")

    features = [
        "wasted_under5_pct",
        "severely_wasted_under5_pct",
        "underweight_under5_pct",
        "improved_drinking_water_pct",
        "improved_sanitation_pct",
        "clean_cooking_fuel_pct",
        "households_surveyed"
    ]

    target = "stunted_under5_pct"

    X = df[features]
    y = df[target]

    # Remove rows with missing target values
    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    # Use the same train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Algorithms to compare
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        )
    }

    results = []

    for name, algorithm in models.items():

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", algorithm)
        ])

        # Five-fold cross-validation on training data
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring="r2"
        )

        # Train and test the model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "Algorithm": name,
            "MAE": round(mean_absolute_error(y_test, predictions), 2),
            "RMSE": round(
                np.sqrt(mean_squared_error(y_test, predictions)), 2
            ),
            "Test R2": round(r2_score(y_test, predictions), 4),
            "CV Mean R2": round(float(cv_scores.mean()), 4)
        })

    return pd.DataFrame(results).sort_values(
        by="CV Mean R2", ascending=False
    )

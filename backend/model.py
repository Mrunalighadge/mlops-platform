import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
import os
import json
from datetime import datetime

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_data(df: pd.DataFrame, target_column: str):
    df = df.dropna()
    X = df.drop(columns=[target_column])
    y = df[target_column]
    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    task_type = "classification"
    if y.dtype in ["float64", "int64"] and y.nunique() > 10:
        task_type = "regression"
    else:
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))
        encoders["target"] = le_target
    return X, y, encoders, task_type

def train_model(df: pd.DataFrame, target_column: str, experiment_name: str = "mlops-platform"):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        X, y, encoders, task_type = prepare_data(df, target_column)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if task_type == "classification":
            metrics = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 4),
            }
        else:
            metrics = {
                "r2_score": round(r2_score(y_test, y_pred), 4),
                "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            }
        mlflow.log_params({
            "target_column": target_column,
            "task_type": task_type,
            "n_estimators": 100,
            "train_size": len(X_train),
            "test_size": len(X_test),
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"model_{version}.pkl")
        meta_path = os.path.join(MODEL_DIR, f"meta_{version}.json")
        joblib.dump(model, model_path)
        meta = {
            "version": version,
            "target_column": target_column,
            "task_type": task_type,
            "features": list(X.columns),
            "metrics": metrics,
            "trained_at": datetime.now().isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        with open(os.path.join(MODEL_DIR, "latest.json"), "w") as f:
            json.dump({"version": version, "model_path": model_path, "meta_path": meta_path}, f)
        return {"version": version, "task_type": task_type, "metrics": metrics, "features": list(X.columns)}

def predict(input_data: dict):
    latest_path = os.path.join(MODEL_DIR, "latest.json")
    if not os.path.exists(latest_path):
        raise Exception("No trained model found. Please train a model first.")
    with open(latest_path) as f:
        latest = json.load(f)
    model = joblib.load(latest["model_path"])
    with open(latest["meta_path"]) as f:
        meta = json.load(f)
    input_df = pd.DataFrame([input_data])
    for col in meta["features"]:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[meta["features"]]
    prediction = model.predict(input_df)[0]
    probability = None
    if meta["task_type"] == "classification":
        proba = model.predict_proba(input_df)[0]
        probability = round(float(max(proba)) * 100, 1)
    return {
        "prediction": str(prediction),
        "probability": probability,
        "task_type": meta["task_type"],
        "model_version": latest["version"],
    }

def get_all_models():
    models = []
    if not os.path.exists(MODEL_DIR):
        return models
    for f in os.listdir(MODEL_DIR):
        if f.startswith("meta_") and f.endswith(".json"):
            with open(os.path.join(MODEL_DIR, f)) as file:
                models.append(json.load(file))
    return sorted(models, key=lambda x: x["trained_at"], reverse=True)
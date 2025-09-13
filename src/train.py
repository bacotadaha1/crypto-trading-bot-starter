import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from src.config import settings

FEATURES = ["ret_s", "ret_l", "vol_l", "rsi14"]
TARGET = "future_return"

def load_training(csv_path=None):
    csv_path = csv_path or str(Path(settings.data_dir) / "training_data.csv")
    df = pd.read_csv(csv_path, parse_dates=["ts"], index_col="ts")
    y = (df[TARGET] > 0).astype(int)
    X = df[FEATURES].copy()
    X = X.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    return X, y

def train_and_save(csv_path=None, model_path=None):
    csv_path = csv_path or str(Path(settings.data_dir) / "training_data.csv")
    model_path = model_path or "model.pkl"

    X, y = load_training(csv_path)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    score = clf.score(Xte, yte)
    print(f"[MODEL] Test accuracy: {score:.3f}")
    joblib.dump({"model": clf, "features": FEATURES}, model_path)
    print(f"[SAVE] {model_path}")

if _name_ == "_main_":
    train_and_save()

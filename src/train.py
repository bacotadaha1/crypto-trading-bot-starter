# src/train.py
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.config import settings
from src.ingestion import make_client

DATA_DIR = settings.data_dir
CSV_PATH = DATA_DIR / "training_data.csv"
MODELS_DIR = Path("models")


# ----------------- utils OHLCV -----------------
def _parse_symbols() -> list[str]:
    if isinstance(settings.symbols, (list, tuple)):
        return [s.strip() for s in settings.symbols if s.strip()]
    if isinstance(settings.symbols, str):
        return [s.strip() for s in settings.symbols.split(",") if s.strip()]
    return ["BTC/USDT"]

def _safe_fetch_ohlcv(ex, symbol: str, timeframe: str, limit: int, retries: int = 4, backoff: float = 2.0):
    for i in range(retries):
        try:
            return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(backoff * (i + 1))

def ensure_training_csv() -> None:
    """
    Si data/training_data.csv n'existe pas, on le génère depuis l'exchange.
    """
    if CSV_PATH.exists():
        print(f"✅ {CSV_PATH} existe déjà — on continue.")
        return

    print("⚠️  training_data.csv introuvable — génération automatique depuis l'exchange…")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ex = make_client()
    symbols = _parse_symbols()
    timeframe = settings.timeframe
    limit = settings.limit

    frames = []
    for sym in symbols:
        ohlcv = _safe_fetch_ohlcv(ex, sym, timeframe=timeframe, limit=limit)
        if not ohlcv:
            print(f"⚠️  Aucune donnée pour {sym} — ignoré.")
            continue

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["symbol"] = sym
        frames.append(df)

    if not frames:
        raise RuntimeError("Aucune donnée téléchargée — impossible de créer training_data.csv")

    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])
    out.to_csv(CSV_PATH, index=False)
    print(f"✅ Données créées : {CSV_PATH.resolve()} (rows={len(out)})")


# ----------------- features & dataset -----------------
def _build_features_common(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """
    Construit EXACTEMENT les mêmes features que predict.py attend.
    Retourne (X, df_full_aligne).
    """
    df = df.copy()

    # ret_1
    df["ret_1"] = df["close"].pct_change()

    # rolling mean/std
    df["roll_mean"] = df["close"].rolling(vol_window).mean()
    df["roll_std"]  = df["close"].rolling(vol_window).std()

    # RSI(14) simple
    delta = df["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=df.index).rolling(14).mean()
    roll_down = pd.Series(down, index=df.index).rolling(14).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    df["rsi"] = 100 - (100 / (1 + rs))

    # high/low range relatif
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    # z-score du prix
    df["price_z"] = (df["close"] - df["roll_mean"]) / (df["roll_std"].replace(0, np.nan))

    # dropna et ordre des colonnes
    df = df.dropna().copy()
    feats = ["ret_1", "roll_mean", "roll_std", "rsi", "hl_range", "price_z", "volume"]
    X = df[feats].copy()
    return X, df

def _make_xy_for_training(df_symbol: pd.DataFrame, vol_window: int):
    """
    Construit X, y pour l'entraînement à partir d'un df OHLCV d'un symbole.
    Cible = direction du prochain close (retour futur).
    """
    X, df_feat = _build_features_common(df_symbol, vol_window)
    # y = +1 si close(t+1) >= close(t), sinon 0
    future_ret = df_feat["close"].pct_change().shift(-1)
    y = (future_ret >= 0).astype(int)

    # aligner X et y (on perd la dernière ligne car shift(-1))
    X = X.iloc[:-1, :].copy()
    y = y.iloc[:-1].copy()

    # sécurité en cas de classe unique
    if y.nunique() < 2:
        # force un minime bruit pour éviter les erreurs (peu probable sur 4h 1500 bars)
        y.iloc[-1] = 1 - y.iloc[-1]

    return X.values, y.values


# ----------------- entraînement -----------------
def train_models():
    df_all = pd.read_csv(CSV_PATH)
    print(f"📦 training_data.csv lu : {len(df_all)} lignes, {df_all['symbol'].nunique()} symboles.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rows_report = []
    symbols = sorted(df_all["symbol"].unique())

    for symbol in symbols:
        df_sym = df_all[df_all["symbol"] == symbol].copy()
        if len(df_sym) < 300:  # garde-fou
            print(f"⚠️  Pas assez de données pour {symbol} — skip.")
            continue

        # X, y
        X, y = _make_xy_for_training(df_sym, settings.vol_window)

        # split chronologique 80/20
        n = len(X)
        split = int(n * 0.8)
        X_tr, y_tr = X[:split], y[:split]
        X_te, y_te = X[split:], y[split:]

        # pipeline : scaler + random forest
        pipe = Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("rf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])

        pipe.fit(X_tr, y_tr)

        # métriques simples
        y_hat = pipe.predict(X_te)
        acc = float(accuracy_score(y_te, y_hat))
        try:
            proba = pipe.predict_proba(X_te)[:, 1]
            auc = float(roc_auc_score(y_te, proba))
        except Exception:
            auc = float("nan")

        # sauvegarde modèle
        model_path = MODELS_DIR / f"{symbol.replace('/','_')}_model.pkl"
        joblib.dump(pipe, model_path)
        print(f"✅ Modèle sauvegardé : {model_path} | test_acc={acc:.3f} | roc_auc={auc if auc==auc else 'nan'}")

        rows_report.append({
            "symbol": symbol,
            "n_samples": n,
            "train_size": split,
            "test_size": n - split,
            "test_acc": round(acc, 4),
            "test_roc_auc": (round(auc, 4) if auc == auc else None),  # nan-safe
        })

    # petit rapport
    rep = pd.DataFrame(rows_report)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rep.to_csv(DATA_DIR / "ml_training_report.csv", index=False, encoding="utf-8")
    print("✅ Entraînement terminé — rapport écrit dans data/ml_training_report.csv")


def main():
    ensure_training_csv()
    train_models()


if __name__ == "__main__":
    main()

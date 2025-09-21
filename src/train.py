# src/train.py
from __future__ import annotations
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.config import settings
from src.ingestion import make_client

DATA_DIR = settings.data_dir
CSV_PATH = DATA_DIR / "training_data.csv"
MODELS_DIR = Path("models")


# ---------- helpers généraux ----------
def _parse_symbols() -> list[str]:
    syms = settings.symbols
    if isinstance(syms, str):
        return [s.strip() for s in syms.split(",") if s.strip()]
    if isinstance(syms, (list, tuple)):
        return [s.strip() for s in syms if s.strip()]
    return ["BTC/USDT"]


def _exchange_symbol(exchange_id: str, human_symbol: str) -> str:
    """
    Convertit le symbole humain 'BTC/USDT' vers la notation exigée
    par certains exchanges (KuCoin/OKX/Bybit => 'BTC-USDT').
    """
    dash_exchanges = {"kucoin", "okx", "bybit"}
    if exchange_id in dash_exchanges:
        return human_symbol.replace("/", "-")
    return human_symbol


def _safe_fetch_ohlcv(ex, symbol: str, timeframe: str, limit: int,
                      retries: int = 4, backoff: float = 2.0):
    for i in range(retries):
        try:
            return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(backoff * (i + 1))


# ---------- features (identiques à predict.py) ----------
def _build_features_train(df: pd.DataFrame, vol_window: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Reproduit EXACTEMENT les features de src/predict.py et crée la cible binaire:
    target_up = 1 si close(t+1) > close(t) sinon 0.
    Retourne X (features) et y (target_up) alignés, sans NaN et sans fuite.
    """
    df = df.copy()

    # Features
    df["ret_1"] = df["close"].pct_change()
    df["roll_mean"] = df["close"].rolling(vol_window).mean()
    df["roll_std"] = df["close"].rolling(vol_window).std()

    delta = df["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=df.index).rolling(14).mean()
    roll_down = pd.Series(down, index=df.index).rolling(14).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    df["rsi"] = 100 - (100 / (1 + rs))

    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["price_z"] = (df["close"] - df["roll_mean"]) / (df["roll_std"].replace(0, np.nan))

    # Cible: mouvement du prochain pas de temps
    df["target_up"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Nettoyage
    df = df.dropna().copy()

    feats = ["ret_1", "roll_mean", "roll_std", "rsi", "hl_range", "price_z", "volume"]
    X = df[feats].copy()
    y = df["target_up"].copy()

    # Aligne pour ne pas utiliser d'info future:
    # (on prédit t+1 avec x_t, donc on retire le dernier x qui n'a pas y)
    if len(X) > 1 and len(y) > 1:
        X = X.iloc[:-1, :].reset_index(drop=True)
        y = y.iloc[:-1].reset_index(drop=True)

    return X, y


# ---------- ingestion CSV ----------
def ensure_training_csv() -> None:
    """
    Si data/training_data.csv n'existe pas, on le génère depuis l'exchange
    pour tous les symboles listés dans settings.symbols.
    """
    if CSV_PATH.exists():
        print(f"✅ {CSV_PATH} existe déjà — on continue.")
        return

    print("⚠️  training_data.csv introuvable — génération automatique depuis l'exchange…")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ex = make_client()
    ex_id = getattr(ex, "id", str(settings.exchange)).lower()
    symbols = _parse_symbols()
    timeframe = settings.timeframe
    limit = settings.limit

    frames = []
    for human_sym in symbols:
        ex_sym = _exchange_symbol(ex_id, human_sym)
        try:
            ohlcv = _safe_fetch_ohlcv(ex, ex_sym, timeframe=timeframe, limit=limit)
        except Exception as e:
            print(f"⚠️  Échec fetch {human_sym} ({ex_sym}) : {e}")
            continue

        if not ohlcv:
            print(f"⚠️  Aucune donnée pour {human_sym} — ignoré.")
            continue

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["symbol"] = human_sym  # on garde toujours la forme humaine dans le CSV
        frames.append(df)

    if not frames:
        raise RuntimeError("Aucune donnée téléchargée — impossible de créer training_data.csv.")

    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])
    out.to_csv(CSV_PATH, index=False)
    print(f"✅ Données créées : {CSV_PATH.resolve()} (rows={len(out)})")


# ---------- entraînement ----------
def train_models():
    """
    Pour chaque symbole:
      - recharge le CSV (ordonné dans le temps),
      - construit les features identiques à la prédiction,
      - split temporel simple (80/20) pour éviter la fuite,
      - fit un pipeline (StandardScaler + LogisticRegression class_weight='balanced'),
      - sauvegarde models/<SYMBOL>_model.pkl.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(CSV_PATH)
    print(f"📦 training_data.csv lu : {len(df_all)} lignes, {df_all['symbol'].nunique()} symboles.")

    symbols = sorted(df_all["symbol"].unique())
    metrics_rows = []

    for sym in symbols:
        df_sym = df_all[df_all["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
        if len(df_sym) < 200:
            print(f"⚠️ Trop peu de données pour {sym} ({len(df_sym)}) — on saute.")
            continue

        # Features/target
        X, y = _build_features_train(df_sym, settings.vol_window)
        n = len(X)
        if n < 100 or y.nunique() < 2:
            print(f"⚠️ Données/variabilité insuffisantes pour {sym} — on saute.")
            continue

        # Split temporel 80/20
        split = int(n * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_valid, y_valid = X.iloc[split:], y.iloc[split:]

        # Pipeline
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None))
        ])

        pipe.fit(X_train.values, y_train.values)

        # Petite métrique de sanity-check (accuracy sur la fin)
        acc = float((pipe.predict(X_valid.values) == y_valid.values).mean())
        print(f"✅ {sym}: valid_acc={acc:.3f}  (n={len(y_valid)})")
        metrics_rows.append({"symbol": sym, "valid_acc": acc, "n_valid": int(len(y_valid))})

        # Sauvegarde
        model_path = MODELS_DIR / f"{sym.replace('/', '_')}_model.pkl"
        joblib.dump(pipe, model_path)
        print(f"💾 Modèle sauvegardé : {model_path}")

    # Rapport d'entraînement
    rep_path = DATA_DIR / "ml_training_report.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(rep_path, index=False)
    print(f"📝 Rapport écrit : {rep_path.resolve()}")


def main():
    ensure_training_csv()
    train_models()
    print("🎉 Entraînement terminé.")


if __name__ == "__main__":
    main()

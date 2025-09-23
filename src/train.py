# src/train.py
from __future__ import annotations
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.config import settings
from src.ingestion import make_client
from src.alt_data import build_alt_features  # <<< NEW

DATA_DIR = settings.data_dir
CSV_PATH = DATA_DIR / "training_data.csv"
MODELS_DIR = Path("models")


# ---------- utilitaires ----------
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
    Crée data/training_data.csv si absent en téléchargeant l'OHLCV.
    Sauvegarde colonnes: [timestamp(ms), open, high, low, close, volume, symbol]
    """
    if CSV_PATH.exists():
        print(f"✅ {CSV_PATH} existe déjà — on continue.")
        return

    print("⚠️  training_data.csv introuvable — génération automatique depuis l'exchange...")
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


def _prepare_df_symbol(df_sym: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise le DataFrame d'un symbole:
      - crée 'ts' (datetime UTC) à partir de 'timestamp' (ms) si nécessaire
      - tri + colonnes minimales
    """
    df = df_sym.copy()
    if "ts" not in df.columns:
        # le CSV a 'timestamp' en millisecondes
        df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df[["ts", "open", "high", "low", "close", "volume"]]


def _build_features_train(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construit les mêmes features que la prédiction (prix + alt-data) et la cible (UP/DOWN à 1 step).
    Robuste si alt-data absente : remplissage à 0.0.
    """
    df = df_in.copy()

    # ----- prix
    df["ret_1"] = df["close"].pct_change()
    roll = int(getattr(settings, "vol_window", 12))
    df["roll_mean"] = df["close"].rolling(roll).mean()
    df["roll_std"]  = df["close"].rolling(roll).std()

    delta = df["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    r_up = pd.Series(up, index=df.index).rolling(14).mean()
    r_dn = pd.Series(down, index=df.index).rolling(14).mean()
    rs = r_up / r_dn.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["price_z"] = (df["close"] - df["roll_mean"]) / (df["roll_std"].replace(0, np.nan))

    # ----- alt-data alignée ts
    alt = build_alt_features(df[["ts", "close", "volume"]])
    df = df.set_index("ts").join(alt, how="left").reset_index()

    alt_cols = ["sent_hour", "fg_idx", "rv_w6", "rv_w36", "vol_z"]
    for col in alt_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # cible = signe du retour futur
    df["future_ret"] = df["close"].pct_change().shift(-1)
    y = (df["future_ret"] > 0).astype("int8")

    # drop uniquement sur le noyau "prix" nécessaire
    core_needed = ["ret_1", "roll_mean", "roll_std", "rsi", "future_ret"]
    df = df.dropna(subset=core_needed).copy()

    feats = [
        "ret_1", "roll_mean", "roll_std", "rsi", "hl_range", "price_z", "volume",
        "sent_hour", "fg_idx", "rv_w6", "rv_w36", "vol_z",
    ]
    X = df[feats].astype("float32")
    y = y.loc[X.index].astype("int8")

    return X, y


def train_models() -> None:
    """
    Entraîne un modèle compact par symbole et le sauvegarde compressé (<100MB).
    - RandomForest limité en profondeur
    - compression joblib
    - CV TimeSeriesSplit (log d’info)
    """
    df_all = pd.read_csv(CSV_PATH)
    print(f"📦 training_data.csv lu : {len(df_all)} lignes, {df_all['symbol'].nunique()} symboles.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    symbols = sorted(df_all["symbol"].unique())
    for symbol in symbols:
        base = _prepare_df_symbol(df_all[df_all["symbol"] == symbol])
        X, y = _build_features_train(base)
        if len(X) < 200:
            print(f"⚠️  Trop peu d'observations pour {symbol} — skip.")
            continue

        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )

        # (optionnel) CV pour info
        tscv = TimeSeriesSplit(n_splits=5)
        try:
            scores = cross_val_score(model, X.values, y.values, cv=tscv, scoring="roc_auc", n_jobs=-1)
            print(f"📈 {symbol} | AUC CV (5 folds): {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as e:
            print(f"[WARN] CV échouée pour {symbol}: {e}")

        model.fit(X.values, y.values)

        # --- sauvegarde compressée ---
        model_path = MODELS_DIR / f"{symbol.replace('/','_')}_model.pkl"
        joblib.dump(model, model_path, compress=3)
        print(f"✅ Modèle sauvegardé (compressé) : {model_path}")

    # petit rapport
    (DATA_DIR / "ml_training_report.csv").write_text(
        f"rows,{len(df_all)}\nsymbols,{','.join(symbols)}\n", encoding="utf-8"
    )
    print("✅ Entraînement terminé (rapport écrit).")


def main():
    ensure_training_csv()
    train_models()


if __name__ == "__main__":
    main()

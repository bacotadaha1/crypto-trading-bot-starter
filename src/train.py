# src/train.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier  # compatible online_update (partial_fit)

from src.config import settings
from src.ingestion import make_client
from src.alt_data import build_alt_features  # <<< NEW

DATA_DIR = settings.data_dir
CSV_PATH = DATA_DIR / "training_data.csv"
MODELS_DIR = Path("models")


# -------------------- utils --------------------
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
    Crée data/training_data.csv si absent en téléchargeant l'OHLCV (tous symboles).
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


# -------------------- features --------------------
_BASE_FEATS = ["ret_1", "roll_mean", "roll_std", "rsi", "hl_range", "price_z", "volume"]
_ALT_FEATS  = ["sent_hour", "fg_idx", "rv_w6", "rv_w36", "vol_z"]
_ALL_FEATS  = _BASE_FEATS + _ALT_FEATS  # 12 features au total

def _build_tech_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les features 'techniques' sur le dataframe de prix (une ligne = une bougie).
    """
    d = df.copy()
    d["ret_1"] = d["close"].pct_change()

    roll = int(getattr(settings, "vol_window", 12))
    d["roll_mean"] = d["close"].rolling(roll).mean()
    d["roll_std"]  = d["close"].rolling(roll).std()

    # RSI (14)
    delta = d["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    r_up = pd.Series(up, index=d.index).rolling(14).mean()
    r_dn = pd.Series(down, index=d.index).rolling(14).mean()
    rs = r_up / r_dn.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))

    d["hl_range"] = (d["high"] - d["low"]) / d["close"].replace(0, np.nan)
    d["price_z"] = (d["close"] - d["roll_mean"]) / d["roll_std"].replace(0, np.nan)

    d = d.dropna().copy()
    return d


def _build_dataset_for_symbol(df_sym: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Pour un symbole donné :
      - fabrique les features techniques
      - fabrique les alt-features alignées (sentiment RSS, Fear&Greed, régimes)
      - merge sur 'ts'
      - crée la cible binaire (UP = 1 si close[t+1] > close[t])
    Retourne X (12 colonnes) et y.
    """
    # rename 'timestamp' -> 'ts' et passer en UTC
    df = df_sym.rename(columns={"timestamp": "ts"}).copy()
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

    # features techniques
    tech = _build_tech_features(df)

    # alt features alignées sur les timestamps des bougies
    alt = build_alt_features(df[["ts", "close", "volume"]])

    # merge (inner) sur ts
    merged = (
        tech.merge(alt, left_on=pd.to_datetime(tech["ts"], utc=True),
                   right_on=alt.index, how="inner")
    )
    # après merge, 'key_0' est l'index alt; on réinstalle 'ts'
    merged = merged.drop(columns=["key_0"])
    merged = merged.set_index(pd.to_datetime(merged["ts"], utc=True)).sort_index()

    # cible : direction de la prochaine bougie
    merged["future_ret"] = merged["close"].pct_change().shift(-1)
    y = (merged["future_ret"] > 0).astype("int8")

    X = merged[_ALL_FEATS].copy()

    xy = pd.concat([X, y.rename("y")], axis=1).dropna()
    X = xy[_ALL_FEATS].astype("float32")
    y = xy["y"].astype("int8")

    return X, y


# -------------------- train --------------------
def train_models() -> None:
    """
    Entraîne un modèle **SGDClassifier dans un Pipeline(StandardScaler)** pour chaque symbole,
    sur **12 features** (7 techniques + 5 alt-data). Sauvegarde compressée.
    """
    df_all = pd.read_csv(CSV_PATH)
    print(f"📦 training_data.csv lu : {len(df_all)} lignes, {df_all['symbol'].nunique()} symboles.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    symbols = sorted(df_all["symbol"].unique())
    for symbol in symbols:
        df = df_all[df_all["symbol"] == symbol].copy()
        X, y = _build_dataset_for_symbol(df)

        if len(X) < 300:
            print(f"⚠️  Trop peu d'observations pour {symbol} — skip ({len(X)}).")
            continue

        # Pipeline compatible avec online_update (partial_fit sur SGD)
        pipe = Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", SGDClassifier(loss="log_loss",
                                  alpha=1e-4,
                                  max_iter=1000,
                                  tol=1e-3,
                                  random_state=42))
        ])

        # CV en TimeSeriesSplit (facultatif)
        try:
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(pipe, X.values, y.values, cv=tscv,
                                     scoring="roc_auc", n_jobs=-1)
            print(f"📈 {symbol} | AUC CV (5 folds): {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as e:
            print(f"[WARN] CV échouée pour {symbol}: {e}")

        pipe.fit(X.values, y.values)

        model_path = MODELS_DIR / f"{symbol.replace('/','_')}_model.pkl"
        joblib.dump(pipe, model_path, compress=3)
        print(f"✅ Modèle sauvegardé (12 feats, compressé) : {model_path}")

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

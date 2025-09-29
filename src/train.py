# src/train.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple
from datetime import datetime, timezone
import time

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.ingestion import make_client
from src.alt_data import build_alt_features

DATA_DIR = settings.data_dir
CSV_PATH = DATA_DIR / "training_data.csv"
MODELS_DIR = Path("models")

# ---------------- helpers ----------------
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
    if CSV_PATH.exists():
        print(f"✅ {CSV_PATH} existe déjà — on continue.")
        return
    print("⚠️  training_data.csv introuvable — génération depuis l'exchange...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ex = make_client()
    symbols = _parse_symbols()
    frames = []
    for sym in symbols:
        ohlcv = _safe_fetch_ohlcv(ex, sym, settings.timeframe, settings.limit)
        if not ohlcv:
            print(f"⚠️  Aucune donnée pour {sym} — ignoré.")
            continue
        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df["symbol"] = sym
        frames.append(df)
    if not frames:
        raise RuntimeError("Aucune donnée téléchargée — impossible de créer training_data.csv")
    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "ts"])
    out.to_csv(CSV_PATH, index=False)
    print(f"✅ Données créées : {CSV_PATH.resolve()} (rows={len(out)})")

# ---------- features ----------
def _build_features_tech(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    d = df.copy()
    d["ret_1"] = d["close"].pct_change()
    d["roll_mean"] = d["close"].rolling(vol_window, min_periods=max(2, vol_window//2)).mean()
    d["roll_std"] = d["close"].rolling(vol_window, min_periods=max(2, vol_window//2)).std()
    delta = d["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    r_up = pd.Series(up, index=d.index).rolling(14, min_periods=7).mean()
    r_dn = pd.Series(down, index=d.index).rolling(14, min_periods=7).mean()
    rs = r_up / r_dn.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))
    d["hl_range"] = (d["high"] - d["low"]) / d["close"].replace(0, np.nan)
    d["price_z"] = (d["close"] - d["roll_mean"]) / d["roll_std"].replace(0, np.nan)
    tech = d[["ts","ret_1","roll_mean","roll_std","rsi","hl_range","price_z","volume"]]
    return tech

def _build_full_features(df_price: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    tech = _build_features_tech(df_price, vol_window=vol_window).dropna().copy()
    alt = build_alt_features(df_price[["ts","close","volume"]]).reset_index()
    full = (tech.merge(alt, on="ts", how="left")
                 .merge(df_price[["ts","close"]], on="ts", how="left"))
    full = full.sort_values("ts").ffill().dropna().reset_index(drop=True)
    return full

def _make_Xy(df_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feat_cols = [
        "ret_1","roll_mean","roll_std","rsi","hl_range","price_z","volume",
        "sent_hour","fg_idx","rv_w6","rv_w36","vol_z"
    ]
    d = df_full.copy()
    d["future_ret"] = d["close"].pct_change().shift(-1)
    y = (d["future_ret"] > 0).astype("int8")
    X = d[feat_cols].astype("float32")
    xy = pd.concat([X, y.rename("y")], axis=1).dropna()
    return xy[feat_cols], xy["y"]

def train_models() -> None:
    df_all = pd.read_csv(CSV_PATH, parse_dates=["ts"])
    df_all["ts"] = pd.to_datetime(df_all["ts"], utc=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sorted(df_all["symbol"].unique())
    vol_window = int(getattr(settings, "vol_window", 12))

    for symbol in symbols:
        src = df_all[df_all["symbol"] == symbol].copy()
        if len(src) < 300:
            print(f"⚠️  Trop peu d'observations pour {symbol} — skip.")
            continue

        full = _build_full_features(src[["ts","open","high","low","close","volume"]], vol_window)
        if len(full) < 200:
            print(f"⚠️  Pas assez de lignes après feature engineering pour {symbol} — skip.")
            continue

        X, y = _make_Xy(full)

        # Pipeline + calibration CV=3 (isotonic) — robuste et évite proba extrêmes
        base = make_pipeline(
            StandardScaler(with_mean=True),
            SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                random_state=42,
                max_iter=5000,
                tol=1e-4,
            ),
        )
        # Calibration croisée — refit sur chaque fold, calibration sur les out-of-fold
        calib = CalibratedClassifierCV(base, cv=3, method="isotonic")
        calib.fit(X.values, y.values)

        payload = {
            "model": calib,
            "features": list(X.columns),
            "vol_window": vol_window,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        model_path = MODELS_DIR / f"{symbol.replace('/','_')}_model.pkl"
        joblib.dump(payload, model_path, compress=3)
        print(f"✅ Modèle calibré sauvegardé : {model_path} (rows={len(X)})")

    (DATA_DIR / "ml_training_report.csv").write_text(
        f"rows,{len(df_all)}\nsymbols,{','.join(symbols)}\n", encoding="utf-8"
    )
    print("✅ Entraînement terminé (calibré isotone CV=3).")

def main():
    ensure_training_csv()
    train_models()

if __name__ == "__main__":
    main()

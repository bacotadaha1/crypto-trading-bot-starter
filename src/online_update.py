from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.ingestion import make_client

MODELS_DIR = Path("models")
STATE_DIR = Path("models_state")   # mémo par symbole (dernier ts appris)
STATE_DIR.mkdir(parents=True, exist_ok=True)

def _sym_key(symbol: str) -> str:
    return symbol.replace("/", "_").replace("-", "_").upper()

def _build_features(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = df["close"].pct_change()
    roll = int(getattr(settings, "vol_window", vol_window))
    df["roll_mean"] = df["close"].rolling(roll).mean()
    df["roll_std"]  = df["close"].rolling(roll).std()
    d = df["close"].diff()
    up = np.where(d > 0, d, 0.0); down = np.where(d < 0, -d, 0.0)
    roll_up = pd.Series(up, index=df.index).rolling(14).mean()
    roll_down = pd.Series(down, index=df.index).rolling(14).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    df["rsi"] = 100 - (100 / (1 + rs))
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["price_z"] = (df["close"] - df["roll_mean"]) / (df["roll_std"].replace(0, np.nan))
    # cible: direction de la bougie suivante
    df["future_ret"] = df["close"].pct_change().shift(-1)
    df["y"] = (df["future_ret"] > 0).astype("int8")
    feats = ["ret_1", "roll_mean", "roll_std", "rsi", "hl_range", "price_z", "volume"]
    out = df[["ts", *feats, "y"]].dropna().copy()
    return out, feats

def _fetch_df(ex, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def _load_state(symbol: str) -> dict:
    p = STATE_DIR / f"{_sym_key(symbol)}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"last_ts": None}

def _save_state(symbol: str, state: dict) -> None:
    p = STATE_DIR / f"{_sym_key(symbol)}.json"
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_or_bootstrap(symbol: str, X: np.ndarray, y: np.ndarray):
    base = MODELS_DIR / _sym_key(symbol)
    model_path = base.with_name(base.name + "_model.pkl")
    scaler_path = base.with_name(base.name + "_scaler.pkl")
    model = joblib.load(model_path) if model_path.exists() else None
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    if model is None or scaler is None:
        scaler = StandardScaler(with_mean=True, with_std=True)
        model = SGDClassifier(loss="log_loss", random_state=42)
        scaler.partial_fit(X)
        X0 = scaler.transform(X)
        model.partial_fit(X0, y, classes=np.array([0,1], dtype=np.int32))
    return model, scaler

def _incremental_fit_one(ex, symbol: str) -> str:
    df_raw = _fetch_df(ex, symbol, settings.timeframe, settings.limit)
    if df_raw.empty or len(df_raw) < 50:
        return f"{symbol}: pas assez de données (n={len(df_raw)})"
    feats_df, cols = _build_features(df_raw, settings.vol_window)
    if feats_df.empty:
        return f"{symbol}: features vides"

    st = _load_state(symbol)
    if st["last_ts"]:
        cutoff = pd.to_datetime(st["last_ts"], utc=True, errors="coerce")
        new_df = feats_df[feats_df["ts"] > cutoff].copy()
    else:
        new_df = feats_df.tail(500).copy()  # warmup initial
    if new_df.empty:
        return f"{symbol}: rien de nouveau depuis {st['last_ts']}"

    X = new_df[cols].astype("float32").values
    y = new_df["y"].astype("int8").values
    model, scaler = _load_or_bootstrap(symbol, X, y)
    scaler.partial_fit(X)
    Xs = scaler.transform(X)
    model.partial_fit(Xs, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / f"{_sym_key(symbol)}_model.pkl", compress=3)
    joblib.dump(scaler, MODELS_DIR / f"{_sym_key(symbol)}_scaler.pkl", compress=3)

    st["last_ts"] = str(new_df["ts"].max())
    _save_state(symbol, st)
    return f"{symbol}: +{len(new_df)} obs, last_ts={st['last_ts']}"

def main():
    syms = settings.symbols
    if isinstance(syms, str):
        syms = [s.strip() for s in syms.split(",") if s.strip()]
    ex = make_client()
    logs = []
    for s in syms:
        try:
            logs.append(_incremental_fit_one(ex, s))
        except Exception as e:
            logs.append(f"{s}: ERREUR {e}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "online_update.log").write_text("\n".join(logs), encoding="utf-8")

if __name__ == "__main__":
    main()

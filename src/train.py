# src/train.py
from __future__ import annotations
import time
from pathlib import Path
from typing import Tuple

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

# ----------------- helpers -----------------
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
    (Re)crée data/training_data.csv si absent en téléchargeant l'OHLCV.
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
        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df["symbol"] = sym
        frames.append(df)

    if not frames:
        raise RuntimeError("Aucune donnée téléchargée — impossible de créer training_data.csv")

    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "ts"])
    out.to_csv(CSV_PATH, index=False)
    print(f"✅ Données créées : {CSV_PATH.resolve()} (rows={len(out)})")

# ---------- features robustes ----------
def _build_features_tech(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    d = df.copy()

    # moyennes/écarts tolérants (min_periods)
    d["ret_1"] = d["close"].pct_change()
    d["roll_mean"] = d["close"].rolling(vol_window, min_periods=max(2, vol_window//2)).mean()
    d["roll_std"]  = d["close"].rolling(vol_window, min_periods=max(2, vol_window//2)).std()

    # RSI 14 tolérant
    delta = d["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    r_up = pd.Series(up, index=d.index).rolling(14, min_periods=7).mean()
    r_dn = pd.Series(down, index=d.index).rolling(14, min_periods=7).mean()
    rs = r_up / r_dn.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))

    d["hl_range"] = (d["high"] - d["low"]) / d["close"].replace(0, np.nan)
    d["price_z"]  = (d["close"] - d["roll_mean"]) / d["roll_std"].replace(0, np.nan)

    out = d[["ts","open","high","low","close","volume",
             "ret_1","roll_mean","roll_std","rsi","hl_range","price_z"]].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out

def _build_full_features(df_price: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """
    Fusionne features techniques + alt-data (sentiment/FG/régime).
    Alt-data est forward/backward-filled et 0 si manquant, pour éviter de
    jeter toute la ligne.
    """
    tech = _build_features_tech(df_price, vol_window)
    alt  = build_alt_features(df_price[["ts","close","volume"]])

    alt = alt.reset_index() if isinstance(alt.index, pd.DatetimeIndex) else alt
    full = tech.merge(alt, on="ts", how="left").sort_values("ts")

    # colonnes alt garanties présentes
    for col in ["sent_hour","fg_idx","rv_w6","rv_w36","vol_z"]:
        if col not in full.columns:
            full[col] = np.nan

    full[["sent_hour","fg_idx","rv_w6","rv_w36","vol_z"]] = (
        full[["sent_hour","fg_idx","rv_w6","rv_w36","vol_z"]]
        .ffill().bfill().fillna(0.0)
    )

    # On ne drop que ce qui est indispensable côté technique
    full = full.dropna(subset=["ret_1","roll_mean","roll_std","rsi","hl_range","price_z","volume"])
    return full

def _make_Xy(df_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    d = df_full.copy()
    d["future_ret"] = d["close"].pct_change().shift(-1)
    y = (d["future_ret"] > 0).astype("int8")

    feat_cols = [
        "ret_1","roll_mean","roll_std","rsi","hl_range","price_z","volume",
        "sent_hour","fg_idx","rv_w6","rv_w36","vol_z"
    ]
    X = (
        d[feat_cols]
        .astype("float32")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    xy = pd.concat([X, y.rename("y")], axis=1).dropna(subset=["y"])
    return xy[feat_cols], xy["y"]

# ---------- entraînement ----------
def train_models() -> None:
    df_all = pd.read_csv(CSV_PATH, parse_dates=["ts"])
    df_all["ts"] = pd.to_datetime(df_all["ts"], utc=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sorted(df_all["symbol"].unique())
    vol_window = int(getattr(settings, "vol_window", 12))

    saved_any = False
    for symbol in symbols:
        src = df_all[df_all["symbol"] == symbol].copy()
        if len(src) < 300:
            print(f"⚠️  Trop peu d'observations pour {symbol} — skip.")
            continue

        full = _build_full_features(src[["ts","open","high","low","close","volume"]], vol_window)
        # sécurité : on garde 'close' pour la cible (déjà dans full)
        if "close" not in full.columns:
            full = full.merge(src[["ts","close"]], on="ts", how="left").ffill()

        X, y = _make_Xy(full)
        if len(X) < 200:
            print(f"⚠️  Pas assez de lignes après features pour {symbol} — skip.")
            continue

        # split chrono 80/20 pour calibration
        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_cal,   y_cal   = X.iloc[split:], y.iloc[split:]

        base = make_pipeline(
            StandardScaler(with_mean=True),
            SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                random_state=42,
                max_iter=2000,
                tol=1e-3,
            )
        )
        base.fit(X_train, y_train)

        clf = CalibratedClassifierCV(base, cv="prefit", method="sigmoid")
        clf.fit(X_cal, y_cal)

        model_path = MODELS_DIR / f"{symbol.replace('/','_')}_model.pkl"
        payload = {"model": clf, "features": list(X.columns), "vol_window": vol_window}
        joblib.dump(payload, model_path, compress=3)
        print(f"✅ Modèle calibré sauvegardé : {model_path} (rows={len(X)})")
        saved_any = True

    (DATA_DIR / "ml_training_report.csv").write_text(
        f"rows,{len(df_all)}\nsymbols,{','.join(symbols)}\n", encoding="utf-8"
    )
    if saved_any:
        print("✅ Entraînement terminé (calibré).")
    else:
        print("⚠️  Aucun modèle sauvegardé (trop peu de lignes après features).")

def main():
    ensure_training_csv()
    train_models()

if __name__ == "__main__":
    main()

# src/train.py
from __future__ import annotations
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score

from src.config import settings
from src.ingestion import make_client

DATA_DIR = settings.data_dir
CSV_PATH = DATA_DIR / "training_data.csv"
MODELS_DIR = Path("models")

# ---------- features identiques train/predict ----------
def build_features(df: pd.DataFrame, vol_window: int) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["ret_1"] = df["close"].pct_change()
    roll = df["close"].rolling(vol_window)
    df["roll_mean"] = roll.mean()
    df["roll_std"]  = roll.std()

    # RSI simple (14)
    delta = df["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=df.index).rolling(14).mean()
    roll_down = pd.Series(down, index=df.index).rolling(14).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    df["rsi"] = 100 - (100 / (1 + rs))

    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["price_z"] = (df["close"] - df["roll_mean"]) / (df["roll_std"].replace(0, np.nan))

    # cible: up si close(t+1) > close(t)
    df["target_up"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df = df.dropna().copy()
    feats = ["ret_1", "roll_mean", "roll_std", "rsi", "hl_range", "price_z", "volume"]
    X = df[feats].astype(float).values
    y = df["target_up"].astype(int).values
    return X, y, df

# ---------- téléchargement OHLCV ~4 ans (pagination robuste) ----------
_MS_IN_MIN = 60_000

def _timeframe_to_ms(tf: str) -> int:
    # ccxt timeframe like "4h", "1h", "1d"
    unit = tf[-1]
    n = int(tf[:-1])
    if unit == "m":
        return n * _MS_IN_MIN
    if unit == "h":
        return n * 60 * _MS_IN_MIN
    if unit == "d":
        return n * 24 * 60 * _MS_IN_MIN
    raise ValueError(f"timeframe inconnu: {tf}")

def fetch_ohlcv_since(ex, symbol: str, timeframe: str, since_ms: int, limit_per_call: int = 1000):
    out = []
    tf_ms = _timeframe_to_ms(timeframe)
    cursor = since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit_per_call)
        if not batch:
            break
        out.extend(batch)
        last_ts = batch[-1][0]
        # avance d'un pas de timeframe pour éviter de retélécharger le dernier point
        cursor = last_ts + tf_ms
        # sécurité anti-rate limit
        time.sleep(ex.rateLimit / 1000)
        # si on n'avance plus, on sort
        if len(batch) < limit_per_call:
            break
    return out

def ensure_training_csv_years(years: int = 4) -> None:
    if CSV_PATH.exists():
        print(f"✅ {CSV_PATH} existe déjà — on continue.")
        return

    print(f"⚠️  training_data.csv introuvable — téléchargement ~{years} ans…")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ex = make_client()
    symbols = [s.strip() for s in (settings.symbols if isinstance(settings.symbols, (list, tuple)) else settings.symbols.split(","))]
    timeframe = settings.timeframe

    since_dt = datetime.now(timezone.utc) - timedelta(days=365*years + 7)  # marge
    since_ms = int(since_dt.timestamp() * 1000)

    frames = []
    for sym in symbols:
        ohlcv = fetch_ohlcv_since(ex, sym, timeframe, since_ms)
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

# ---------- entraînement avec TSS(5) + sauvegarde ----------
def train_models():
    df_all = pd.read_csv(CSV_PATH)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    report = []
    for sym in sorted(df_all["symbol"].unique()):
        df = df_all[df_all["symbol"] == sym].copy()
        # build features / target
        X, y, df_feat = build_features(df, settings.vol_window)
        if len(df_feat) < 200:
            print(f"⚠️  {sym}: pas assez de points après features — skip.")
            continue

        tss = TimeSeriesSplit(n_splits=5)
        accs, f1s = [], []
        for tr_idx, te_idx in tss.split(X):
            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]
            clf = RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1
            )
            clf.fit(Xtr, ytr)
            yhat = clf.predict(Xte)
            accs.append(accuracy_score(yte, yhat))
            f1s.append(f1_score(yte, yhat))

        acc_cv, f1_cv = float(np.mean(accs)), float(np.mean(f1s))
        print(f"📊 {sym} | acc_cv={acc_cv:.3f} | f1_cv={f1_cv:.3f} | n={len(df_feat)}")

        # modèle final sur tout l'historique
        final_clf = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        final_clf.fit(X, y)
        path = MODELS_DIR / f"{sym.replace('/','_')}_model.pkl"
        joblib.dump(final_clf, path)
        print(f"✅ Modèle sauvegardé : {path}")

        report.append({"symbol": sym, "n_samples": len(df_feat), "acc_cv": acc_cv, "f1_cv": f1_cv})

    pd.DataFrame(report).to_csv(DATA_DIR / "ml_training_report.csv", index=False)
    print("✅ Entraînement terminé.")

def main():
    ensure_training_csv_years(years=4)  # ~4 ans
    train_models()

if __name__ == "__main__":
    main()

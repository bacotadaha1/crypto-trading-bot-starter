# src/train.py
import time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import settings
from src.ingestion import make_client

DATA_DIR = settings.data_dir
CSV_PATH = DATA_DIR / "training_data.csv"
MODELS_DIR = Path("models")


def _parse_symbols() -> list[str]:
    syms = settings.symbols
    if isinstance(syms, str):
        return [s.strip() for s in syms.split(",") if s.strip()]
    if isinstance(syms, (list, tuple)):
        return [s.strip() for s in syms if s.strip()]
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


def _build_features_train(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """
    Reconstruit EXACTEMENT les mêmes features que dans predict.py
    et crée une target binaire: close_{t+1} > close_t.
    """
    df = df.copy()
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

    # target: up if close(t+1) > close(t)
    df["target_up"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df = df.dropna().copy()

    feats = ["ret_1", "roll_mean", "roll_std", "rsi", "hl_range", "price_z", "volume"]
    return df[feats + ["target_up"]]


def train_models():
    """
    Pour chaque symbole, entraîne un pipeline StandardScaler + LogisticRegression
    sur les features alignées avec predict.py, puis sauvegarde models/<SYMBOL>_model.pkl
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(CSV_PATH)
    print(f"📦 training_data.csv lu : {len(df_all)} lignes, {df_all['symbol'].nunique()} symboles.")

    symbols = sorted(df_all["symbol"].unique())
    for symbol in symbols:
        df_sym = df_all[df_all["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)
        if len(df_sym) < 100:
            print(f"⚠️ Trop peu de données pour {symbol} ({len(df_sym)}) — on saute.")
            continue

        df_feat = _build_features_train(df_sym, settings.vol_window)
        if df_feat["target_up"].nunique() < 2:
            print(f"⚠️ Pas assez de variation de la cible pour {symbol} — on saute.")
            continue

        X = df_feat.drop(columns=["target_up"]).values
        y = df_feat["target_up"].values

        # simple split time-based (80/20)
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]

        # pipeline: standardize + logistic regression
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=1000))
        ])
        pipe.fit(X_train, y_train)

        model_path = MODELS_DIR / f"{symbol.replace('/', '_')}_model.pkl"
        joblib.dump(pipe, model_path)
        print(f"✅ Modèle sauvegardé : {model_path}")

    # petit rapport
    (DATA_DIR / "ml_training_report.csv").write_text(
        f"rows,{len(df_all)}\nsymbols,{','.join(symbols)}\n",
        encoding="utf-8"
    )
    print("✅ Entraînement terminé (rapport écrit).")


def main():
    ensure_training_csv()
    train_models()


if __name__ == "__main__":
    main()

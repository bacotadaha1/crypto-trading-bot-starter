# src/train.py
from __future__ import annotations

import time
from pathlib import Path
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier  # simple & dispo dans scikit-learn

from src.config import settings
from src.ingestion import make_client

# -------------------------------------------------------------------
# Helpers (symboles & chemins)
# -------------------------------------------------------------------
def _canonical(symbol: str) -> str:
    """Force le format CCXT avec des slashes (ex: BTC/USDT)."""
    return symbol.replace("-", "/").strip()

def _storage_key(symbol: str) -> str:
    """Version safe pour les noms de fichiers (ex: BTC_USDT)."""
    return _canonical(symbol).replace("/", "_")

# Dossiers / chemins
DATA_DIR = settings.data_dir
CSV_PATH = DATA_DIR / "training_data.csv"
MODELS_DIR = Path("models")


# -------------------------------------------------------------------
# Parsing paramètres
# -------------------------------------------------------------------
def _parse_symbols() -> list[str]:
    if isinstance(settings.symbols, (list, tuple)):
        return [s for s in (str(x).strip() for x in settings.symbols) if s]
    if isinstance(settings.symbols, str):
        return [s for s in (x.strip() for x in settings.symbols.split(",")) if s]
    return ["BTC/USDT"]


# -------------------------------------------------------------------
# IO bourse (robuste)
# -------------------------------------------------------------------
def _safe_fetch_ohlcv(ex, symbol: str, timeframe: str, limit: int, retries: int = 4, backoff: float = 2.0):
    for i in range(retries):
        try:
            return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(backoff * (i + 1))


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
def ensure_training_csv() -> None:
    """
    Crée data/training_data.csv si absent en agrégeant l’OHLCV multi-symboles.
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
        sym_ccxt = _canonical(sym)
        ohlcv = _safe_fetch_ohlcv(ex, sym_ccxt, timeframe=timeframe, limit=limit)
        if not ohlcv:
            print(f"⚠️  Aucune donnée pour {sym_ccxt} — ignoré.")
            continue
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["symbol"] = sym_ccxt
        frames.append(df)

    if not frames:
        raise RuntimeError("Aucune donnée téléchargée — impossible de créer training_data.csv")

    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])
    out.to_csv(CSV_PATH, index=False)
    print(f"✅ Données créées : {CSV_PATH.resolve()} (rows={len(out)})")


def _build_features_train(df: pd.DataFrame, vol_window: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construit les features (identiques à celles utilisées dans predict.py) + la cible y.
    y = direction du prochain pas (UP si ret>0).
    """
    df = df.copy()
    df["ret_1"] = df["close"].pct_change()
    df["roll_mean"] = df["close"].rolling(vol_window).mean()
    df["roll_std"] = df["close"].rolling(vol_window).std()

    # RSI minimal
    delta = df["close"].diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down.replace(0, pd.NA)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, pd.NA)
    df["price_z"] = (df["close"] - df["roll_mean"]) / df["roll_std"].replace(0, pd.NA)

    # cible = signe de la variation future (on décale -1)
    y = df["close"].pct_change().shift(-1)
    y = (y > 0).astype("Int64")  # 1 si UP, 0 sinon

    feats = ["ret_1", "roll_mean", "roll_std", "rsi", "hl_range", "price_z", "volume"]
    X = df[feats]

    # alignement & dropna
    m = pd.concat([X, y.rename("y")], axis=1).dropna()
    return m[feats], m["y"].astype(int)


# -------------------------------------------------------------------
# Entraînement & sauvegarde
# -------------------------------------------------------------------
def train_models():
    df = pd.read_csv(CSV_PATH)
    print(f"📦 training_data.csv lu : {len(df)} lignes, {df['symbol'].nunique()} symboles.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    symbols = sorted(df["symbol"].unique())
    for symbol in symbols:
        df_sym = df[df["symbol"] == symbol].copy()
        X, y = _build_features_train(df_sym, settings.vol_window)
        if len(X) < 50 or y.nunique() < 2:
            print(f"⚠️  Pas assez de données/variabilité pour {symbol} — skip.")
            continue

        # Modèle simple (remplace par ton vrai pipeline plus tard)
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X.values, y.values)

        model_path = MODELS_DIR / f"{_storage_key(symbol)}_model.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Modèle sauvegardé : {model_path}")

    # petit rapport
    (DATA_DIR / "ml_training_report.csv").write_text(
        f"rows,{len(df)}\nsymbols,{','.join(symbols)}\n",
        encoding="utf-8"
    )
    print("✅ Entraînement terminé (rapport écrit).")


def main():
    ensure_training_csv()
    train_models()


if __name__ == "__main__":
    main()

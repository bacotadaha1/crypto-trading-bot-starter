# src/train.py
import time
from pathlib import Path
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier  # exemple simple, remplace par ton vrai modèle

from src.config import settings
from src.ingestion import make_client

DATA_DIR = settings.data_dir
CSV_PATH = DATA_DIR / "training_data.csv"
MODELS_DIR = Path("models")

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

def train_models():
    """Entraînement minimal : pour chaque symbole on crée un DummyClassifier et on le sauve.
       Remplace la logique par ton vrai training."""
    df = pd.read_csv(CSV_PATH)
    print(f"📦 training_data.csv lu : {len(df)} lignes, {df['symbol'].nunique()} symboles.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # pour chaque symbol, exemple : save un dummy (tu mettras ton pipeline scikit)
    symbols = sorted(df["symbol"].unique())
    for symbol in symbols:
        df_sym = df[df["symbol"] == symbol].copy()
        # exemple features triviales pour démonstration:
        X = df_sym[["close", "volume"]].fillna(0).values
        # target bidon: up/down suivant close change
        y = (df_sym["close"].pct_change().fillna(0) > 0).astype(int).values
        if len(X) < 2:
            print(f"⚠️ pas assez de données pour {symbol} — skip.")
            continue
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X, y)
        model_path = MODELS_DIR / f"{symbol.replace('/','_')}_model.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Modèle sauvegardé : {model_path}")

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

import time
from pathlib import Path
import pandas as pd
import joblib   # <-- AJOUT
from sklearn.linear_model import LinearRegression  # <-- AJOUT pour un modèle simple

from src.config import settings
from src.ingestion import make_client

DATA_DIR = settings.data_dir
CSV_PATH = DATA_DIR / "training_data.csv"


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
    Si data/training_data.csv n'existe pas, on le génère depuis l'exchange.
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


# ====== ta logique d’entraînement existante ======
# Si tu as déjà un code d'entraînement dans un autre module (ex: src.ml_dataset / src.train_model),
# appelle-le ici. À défaut, on fait un stub minimal.

def train_models():
    """
    Exemple simple : pour chaque symbole, on entraîne une régression linéaire
    close(t-1) -> close(t) et on sauvegarde un modèle dans ./models/
    """
    df = pd.read_csv(CSV_PATH)
    print(f"📦 training_data.csv lu : {len(df)} lignes, {df['symbol'].nunique()} symboles.")

    # Créer le dossier models/ si absent
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for sym, grp in df.groupby("symbol"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        grp["close_lag1"] = grp["close"].shift(1)
        grp = grp.dropna().reset_index(drop=True)

        if len(grp) < 50:
            print(f"⚠️ Trop peu de données pour {sym}, on saute.")
            continue

        X = grp[["close_lag1"]].values
        y = grp["close"].values

        model = LinearRegression()
        model.fit(X, y)

        model_path = MODELS_DIR / f"{sym.replace('/', '_')}_model.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Modèle sauvegardé : {model_path}")



def main():
    ensure_training_csv()
    train_models()


if __name__ == "__main__":
    main()

import time
from pathlib import Path
import pandas as pd

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
    Exemple très simple : charge le CSV et simule un entraînement.
    Remplace par ton vrai code d’entraînement si déjà présent dans le repo.
    """
    df = pd.read_csv(CSV_PATH)
    print(f"📦 training_data.csv lu : {len(df)} lignes, {df['symbol'].nunique()} symboles.")
    # ... Ici ton code: features, split, fit, save models ...
    # Sauvegardons un petit fichier rapport pour tracer le run :
    (DATA_DIR / "ml_training_report.csv").write_text(
        f"rows,{len(df)}\nsymbols,{','.join(sorted(df['symbol'].unique()))}\n",
        encoding="utf-8"
    )
    print("✅ Entraînement terminé (rapport écrit).")


def main():
    ensure_training_csv()
    train_models()


if __name__ == "__main__":
    main()

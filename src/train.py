# src/train.py
# =============================================================================
# Entraînement quotidien (robuste) + création automatique du CSV si absent
# - Crée data/training_data.csv depuis l'exchange si besoin
# - Construit des features simples
# - Entraîne 1 modèle / symbole (LogisticRegression)
# - Sauvegarde les modèles -> models/model_<SYMBOL>.pkl
# - Ecrit un rapport -> data/ml_training_report.csv
# =============================================================================

import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import settings
from src.ingestion import make_client

# =========================
# 1) Gestion des données
# =========================
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
CSV_PATH = DATA_DIR / "training_data.csv"
REPORT_PATH = DATA_DIR / "ml_training_report.csv"


def _parse_symbols() -> List[str]:
    # On accepte "BTC/USDT,ETH/USDT" ou liste déjà fournie
    symbols_env = getattr(settings, "symbols", None)
    if isinstance(symbols_env, str):
        syms = [s.strip() for s in symbols_env.split(",") if s.strip()]
        return syms if syms else ["BTC/USDT"]
    if isinstance(symbols_env, (list, tuple)) and len(symbols_env) > 0:
        return list(symbols_env)
    return ["BTC/USDT"]


def _get_timeframe() -> str:
    return getattr(settings, "timeframe", "4h")


def _get_limit() -> int:
    try:
        return int(getattr(settings, "limit", 2000))
    except Exception:
        return 2000


def _safe_fetch_ohlcv(ex, symbol: str, timeframe: str, limit: int,
                      retries: int = 3, backoff: float = 2.0):
    """Appel CCXT avec retries basiques (rate limit/erreurs transitoires)."""
    for i in range(retries):
        try:
            return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(backoff * (i + 1))


def ensure_training_csv() -> None:
    """
    Si data/training_data.csv n'existe pas, on le génère à partir des OHLCV.
    Colonnes: timestamp, open, high, low, close, volume, symbol
    """
    if CSV_PATH.exists():
        print(f"✅ {CSV_PATH} existe déjà — on continue.")
        return

    print("⚠️  training_data.csv introuvable — génération automatique depuis l'exchange...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ex = make_client()
    symbols = _parse_symbols()
    timeframe = _get_timeframe()
    limit = _get_limit()

    frames = []
    for sym in symbols:
        ohlcv = _safe_fetch_ohlcv(ex, sym, timeframe=timeframe, limit=limit)
        if not ohlcv or len(ohlcv) == 0:
            print(f"⚠️  Aucune donnée OHLCV pour {sym} — ignoré.")
            continue

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["symbol"] = sym
        frames.append(df)

    if not frames:
        raise RuntimeError("Aucune donnée téléchargée — impossible de créer training_data.csv")

    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])
    out.to_csv(CSV_PATH, index=False)
    print(f"✅ Données créées : {CSV_PATH.resolve()} (rows={len(out)})")


# =========================
# 2) Features & dataset
# =========================
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit des features simples par symbole.
    - returns (close pct)
    - MA 10 / MA 20
    - volatilité (std rolling 10)
    - label binaire: close(t+1) > close(t)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values(["symbol", "timestamp"])

    def _feat(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["ret_1"] = g["close"].pct_change()
        g["ma_10"] = g["close"].rolling(10).mean()
        g["ma_20"] = g["close"].rolling(20).mean()
        g["vol_10"] = g["close"].pct_change().rolling(10).std()
        # label: mouvement de la bougie suivante
        g["future_close"] = g["close"].shift(-1)
        g["y"] = (g["future_close"] > g["close"]).astype(int)
        return g

    df = df.groupby("symbol", group_keys=False).apply(_feat)
    # drop lignes avec NaN (au début des rollings ou dernière ligne pour y)
    df = df.dropna().reset_index(drop=True)
    return df


def train_test_split_by_time(df: pd.DataFrame, test_ratio: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporel pour éviter les fuites: début = train, fin = test.
    """
    n = len(df)
    cut = int(n * (1 - test_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# =========================
# 3) Entraînement par symbole
# =========================
def train_one_symbol(sym_df: pd.DataFrame, symbol: str) -> dict:
    """
    Entraîne un modèle binaire 'UP vs DOWN' pour un symbole.
    Sauvegarde le modèle dans models/model_<SYMBOL>.pkl
    Retourne un dict de métriques.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    import joblib

    features = ["ret_1", "ma_10", "ma_20", "vol_10"]
    target = "y"

    train_df, test_df = train_test_split_by_time(sym_df, test_ratio=0.25)
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    # Modèle simple et rapide
    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    clf.fit(X_train, y_train)

    # Eval
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    try:
        proba = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = float("nan")

    # Sauvegarde
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # normalise le nom pour le fichier
    safe_sym = symbol.replace("/", "_")
    model_path = MODELS_DIR / f"model_{safe_sym}.pkl"
    joblib.dump(clf, model_path)

    return {
        "symbol": symbol,
        "rows": int(len(sym_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "accuracy": round(acc, 4),
        "auc": round(auc, 4) if not np.isnan(auc) else None,
        "model_path": str(model_path),
    }


def main():
    # 1) Assure la présence des données
    ensure_training_csv()

    # 2) Charge et prépare les features
    raw = pd.read_csv(CSV_PATH)
    df = make_features(raw)

    # 3) Entraîne un modèle par symbole
    symbols = sorted(df["symbol"].unique().tolist())
    print(f"🔧 Entraînement pour {len(symbols)} symbole(s): {symbols}")

    report_rows = []
    for sym in symbols:
        sym_df = df[df["symbol"] == sym].copy()
        if len(sym_df) < 100:  # garde-fou
            print(f"⚠️ Trop peu de données pour {sym} ({len(sym_df)} lignes) — ignoré.")
            continue
        metrics = train_one_symbol(sym_df, sym)
        print("—", metrics)
        report_rows.append(metrics)

    # 4) Sauvegarde le rapport
    if report_rows:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report_rows).to_csv(REPORT_PATH, index=False)
        print(f"📝 Rapport écrit : {REPORT_PATH.resolve()}")
    else:
        print("⚠️ Aucun modèle entraîné (données insuffisantes ?)")
        # On ne lève pas d'exception pour laisser le workflow passer,
        # mais n'hésite pas à raise si tu préfères.
        # raise RuntimeError("No model trained.")


if __name__ == "__main__":
    main()

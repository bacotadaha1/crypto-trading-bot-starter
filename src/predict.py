# src/predict.py
from __future__ import annotations

import math
import json
import csv
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from pandas.api.types import is_datetime64_any_dtype

from src.config import settings
from src.ingestion import make_client  # client CCXT
from src.alt_data import build_alt_features  # Fear&Greed + sentiment RSS + regime


# ============================== Constantes ==============================

# Les 7 features "core" calculées depuis l'OHLCV (utilisées à l'entraînement)
CORE_FEATS = ["ret_1", "roll_mean", "roll_std", "rsi", "hl_range", "price_z", "volume"]


# ============================== Utilitaires ==============================

def _sym_to_fname(symbol: str) -> str:
    return symbol.replace("/", "_").replace("-", "_").upper()

def _send_telegram(msg: str) -> None:
    token = getattr(settings, "telegram_bot_token", "") or ""
    chat_id = getattr(settings, "telegram_chat_id", "") or ""
    if not token or not chat_id:
        print("[INFO] Telegram non configuré. Message:\n", msg)
        return
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=data, timeout=20)
        if r.status_code != 200:
            print("[WARN] Telegram status:", r.status_code, r.text)
    except Exception as e:
        print("[WARN] Telegram error:", e)

def fetch_ohlcv_df(ex, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """
    Récupère l'OHLCV et retourne DataFrame avec ts (UTC), open, high, low, close, volume.
    """
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def load_model_for(symbol: str):
    """
    Charge le modèle sauvegardé pour un symbole : models/BTC_USDT_model.pkl
    """
    path = Path("models") / f"{_sym_to_fname(symbol)}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Aucun modèle trouvé pour {symbol} dans ./models/")
    return joblib.load(path), path


# ============================== Features ==============================

def _build_core_features(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """
    Construit les 7 features 'core' utilisées par le training.
    """
    out = df.copy()

    # sécurités types
    if not is_datetime64_any_dtype(out["ts"]):
        out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")

    out["ret_1"] = out["close"].pct_change()

    roll = int(vol_window)
    out["roll_mean"] = out["close"].rolling(roll).mean()
    out["roll_std"] = out["close"].rolling(roll).std()

    delta = out["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    r_up = pd.Series(up, index=out.index).rolling(14).mean()
    r_dn = pd.Series(down, index=out.index).rolling(14).mean()
    rs = r_up / r_dn.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    out["hl_range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["price_z"] = (out["close"] - out["roll_mean"]) / out["roll_std"].replace(0, np.nan)

    # on ne droppe pas tout : on laissera le nettoyage ciblé après le join
    return out[["ts"] + CORE_FEATS]

def _build_full_feature_table(df_price: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """
    Assemble 'core' + alt-features alignées sur ts.
    - Alt-features peuvent être partielles : on ffill/bfill.
    - On ne droppe que si les features 'core' manquent (pas à cause des alt).
    """
    core = _build_core_features(df_price, vol_window).copy()

    if not is_datetime64_any_dtype(core["ts"]):
        core["ts"] = pd.to_datetime(core["ts"], utc=True)

    core_idxed = core.set_index("ts").sort_index()

    # alt-data indexées par ts (peuvent être vides si indispo)
    alt = build_alt_features(df_price)  # index=ts (sent_hour, fg_idx, rv_w6, rv_w36, vol_z)
    full = core_idxed.join(alt, how="left")

    # Remplissage souple pour éviter les trous : d'abord ffill puis bfill
    full = full.ffill().bfill()

    # On NE droppe que si les features 'core' manquent
    full = full.dropna(subset=CORE_FEATS)

    full = full.copy()
    full.reset_index(inplace=True)  # remet ts en colonne
    return full

def _align_features_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Aligne X sur l'ordre/ensemble des colonnes attendues par le modèle.
    - Si le modèle expose feature_names_in_, on les suit.
    - Sinon, on suppose que les 7 'core' ont été utilisées à l'entraînement.
    - Toute colonne absente est remplie par 0.
    """
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif hasattr(model, "n_features_in_") and X is not None:
        # fallback : si pas de noms, on prend d'abord les CORE_FEATS, puis on complète
        expected = [c for c in CORE_FEATS if c in X.columns]
        # si le modèle en attend plus, on complète par les colonnes de X dans l'ordre
        for c in X.columns:
            if c not in expected and len(expected) < int(model.n_features_in_):
                expected.append(c)
        # si encore trop court, on force à CORE_FEATS uniquement
        if len(expected) < int(model.n_features_in_):
            # pad avec zéros virtuels (on les créera juste après)
            expected = expected + [f"__pad_{i}" for i in range(int(model.n_features_in_)-len(expected))]
    else:
        expected = [c for c in CORE_FEATS if c in X.columns]

    X_aligned = X.copy()
    # crée colonnes manquantes (0) sans casser l'ordre
    for col in expected:
        if col not in X_aligned.columns:
            X_aligned[col] = 0.0

    return X_aligned[expected]


# ============================== Prédiction ==============================

def predict_one_symbol(ex, symbol: str) -> dict:
    df_price = fetch_ohlcv_df(ex, symbol, settings.timeframe, settings.limit)
    if df_price is None or len(df_price) == 0:
        raise RuntimeError("Téléchargement OHLCV vide.")

    full = _build_full_feature_table(df_price, settings.vol_window)

    if full.empty or len(full) == 0:
        raise RuntimeError(
            "Pas de ligne exploitable pour la prédiction (features vides). "
            "Augmente LIMIT (ex: 5000) ou vérifie le téléchargement OHLCV."
        )

    # X complet (core + alt), puis on alignera aux attentes du modèle
    X_all = full.drop(columns=["ts"])
    x_last_all = X_all.iloc[-1:].copy()

    model, model_path = load_model_for(symbol)
    x_last = _align_features_to_model(x_last_all, model)

    if x_last.shape[0] == 0:
        raise RuntimeError("Aucune ligne finale après alignement des features.")

    # Prédiction
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_last.values)
        p_up = float(proba[0, -1])  # proba classe "UP"
        direction = "UP" if p_up >= 0.5 else "DOWN"
    else:
        y_val = float(model.predict(x_last.values)[0])
        p_up = 1.0 / (1.0 + math.exp(-10.0 * y_val))  # squashing pour pseudo-proba
        direction = "UP" if y_val >= 0 else "DOWN"

    last_close = float(df_price["close"].iloc[-1])
    tstamp = pd.to_datetime(df_price["ts"].iloc[-1], utc=True).strftime("%Y-%m-%d %H:%M UTC")

    return {
        "symbol": symbol,
        "time": tstamp,
        "last_close": last_close,
        "p_up": p_up,
        "direction": direction,
        "model_path": str(model_path),
    }


# ============================== Main ==============================

def main():
    ex = make_client()

    # s'assurer que settings.symbols soit bien une liste
    symbols = settings.symbols
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]

    results: list[dict] = []
    for symbol in symbols:
        try:
            res = predict_one_symbol(ex, symbol)
            results.append(res)
        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})

    # ——— message Telegram
    limit_txt = f"  Limit: {settings.limit}" if hasattr(settings, "limit") else ""
    title = "*Signal quotidien (ML + Sentiment)*"
    lines = [
        title,
        f"Exchange: `kucoin`   testnet: `{settings.use_testnet}`",
        f"Timeframe: `4h`{limit_txt}",
        ""
    ]
    for r in results:
        if "error" in r:
            lines.append(f"• *{r['symbol']}* → `ERREUR`: {r['error']}")
            continue
        conf = int(round(r["p_up"] * 100))
        emoji = "🟢⬆️" if r["direction"] == "UP" else "🔴⬇️"
        lines.append(
            f"• *{r['symbol']}* @ {r['time']} {emoji}\n"
            f"  Direction: *{r['direction']}*  | Confiance: *{conf}%*  | Close: `{r['last_close']}`"
        )

    msg = "\n".join(lines)
    print(msg)
    _send_telegram(msg)

    # ——— log JSON (facultatif)
    Path("data").mkdir(parents=True, exist_ok=True)
    out_json = Path("data") / f"predictions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ——— journalisation CSV pour l’évaluation ultérieure
    log_path = Path("data") / "preds_log.csv"
    rows = []
    ts_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for r in results:
        if "error" in r:
            rows.append({
                "ts_utc": ts_now,
                "exchange": settings.exchange,
                "timeframe": settings.timeframe,
                "symbol": r.get("symbol"),
                "last_close": None,
                "p_up": None,
                "direction": None,
                "error": r.get("error"),
            })
        else:
            rows.append({
                "ts_utc": ts_now,
                "exchange": settings.exchange,
                "timeframe": settings.timeframe,
                "symbol": r["symbol"],
                "last_close": r["last_close"],
                "p_up": r["p_up"],
                "direction": r["direction"],
                "error": None,
            })

    if rows:
        write_header = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()

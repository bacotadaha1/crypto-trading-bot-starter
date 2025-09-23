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
from src.ingestion import make_client
from src.alt_data import build_alt_features  # <- alt-data (sentiment, fear&greed, regime)

# ------------------------- utils -------------------------

CORE_FEATS = ["ret_1", "roll_mean", "roll_std", "rsi", "hl_range", "price_z", "volume"]

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
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    # ts en datetime UTC
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

# ------------------------- features -------------------------

def _build_core_features(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """
    Construit les 7 features "core" alignées au training.
    Retourne un DataFrame avec colonnes: ['ts'] + CORE_FEATS
    """
    x = df.copy()
    # sécurité temps
    if not is_datetime64_any_dtype(x["ts"]):
        x["ts"] = pd.to_datetime(x["ts"], utc=True)

    x["ret_1"] = x["close"].pct_change()
    x["roll_mean"] = x["close"].rolling(vol_window).mean()
    x["roll_std"] = x["close"].rolling(vol_window).std()

    delta = x["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=x.index).rolling(14).mean()
    roll_down = pd.Series(down, index=x.index).rolling(14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))

    x["hl_range"] = (x["high"] - x["low"]) / x["close"].replace(0, np.nan)
    x["price_z"] = (x["close"] - x["roll_mean"]) / (x["roll_std"].replace(0, np.nan))

    x = x.dropna().copy()
    return x[["ts"] + CORE_FEATS].copy()

def _build_full_feature_table(df_price: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """
    Assemble core (7) + alt-data (sentiment/hourly, fg_idx, regime) alignés sur ts.
    """
    core = _build_core_features(df_price, vol_window).copy()

    # normalise ts proprement (évite l'erreur 'datetime64ns, UTC as a data type')
    if not is_datetime64_any_dtype(core["ts"]):
        core["ts"] = pd.to_datetime(core["ts"], utc=True)

    core_idxed = core.set_index("ts").sort_index()

    # alt-data alignées sur les mêmes timestamps
    alt = build_alt_features(df_price)  # index = ts (UTC)
    # join et ffill pour compléter les trous
    full = core_idxed.join(alt, how="left").ffill()

    full = full.dropna().copy()
    full.reset_index(inplace=True)  # remet 'ts' en colonne
    return full  # colonnes: ['ts'] + CORE_FEATS + alt-cols

# ------------------------- modèle -------------------------

def load_model_for(symbol: str):
    path = Path("models") / f"{_sym_to_fname(symbol)}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Aucun modèle trouvé pour {symbol} dans ./models/")
    return joblib.load(path), path

def _align_features_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Aligne X sur les features apprises par le modèle:
      - si feature_names_in_ dispo: on reindex sur ces colonnes (remplissage 0 si manque)
      - sinon si n_features_in_ dispo: on prend les n premières colonnes de X
      - sinon: on renvoie X tel quel
    """
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return X.reindex(columns=list(names), fill_value=0)

    n = getattr(model, "n_features_in_", None)
    if isinstance(n, (int, np.integer)) and n > 0:
        # garde l'ordre actuel des colonnes
        return X.iloc[:, :int(n)]

    return X

# ------------------------- prédiction -------------------------

def predict_one_symbol(ex, symbol: str) -> dict:
    df_price = fetch_ohlcv_df(ex, symbol, settings.timeframe, settings.limit)

    # table complète (core + alt)
    full = _build_full_feature_table(df_price, settings.vol_window)
    # X = toutes les colonnes sauf 'ts'
    X_all = full.drop(columns=["ts"])
    x_last_all = X_all.iloc[-1:].copy()

    model, model_path = load_model_for(symbol)
    # aligne les colonnes avec le modèle
    x_last = _align_features_to_model(x_last_all, model)

    # predict / predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_last.values)
        p_up = float(proba[0, -1])
        direction = "UP" if p_up >= 0.5 else "DOWN"
    else:
        y_val = float(model.predict(x_last.values)[0])
        p_up = 1.0 / (1.0 + math.exp(-10.0 * y_val))
        direction = "UP" if y_val >= 0 else "DOWN"

    last_close = float(df_price["close"].iloc[-1])
    tstamp = pd.to_datetime(df_price["ts"].iloc[-1]).strftime("%Y-%m-%d %H:%M UTC")

    return {
        "symbol": symbol,
        "time": tstamp,
        "last_close": last_close,
        "p_up": p_up,
        "direction": direction,
        "model_path": str(model_path),
    }

# ------------------------- main -------------------------

def main():
    ex = make_client()

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

    # message Telegram
    limit_txt = f"  Limit: {settings.limit}" if hasattr(settings, "limit") else ""
    lines = [
        "*Signal quotidien (ML + Sentiment)*",
        f"Exchange: `kucoin`   testnet: `{settings.use_testnet}`",
        f"Timeframe: `{settings.timeframe}`{limit_txt}",
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

    # logs
    Path("data").mkdir(parents=True, exist_ok=True)

    out_json = Path("data") / f"predictions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # journal CSV pour l'évaluation
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

    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main()

# src/predict.py
from __future__ import annotations

import math
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import joblib

from src.config import settings
from src.ingestion import make_client

# -------------------------------------------------------------------
# Helpers (symboles & chemins)
# -------------------------------------------------------------------
def _canonical(symbol: str) -> str:
    """Toujours utiliser le format CCXT avec des slashes."""
    return symbol.replace("-", "/").strip()

def _storage_key(symbol: str) -> str:
    """Clé adaptée pour les noms de fichiers (avec underscores)."""
    return _canonical(symbol).replace("/", "_")


# -------------------------------------------------------------------
# Features identiques à l'entraînement
# -------------------------------------------------------------------
def _build_features(df: pd.DataFrame, vol_window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    df = df.dropna().copy()
    feats = ["ret_1", "roll_mean", "roll_std", "rsi", "hl_range", "price_z", "volume"]
    return df[feats].copy(), df


def _send_telegram(msg: str):
    token = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id
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
    sym = _canonical(symbol)
    ohlcv = ex.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def load_model_for(symbol: str):
    path = Path("models") / f"{_storage_key(symbol)}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Aucun modèle trouvé pour {symbol} dans ./models/")
    return joblib.load(path), path


def predict_one_symbol(ex, symbol: str) -> dict:
    df = fetch_ohlcv_df(ex, symbol, settings.timeframe, settings.limit)
    X, df_full = _build_features(df, settings.vol_window)

    model, model_path = load_model_for(symbol)

    # dernière ligne
    x_last = X.iloc[-1:].copy().values

    # proba / direction
    proba_up = None
    y_hat = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_last)
        proba_up = float(proba[0, -1])
        y_hat = int(proba_up >= 0.5)
    else:
        y_val = float(model.predict(x_last)[0])
        proba_up = 1.0 / (1.0 + math.exp(-10 * y_val))
        y_hat = int(y_val >= 0)

    last_close = float(df_full["close"].iloc[-1])
    tstamp = df_full["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")

    return {
        "symbol": _canonical(symbol),
        "time": tstamp,
        "last_close": last_close,
        "p_up": proba_up,
        "direction": "UP" if y_hat == 1 else "DOWN",
        "model_file": str(model_path),  # évite le warning pydantic
    }


def main():
    ex = make_client()
    results = []
    for symbol in settings.symbols:
        try:
            r = predict_one_symbol(ex, symbol)
            results.append(r)
        except Exception as e:
            results.append({"symbol": _canonical(symbol), "error": str(e)})

    # message Telegram
    lines = [
        "*Signal quotidien (ML)*",
        f"Exchange: `***`  testnet: `{settings.use_testnet}`",
        f"Timeframe: `***`  Limit: `{settings.limit}`",
        "",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"• *{r['symbol']}* → `ERREUR`: {r['error']}")
            continue
        conf = int(round(r["p_up"] * 100))
        lines.append(
            f"• *{r['symbol']}* @ {r['time']}  🔴⬇️\n"
            if r["direction"] == "DOWN"
            else f"• *{r['symbol']}* @ {r['time']}  🟢⬆️\n"
            + f"  Direction: *{r['direction']}*  | Confiance: *{conf}%*  | Close: `{r['last_close']}`"
        )
        # réécrit proprement la ligne complète
        lines[-1] = (
            f"• *{r['symbol']}* @ {r['time']}  "
            f"{'🟢⬆️' if r['direction']=='UP' else '🔴⬇️'}\n"
            f"  Direction: *{r['direction']}*  | Confiance: *{conf}%*  | Close: `{r['last_close']}`"
        )

    msg = "\n".join(lines)
    print(msg)
    _send_telegram(msg)

    # log JSON
    Path("data").mkdir(parents=True, exist_ok=True)
    out = Path("data") / f"predictions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

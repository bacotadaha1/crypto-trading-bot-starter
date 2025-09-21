from __future__ import annotations
import math
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from src.config import settings
from src.ingestion import make_client  # client CCXT

# -------- small helpers --------
def _sym_to_fname(symbol: str) -> str:
    return symbol.replace("/", "_")

def _build_features(df: pd.DataFrame, vol_window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Features simples et robustes, cohérentes avec l'entraînement quick-start :
    - returns 1 step
    - rolling mean & std sur vol_window
    - RSI basique
    - % change high/low
    """
    df = df.copy()
    df["ret_1"] = df["close"].pct_change()
    df["roll_mean"] = df["close"].rolling(vol_window).mean()
    df["roll_std"]  = df["close"].rolling(vol_window).std()

    # RSI très simple
    delta = df["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=df.index).rolling(14).mean()
    roll_down = pd.Series(down, index=df.index).rolling(14).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    df["rsi"] = 100 - (100 / (1 + rs))

    # shape features
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    # normalisations simples
    df["price_z"] = (df["close"] - df["roll_mean"]) / (df["roll_std"].replace(0, np.nan))

    # drop na
    df = df.dropna().copy()

    feats = ["ret_1", "roll_mean", "roll_std", "rsi", "hl_range", "price_z", "volume"]
    dfX = df[feats].copy()
    return dfX, df

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
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def load_model_for(symbol: str) -> tuple[object, Path]:
    """
    Charge le modèle sauvegardé pour un symbole et retourne (model, path).
    Ex: models/BTC_USDT_model.pkl
    """
    path = Path("models") / f"{symbol.replace('/', '_')}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Aucun modèle trouvé pour {symbol} dans ./models/")
    model = joblib.load(path)
    return model, path

def predict_one_symbol(ex, symbol: str) -> dict:
    df = fetch_ohlcv_df(ex, symbol, settings.timeframe, settings.limit)
    X, df_full = _build_features(df, settings.vol_window)

    model, model_path = load_model_for(symbol)

    # on prend la dernière ligne de features
    x_last = X.iloc[-1:].copy()

    # 1) si le modèle expose predict_proba (classif binaire up/down)
    proba_up = None
    y_hat = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_last.values)
        proba_up = float(proba[0, -1])   # suppose dernière colonne = « up »
        y_hat = int(proba_up >= 0.5)
    else:
        # 2) sinon, régression : on interprète le signe comme direction
        y_val = float(model.predict(x_last.values)[0])
        proba_up = 1.0 / (1.0 + math.exp(-10 * y_val))  # squash pour pseudo-proba
        y_hat = int(y_val >= 0)

    last_close = float(df_full["close"].iloc[-1])
    tstamp = df_full["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")

    return {
        "symbol": symbol,
        "time": tstamp,
        "last_close": last_close,
        "p_up": proba_up,
        "direction": "UP" if y_hat == 1 else "DOWN",
        "model_path": str(model_path),
    }

def main():
    ex = make_client()
    results = []
    for symbol in settings.symbols:
        try:
            r = predict_one_symbol(ex, symbol)
            results.append(r)
        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})

    # message Telegram
    lines = [
        "Signal quotidien (ML)",
        f"Exchange: {settings.exchange}  testnet: {settings.use_testnet}",
        f"Timeframe: {settings.timeframe}",
        "",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"• {r['symbol']} → ERREUR: {r['error']}")
            continue
        conf = int(round(r["p_up"] * 100))
        lines.append(
            f"• {r['symbol']} @ {r['time']}  \n  Direction: {r['direction']}  | Confiance: {conf}%  | Close: {r['last_close']}"
        )

    msg = "\n".join(lines)
    print(msg)
    _send_telegram(msg)

    # log JSON si besoin
    Path("data").mkdir(parents=True, exist_ok=True)
    out = Path("data") / f"predictions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

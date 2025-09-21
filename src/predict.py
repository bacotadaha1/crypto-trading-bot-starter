# src/predict.py
from __future__ import annotations
import math
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from src.config import settings
from src.ingestion import make_client


# ---------- helpers ----------
def _exchange_symbol(exchange_id: str, human_symbol: str) -> str:
    """BTC/USDT -> BTC-USDT pour kucoin/okx/bybit, sinon laisse tel quel."""
    dash_exchanges = {"kucoin", "okx", "bybit"}
    if exchange_id in dash_exchanges:
        return human_symbol.replace("/", "-")
    return human_symbol


def _model_path(human_symbol: str) -> Path:
    """Chemin du modèle pour un symbole humain (ex: models/BTC_USDT_model.pkl)."""
    return Path("models") / f"{human_symbol.replace('/', '_')}_model.pkl"


def _build_features(df: pd.DataFrame, vol_window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    EXACTEMENT les mêmes features que dans train.py :
    ret_1, roll_mean, roll_std, rsi, hl_range, price_z, volume.
    Retourne (X, df_full_après_dropna).
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

    df = df.dropna().copy()
    feats = ["ret_1", "roll_mean", "roll_std", "rsi", "hl_range", "price_z", "volume"]
    X = df[feats].copy()
    return X, df


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


def _fetch_ohlcv_df(ex, human_symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ex_id = getattr(ex, "id", str(settings.exchange)).lower()
    ex_symbol = _exchange_symbol(ex_id, human_symbol)
    ohlcv = ex.fetch_ohlcv(ex_symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def _load_model(human_symbol: str):
    path = _model_path(human_symbol)
    if not path.exists():
        raise FileNotFoundError(f"Aucun modèle trouvé pour {human_symbol} dans ./models/")
    return joblib.load(path), path


def _predict_one(ex, human_symbol: str) -> dict:
    # 1) données récentes
    df = _fetch_ohlcv_df(ex, human_symbol, settings.timeframe, settings.limit)
    if df.empty or len(df) < 50:
        raise RuntimeError(f"Trop peu de données OHLCV pour {human_symbol}")

    # 2) features (identiques à train)
    X, df_full = _build_features(df, settings.vol_window)
    if len(X) == 0:
        raise RuntimeError(f"Aucune ligne de features après dropna pour {human_symbol}")

    # 3) charger le modèle du symbole
    model, path = _load_model(human_symbol)

    # 4) prédire la dernière bougie
    x_last = X.iloc[-1:].values

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_last)
        p_up = float(proba[0, -1])  # dernière colonne = classe "UP"
        direction = "UP" if p_up >= 0.5 else "DOWN"
    else:
        # fallback régression -> signe = direction, squash -> [0,1]
        y_val = float(model.predict(x_last)[0])
        p_up = 1.0 / (1.0 + math.exp(-10 * y_val))
        direction = "UP" if y_val >= 0 else "DOWN"

    last_close = float(df_full["close"].iloc[-1])
    tstamp = df_full["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")

    return {
        "symbol": human_symbol,
        "time": tstamp,
        "last_close": last_close,
        "p_up": p_up,
        "direction": direction,
        "model_path": str(path),
    }


def main():
    ex = make_client()

    results = []
    for sym in settings.symbols:
        try:
            results.append(_predict_one(ex, sym))
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})

    # message Telegram
    lines = [
        "*Signal quotidien (ML)*",
        f"Exchange: `{settings.exchange}`   testnet: `{settings.use_testnet}`",
        f"Timeframe: `{settings.timeframe}`   Limit: `{settings.limit}`",
        "",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"• *{r['symbol']}* → `ERREUR`: {r['error']}")
        else:
            conf = int(round(r["p_up"] * 100))
            arrow = "🟢⬆️" if r["direction"] == "UP" else "🔴⬇️"
            lines.append(
                f"• *{r['symbol']}* @ {r['time']}  {arrow}\n"
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

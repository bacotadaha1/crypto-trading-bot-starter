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

from src.config import settings
from src.ingestion import make_client
from src.alt_data import build_alt_features  # <<< NEW


# ------------------------- helpers -------------------------

def _sym_to_fname(symbol: str) -> str:
    return symbol.replace("/", "_").replace("-", "_").upper()

def _build_features(df: pd.DataFrame, vol_window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Features techniques + alt-features (sentiment, fear&greed, régime).
    """
    df = df.copy()
    # ts
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # techniques
    df["ret_1"] = df["close"].pct_change()
    df["roll_mean"] = df["close"].rolling(vol_window).mean()
    df["roll_std"]  = df["close"].rolling(vol_window).std()

    delta = df["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=df.index).rolling(14).mean()
    roll_down = pd.Series(down, index=df.index).rolling(14).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    df["rsi"] = 100 - (100 / (1 + rs))

    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["price_z"] = (df["close"] - df["roll_mean"]) / (df["roll_std"].replace(0, np.nan))

    base_feats = ["ret_1", "roll_mean", "roll_std", "rsi", "hl_range", "price_z", "volume"]
    X_base = df[["ts", *base_feats]].copy()

    # alt-features alignées
    alt = build_alt_features(df_price=df[["ts", "close", "volume"]])
    X = pd.merge(X_base.set_index("ts"), alt, left_index=True, right_index=True, how="left").dropna()

    return X, df.set_index("ts")


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
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def load_model_for(symbol: str):
    path = Path("models") / f"{_sym_to_fname(symbol)}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Aucun modèle trouvé pour {symbol} dans ./models/")
    return joblib.load(path), path


# ------------------------- prédiction -------------------------

def predict_one_symbol(ex, symbol: str) -> dict:
    df = fetch_ohlcv_df(ex, symbol, settings.timeframe, settings.limit)
    X, df_full = _build_features(df, settings.vol_window)

    model, model_path = load_model_for(symbol)

    # dernière observation prête
    x_last = X.iloc[-1:].copy().values

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_last)
        p_up = float(proba[0, -1])
        direction = "UP" if p_up >= 0.5 else "DOWN"
    else:
        y_val = float(model.predict(x_last)[0])
        p_up = 1.0 / (1.0 + math.exp(-10.0 * y_val))
        direction = "UP" if y_val >= 0 else "DOWN"

    last_close = float(df_full["close"].iloc[-1])
    tstamp = df_full.index[-1].strftime("%Y-%m-%d %H:%M UTC")

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

    limit_txt = f"  Limit: {settings.limit}" if hasattr(settings, "limit") else ""
    lines = [
        "*Signal quotidien (ML + Sentiment)*",
        f"Exchange: `{settings.exchange}`   testnet: `{settings.use_testnet}`",
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

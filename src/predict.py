# src/predict.py
from __future__ import annotations

import math
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import joblib

from src.config import settings
from src.ingestion import make_client

# ---------------------- constantes & chemins ----------------------
LOG_PATH = Path("data") / "predictions_log.csv"


# ---------------------- helpers symboles / temps ------------------
def _canonical(symbol: str) -> str:
    """Force le format CCXT avec des slashes (ex: BTC/USDT)."""
    return symbol.replace("-", "/").strip()

def _storage_key(symbol: str) -> str:
    """Version safe pour les noms de fichiers (ex: BTC_USDT)."""
    return _canonical(symbol).replace("/", "_")

def _timeframe_to_ms(tf: str) -> int:
    unit = tf[-1]
    n = int(tf[:-1])
    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 60 * 60_000
    if unit == "d":
        return n * 24 * 60 * 60_000
    raise ValueError(f"timeframe inconnu: {tf}")


# ---------------------- features (identiques au train) ------------
def _build_features(df: pd.DataFrame, vol_window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit EXACTEMENT les features utilisées à l'entraînement :
    ret_1, roll_mean, roll_std, rsi, hl_range, price_z, volume.
    Retourne (X, df_full_apres_dropna).
    """
    df = df.copy()

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

    df = df.dropna().copy()
    feats = ["ret_1", "roll_mean", "roll_std", "rsi", "hl_range", "price_z", "volume"]
    X = df[feats].copy()
    return X, df


# ---------------------- IO marché & modèles -----------------------
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


# ---------------------- Telegram ---------------------------------
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


# ---------------------- Journalisation & étiquetage ---------------
def _append_log(rows: list[dict]):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if LOG_PATH.exists():
        old = pd.read_csv(LOG_PATH)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)

def _label_yesterday_predictions(ex):
    """
    Pour chaque ligne du log sans 'actual_dir', on récupère le close du
    chandelier suivant et on marque CORRECT/INCORRECT.
    """
    if not LOG_PATH.exists():
        return
    df = pd.read_csv(LOG_PATH)
    if "actual_dir" not in df.columns:
        df["actual_dir"] = np.nan
        df["correct"] = np.nan

    timefmt = "%Y-%m-%d %H:%M UTC"
    updated = False

    for i, row in df.iterrows():
        if pd.notna(row.get("actual_dir")):
            continue  # déjà étiqueté

        sym = _canonical(str(row["symbol"]))
        try:
            t0 = datetime.strptime(str(row["time"]), timefmt).replace(tzinfo=timezone.utc)
        except Exception:
            # format inconnu -> skip
            continue

        tf = settings.timeframe
        tf_ms = _timeframe_to_ms(tf)

        # On récupère quelques barres autour de t0 pour trouver t0 et t0+1
        since_ms = int(t0.timestamp() * 1000) - tf_ms
        try:
            ohlcv = ex.fetch_ohlcv(sym, timeframe=tf, since=since_ms, limit=5)
        except Exception:
            continue
        if not ohlcv:
            continue

        closes = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        closes["ts"] = pd.to_datetime(closes["ts"], unit="ms", utc=True)

        # Index pile sur t0 si possible
        idx = closes.index[closes["ts"] == t0]
        if len(idx) == 0:
            # sinon, prendre le dernier <= t0
            mask = (closes["ts"] <= t0)
            if not mask.any():
                continue
            idx = [int(np.where(mask.to_numpy())[0].max())]

        j0 = idx[0]
        j1 = j0 + 1  # barre suivante = verdict
        if j1 >= len(closes):
            continue  # pas encore dispo

        c0 = float(closes.loc[j0, "close"])
        c1 = float(closes.loc[j1, "close"])
        actual_dir = "UP" if c1 > c0 else "DOWN"
        df.at[i, "actual_dir"] = actual_dir
        df.at[i, "correct"] = (actual_dir == row["direction"])
        updated = True

    if updated:
        df.to_csv(LOG_PATH, index=False)


# ---------------------- Prédiction par symbole --------------------
def predict_one_symbol(ex, symbol: str) -> dict:
    df = fetch_ohlcv_df(ex, symbol, settings.timeframe, settings.limit)
    if df.empty or len(df) < 50:
        raise RuntimeError(f"Trop peu de données OHLCV pour {symbol}")

    X, df_full = _build_features(df, settings.vol_window)
    if len(X) == 0:
        raise RuntimeError(f"Aucune ligne de features après dropna pour {symbol}")

    model, model_path = load_model_for(symbol)

    x_last = X.iloc[-1:].values  # dernière bougie

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_last)
        p_up = float(proba[0, -1])  # proba classe 'UP'
        direction = "UP" if p_up >= 0.5 else "DOWN"
    else:
        # fallback régression -> signe => direction, squash en [0,1]
        y_val = float(model.predict(x_last)[0])
        p_up = 1.0 / (1.0 + math.exp(-10 * y_val))
        direction = "UP" if y_val >= 0 else "DOWN"

    last_close = float(df_full["close"].iloc[-1])
    tstamp = df_full["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")

    return {
        "symbol": _canonical(symbol),
        "time": tstamp,
        "last_close": last_close,
        "p_up": p_up,
        "direction": direction,
        "model_file": str(model_path),
    }


# ---------------------- main -------------------------------------
def main():
    ex = make_client()

    # 1) Étiquette (si possible) les prédictions passées
    _label_yesterday_predictions(ex)

    # 2) Prédictions du jour
    results = []
    rows_for_log = []
    syms = settings.symbols if isinstance(settings.symbols, (list, tuple)) else str(settings.symbols).split(",")
    for symbol in [s.strip() for s in syms if str(s).strip()]:
        try:
            r = predict_one_symbol(ex, symbol)
            results.append(r)
            # préparer ligne pour le journal
            rows_for_log.append({
                "symbol": r["symbol"],
                "time": r["time"],
                "direction": r["direction"],
                "p_up": r["p_up"],
                "close_at_pred": r["last_close"],
                "model_file": r["model_file"],
            })
        except Exception as e:
            results.append({"symbol": _canonical(symbol), "error": str(e)})

    # 3) Message Telegram
    lines = [
        "*Signal quotidien (ML)*",
        f"Exchange: `{settings.exchange}`   testnet: `{settings.use_testnet}`",
        f"Timeframe: `{settings.timeframe}`   Limit: `{settings.limit}`",
        "",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"• *{r['symbol']}* → `ERREUR`: {r['error']}")
            continue
        conf = int(round(r["p_up"] * 100))
        arrow = "🟢⬆️" if r["direction"] == "UP" else "🔴⬇️"
        lines.append(
            f"• *{r['symbol']}* @ {r['time']}  {arrow}\n"
            f"  Direction: *{r['direction']}*  | Confiance: *{conf}%*  | Close: `{r['last_close']}`"
        )

    msg = "\n".join(lines)
    print(msg)
    _send_telegram(msg)

    # 4) Log JSON + journal CSV
    Path("data").mkdir(parents=True, exist_ok=True)
    out = Path("data") / f"predictions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if rows_for_log:
        _append_log(rows_for_log)


if __name__ == "__main__":
    main()

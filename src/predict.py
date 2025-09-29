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
from src.alt_data import build_alt_features

# ---- helpers ----
def _sym_to_fname(symbol: str) -> str:
    return symbol.replace("/", "_").replace("-", "_").upper()

def _build_tech_features(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    d = df.copy()
    d["ret_1"] = d["close"].pct_change()
    d["roll_mean"] = d["close"].rolling(vol_window, min_periods=max(2, vol_window//2)).mean()
    d["roll_std"]  = d["close"].rolling(vol_window, min_periods=max(2, vol_window//2)).std()
    delta = d["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    r_up = pd.Series(up, index=d.index).rolling(14, min_periods=7).mean()
    r_dn = pd.Series(down, index=d.index).rolling(14, min_periods=7).mean()
    rs = r_up / r_dn.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))
    d["hl_range"] = (d["high"] - d["low"]) / d["close"].replace(0, np.nan)
    d["price_z"] = (d["close"] - d["roll_mean"]) / d["roll_std"].replace(0, np.nan)
    return d[["ts","ret_1","roll_mean","roll_std","rsi","hl_range","price_z","volume"]]

def _build_all_features(df_price: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    tech = _build_tech_features(df_price, vol_window).dropna().copy()
    alt = build_alt_features(df_price[["ts","close","volume"]])
    tech["ts"] = pd.to_datetime(tech["ts"], utc=True)
    tech = tech.set_index("ts").sort_index()
    all_feats = tech.join(alt, how="left")

    # créer colonnes alt manquantes si besoin
    for col in ["sent_hour","fg_idx","rv_w6","rv_w36","vol_z"]:
        if col not in all_feats.columns:
            all_feats[col] = np.nan
    # ffill puis replace na restants par 0 (alt-data seulement)
    all_feats[["sent_hour","fg_idx","rv_w6","rv_w36","vol_z"]] = (
        all_feats[["sent_hour","fg_idx","rv_w6","rv_w36","vol_z"]].ffill().fillna(0.0)
    )
    return all_feats

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
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def load_model_for(symbol: str):
    path = Path("models") / f"{_sym_to_fname(symbol)}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Aucun modèle trouvé pour {symbol} dans ./models/")
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        feat_cols = list(obj.get("features", []))
        vol_window = int(obj.get("vol_window", getattr(settings, "vol_window", 12)))
        return model, feat_cols, vol_window, path
    # fallback anciens modèles: 7 features techniques
    return obj, ["ret_1","roll_mean","roll_std","rsi","hl_range","price_z","volume"], int(getattr(settings, "vol_window", 12)), path

def _proba_up_from_model(model, X_last: np.ndarray) -> float:
    try:
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X_last)[0, -1])
        if hasattr(model, "decision_function"):
            z = float(model.decision_function(X_last)[0])
            return 1.0 / (1.0 + math.exp(-z))
        yv = float(model.predict(X_last)[0])
        return 1.0 / (1.0 + math.exp(-2.5 * yv))
    except Exception as e:
        print("[WARN] proba_from_model:", e)
        return 0.5

# ---- prediction ----
def predict_one_symbol(ex, symbol: str) -> dict:
    dfp = fetch_ohlcv_df(ex, symbol, settings.timeframe, settings.limit)

    model, feat_cols_expected, vol_window, model_path = load_model_for(symbol)
    feats_all = _build_all_features(dfp, vol_window)

    # s'assurer que TOUTES les colonnes attendues existent
    for col in feat_cols_expected:
        if col not in feats_all.columns:
            feats_all[col] = 0.0

    X = feats_all[feat_cols_expected].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if len(X) == 0:
        raise ValueError(f"Pas assez d’observations pour construire {len(feat_cols_expected)} features.")

    x_last = X.iloc[-1:].to_numpy(dtype=float)
    p_up = _proba_up_from_model(model, x_last)

    # clip doux pour éviter 0/1 affichés si le modèle dérape
    eps = float(getattr(settings, "proba_clip_eps", 0.01))
    p_up = float(np.clip(p_up, eps, 1.0 - eps))

    direction = "UP" if p_up >= 0.5 else "DOWN"
    last_close = float(dfp["close"].iloc[-1])
    tstamp = dfp["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")

    return {
        "symbol": symbol,
        "time": tstamp,
        "last_close": last_close,
        "p_up": p_up,
        "direction": direction,
        "model_path": str(model_path),
        "used_features": feat_cols_expected,
    }

def main():
    ex = make_client()
    symbols = settings.symbols
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]

    results: list[dict] = []
    for symbol in symbols:
        try:
            results.append(predict_one_symbol(ex, symbol))
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
        else:
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
                "ts_utc": ts_now, "exchange": settings.exchange, "timeframe": settings.timeframe,
                "symbol": r.get("symbol"), "last_close": None, "p_up": None, "direction": None, "error": r.get("error"),
            })
        else:
            rows.append({
                "ts_utc": ts_now, "exchange": settings.exchange, "timeframe": settings.timeframe,
                "symbol": r["symbol"], "last_close": r["last_close"], "p_up": r["p_up"],
                "direction": r["direction"], "error": None,
            })
    if rows:
        write_header = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header: w.writeheader()
            w.writerows(rows)

if __name__ == "__main__":
    main()

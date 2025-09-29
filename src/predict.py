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
from src.ingestion import make_client                 # client CCXT
from src.alt_data import build_alt_features           # sentiment + regime features


# ============================ small utils ============================

def _sym_to_fname(symbol: str) -> str:
    return symbol.replace("/", "_").replace("-", "_").upper()

def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))

def _inv_logit(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

def _apply_temperature(p: float, T: float) -> float:
    """Adoucit une proba via temperature scaling (T>1 => proba moins extrêmes)."""
    return float(_inv_logit(_logit(p) / max(T, 1e-6)))


# ============================ features ============================

def _build_tech_features(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """
    7 features techniques alignées sur ts (UTC).
    """
    d = df.copy()
    d["ret_1"] = d["close"].pct_change()
    d["roll_mean"] = d["close"].rolling(vol_window, min_periods=vol_window // 2 or 1).mean()
    d["roll_std"]  = d["close"].rolling(vol_window, min_periods=vol_window // 2 or 1).std()

    delta = d["close"].diff()
    up   = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    r_up = pd.Series(up, index=d.index).rolling(14, min_periods=7).mean()
    r_dn = pd.Series(down, index=d.index).rolling(14, min_periods=7).mean()
    rs = r_up / r_dn.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))

    d["hl_range"] = (d["high"] - d["low"]) / d["close"].replace(0, np.nan)
    d["price_z"]  = (d["close"] - d["roll_mean"]) / d["roll_std"].replace(0, np.nan)

    tech = d[["ts","ret_1","roll_mean","roll_std","rsi","hl_range","price_z","volume"]].copy()
    tech["ts"] = pd.to_datetime(tech["ts"], utc=True)
    return tech.dropna()


def _build_all_features(df_price: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """
    Construit le *set complet* (12 colonnes possibles) indexé par ts:
      - 7 techniques: ret_1, roll_mean, roll_std, rsi, hl_range, price_z, volume
      - 5 alt-data  : sent_hour, fg_idx, rv_w6, rv_w36, vol_z
    """
    tech = _build_tech_features(df_price, vol_window).set_index("ts").sort_index()

    # alt features (déjà indexées par ts)
    alt = build_alt_features(df_price[["ts", "close", "volume"]])

    feats = tech.join(alt, how="left")
    for col in ["sent_hour", "fg_idx", "rv_w6", "rv_w36", "vol_z"]:
        if col not in feats.columns:
            feats[col] = np.nan

    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(method="ffill")
    feats = feats.fillna(0.0)
    return feats


# ============================ IO helpers ============================

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
    """
    Retourne (estimator, feat_cols, vol_window, temperature, path).
    Compatible nouveaux modèles (dict avec clés) et anciens (estimator seul).
    """
    path = Path("models") / f"{_sym_to_fname(symbol)}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Aucun modèle trouvé pour {symbol} dans ./models/")

    obj = joblib.load(path)

    # Nouveau format (dict)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        feat_cols = list(obj.get("features", []))
        vol_window = int(obj.get("vol_window", getattr(settings, "vol_window", 12)))
        temperature = float(obj.get("temperature", 1.0))
        if not feat_cols:
            feat_cols = [
                "ret_1","roll_mean","roll_std","rsi","hl_range","price_z","volume",
                "sent_hour","fg_idx","rv_w6","rv_w36","vol_z"
            ]
        return model, feat_cols, vol_window, temperature, path

    # Ancien format
    legacy_cols = ["ret_1","roll_mean","roll_std","rsi","hl_range","price_z","volume"]
    return obj, legacy_cols, int(getattr(settings, "vol_window", 12)), 1.0, path


def _proba_up_from_model(model, X_last: np.ndarray, temperature: float = 1.0) -> float:
    """
    Proba d'UP avec température et léger clipping pour éviter 0/1 exacts.
    """
    try:
        if hasattr(model, "predict_proba"):
            p = float(model.predict_proba(X_last)[0, -1])
        elif hasattr(model, "decision_function"):
            z = float(model.decision_function(X_last)[0])
            p = _inv_logit(z)
        else:
            yv = float(model.predict(X_last)[0])
            p = _inv_logit(2.5 * yv)
    except Exception as e:
        print("[WARN] proba_from_model:", e)
        p = 0.5

    p = _apply_temperature(p, temperature)
    p = float(np.clip(p, 0.02, 0.98))  # bornes douces
    return p


# ============================ core ============================

def predict_one_symbol(ex, symbol: str) -> dict:
    # 1) Prix bruts
    df_price = fetch_ohlcv_df(ex, symbol, settings.timeframe, settings.limit)

    # 2) Modèle + colonnes attendues
    model, feat_cols_expected, vol_window, temperature, model_path = load_model_for(symbol)

    # 3) Features complètes -> sélection des colonnes du modèle
    feats_all = _build_all_features(df_price, vol_window)

    for col in feat_cols_expected:
        if col not in feats_all.columns:
            feats_all[col] = 0.0

    X = feats_all[feat_cols_expected].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if len(X) == 0:
        raise ValueError(f"Pas assez d’observations pour construire {len(feat_cols_expected)} features.")

    # 4) Prédiction sur la dernière ligne
    x_last = X.iloc[-1:].to_numpy(dtype=float)
    p_up = _proba_up_from_model(model, x_last, temperature=temperature)
    direction = "UP" if p_up >= 0.5 else "DOWN"

    last_close = float(df_price["close"].iloc[-1])
    tstamp = df_price["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")

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

    # s'assurer que settings.symbols soit une liste
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

    # --- message Telegram
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
        conf = max(1, min(conf, 99))  # jamais 0/100 visuellement
        emoji = "🟢⬆️" if r["direction"] == "UP" else "🔴⬇️"
        lines.append(
            f"• *{r['symbol']}* @ {r['time']} {emoji}\n"
            f"  Direction: *{r['direction']}*  | Confiance: *{conf}%*  | Close: `{r['last_close']}`"
        )

    msg = "\n".join(lines)
    print(msg)
    _send_telegram(msg)

    # --- log JSON (facultatif)
    Path("data").mkdir(parents=True, exist_ok=True)
    out_json = Path("data") / f"predictions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # --- journalisation CSV (pour l'évaluation/diagnostic)
    log_path = Path("data") / "preds_log.csv"
    rows = []
    ts_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for r in results:
        if "error" in r:
            rows.append({
                "ts_utc": ts_now, "exchange": settings.exchange, "timeframe": settings.timeframe,
                "symbol": r.get("symbol"), "last_close": None, "p_up": None,
                "direction": None, "error": r.get("error"),
            })
        else:
            rows.append({
                "ts_utc": ts_now, "exchange": settings.exchange, "timeframe": settings.timeframe,
                "symbol": r["symbol"], "last_close": r["last_close"],
                "p_up": r["p_up"], "direction": r["direction"], "error": None,
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

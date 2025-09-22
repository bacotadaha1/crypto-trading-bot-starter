# src/evaluate.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import settings
from src.ingestion import make_client


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


def _load_log(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        print("Aucun data/preds_log.csv — rien à évaluer.")
        return None
    df = pd.read_csv(path)
    # normalise les noms de colonnes
    df.columns = [c.strip().lower() for c in df.columns]

    # harmonise le nom de la colonne 'symbol'
    if "symbol" not in df.columns:
        for alt in ("pair", "ticker", "market", "instrument"):
            if alt in df.columns:
                df = df.rename(columns={alt: "symbol"})
                break
    if "symbol" not in df.columns:
        print(f"Colonnes disponibles: {list(df.columns)}")
        raise KeyError(
            "Le fichier preds_log.csv ne contient pas de colonne 'symbol'. "
            "Vérifie l’écriture du log dans src/predict.py."
        )

    # parse timestamp
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    elif "time" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    else:
        df["ts_utc"] = pd.NaT

    # nettoie les lignes complètement vides
    df = df.dropna(subset=["symbol", "ts_utc"], how="any")
    return df


def _next_close_for(ex, symbol: str, timeframe: str, ref_ts: pd.Timestamp) -> float | None:
    """
    Récupère le close de la bougie **suivante** après ref_ts.
    """
    limit = 5
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    d = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    d["ts"] = pd.to_datetime(d["ts"], unit="ms", utc=True)
    # trouve la première bougie strictement > ref_ts
    nxt = d[d["ts"] > ref_ts].head(1)
    if len(nxt) == 0:
        return None
    return float(nxt["close"].iloc[0])


def _compute_metrics(df: pd.DataFrame, days: int) -> tuple[str, dict]:
    """
    Filtre sur les N derniers jours et calcule la précision globale + par symbole.
    """
    since = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=days)
    recent = df[df["ts_utc"] >= since].copy()
    if recent.empty:
        return f"Aucune prédiction dans les {days} derniers jours.", {}

    # client exchange une seule fois
    ex = make_client()

    # on s'assure que les champs nécessaires existent
    for col in ("p_up", "direction", "last_close"):
        if col not in recent.columns:
            recent[col] = np.nan

    results = []
    for _, row in recent.iterrows():
        sym = str(row["symbol"])
        ref_ts = row["ts_utc"]
        pred_dir = str(row.get("direction", "")).upper()
        try:
            last_close = float(row.get("last_close"))
        except Exception:
            last_close = np.nan

        close_next = None
        ok = None
        err = None
        try:
            close_next = _next_close_for(ex, sym, settings.timeframe, ref_ts)
            if close_next is not None and not np.isnan(last_close):
                realized = "UP" if close_next >= last_close else "DOWN"
                ok = (realized == pred_dir)
            else:
                ok = None
        except Exception as e:
            err = str(e)

        results.append({
            "symbol": sym,
            "ts_utc": ref_ts,
            "pred_dir": pred_dir,
            "last_close": last_close,
            "close_next": close_next,
            "ok": ok,
            "error": err,
        })

    resdf = pd.DataFrame(results)

    # précision globale (ignore None)
    valid = resdf[resdf["ok"].notna()]
    global_acc = float(valid["ok"].mean()) if not valid.empty else np.nan

    # précision par symbole
    by_sym = {}
    for sym, grp in valid.groupby("symbol"):
        by_sym[sym] = float(grp["ok"].mean())

    # message
    lines = [f"*Évaluation sur {days} jour(s)*",
             f"Précision globale: *{0 if np.isnan(global_acc) else round(global_acc*100):.0f}%*",
             ""]
    if by_sym:
        for sym, acc in sorted(by_sym.items()):
            lines.append(f"• {sym}: *{round(acc*100):.0f}%*")
    else:
        lines.append("_Pas assez de données valides pour calculer une précision._")

    return "\n".join(lines), by_sym


def main():
    days = int(os.getenv("EVAL_DAYS", "7"))
    log_path = Path("data") / "preds_log.csv"
    df = _load_log(log_path)
    if df is None or df.empty:
        return

    msg, _ = _compute_metrics(df, days)
    print(msg)
    _send_telegram(msg)


if __name__ == "__main__":
    main()

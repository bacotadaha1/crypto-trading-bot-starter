from __future__ import annotations
import math
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from src.config import settings
from src.ingestion import make_client

def _fetch_last_two(ex, symbol: str, timeframe: str):
    """Récupère les 2 dernières bougies pour connaître la bougie suivante."""
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=2)
    if not ohlcv or len(ohlcv) < 2:
        return None
    # [ [ts, o, h, l, c, v], [ts, ...] ]
    return ohlcv[-2], ohlcv[-1]

def evaluate_last_predictions():
    log_path = Path("data") / "preds_log.csv"
    if not log_path.exists():
        print("Aucun preds_log.csv — rien à évaluer.")
        return None, []

    # on ne prend que le dernier “batch” (même timestamp le plus récent)
    df = pd.read_csv(log_path)
    if df.empty:
        print("preds_log.csv vide.")
        return None, []

    # dernier timestamp d’écriture
    last_ts = df["ts_utc"].iloc[-1]
    batch = df[df["ts_utc"] == last_ts].copy()

    ex = make_client()
    timeframe = str(batch["timeframe"].iloc[0])

    report_rows = []
    n_ok, n_tot = 0, 0

    for _, r in batch.iterrows():
        symbol = r["symbol"]
        if pd.isna(symbol) or r.get("error"):
            report_rows.append((symbol, "SKIP", None, None, "error during predict"))
            continue

        candles = _fetch_last_two(ex, symbol, timeframe)
        if not candles:
            report_rows.append((symbol, "SKIP", None, None, "no candles"))
            continue

        prev_c, next_c = candles[0][4], candles[1][4]
        realized_up = int(next_c >= prev_c)  # 1 si hausse, 0 si baisse
        pred_up = 1 if str(r["direction"]).upper() == "UP" else 0

        correct = int(pred_up == realized_up)
        n_ok += correct; n_tot += 1

        report_rows.append((
            symbol,
            "OK" if correct else "MISS",
            float(r["p_up"]) if not pd.isna(r["p_up"]) else None,
            float(next_c),
            None
        ))

    acc = (n_ok / n_tot) if n_tot > 0 else None
    return acc, report_rows

def _send_telegram(msg: str):
    token = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id
    if not token or not chat_id:
        print("[INFO] Telegram non configuré. Message:\n", msg); return
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=data, timeout=20)
        if r.status_code != 200:
            print("[WARN] Telegram status:", r.status_code, r.text)
    except Exception as e:
        print("[WARN] Telegram error:", e)

def main():
    acc, rows = evaluate_last_predictions()
    lines = []
    lines.append("*Bilan prédictions d’hier*")
    lines.append(f"Exchange: `{settings.exchange}`   timeframe: `{settings.timeframe}`")
    lines.append("")

    if acc is None:
        lines.append("_Aucune donnée à évaluer._")
    else:
        lines.append(f"Accuracy: *{round(acc*100):d}%*")
        for (sym, status, p_up, next_close, note) in rows:
            if status == "SKIP":
                lines.append(f"• *{sym}* → `SKIP` ({note})")
            else:
                conf = f"{int(round(p_up*100))}%" if p_up is not None else "—"
                emoji = "✅" if status == "OK" else "❌"
                lines.append(f"• *{sym}* → {emoji}  | Confiance: `{conf}` | Next close: `{next_close}`")

    msg = "\n".join(lines)
    print(msg)
    _send_telegram(msg)

if __name__ == "__main__":
    main()

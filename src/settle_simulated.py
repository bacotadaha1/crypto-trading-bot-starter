# src/settle_simulated.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import csv

import pandas as pd
import numpy as np

from src.config import settings
from src.ingestion import make_client
from src.strategy import compute_atr
from src.predict import fetch_ohlcv_df

ORDERS = Path("data") / "orders_log.csv"
FILLS  = Path("data") / "fills_log.csv"

def _read_orders() -> list[dict]:
    if not ORDERS.exists():
        return []
    out = []
    with ORDERS.open("r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            out.append(row)
    return out

def _append_fill(row: dict):
    FILLS.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FILLS.exists()
    with FILLS.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)

def _as_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _parse_ts(ts_str: str):
    # ts_utc = "2025-09-24T06:12:00+00:00" ou "2025-09-24 06:12:00"
    try:
        return datetime.fromisoformat(ts_str.replace("Z","+00:00")).astimezone(timezone.utc)
    except Exception:
        return None

def settle_one(ex, o: dict):
    symbol = o["symbol"]
    side   = o["side"]
    price  = _as_float(o["price"], 0.0) if "price" in o else None  # dans nos logs initiaux on n’avait pas la colonne price
    qty    = _as_float(o["qty"], 0.0)
    sl     = _as_float(o["sl"], 0.0)
    tp     = _as_float(o["tp"], 0.0)
    ts     = _parse_ts(o.get("ts_utc", ""))

    if qty <= 0 or sl <= 0 or tp <= 0 or ts is None:
        return

    # on récupère les bougies APRÈS l’entrée pour vérifier si tp/sl touché
    # on prend une marge de 20 bougies (≈3,3 jours sur 4h)
    df = fetch_ohlcv_df(ex, symbol, settings.timeframe, limit=settings.limit)
    df = df.sort_values("ts").reset_index(drop=True)
    # on cherche la 1ère bougie APRÈS l’entrée
    idx = df.index[df["ts"] > ts]
    if len(idx) == 0:
        return
    start = int(idx.min())
    look_ahead = df.iloc[start:start+20].copy()
    if look_ahead.empty:
        return

    outcome = "open"
    exit_ts = None
    exit_px = None

    for _, r in look_ahead.iterrows():
        h = float(r["high"]); l = float(r["low"])
        if side == "buy":
            if l <= sl:
                outcome = "stopped"
                exit_px = sl
                exit_ts = r["ts"]
                break
            if h >= tp:
                outcome = "takeprofit"
                exit_px = tp
                exit_ts = r["ts"]
                break
        else:  # sell
            if h >= sl:
                outcome = "stopped"
                exit_px = sl
                exit_ts = r["ts"]
                break
            if l <= tp:
                outcome = "takeprofit"
                exit_px = tp
                exit_ts = r["ts"]
                break

    if outcome == "open":
        # position toujours ouverte => on marque à la dernière clôture observée
        exit_ts = look_ahead["ts"].iloc[-1]
        exit_px = float(look_ahead["close"].iloc[-1])
        outcome = "mark"

    # PnL (en unités quote, ex: USDT)
    if side == "buy":
        pnl = (exit_px - price) * qty if price else 0.0
    else:
        pnl = (price - exit_px) * qty if price else 0.0

    row = {
        "symbol": symbol,
        "side": side,
        "entry_ts": ts.isoformat(timespec="seconds"),
        "entry_price": round(price or 0.0, 6),
        "qty": round(qty, 6),
        "exit_ts": exit_ts.isoformat() if isinstance(exit_ts, pd.Timestamp) else str(exit_ts),
        "exit_price": round(exit_px or 0.0, 6),
        "outcome": outcome,
        "pnl_usdt": round(float(pnl), 4),
    }
    _append_fill(row)

def main():
    ex = make_client()
    orders = _read_orders()
    if not orders:
        print("Aucun ordre à régler.")
        return
    for o in orders[-30:]:   # ne traite que les plus récents pour éviter les doublons massifs
        try:
            settle_one(ex, o)
        except Exception as e:
            print("Settle error", o.get("symbol"), e)

if __name__ == "__main__":
    main()

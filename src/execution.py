# src/execution.py
from __future__ import annotations
from pathlib import Path
import csv
from datetime import datetime, timezone
from src.ingestion import make_client
from src.config import settings

def place_or_log(order):
    Path("data").mkdir(exist_ok=True, parents=True)
    row = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbol": order.symbol,
        "side": order.side,
        "qty": round(order.qty, 6),
        "sl": round(order.sl, 6),
        "tp": round(order.tp, 6),
        "live": bool(getattr(settings, "trade_enabled", False)),
    }
    with open("data/orders_log.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if f.tell() == 0: w.writeheader()
        w.writerow(row)

    if not getattr(settings, "trade_enabled", False):
        return "SIMULATED"

    ex = make_client()  # CCXT configuré via settings
    # market order simple (testnet d’abord)
    params = {}
    if order.side == "buy":
        res = ex.create_market_buy_order(order.symbol, order.qty, params)
    else:
        res = ex.create_market_sell_order(order.symbol, order.qty, params)
    # SL/TP: ajouter ensuite via OCO si l’exchange le supporte, sinon surveiller côté bot
    return "PLACED"

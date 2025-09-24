# src/portfolio.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import math

RISK_PER_TRADE = 0.006   # 0.6% du capital
SL_ATR = 2.5
TP_ATR = 3.5

@dataclass
class OrderPlan:
    symbol: str
    side: str
    qty: float
    sl: float
    tp: float

def plan_order(signal, equity_usdt: float) -> OrderPlan | None:
    if signal.side == "flat" or signal.atr <= 0:
        return None
    risk_value = equity_usdt * RISK_PER_TRADE
    stop_dist = signal.atr * SL_ATR
    if stop_dist <= 0:
        return None
    qty = risk_value / stop_dist
    qty = max(qty, 0.0)
    if qty == 0:
        return None

    if signal.side == "buy":
        sl = signal.price - stop_dist
        tp = signal.price + signal.atr * TP_ATR
    else:
        sl = signal.price + stop_dist
        tp = signal.price - signal.atr * TP_ATR
    return OrderPlan(signal.symbol, signal.side, qty, sl, tp)

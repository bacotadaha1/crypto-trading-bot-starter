# src/action_strategy.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Signal:
    symbol: str
    side: str         # "buy" | "sell" | "flat"
    p_up: float
    sent_hour: float
    fg_idx: float
    price: float
    atr: float

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    d = df.copy()
    d["tr1"] = d["high"] - d["low"]
    d["tr2"] = (d["high"] - d["close"].shift()).abs()
    d["tr3"] = (d["low"] - d["close"].shift()).abs()
    d["tr"]  = d[["tr1","tr2","tr3"]].max(axis=1)
    return float(d["tr"].rolling(period).mean().iloc[-1])

def decide_action(symbol: str, last_row: pd.Series) -> Signal:
    # on borne la proba (affichage et robustesse)
    p_up = float(np.clip(last_row.get("p_up", 0.5), 0.05, 0.95))
    sent = float(last_row.get("sent_hour", 0.0))
    fg   = float(last_row.get("fg_idx", 50.0))
    price= float(last_row.get("close"))
    atr  = float(last_row.get("atr", 0.0))

    # règles simples et lisibles
    side = "flat"
    if p_up >= 0.58 and sent > 0 and fg >= 45:
        side = "buy"
    elif p_up <= 0.42 and sent < 0 and fg <= 55:
        side = "sell"

    return Signal(symbol, side, p_up, sent, fg, price, atr)

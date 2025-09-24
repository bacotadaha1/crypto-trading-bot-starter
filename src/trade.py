# src/trade.py
from __future__ import annotations
import numpy as np
import pandas as pd

from src.config import settings
from src.ingestion import make_client
from src.alt_data import build_alt_features
from src.predict import fetch_ohlcv_df, load_model_for, _proba_up_from_model, _build_all_features

from src.action_strategy import compute_atr, decide_action   # ← NEW: on importe ici
from src.portfolio import plan_order
from src.execution import place_or_log

def _symbols_list():
    syms = settings.symbols
    if isinstance(syms, str):
        return [s.strip() for s in syms.split(",") if s.strip()]
    return list(syms)

def main():
    ex = make_client()
    symbols = _symbols_list()
    equity_usdt = float(getattr(settings, "equity_usdt", 1000))  # capital fictif pour DRY-RUN

    for sym in symbols:
        # 1) données récentes
        dfp = fetch_ohlcv_df(ex, sym, settings.timeframe, settings.limit)
        atr = compute_atr(dfp.copy(), 14)

        # 2) features & proba (exactement comme predict)
        model, cols, volw, _ = load_model_for(sym)
        feats = _build_all_features(dfp, volw)
        for c in cols:
            if c not in feats.columns:
                feats[c] = 0.0
        X = feats[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x_last = X.iloc[-1:].to_numpy(dtype=float)
        p_up = _proba_up_from_model(model, x_last)

        # 3) ligne finale pour décider
        last = feats.iloc[-1:].copy()
        last["p_up"] = p_up
        last["close"] = dfp["close"].iloc[-1]
        last["atr"] = atr

        sig = decide_action(sym, last.iloc[0])
        plan = plan_order(sig, equity_usdt)
        if plan:
            status = place_or_log(plan)   # "SIMULATED" si TRADE_ENABLED=false
            print(sym, sig.side, plan.qty, status)
        else:
            print(sym, "flat")

if __name__ == "__main__":
    main()

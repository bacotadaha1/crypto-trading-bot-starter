# src/trade.py
from __future__ import annotations
import pandas as pd
from src.config import settings
from src.ingestion import make_client
from src.alt_data import build_alt_features
from src.strategy import compute_atr, decide_action
from src.portfolio import plan_order
from src.execution import place_or_log
from src.predict import fetch_ohlcv_df, load_model_for, _build_all_features, _proba_up_from_model

def main():
    ex = make_client()
    symbols = settings.symbols if isinstance(settings.symbols, list) else [s.strip() for s in settings.symbols.split(",")]
    equity_usdt = float(getattr(settings, "equity_usdt", 1000))  # valeur fictive au début

    for sym in symbols:
        # données récentes
        dfp = fetch_ohlcv_df(ex, sym, settings.timeframe, settings.limit)
        atr = compute_atr(dfp.copy(), 14)

        # features & proba (même que predict)
        model, cols, volw, _ = load_model_for(sym)
        feats = _build_all_features(dfp, volw)
        for c in cols:
            if c not in feats.columns: feats[c] = 0.0
        X = feats[cols].fillna(0.0)
        x_last = X.iloc[-1:].to_numpy(float)
        p_up = _proba_up_from_model(model, x_last)

        last = feats.iloc[-1:].copy()
        last["p_up"] = p_up
        last["close"] = dfp["close"].iloc[-1]
        last["atr"] = atr

        sig = decide_action(sym, last.iloc[0])
        plan = plan_order(sig, equity_usdt)
        if plan:
            status = place_or_log(plan)
            print(sym, sig.side, plan.qty, status)
        else:
            print(sym, "flat")

if __name__ == "__main__":
    main()

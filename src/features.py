import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from .config import settings

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    r = df["close"].pct_change()
    df["ret_s"] = df["close"].pct_change(settings.ret_short)
    df["ret_l"] = df["close"].pct_change(settings.ret_long)
    df["vol_l"] = r.rolling(settings.vol_window).std() * np.sqrt(365*6)  # ~6 bougies 4h/jour
    df["rsi14"] = RSIIndicator(df["close"], 14).rsi()
    return df.dropna()

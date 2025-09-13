import pandas as pd
import numpy as np
from .config import settings

def momentum_signal(df: pd.DataFrame) -> pd.Series:
    score = 0.6*np.sign(df["ret_l"]) + 0.4*np.sign(df["ret_s"])
    return (score > 0).astype(int)

def vol_target_sizing(df: pd.DataFrame) -> pd.Series:
    vol = df["close"].pct_change().rolling(settings.vol_window).std()
    ann = (365*6) ** 0.5
    size = (settings.target_vol_annual / (vol*ann).replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    return size

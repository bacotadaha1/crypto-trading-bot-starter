import pandas as pd
from .config import settings
from .ingestion import make_client
from .features import build_features
from .strategy import momentum_signal, vol_target_sizing

class Trader:
    def __init__(self):
        # crée immédiatement le client CCXT
        self.ex = make_client()

    def compute_signal(self, symbol: str):
        raw = self.ex.fetch_ohlcv(symbol, timeframe=settings.timeframe, limit=settings.limit)
        if not raw or len(raw) == 0:
            raise ValueError(f"Aucune donnée OHLCV pour {symbol} (timeframe={settings.timeframe}).")

        df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)

        feat = build_features(df.copy())
        if feat.empty:
            raise ValueError(f"Features vides pour {symbol} (données insuffisantes).")

        sig = int(momentum_signal(feat).iloc[-1])
        size = float(vol_target_sizing(feat).iloc[-1])
        px = float(df["close"].iloc[-1])
        return df, sig, size, px

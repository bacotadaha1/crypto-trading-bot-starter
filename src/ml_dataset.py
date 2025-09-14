import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from src.config import settings
from src.features import build_features

def _make_client():
    ex_cls = getattr(ccxt, settings.exchange)
    ex = ex_cls({"enableRateLimit": True})
    return ex

def _since_ms(years=4):
    dt = datetime.now(timezone.utc) - timedelta(days=365*years)
    return int(dt.timestamp() * 1000)

def fetch_ohlcv_range(ex, symbol: str, timeframe: str, since_ms: int, limit=1000):
    all_rows = []
    since = since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < limit:
            break
        since = batch[-1][0] + 1
        if len(all_rows) > 200_000:
            break
    df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts").sort_index()

def build_labeled_frame(df: pd.DataFrame, horizon=1):
    feat = build_features(df.copy())
    fut_ret = feat["close"].pct_change().shift(-horizon)
    feat["future_return"] = fut_ret
    feat = feat.dropna()
    return feat

def create_dataset(symbols=None, years=4, horizon=1, timeframe=None, out_csv=None):
    symbols = symbols or settings.symbols
    timeframe = timeframe or settings.timeframe
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_csv or str(data_dir / "training_data.csv")

    ex = _make_client()
    frames = []
    since = _since_ms(years=years)
    for sym in symbols:
        try:
            df = fetch_ohlcv_range(ex, sym, timeframe, since)
            if df.empty:
                print(f"[WARN] Pas de données pour {sym}")
                continue
            lab = build_labeled_frame(df, horizon=horizon).assign(symbol=sym)
            frames.append(lab)
            print(f"[OK] {sym}: {len(lab)} lignes")
        except Exception as e:
            print(f"[WARN] {sym} KO: {e}")

    if not frames:
        raise RuntimeError("Aucune donnée ML construite.")
    all_df = pd.concat(frames).sort_index()
    all_df.to_csv(out_csv, index=True)
    print(f"[DONE] Dataset ML -> {out_csv} ({len(all_df)} lignes)")

if __name__ == "__main__":
    create_dataset()


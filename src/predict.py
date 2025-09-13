import os
import pandas as pd
from pathlib import Path
from src.config import settings
from src.ingestion import make_client
from src.features import build_features
from src.strategy import momentum_signal, vol_target_sizing
from src.notify import send_telegram
import joblib

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.pkl"))
USE_ML = MODEL_PATH.exists()
ML_META = None
if USE_ML:
    try:
        ML_META = joblib.load(MODEL_PATH)
    except Exception:
        ML_META = None
        USE_ML = False

def predict_once():
    ex = make_client()
    results = []
    for sym in settings.symbols:
        try:
            raw = ex.fetch_ohlcv(sym, timeframe=settings.timeframe, limit=settings.limit)
            df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df.set_index("ts", inplace=True)

            feat = build_features(df.copy())
            px = float(feat["close"].iloc[-1])

            if USE_ML and ML_META is not None:
                features = ML_META.get("features", ["ret_s","ret_l","vol_l","rsi14"])
                row = feat[features].replace([float("inf"), float("-inf")], 0.0).fillna(0.0).iloc[-1]
                proba = ML_META["model"].predict_proba([row.values])[0][1]
                sig = int(proba >= 0.55)
            else:
                sig = int(momentum_signal(feat).iloc[-1])

            size = float(vol_target_sizing(feat).iloc[-1])
            results.append({"symbol": sym, "signal": sig, "size": size, "price": px})
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})

    outdir = Path(settings.data_dir) / "predictions"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"pred_{pd.Timestamp.utcnow().strftime('%Y-%m-%d')}.csv"
    pd.DataFrame(results).to_csv(outfile, index=False)

    tag = "ML" if USE_ML and ML_META is not None else "RULES"
    lines = []
    for r in results:
        if "error" in r:
            lines.append(f"{r['symbol']}: ERROR {r['error']}")
        else:
            lines.append(f"{r['symbol']}: signal={r['signal']} size={r['size']:.4f} px={r['price']}")
    send_telegram(f"[PREDICT-{tag}] {os.getenv('TIMEFRAME','')} {os.getenv('SYMBOLS','')}\n" + "\n".join(lines))

    print("predict_once: OK")
    return results

if _name_ == "_main_":
    predict_once()

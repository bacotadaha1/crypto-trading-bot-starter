from fastapi import FastAPI, Header, HTTPException, Depends
from ..config import settings
from ..live import Trader
from ..notify import send_telegram

from ..ml_dataset import create_dataset
from ..train import train_and_save
from ..eval_daily import eval_daily
from ..predict import predict_once

app = FastAPI(title="Crypto Bot API", docs_url=None, redoc_url=None)
trader = Trader()

def verify_key(x_api_key: str = Header(None)):
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/signal", dependencies=[Depends(verify_key)])
def signal():
    out = {}
    for sym in settings.symbols:
        try:
            _, sig, size, px = trader.compute_signal(sym)
            out[sym] = {"signal": int(sig), "size": float(size), "price": float(px)}
        except Exception as e:
            out[sym] = {"error": str(e)}
    return out

# ---------- admin ----------
@app.post("/admin/update", dependencies=[Depends(verify_key)])
def admin_update():
    from ..ingestion import make_client
    from ..features import build_features
    import pandas as pd
    from pathlib import Path
    ex = make_client()
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    for sym in settings.symbols:
        raw = ex.fetch_ohlcv(sym, timeframe=settings.timeframe, limit=settings.limit)
        if not raw:
            continue
        df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        feat = build_features(df.copy())
        out = Path(settings.data_dir) / f"{sym.replace('/','')}{settings.timeframe}.parquet"
        feat.to_parquet(out)
    return {"ok": True, "msg": "update done"}

@app.post("/admin/predict", dependencies=[Depends(verify_key)])
def admin_predict():
    res = predict_once()
    return {"ok": True, "n": len(res)}

@app.post("/admin/eval", dependencies=[Depends(verify_key)])
def admin_eval():
    df = eval_daily()
    return {"ok": True, "n": 0 if df is None else len(df)}

@app.post("/admin/ml_dataset", dependencies=[Depends(verify_key)])
def admin_ml_dataset():
    create_dataset(years=4)
    return {"ok": True}

@app.post("/admin/train", dependencies=[Depends(verify_key)])
def admin_train():
    model_path = (settings.data_dir / "model.pkl").resolve()
    train_and_save(csv_path=str(settings.data_dir / "training_data.csv"),
                   model_path=str(model_path))
    send_telegram("[TRAIN] model.pkl mis à jour")
    return {"ok": True}

import pandas as pd
from pathlib import Path
from src.config import settings
from src.ingestion import load_all_symbols
from src.features import build_features

def main():
    dfs = load_all_symbols()
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    for name, df in dfs.items():
        feat = build_features(df)
        out = Path(settings.data_dir) / f"{name}_{settings.timeframe}.parquet"
        feat.to_parquet(out)
    print("daily_update: OK")

if __name__ == "__main__":
    main()

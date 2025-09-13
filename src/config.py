from pathlib import Path
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseModel):
    exchange: str = os.getenv("EXCHANGE", "binance")
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    use_testnet: bool = os.getenv("USE_TESTNET", "true").lower() == "true"

    symbols: list[str] = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",")
    timeframe: str = os.getenv("TIMEFRAME", "4h")
    limit: int = int(os.getenv("LIMIT", 500))

    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))

    ret_short: int = int(os.getenv("RET_SHORT", 3))
    ret_long: int = int(os.getenv("RET_LONG", 12))
    vol_window: int = int(os.getenv("VOL_WINDOW", 12))

    fee_bps: float = float(os.getenv("FEE_BPS", 8))
    slippage: float = float(os.getenv("SLIPPAGE", "0.0002").replace("%","")) / (100 if "%" in os.getenv("SLIPPAGE","") else 1)

    target_vol_annual: float = float(os.getenv("TARGET_VOL_ANNUAL", 0.10))
    max_drawdown_kill: float = float(os.getenv("MAX_DRAWDOWN_KILL", 0.12))

    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", 8000))
    api_key: str = os.getenv("API_KEY", "")

    trade_enabled: bool = os.getenv("TRADE_ENABLED", "false").lower() == "true"

settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)

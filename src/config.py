# src/config.py
from pathlib import Path
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _symbols_env(default: str) -> list[str]:
    raw = os.getenv("SYMBOLS", default)
    return [s.strip() for s in raw.split(",") if s.strip()]

class Settings(BaseModel):
    # --- Exchange ---
    exchange: str = os.getenv("EXCHANGE", "binance")  # <- par défaut binance
    use_testnet: bool = _bool_env("USE_TESTNET", False)  # <- False par défaut (Binance spot)

    # --- Clés API (optionnelles) ---
    # On accepte API_KEY/API_SECRET (génériques) OU BINANCE_API_KEY/BINANCE_API_SECRET
    api_key: str = os.getenv("API_KEY", "") or os.getenv("BINANCE_API_KEY", "")
    api_secret: str = os.getenv("API_SECRET", "") or os.getenv("BINANCE_API_SECRET", "")

    # --- Dataset / features ---
    symbols: list[str] = _symbols_env("BTC/USDT,ETH/USDT,SOL/USDT")
    timeframe: str = os.getenv("TIMEFRAME", "4h")
    limit: int = _int_env("LIMIT", 1200)

    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))

    ret_short: int = _int_env("RET_SHORT", 3)
    ret_long: int = _int_env("RET_LONG", 12)
    vol_window: int = _int_env("VOL_WINDOW", 12)

    fee_bps: float = _float_env("FEE_BPS", 8.0)

    # SLIPPAGE peut être entré en décimal (0.0002) ou en pourcentage ("0.02%")
    slippage: float = (
        (float(os.getenv("SLIPPAGE", "0.0002").replace("%", "")) / 100.0)
        if "%" in os.getenv("SLIPPAGE", "")
        else _float_env("SLIPPAGE", 0.0002)
    )

    # --- API locale (si tu exposes FastAPI quelque part) ---
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = _int_env("API_PORT", 8000)
    api_key_header: str = os.getenv("API_KEY", "")  # si tu utilises une clé d'API HTTP

    # --- Trading live (désactivé par défaut) ---
    trade_enabled: bool = _bool_env("TRADE_ENABLED", False)

settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)

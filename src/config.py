from pathlib import Path
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    # === Exchange & données ===
    # Tu peux changer "kraken" en "kucoin" ou "coinbase" si tu préfères.
    exchange: str = os.getenv("EXCHANGE", "kraken")

    # Clés API optionnelles (laisse vide si tu ne trades pas en live)
    api_key: str = os.getenv("API_KEY", "")
    api_secret: str = os.getenv("API_SECRET", "")
    use_testnet: bool = os.getenv("USE_TESTNET", "false").lower() == "true"

    symbols: list[str] = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",")
    timeframe: str = os.getenv("TIMEFRAME", "4h")
    limit: int = int(os.getenv("LIMIT", 1500))

    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))

    # === Features/Strat ===
    ret_short: int = int(os.getenv("RET_SHORT", 3))
    ret_long: int = int(os.getenv("RET_LONG", 12))
    vol_window: int = int(os.getenv("VOL_WINDOW", 12))

    fee_bps: float = float(os.getenv("FEE_BPS", 8))
    slippage: float = float(os.getenv("SLIPPAGE", "0.02").replace("%", "")) / (
        100 if "%" in os.getenv("SLIPPAGE", "") else 1
    )

    target_vol_annual: float = float(os.getenv("TARGET_VOL_ANNUAL", 0.10))
    max_drawdown_kill: float = float(os.getenv("MAX_DRAWDOWN_KILL", 0.12))

    # === API interne éventuelle ===
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", 8000))

    # Exécution de trades (toujours False pour la prédiction auto)
    trade_enabled: bool = os.getenv("TRADE_ENABLED", "false").lower() == "true"

    # Telegram (optionnel pour recevoir un message)
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)

# src/config.py
from pathlib import Path
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# charge .env si présent (local)
load_dotenv()

class Settings(BaseModel):
    # exchange logique (nom simple : binance, bybit, kucoin, okx, etc.)
    exchange: str = os.getenv("EXCHANGE", "binance").lower()

    # clés par exchange (si tu en as)
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    bybit_api_key: str = os.getenv("BYBIT_API_KEY", "")
    bybit_api_secret: str = os.getenv("BYBIT_API_SECRET", "")
    kucoin_api_key: str = os.getenv("KUCOIN_API_KEY", "")
    kucoin_api_secret: str = os.getenv("KUCOIN_API_SECRET", "")
    okx_api_key: str = os.getenv("OKX_API_KEY", "")
    okx_api_secret: str = os.getenv("OKX_API_SECRET", "")

    # testnet flag global (utile si l'exchange supporte)
    use_testnet: bool = os.getenv("USE_TESTNET", "false").lower() == "true"

    # symbols et timeframe
    symbols: list[str] = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",")
    timeframe: str = os.getenv("TIMEFRAME", "4h")
    limit: int = int(os.getenv("LIMIT", 500))

    # data dir
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))

    # features / strategy params
    ret_short: int = int(os.getenv("RET_SHORT", 3))
    ret_long: int = int(os.getenv("RET_LONG", 12))
    vol_window: int = int(os.getenv("VOL_WINDOW", 12))

    # telegram optional
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # model path fallback (optionnel)
    model_path: str = os.getenv("MODEL_PATH", "models/model1.pkl")

    # helper -> renvoie id ccxt (ex: 'binance', 'bybit', 'kucoin', 'okx')
    def exchange_id(self) -> str:
        valid = {"binance", "bybit", "kucoin", "okx"}
        ex = (self.exchange or "").lower()
        if ex not in valid:
            raise ValueError(f"Exchange '{self.exchange}' non supporté. Choisis parmi {sorted(valid)}.")
        return ex

    # helper -> kwargs pour initialiser le client ccxt
    def ccxt_kwargs(self) -> dict:
        ex = self.exchange_id()
        kwargs = {"enableRateLimit": True}
        if ex == "binance":
            if self.binance_api_key and self.binance_api_secret:
                kwargs.update({"apiKey": self.binance_api_key, "secret": self.binance_api_secret})
                # example testnet flag for binance futures (si utilisé)
                if self.use_testnet:
                    kwargs["options"] = {"defaultType": "future", "adjustForTimeDifference": True}
        elif ex == "bybit":
            if self.bybit_api_key and self.bybit_api_secret:
                kwargs.update({"apiKey": self.bybit_api_key, "secret": self.bybit_api_secret})
            if self.use_testnet:
                kwargs["test"] = True
        elif ex == "kucoin":
            if self.kucoin_api_key and self.kucoin_api_secret:
                kwargs.update({"apiKey": self.kucoin_api_key, "secret": self.kucoin_api_secret})
        elif ex == "okx":
            if self.okx_api_key and self.okx_api_secret:
                kwargs.update({"apiKey": self.okx_api_key, "secret": self.okx_api_secret})
        return kwargs

# instance globale
settings = Settings()
# assure le dossier data
settings.data_dir.mkdir(parents=True, exist_ok=True)

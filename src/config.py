# src/config.py
from __future__ import annotations
from pathlib import Path
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Charge .env (sans erreur si absent)
load_dotenv()

def _getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    """Petit helper pour centraliser os.getenv (trim + None si vide)."""
    val = os.getenv(name, default)
    if val is None:
        return None
    val = val.strip()
    return val if val != "" else None

def _split_symbols(raw: Optional[str], fallback: str = "BTC/USDT,ETH/USDT") -> list[str]:
    raw = raw or fallback
    return [s.strip() for s in raw.split(",") if s.strip()]

class Settings:
    """
    Configuration unique du projet. 
    - On choisit l'exchange avec EXCHANGE (binance|bybit|kucoin|okx).
    - Les clés API existent en parallèle pour chaque exchange :
        BINANCE_API_KEY / BINANCE_API_SECRET
        BYBIT_API_KEY   / BYBIT_API_SECRET
        KUCOIN_API_KEY  / KUCOIN_API_SECRET  / KUCOIN_API_PASSPHRASE
        OKX_API_KEY     / OKX_API_SECRET     / OKX_API_PASSPHRASE
    - USE_TESTNET=true|false gère le testnet quand l'exchange le supporte.
    - Les autres hyperparamètres restent inchangés.
    """

    # ===== Sélection & généraux =====
    exchange: str
    use_testnet: bool

    # Série & données
    symbols: list[str]
    timeframe: str
    limit: int
    data_dir: Path

    # Features / stratégie
    ret_short: int
    ret_long: int
    vol_window: int
    fee_bps: float
    slippage: float
    target_vol_annual: float
    max_drawdown_kill: float

    # API interne (facultatif)
    api_host: str
    api_port: int
    api_key: str

    # Mode trade
    trade_enabled: bool

    # Telegram (facultatif)
    telegram_bot_token: Optional[str]
    telegram_chat_id: Optional[str]

    # ===== Clés par exchange (on les stocke toutes ici, on choisira au runtime) =====
    binance_api_key: Optional[str]
    binance_api_secret: Optional[str]

    bybit_api_key: Optional[str]
    bybit_api_secret: Optional[str]

    kucoin_api_key: Optional[str]
    kucoin_api_secret: Optional[str]
    kucoin_api_passphrase: Optional[str]

    okx_api_key: Optional[str]
    okx_api_secret: Optional[str]
    okx_api_passphrase: Optional[str]

    # -------- Initialisation --------
    def __init__(self) -> None:
        # Choix de l’exchange (normalisé)
        self.exchange = (_getenv("EXCHANGE", "kucoin") or "kucoin").lower()
        self.use_testnet = (_getenv("USE_TESTNET", "false") or "false").lower() == "true"

        # Série & données
        self.symbols   = _split_symbols(_getenv("SYMBOLS"), "BTC/USDT,ETH/USDT")
        self.timeframe = _getenv("TIMEFRAME", "4h") or "4h"
        self.limit     = int(_getenv("LIMIT", "500") or "500")
        self.data_dir  = Path(_getenv("DATA_DIR", "./data") or "./data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Features / stratégie
        self.ret_short = int(_getenv("RET_SHORT", "3") or "3")
        self.ret_long  = int(_getenv("RET_LONG",  "12") or "12")
        self.vol_window = int(_getenv("VOL_WINDOW", "12") or "12")

        self.fee_bps  = float(_getenv("FEE_BPS", "8") or "8")
        # SLIPPAGE peut être "0.02%" ou "0.0002"
        raw_slip = _getenv("SLIPPAGE", "0.0002") or "0.0002"
        if raw_slip.endswith("%"):
            self.slippage = float(raw_slip[:-1]) / 100.0
        else:
            self.slippage = float(raw_slip)

        self.target_vol_annual  = float(_getenv("TARGET_VOL_ANNUAL", "0.10") or "0.10")
        self.max_drawdown_kill  = float(_getenv("MAX_DRAWDOWN_KILL", "0.12") or "0.12")

        # API interne (facultatif)
        self.api_host = _getenv("API_HOST", "127.0.0.1") or "127.0.0.1"
        self.api_port = int(_getenv("API_PORT", "8000") or "8000")
        self.api_key  = _getenv("API_KEY", "") or ""

        # Mode trade (désactivé par défaut)
        self.trade_enabled = (_getenv("TRADE_ENABLED", "false") or "false").lower() == "true"

        # Telegram (facultatif)
        self.telegram_bot_token = _getenv("TELEGRAM_BOT_TOKEN", None)
        self.telegram_chat_id   = _getenv("TELEGRAM_CHAT_ID", None)

        # Clés API pour chaque exchange (lisibles une fois pour toutes)
        self.binance_api_key    = _getenv("BINANCE_API_KEY")
        self.binance_api_secret = _getenv("BINANCE_API_SECRET")

        self.bybit_api_key    = _getenv("BYBIT_API_KEY")
        self.bybit_api_secret = _getenv("BYBIT_API_SECRET")

        self.kucoin_api_key        = _getenv("KUCOIN_API_KEY")
        self.kucoin_api_secret     = _getenv("KUCOIN_API_SECRET")
        self.kucoin_api_passphrase = _getenv("KUCOIN_API_PASSPHRASE")

        self.okx_api_key        = _getenv("OKX_API_KEY")
        self.okx_api_secret     = _getenv("OKX_API_SECRET")
        self.okx_api_passphrase = _getenv("OKX_API_PASSPHRASE")

    # -------- Utilitaires orientés CCXT --------
    def exchange_id(self) -> str:
        """
        Retourne l'ID CCXT à utiliser (identique à l'exchange en minuscule
        pour ceux listés ici).
        """
        valid = {"binance", "bybit", "kucoin", "okx"}
        ex = self.exchange.lower()
        if ex not in valid:
            raise ValueError(f"Exchange '{self.exchange}' non supporté. Choisis parmi {sorted(valid)}.")
        return ex

    def active_api(self) -> Dict[str, str]:
        """
        Retourne un dict avec les credentials pour l'exchange sélectionné.
        Clés absentes si non nécessaires (ex: passphrase).
        """
        ex = self.exchange_id()
        if ex == "binance":
            return {
                "apiKey": self.binance_api_key or "",
                "secret": self.binance_api_secret or "",
            }
        if ex == "bybit":
            return {
                "apiKey": self.bybit_api_key or "",
                "secret": self.bybit_api_secret or "",
            }
        if ex == "kucoin":
            out = {
                "apiKey": self.kucoin_api_key or "",
                "secret": self.kucoin_api_secret or "",
            }
            if self.kucoin_api_passphrase:
                out["password"] = self.kucoin_api_passphrase
            return out
        if ex == "okx":
            out = {
                "apiKey": self.okx_api_key or "",
                "secret": self.okx_api_secret or "",
            }
            if self.okx_api_passphrase:
                out["password"] = self.okx_api_passphrase
            return out
        # fallback sécurisé
        return {}

    def ccxt_kwargs(self) -> Dict[str, Any]:
        """
        Prépare les kwargs usuels pour instancier un client CCXT.
        Exemple:
            import ccxt
            ex = getattr(ccxt, settings.exchange_id())(settings.ccxt_kwargs())
        """
        creds = self.active_api()
        kwargs: Dict[str, Any] = {
            "enableRateLimit": True,
        }
        # injecte credentials s'ils existent
        kwargs.update({k: v for k, v in creds.items() if v})

        # Options testnet par exchange (seulement si l'exchange le supporte côté public)
        ex = self.exchange_id()
        if self.use_testnet:
            if ex == "bybit":
                kwargs.setdefault("options", {})["defaultType"] = "spot"
                kwargs.setdefault("urls", {})["api"] = {
                    "public": "https://api-testnet.bybit.com",
                    "private": "https://api-testnet.bybit.com",
                }
            elif ex == "okx":
                kwargs.setdefault("options", {})["sandboxMode"] = True
            elif ex == "binance":
                # Attention : testnet spot a des limites/géos. À activer si utile :
                # kwargs.setdefault("options", {})["sandboxMode"] = True
                pass
            # Kucoin: testnet via 'kucoinfutures' ou sandbox WS; pour spot public,
            # on garde la prod tant que CCXT ne propose pas d'URL testnet spot stable.
        return kwargs

    # Confort: représentation courte
    def __repr__(self) -> str:
        return (
            f"Settings(exchange={self.exchange!r}, use_testnet={self.use_testnet}, "
            f"symbols={self.symbols}, timeframe={self.timeframe}, limit={self.limit})"
        )

# Instance globale importée partout
settings = Settings()

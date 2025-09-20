import ccxt
from .config import settings

def make_client():
    ex_id = settings.exchange.lower()
    if not hasattr(ccxt, ex_id):
        raise ValueError(f"Exchange inconnu dans CCXT: {ex_id}")

    ex_cls = getattr(ccxt, ex_id)

    # N'utilise apiKey/secret que si fournis (Binance spot n'en a pas besoin pour OHLCV)
    kwargs = {"enableRateLimit": True}
    if getattr(settings, "api_key", "") and getattr(settings, "api_secret", ""):
        kwargs["apiKey"] = settings.api_key
        kwargs["secret"] = settings.api_secret

    ex = ex_cls(kwargs)

    # Testnet seulement si tu réactives plus tard (pas utile pour Binance spot)
    if getattr(settings, "use_testnet", False) and hasattr(ex, "set_sandbox_mode"):
        ex.set_sandbox_mode(True)

    ex.load_markets()

    if not ex.has.get("fetchOHLCV", False):
        raise ValueError(f"L'exchange {ex_id} ne supporte pas fetchOHLCV")

    return ex

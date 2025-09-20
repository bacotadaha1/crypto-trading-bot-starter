import ccxt
from .config import settings

def make_client():
    ex_id = settings.exchange.lower()
    if not hasattr(ccxt, ex_id):
        raise ValueError(f"Exchange inconnu dans CCXT: {ex_id}")

    ex_cls = getattr(ccxt, ex_id)

    kwargs = {"enableRateLimit": True}
    if settings.api_key and settings.api_secret:
        kwargs["apiKey"] = settings.api_key
        kwargs["secret"] = settings.api_secret

    ex = ex_cls(kwargs)

    if settings.use_testnet and hasattr(ex, "set_sandbox_mode"):
        ex.set_sandbox_mode(True)

    ex.load_markets()

    if not ex.has.get("fetchOHLCV", False):
        raise ValueError(f"L'exchange {ex_id} ne supporte pas fetchOHLCV")

    return ex

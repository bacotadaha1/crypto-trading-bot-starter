import ccxt
from .config import settings

def make_client():
    ex_id = settings.exchange.lower()

if not hasattr(ccxt, ex_id):
        raise ValueError(f"Exchange inconnu dans ccxt: {ex_id}")
    ex_cls = getattr(ccxt, ex_id)
    ex = ex_cls({"enableRateLimit": True})
    ex.load_markets()
    if not ex.has.get("fetchOHLCV", False):
        raise ValueError(f"L'exchange {ex_id} ne supporte pas fetchOHLCV")
    return ex

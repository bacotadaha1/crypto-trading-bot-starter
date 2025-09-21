# src/ingestion.py
import ccxt
from src.config import settings

def make_client():
    ex_id = settings.exchange_id()  # lève ValueError si exchange non supporté
    kwargs = settings.ccxt_kwargs()
    ex_cls = getattr(ccxt, ex_id)
    ex = ex_cls(kwargs)
    # attempt to load markets (certains exchanges bloquent selon IP)
    try:
        ex.load_markets()
    except Exception as e:
        # propage l'exception pour logger dans les workflows
        raise
    return ex

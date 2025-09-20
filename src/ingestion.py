# src/ingestion.py
import ccxt
from src.config import settings


def make_client():
    """
    Crée et retourne un client CCXT configuré selon settings.exchange.
    - Fonctionne sans clés API pour les données OHLCV publiques.
    - Si des clés existent dans les secrets, elles seront utilisées automatiquement.
    - Charge aussi les markets pour éviter les erreurs de symboles.
    """
    ex_id = settings.exchange_id()
    if not hasattr(ccxt, ex_id):
        raise ValueError(f"Exchange {ex_id} non supporté par CCXT.")

    ex_cls = getattr(ccxt, ex_id)
    ex = ex_cls(settings.ccxt_kwargs())

    try:
        ex.load_markets()
    except Exception as e:
        raise RuntimeError(f"Impossible de charger les marchés pour {ex_id}: {e}")

    # Vérifie si l’exchange supporte OHLCV
    if not ex.has.get("fetchOHLCV", False):
        raise ValueError(f"L’exchange {ex_id} ne supporte pas fetchOHLCV (OHLCV indisponible).")

    return ex

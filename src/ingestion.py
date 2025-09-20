import ccxt
from src.config import settings


def make_client():
    """
    Crée un client CCXT pour l'exchange choisi (Kraken par défaut).
    Pas besoin de clés si on récupère juste l'OHLCV.
    """
    ex_id = settings.exchange.lower()
    if not hasattr(ccxt, ex_id):
        raise ValueError(f"Exchange inconnu : {ex_id}")

    ex_cls = getattr(ccxt, ex_id)
    args = {"enableRateLimit": True}

    # Si tu renseignes API_KEY/SECRET (trading live), on les branche :
    if settings.api_key and settings.api_secret:
        args["apiKey"] = settings.api_key
        args["secret"] = settings.api_secret

    ex = ex_cls(args)
    ex.load_markets()

    # Vérification basique qu'on peut bien fetch l'OHLCV
    # (certains exchanges n'ont pas OHLCV pour tous les symboles)
    if not ex.has.get("fetchOHLCV", False):
        raise ValueError(f"L'exchange {ex_id} ne supporte pas fetchOHLCV")

    return ex

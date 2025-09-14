import ccxt
import pandas as pd
from .config import settings

def make_client():
    """Crée et retourne un client CCXT prêt (sandbox si demandé), avec markets chargés."""
    try:
        ex_cls = getattr(ccxt, settings.exchange)
    except AttributeError:
        raise RuntimeError(f"Exchange '{settings.exchange}' inconnu dans ccxt")

    ex = ex_cls({
        "apiKey": settings.binance_api_key,
        "secret": settings.binance_api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    # Sandbox / testnet si supporté
    if settings.use_testnet:
        try:
            ex.set_sandbox_mode(True)
        except Exception:
            # si non supporté, on ignore
            pass

    # Charge les marchés pour valider la connexion
    try:
        ex.load_markets()
    except Exception as e:
        raise RuntimeError(f"Echec load_markets(): {e}")

    return ex

class ExchangeClient:
    """Compat : un wrapper qui expose .client, en s'appuyant sur make_client()."""
    def __init__(self):
        self.client = make_client()

def load_all_symbols() -> dict[str, pd.DataFrame]:
    """Télécharge OHLCV pour tous les symboles définis dans .env et retourne un dict de DataFrames indexés par ts."""
    ex = make_client()
    out = {}
    for sym in settings.symbols:
        try:
            raw = ex.fetch_ohlcv(sym, timeframe=settings.timeframe, limit=settings.limit)
            df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            out[sym.replace("/","_")] = df.set_index("ts")
        except Exception as e:
            print(f"[WARN] fetch_ohlcv({sym}) a échoué : {e}")
            continue
    if not out:
        raise RuntimeError("Aucune donnée chargée : vérifie SYMBOLS/TIMEFRAME/LIMIT ou la connexion Internet.")
    return out


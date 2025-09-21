# src/ingestion.py
import ccxt
from src.config import settings

def make_client():
    ex_id = settings.exchange_id()
    if not hasattr(ccxt, ex_id):
        raise ValueError(f"Exchange inconnu: {ex_id}")
    ex_cls = getattr(ccxt, ex_id)
    ex = ex_cls(settings.ccxt_kwargs())
    ex.load_markets()  # important pour ex.symbols / ex.markets_by_id
    if not ex.has.get("fetchOHLCV", False):
        raise ValueError(f"L'exchange {ex_id} ne supporte pas fetchOHLCV")
    return ex

def resolve_symbol(ex, symbol: str) -> str:
    """
    Retourne un symbole « unifié » accepté par CCXT pour cet exchange.
    Gère les variantes BTC/USDT vs BTC-USDT (KuCoin), etc.
    """
    # 1) exact (déjà unifié)
    if symbol in ex.symbols:
        return symbol

    # 2) id de marché avec tiret -> symbole unifié
    dash_id = symbol.replace("/", "-")
    m_by_id = getattr(ex, "markets_by_id", {})
    if dash_id in m_by_id:
        return m_by_id[dash_id]["symbol"]  # ex: 'BTC/USDT'

    # 3) essai inverse (au cas où on reçoit un tiret)
    slash = symbol.replace("-", "/")
    if slash in ex.symbols:
        return slash

    # 4) matching "souple" (sans séparateur, insensible à la casse)
    want = symbol.replace("/", "").replace("-", "").lower()
    for s in ex.symbols:
        if s.replace("/", "").lower() == want:
            return s

    # 5) en dernier recours: message clair avec exemples
    examples = ", ".join(list(ex.symbols)[:10])
    raise ValueError(f"{ex.id} n'a pas la paire {symbol}. Exemples: {examples}")

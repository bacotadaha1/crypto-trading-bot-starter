# src/alt_data.py
from __future__ import annotations
from datetime import timezone
from typing import Optional
import numpy as np
import pandas as pd
import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_ANALYZER = SentimentIntensityAnalyzer()

_DEF_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://news.yahoo.com/rss/crypto",
    "https://www.reddit.com/r/CryptoCurrency/.rss",
]

def _to_utc(s: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    return pd.DatetimeIndex(dt)

def _reindex_ffill(obj: pd.Series | pd.DataFrame, target_index: pd.DatetimeIndex):
    if isinstance(obj, pd.Series):
        return obj.sort_index().reindex(target_index, method="ffill")
    return obj.sort_index().reindex(target_index, method="ffill")

def fetch_fear_greed(days: int = 90) -> pd.Series:
    try:
        url = f"https://api.alternative.me/fng/?limit={days}&format=json"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return pd.Series(dtype="float64")
        rows = []
        for it in data:
            ts = pd.to_datetime(int(it["timestamp"]), unit="s", utc=True).normalize()
            rows.append((ts, float(it["value"])))
        s = pd.Series(dict(rows)).sort_index()
        s.name = "fg_idx"
        return s
    except Exception:
        return pd.Series(dtype="float64")

def _score_text(text: str) -> float:
    try:
        return float(_ANALYZER.polarity_scores(text or "")["compound"])
    except Exception:
        return 0.0

def fetch_rss_sentiment(hours: int = 72, feeds: Optional[list[str]] = None) -> pd.Series:
    feeds = feeds or _DEF_FEEDS
    cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(hours=hours)
    buckets: dict[pd.Timestamp, list[float]] = {}
    try:
        for url in feeds:
            parsed = feedparser.parse(url)
            for e in parsed.get("entries", []):
                dt = None
                for key in ("published_parsed", "updated_parsed"):
                    if e.get(key):
                        dt = pd.to_datetime(e[key], utc=True)
                        break
                if dt is None or dt < cutoff:
                    continue
                score = _score_text(f"{e.get('title','')}. {e.get('summary','')}")
                hour = dt.floor("h")
                buckets.setdefault(hour, []).append(score)
        if not buckets:
            return pd.Series(dtype="float64")
        s = pd.Series({k: float(np.nanmean(v)) for k, v in buckets.items()}).sort_index()
        s.name = "sent_hour"
        return s
    except Exception:
        return pd.Series(dtype="float64")

def regime_from_prices(df_price: pd.DataFrame) -> pd.DataFrame:
    d = df_price.copy()
    d["ts"] = _to_utc(d["ts"])
    d = d.set_index("ts").sort_index()
    d["ret_1"] = d["close"].pct_change()
    d["rv_w6"] = d["ret_1"].rolling(6, min_periods=3).std()
    d["rv_w36"] = d["ret_1"].rolling(36, min_periods=6).std()
    vol_mean = d["volume"].rolling(20, min_periods=5).mean()
    vol_std = d["volume"].rolling(20, min_periods=5).std()
    d["vol_z"] = (d["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    return d[["rv_w6", "rv_w36", "vol_z"]]

def build_alt_features(df_price: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un DataFrame indexé ts avec colonnes:
    sent_hour, fg_idx, rv_w6, rv_w36, vol_z
    (les séries manquantes sont créées puis ffill sur l’index cible)
    """
    ts = _to_utc(df_price["ts"])
    target_index = pd.DatetimeIndex(pd.Series(ts)).sort_values().unique()

    s_sent = fetch_rss_sentiment(hours=72)
    s_fg = fetch_fear_greed(days=90)
    reg = regime_from_prices(df_price)

    if s_sent.empty:
        s_sent = pd.Series(index=target_index, dtype="float64", name="sent_hour")
    else:
        s_sent = _reindex_ffill(s_sent, target_index)
    if s_fg.empty:
        s_fg = pd.Series(index=target_index, dtype="float64", name="fg_idx")
    else:
        s_fg = _reindex_ffill(s_fg, target_index)
    reg = _reindex_ffill(reg, target_index)

    out = pd.concat([s_sent.rename("sent_hour"), s_fg.rename("fg_idx"), reg], axis=1)
    out.index.name = "ts"
    return out

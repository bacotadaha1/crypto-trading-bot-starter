# src/alt_data.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone, timedelta
import math

import pandas as pd
import numpy as np
import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------- helpers temps ----------
def _to_utc(ts: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    # drop tz-naive if any
    if isinstance(dt, pd.Series):
        return pd.DatetimeIndex(dt)
    return dt

def _resample_to_index(src: pd.Series | pd.DataFrame, target_index: pd.DatetimeIndex, method: str = "ffill"):
    """
    Réindexe une série/dataframe (horaires arbitraires) sur l'index cible (tes bougies),
    en forward-fill pour coller au dernier signal connu.
    """
    if isinstance(src, pd.Series):
        s = src.sort_index()
        return s.reindex(target_index, method=method)
    else:
        df = src.sort_index()
        return df.reindex(target_index, method=method)


# ---------- Fear & Greed (daily) ----------
def fetch_fear_greed(days: int = 90) -> pd.Series:
    """
    API publique alternative.me — retourne une série quotidienne fg_idx (0..100) en UTC.
    Fallback silencieux -> série vide en cas d'erreur.
    """
    try:
        url = f"https://api.alternative.me/fng/?limit={days}&format=json"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
        rows = []
        for it in data:
            # timestamp (s) -> date UTC
            ts = pd.to_datetime(int(it["timestamp"]), unit="s", utc=True).normalize()
            val = float(it["value"])
            rows.append((ts, val))
        if not rows:
            return pd.Series(dtype="float64")
        s = pd.Series({ts: val for ts, val in rows}).sort_index()
        s.name = "fg_idx"
        return s
    except Exception:
        return pd.Series(dtype="float64")


# ---------- Sentiment RSS (hourly) ----------
_ANALYZER = SentimentIntensityAnalyzer()
_DEFAULT_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://news.yahoo.com/rss/crypto",
    "https://www.reddit.com/r/CryptoCurrency/.rss",
]

def _score_text(text: str) -> float:
    try:
        return float(_ANALYZER.polarity_scores(text or "")["compound"])
    except Exception:
        return 0.0

def fetch_rss_sentiment(hours: int = 48, feeds: list[str] | None = None) -> pd.Series:
    """
    Télécharge quelques flux RSS publics et agrège un score VADER horaire (moyenne).
    Fallback silencieux -> série vide si erreur / pas d’articles.
    """
    feeds = feeds or _DEFAULT_FEEDS
    cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(hours=hours)
    buckets: dict[pd.Timestamp, list[float]] = {}
    try:
        for url in feeds:
            parsed = feedparser.parse(url)
            for e in parsed.get("entries", []):
                # timestamp de l'article (si absent -> ignore)
                dt = None
                for key in ("published_parsed", "updated_parsed"):
                    if e.get(key):
                        dt = pd.to_datetime(e[key], utc=True)
                        break
                if dt is None:
                    continue
                if dt < cutoff:
                    continue
                title = e.get("title", "")
                summ = e.get("summary", "")
                score = _score_text(f"{title}. {summ}")
                hour = dt.floor("h")
                buckets.setdefault(hour, []).append(score)
        if not buckets:
            return pd.Series(dtype="float64")
        # moyenne par heure
        s = pd.Series({k: float(np.nanmean(v)) for k, v in buckets.items()}).sort_index()
        s.name = "sent_hour"
        return s
    except Exception:
        return pd.Series(dtype="float64")


# ---------- Régime de marché (depuis tes prix) ----------
def regime_from_prices(df_price: pd.DataFrame) -> pd.DataFrame:
    """
    df_price: colonnes attendues ['ts','close','volume'] (UTC)
    Retourne DataFrame indexé par ts avec:
      - rv_w6  : std des ret_1 sur ~1 jour (6 bougies en 4h)
      - rv_w36 : std des ret_1 sur ~6 jours
      - vol_z  : z-score de volume (window=20)
    """
    df = df_price.copy()
    df["ts"] = _to_utc(df["ts"])
    df = df.set_index("ts").sort_index()
    df["ret_1"] = df["close"].pct_change()
    df["rv_w6"] = df["ret_1"].rolling(6, min_periods=3).std()
    df["rv_w36"] = df["ret_1"].rolling(36, min_periods=6).std()
    vol_mean = df["volume"].rolling(20, min_periods=5).mean()
    vol_std = df["volume"].rolling(20, min_periods=5).std()
    df["vol_z"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    return df[["rv_w6", "rv_w36", "vol_z"]].dropna(how="all")


# ---------- Build alt features alignées sur tes bougies ----------
def build_alt_features(df_price: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un set de features "alt-data" alignées sur l'index ts de df_price:
      - sent_hour  (RSS+VADER, agrégé horaire, ffill sur tes bougies)
      - fg_idx     (Fear & Greed quotidien, ffill)
      - rv_w6, rv_w36, vol_z (régime interne calculé depuis tes prix)
    """
    # index cible
    ts = _to_utc(df_price["ts"])
    target_index = pd.DatetimeIndex(pd.Series(ts)).sort_values()

    # alt: sentiment houre + fear/greed
    s_sent = fetch_rss_sentiment(hours=72)
    s_fg = fetch_fear_greed(days=90)

    # réindexation sur tes bougies (ffill)
    if not s_sent.empty:
        s_sent = _resample_to_index(s_sent, target_index, method="ffill")
    else:
        s_sent = pd.Series(index=target_index, dtype="float64", name="sent_hour")

    if not s_fg.empty:
        # daily -> ffill sur heures
        s_fg = _resample_to_index(s_fg, target_index, method="ffill")
    else:
        s_fg = pd.Series(index=target_index, dtype="float64", name="fg_idx")

    # régime (depuis prix)
    reg = regime_from_prices(df_price)
    reg = _resample_to_index(reg, target_index, method="ffill")

    out = pd.concat([s_sent.rename("sent_hour"), s_fg.rename("fg_idx"), reg], axis=1)
    out.index.name = "ts"
    return out

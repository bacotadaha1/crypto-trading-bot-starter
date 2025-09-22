# src/evaluate.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta
import math
import os

import numpy as np
import pandas as pd

from src.config import settings
from src.ingestion import make_client


LOG_PATH = Path("data") / "preds_log.csv"
TIMEFMT_LOG = "%Y-%m-%d %H:%M:%S"   # format ts_utc dans preds_log.csv
TIMEFMT_BAR = "%Y-%m-%d %H:%M UTC"  # format "time" de predict.py (si besoin)


# ----------------------- utils temps & marché -----------------------
def _timeframe_to_ms(tf: str) -> int:
    unit = tf[-1]
    n = int(tf[:-1])
    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 60 * 60_000
    if unit == "d":
        return n * 24 * 60 * 60_000
    raise ValueError(f"timeframe inconnu: {tf}")

def _parse_utc(s: str) -> datetime:
    # ts_utc: "YYYY-mm-dd HH:MM:SS"
    return datetime.strptime(s, TIMEFMT_LOG).replace(tzinfo=timezone.utc)


# ----------------------- telegram ----------------------------------
def _send_telegram(msg: str):
    token = getattr(settings, "telegram_bot_token", "") or ""
    chat_id = getattr(settings, "telegram_chat_id", "") or ""
    if not token or not chat_id:
        print("[INFO] Telegram non configuré. Message:\n", msg)
        return
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=data, timeout=20)
        if r.status_code != 200:
            print("[WARN] Telegram status:", r.status_code, r.text)
    except Exception as e:
        print("[WARN] Telegram error:", e)


# ----------------------- labellisation (vérité) ---------------------
def _label_missing(ex, df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour toutes les lignes sans 'actual_dir' ni 'correct',
    on récupère la bougie suivante et on marque UP/DOWN + correct.
    """
    if "actual_dir" not in df.columns:
        df["actual_dir"] = pd.NA
    if "correct" not in df.columns:
        df["correct"] = pd.NA

    tf = str(df["timeframe"].iloc[-1] if "timeframe" in df.columns and df["timeframe"].notna().any() else settings.timeframe)
    tf_ms = _timeframe_to_ms(tf)

    # on parcourt uniquement les lignes à compléter
    mask_missing = df["actual_dir"].isna() & df["symbol"].notna() & df["direction"].notna()
    idxs = df.index[mask_missing].tolist()
    if not idxs:
        return df

    for i in idxs:
        row = df.loc[i]
        sym = str(row["symbol"]).strip()
        if not sym:
            continue
        try:
            t0 = _parse_utc(str(row["ts_utc"]))
        except Exception:
            # timestamp illisible, skip
            continue

        since_ms = int(t0.timestamp() * 1000) - tf_ms
        try:
            ohlcv = make_client().fetch_ohlcv(sym, timeframe=tf, since=since_ms, limit=5)
        except Exception as e:
            # marché indisponible: on laisse vide
            continue

        if not ohlcv:
            continue

        closes = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        closes["ts"] = pd.to_datetime(closes["ts"], unit="ms", utc=True)

        # trouver la barre de t0 puis la suivante
        # on prend la dernière barre <= t0, puis j+1
        mask_le = closes["ts"] <= t0
        if not mask_le.any():
            continue
        j0 = int(np.where(mask_le.to_numpy())[0].max())
        j1 = j0 + 1
        if j1 >= len(closes):
            # la bougie suivante n'est pas encore dispo
            continue

        c0 = float(closes.loc[j0, "close"])
        c1 = float(closes.loc[j1, "close"])
        actual_dir = "UP" if c1 > c0 else "DOWN"
        predicted_dir = str(row["direction"]).upper()

        df.at[i, "actual_dir"] = actual_dir
        df.at[i, "correct"] = (predicted_dir == actual_dir)

    return df


# ----------------------- métriques ----------------------------------
def _accuracy(series_bool: pd.Series) -> float | None:
    if series_bool.empty:
        return None
    return float(series_bool.mean())

def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "—"
    return f"{int(round(x * 100))}%"

def _compute_metrics(df: pd.DataFrame, days: int = 7):
    """
    Renvoie:
      - acc_all: accuracy sur toutes les lignes évaluées
      - acc_ndays: accuracy sur les N derniers jours
      - by_symbol_ndays: dict {symbol: accuracy sur N jours}
      - sample_sizes: dict avec tailles d'échantillons
    """
    evaluated = df[df["correct"].notna()].copy()
    acc_all = _accuracy(evaluated["correct"].astype(bool)) if not evaluated.empty else None

    # fenêtre glissante N jours
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    recent = evaluated[ evaluated["ts_utc"].apply(lambda s: _parse_utc(str(s)) >= cutoff) ].copy()
    acc_ndays = _accuracy(recent["correct"].astype(bool)) if not recent.empty else None

    by_symbol_ndays: dict[str, float | None] = {}
    for sym, grp in recent.groupby("symbol"):
        by_symbol_ndays[str(sym)] = _accuracy(grp["correct"].astype(bool))

    sample_sizes = {
        "all": int(len(evaluated)),
        f"last_{days}d": int(len(recent)),
    }
    return acc_all, acc_ndays, by_symbol_ndays, sample_sizes


# ----------------------- entrée principale --------------------------
def main():
    if not LOG_PATH.exists():
        print("Aucun data/preds_log.csv — rien à évaluer.")
        return

    df = pd.read_csv(LOG_PATH)
    if df.empty:
        print("preds_log.csv vide.")
        return

    # 1) compléter les lignes manquantes (vérité terrain)
    ex = make_client()
    df = _label_missing(ex, df)

    # 2) sauvegarder le log mis à jour
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_PATH, index=False)

    # 3) calcul des métriques (N jours paramétrable via env ACC_N_DAYS, défaut 7)
    n_days = int(os.getenv("ACC_N_DAYS", "7"))
    acc_all, acc_ndays, by_sym, sizes = _compute_metrics(df, days=n_days)

    # 4) message Telegram
    lines = []
    lines.append("*Bilan des prédictions*")
    lines.append(f"Exchange: `{settings.exchange}`   timeframe: `{settings.timeframe}`")
    lines.append("")
    lines.append(f"Accuracy (tout l'historique évalué): *{_fmt_pct(acc_all)}*  (n={sizes['all']})")
    lines.append(f"Accuracy {n_days}j: *{_fmt_pct(acc_ndays)}*  (n={sizes[f'last_{n_days}d']})")
    lines.append("")

    if by_sym:
        lines.append(f"*Détail par symbole (sur {n_days}j)*")
        for sym, acc in sorted(by_sym.items()):
            lines.append(f"• {sym}: {_fmt_pct(acc)}")
    else:
        lines.append("_Pas assez de données récentes pour un détail par symbole._")

    msg = "\n".join(lines)
    print(msg)
    _send_telegram(msg)


if __name__ == "__main__":
    main()

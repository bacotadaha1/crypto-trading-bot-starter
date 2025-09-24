# src/settle_simulated.py
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from src.config import settings
# ✅ IMPORTANT : on importe depuis action_strategy (et pas strategy)
from src.action_strategy import compute_atr  # noqa: F401  (utile si tu veux l'utiliser plus tard)


DATA_DIR = settings.data_dir
FILLS_CSV = DATA_DIR / "fills_log.csv"     # créé par le moteur d'exécution si des ordres sont simulés/exécutés
PNL_CSV   = DATA_DIR / "pnl_report.csv"    # rapport de PnL (on le génère ici)


def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        # si le fichier est vide/corrompu, on renvoie un DF vide
        return pd.DataFrame()


def _pair_trades_and_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule un PnL grossier en pairant les BUY/SELL par symbole dans l'ordre chronologique.
    Colonnes attendues idéalement: ['ts','symbol','side','qty','price','fee']
    - S'il manque des colonnes, on essaie des fallback avec des noms proches.
    - Si rien n'est exploitable, on retourne un DF vide -> le script termine OK.
    """
    if df.empty:
        return pd.DataFrame()

    # normalisation colonnes
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_ts     = pick("ts", "time", "timestamp")
    c_sym    = pick("symbol", "sym")
    c_side   = pick("side", "action", "type")
    c_qty    = pick("qty", "quantity", "size", "amount")
    c_price  = pick("price", "px", "fill_price")
    c_fee    = pick("fee", "fees", "commission")

    # colonnes minimales nécessaires
    required = [c_ts, c_sym, c_side, c_qty, c_price]
    if any(x is None for x in required):
        return pd.DataFrame()  # structure inconnue -> on sort proprement

    d = df.copy()
    d[c_ts] = pd.to_datetime(d[c_ts], errors="coerce", utc=True)
    d = d.dropna(subset=[c_ts, c_sym, c_side, c_qty, c_price]).sort_values(c_ts)

    # casting numériques
    for cc in [c_qty, c_price]:
        d[cc] = pd.to_numeric(d[cc], errors="coerce")
    if c_fee:
        d[c_fee] = pd.to_numeric(d[c_fee], errors="coerce").fillna(0.0)
    else:
        d["__fee__"] = 0.0
        c_fee = "__fee__"

    rows = []
    for sym, grp in d.groupby(c_sym):
        pos_qty = 0.0
        avg_cost = 0.0
        cash_pnl = 0.0

        for _, r in grp.iterrows():
            side  = str(r[c_side]).upper()
            qty   = float(r[c_qty])
            price = float(r[c_price])
            fee   = float(r[c_fee])

            if qty <= 0 or price <= 0:
                continue

            if side.startswith("B"):  # BUY
                # moyenne pondérée du coût
                new_qty = pos_qty + qty
                if new_qty > 0:
                    avg_cost = (avg_cost * pos_qty + price * qty) / new_qty
                pos_qty = new_qty
                cash_pnl -= fee
            elif side.startswith("S"):  # SELL
                # on ferme (au moins en partie) la position
                close_qty = min(pos_qty, qty)
                if close_qty > 0:
                    cash_pnl += (price - avg_cost) * close_qty
                    pos_qty -= close_qty
                # si vente à découvert non gérée -> on ignore l'excès
                cash_pnl -= fee
            else:
                # side inconnu
                continue

        rows.append({
            "symbol": sym,
            "pnl_cash": cash_pnl,
            "open_qty": pos_qty,
            "avg_cost": avg_cost if pos_qty > 0 else np.nan,
        })

    out = pd.DataFrame(rows)
    out["ts_utc"] = _now_utc_str()
    cols_order = ["ts_utc", "symbol", "pnl_cash", "open_qty", "avg_cost"]
    return out[cols_order]


def main() -> None:
    _ensure_data_dir()

    fills = _safe_read_csv(FILLS_CSV)
    if fills.empty:
        # ✅ Aucun fill -> on crée un rapport vide et on sorte avec succès
        print(f"[SETTLE] {FILLS_CSV} introuvable ou vide — rien à régler aujourd'hui.")
        PNL_CSV.write_text("ts_utc,symbol,pnl_cash,open_qty,avg_cost\n", encoding="utf-8")
        print(f"[SETTLE] Rapport vide écrit: {PNL_CSV}")
        return

    pnl = _pair_trades_and_pnl(fills)
    if pnl.empty:
        print("[SETTLE] Fichier de fills présent mais structure inconnue — aucun PnL calculé.")
        PNL_CSV.write_text("ts_utc,symbol,pnl_cash,open_qty,avg_cost\n", encoding="utf-8")
        print(f"[SETTLE] Rapport vide écrit: {PNL_CSV}")
        return

    # append si le fichier existe déjà
    write_header = not PNL_CSV.exists()
    pnl.to_csv(PNL_CSV, index=False, mode="a", header=write_header)
    print(f"[SETTLE] Rapport PnL mis à jour -> {PNL_CSV.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # on imprime l'erreur et on retourne un code 1 pour aider au debug
        print("[SETTLE][ERROR]", repr(e))
        sys.exit(1)

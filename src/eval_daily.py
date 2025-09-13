import pandas as pd
from pathlib import Path
from src.config import settings

def eval_daily():
    pred_dir = Path(settings.data_dir) / "predictions"
    files = sorted(pred_dir.glob("pred_*.csv"))
    if not files:
        print("No predictions found.")
        return

    # prend le dernier fichier
    pred_file = files[-1]
    preds = pd.read_csv(pred_file)

    # Ici on ne peut pas vraiment évaluer sans labels (cours futurs),
    # donc on fait un simple export récapitulatif.
    out = Path(settings.data_dir) / "eval_history.csv"
    preds.to_csv(out, mode="a", header=not out.exists(), index=False)
    print(f"eval_daily: appended {pred_file.name}")

if __name__ == "__main__":
    eval_daily()

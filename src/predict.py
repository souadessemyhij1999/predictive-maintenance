"""
Score new machine runs with the trained model and recommend a maintenance action.

Input: a CSV with one row per run and these columns (AI4I names also work):
    type, air_temp_k, process_temp_k, speed_rpm, torque_nm, tool_wear_min

Output: the same table plus failure_risk (0-1) and recommended_action, sorted by risk.
Open it in Excel or Power BI as a daily maintenance worklist.

    python src/predict.py --input samples/new_runs.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from data import add_features

ROOT = Path(__file__).resolve().parent.parent
AI4I_NAMES = {"Type": "type", "Air temperature [K]": "air_temp_k", "Process temperature [K]": "process_temp_k",
              "Rotational speed [rpm]": "speed_rpm", "Torque [Nm]": "torque_nm", "Tool wear [min]": "tool_wear_min"}


def main():
    p = argparse.ArgumentParser(description="Score machine runs for failure risk.")
    p.add_argument("--input", default="samples/new_runs.csv")
    p.add_argument("--output", default="results/maintenance_worklist.csv")
    args = p.parse_args()

    model = joblib.load(ROOT / "models" / "failure_model.joblib")
    card = json.loads((ROOT / "models" / "model_card.json").read_text())
    threshold, tool_limit = card["threshold"], card.get("tool_change_limit_min")

    runs = add_features(pd.read_csv(args.input).rename(columns=AI4I_NAMES))
    runs["failure_risk"] = model.predict_proba(runs[card["features"]])[:, 1].round(3)

    def action(r):
        if r["failure_risk"] >= threshold:
            return "STOP & INSPECT - high failure risk"
        if tool_limit and r["tool_wear_min"] >= tool_limit:
            return f"PLAN TOOL CHANGE - tool wear >= {tool_limit} min"
        return "OK"

    runs["recommended_action"] = runs.apply(action, axis=1)
    cols = [c for c in runs.columns if c not in ("temp_diff_k", "power_w", "strain")]
    out = runs[cols].sort_values("failure_risk", ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Model: {card['model']} | alarm threshold {threshold} | planned tool change at {tool_limit} min\n")
    print(out.to_string(index=False))
    print(f"\n{(out['recommended_action'] != 'OK').sum()} of {len(out)} runs need attention. Saved to {args.output}")


if __name__ == "__main__":
    main()

"""
Data download, loading and feature engineering.

Datasets (downloaded on first use into data/, not committed to the repo):
  * AI4I 2020 Predictive Maintenance Dataset - 10,000 milling-machine runs with 5 failure types
    (Matzka, 2020; UCI Machine Learning Repository, CC BY 4.0)
  * NAB machine_temperature_system_failure - temperature sensor of a real industrial machine,
    5-minute readings, Dec 2013 - Feb 2014, with labelled failure windows
    (Numenta Anomaly Benchmark, Lavin & Ahmad, 2015)
"""

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SOURCES = {
    # UCI is the official home of AI4I; the GitHub mirror is used when UCI is unreachable.
    "ai4i2020.csv": [
        "https://raw.githubusercontent.com/lingchenlijing/AI4I-2020-Predictive-Maintenance-Dataset/master/data/ai4i2020.csv",
    ],
    "machine_temperature_system_failure.csv": [
        "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv",
    ],
    "nab_combined_windows.json": [
        "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json",
    ],
    "nab_combined_labels.json": [
        "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json",
    ],
}

NAB_KEY = "realKnownCause/machine_temperature_system_failure.csv"


def download(name: str) -> Path:
    """Download a dataset file into data/ unless it is already there."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / name
    if path.exists():
        return path
    for url in SOURCES[name]:
        try:
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
            return path
        except Exception as exc:  # try the next mirror
            print(f"  failed from {url}: {exc}")
    raise RuntimeError(f"Could not download {name}")


# ----------------------------------------------------------------------------------------
# AI4I 2020: failure prediction
# ----------------------------------------------------------------------------------------
FAILURE_MODES = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    "RNF": "Random failure",
}

RAW_FEATURES = ["type", "air_temp_k", "process_temp_k", "speed_rpm", "torque_nm", "tool_wear_min"]
ENGINEERED_FEATURES = ["temp_diff_k", "power_w", "strain"]


def load_ai4i() -> pd.DataFrame:
    df = pd.read_csv(download("ai4i2020.csv"))
    df = df.rename(columns={
        "Product ID": "product_id", "Type": "type",
        "Air temperature [K]": "air_temp_k", "Process temperature [K]": "process_temp_k",
        "Rotational speed [rpm]": "speed_rpm", "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min", "Machine failure": "failure",
    })
    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-based features that mirror how each failure mode arises."""
    df = df.copy()
    # Heat dissipation: the machine can't shed heat when process and air temperature are too close
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]
    # Power at the spindle: torque x angular speed (rad/s). Too low or too high power causes failures
    df["power_w"] = df["torque_nm"] * df["speed_rpm"] * 2 * np.pi / 60
    # Overstrain: a worn tool under high torque
    df["strain"] = df["tool_wear_min"] * df["torque_nm"]
    return df


# ----------------------------------------------------------------------------------------
# NAB: anomaly detection on an industrial temperature sensor
# ----------------------------------------------------------------------------------------
def load_nab():
    """Return (series, windows, labels): sensor readings, labelled anomaly windows and exact anomaly times."""
    s = pd.read_csv(download("machine_temperature_system_failure.csv"), parse_dates=["timestamp"])
    # Real sensor data quirk: on 7 Jan 2014 the clock jumps back 55 minutes, repeating 12 timestamps.
    # Sort by time and average readings that share a timestamp.
    s = s.groupby("timestamp")["value"].mean().sort_index().rename("temperature")
    windows = json.loads(download("nab_combined_windows.json").read_text())[NAB_KEY]
    windows = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in windows]
    labels = [pd.Timestamp(t) for t in json.loads(download("nab_combined_labels.json").read_text())[NAB_KEY]]
    return s, windows, labels


if __name__ == "__main__":
    for name in SOURCES:
        download(name)
    df = load_ai4i()
    s, windows, labels = load_nab()
    print(f"AI4I: {len(df):,} runs, {df['failure'].sum()} failures ({df['failure'].mean():.1%})")
    print(f"NAB:  {len(s):,} readings from {s.index.min()} to {s.index.max()}, {len(labels)} labelled anomalies")

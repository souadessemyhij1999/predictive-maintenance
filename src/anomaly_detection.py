"""
Part 2 - Early warning from a single temperature sensor on a real industrial machine (NAB).

No failure labels are used to build the detectors - just like a new machine in a plant, where
you have sensor data but no history of failures. The labels are only used afterwards to score
how early each detector would have warned.

Detectors (all causal: each reading is judged only on data available at that moment)
  1. Static limit        - alarm when temperature falls below the 1st percentile of a calibration week
                            (the classic fixed alarm limit on an HMI panel)
  2. Rolling z-score     - alarm when temperature deviates > 3 standard deviations from its last 24 hours
  3. Isolation Forest    - ML anomaly score on level, short- and long-term deviations, slope and volatility,
                            learned from the calibration week

Alarm logic: a detector must stay in alarm for 30 minutes (6 readings) before an alarm is raised;
alarms closer than 6 hours apart are merged into one event.

    python src/anomaly_detection.py
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from data import load_nab

ROOT = Path(__file__).resolve().parent.parent
RESULTS, ASSETS = ROOT / "results", ROOT / "assets"
SEED = 42
CALIBRATION_DAYS = 7          # first week of data = "normal operation" reference (ends before the first label window)
PERSIST = 6                   # readings (30 min) a condition must last before an alarm is raised
MERGE = pd.Timedelta("6h")    # alarms closer than this belong to the same event
COLOURS = {"Static limit": "#9aa5b8", "Rolling z-score": "#f08c00", "Isolation Forest": "#1c69d4"}


def features(s: pd.Series) -> pd.DataFrame:
    """Causal features: only past readings are used (rolling windows look backwards)."""
    f = pd.DataFrame({"temperature": s})
    for w in ("1h", "6h", "24h"):
        f[f"dev_{w}"] = s - s.rolling(w).mean()
    f["slope_1h"] = s.diff(12)                       # change over the last hour
    f["std_6h"] = s.rolling("6h").std()
    return f.bfill()


def persistent(flag: pd.Series) -> pd.Series:
    """True only once the raw flag has been on for PERSIST consecutive readings."""
    return flag.astype(int).rolling(PERSIST).sum() >= PERSIST


def to_events(alarm: pd.Series) -> list:
    """Group alarm readings into events (start, end)."""
    times = alarm.index[alarm.values]
    events = []
    for t in times:
        if events and t - events[-1][1] <= MERGE:
            events[-1][1] = t
        else:
            events.append([t, t])
    return [tuple(e) for e in events]


def detectors(s: pd.Series):
    calib_end = s.index[0] + pd.Timedelta(days=CALIBRATION_DAYS)
    calib = s[:calib_end]
    out = {}

    # 1. static low-temperature limit
    out["Static limit"] = persistent(s < calib.quantile(0.01))

    # 2. rolling z-score vs. the previous 24 hours
    mu, sd = s.rolling("24h").mean().shift(1), s.rolling("24h").std().shift(1)
    z = ((s - mu) / sd).fillna(0)
    out["Rolling z-score"] = persistent(z.abs() > 3)

    # 3. Isolation Forest on causal features, learned on the calibration week
    f = features(s)
    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(f[:calib_end])
    score = pd.Series(-iso.score_samples(f), index=s.index)      # higher = more unusual
    limit = score[:calib_end].quantile(0.995)
    out["Isolation Forest"] = persistent(score > limit)
    return out, calib_end, score, limit


def evaluate(events, windows, labels, period_end, calib_end):
    rows = []
    for (w_start, w_end), label in zip(windows, labels):
        inside = [e for e in events if e[0] <= w_end and e[1] >= w_start]
        if inside:
            first = max(min(e[0] for e in inside), w_start)   # first alarm inside the window
            rows.append({"event": label.strftime("%d %b %Y %H:%M"), "detected": True,
                         "warning_hours_before_label": round((label - first).total_seconds() / 3600, 1)})
        else:
            rows.append({"event": label.strftime("%d %b %Y %H:%M"), "detected": False,
                         "warning_hours_before_label": np.nan})
    false_alarms = [e for e in events if e[0] > calib_end and not any(e[0] <= b and e[1] >= a for a, b in windows)]
    weeks = (period_end - calib_end).total_seconds() / (7 * 24 * 3600)
    return pd.DataFrame(rows), len(false_alarms), len(false_alarms) / weeks, false_alarms


def main():
    RESULTS.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    s, windows, labels = load_nab()
    alarms, calib_end, iso_score, iso_limit = detectors(s)
    print(f"{len(s):,} readings, {s.index[0]} to {s.index[-1]}; calibration week ends {calib_end}")

    summary, detail, events_by = [], [], {}
    for name, alarm in alarms.items():
        events = to_events(alarm)
        events_by[name] = events
        per_event, n_false, per_week, _ = evaluate(events, windows, labels, s.index[-1], calib_end)
        per_event.insert(0, "detector", name)
        detail.append(per_event)
        summary.append({
            "detector": name,
            "events_detected": f"{int(per_event['detected'].sum())}/{len(labels)}",
            "median_warning_hours": round(per_event["warning_hours_before_label"].median(), 1),
            "false_alarm_events": n_false,
            "false_alarms_per_week": round(per_week, 1),
        })
    summary = pd.DataFrame(summary)
    detail = pd.concat(detail, ignore_index=True)
    summary.to_csv(RESULTS / "anomaly_detector_comparison.csv", index=False)
    detail.to_csv(RESULTS / "anomaly_warning_times.csv", index=False)
    pd.DataFrame([{"detector": n, "alarm_start": a, "alarm_end": b} for n, ev in events_by.items() for a, b in ev]) \
        .to_csv(RESULTS / "anomaly_alarm_events.csv", index=False)
    print("\n" + summary.to_string(index=False))
    print("\n" + detail.to_string(index=False))

    # ---------- chart 1: full timeline ----------
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    ax = axes[0]
    ax.plot(s.index, s.values, color="#4a4a4a", lw=0.6)
    for i, (a, b) in enumerate(windows):
        ax.axvspan(a, b, color="#d6336c", alpha=0.15, label="labelled anomaly window" if i == 0 else None)
    for i, l in enumerate(labels):
        ax.axvline(l, color="#d6336c", lw=1, ls=":", label="labelled anomaly" if i == 0 else None)
    ax.axvspan(s.index[0], calib_end, color="#2e9e44", alpha=0.08, label="calibration week (normal reference)")
    ax.set(ylabel="temperature", title="Industrial machine temperature sensor - 11 weeks, 5-minute readings")
    ax.legend(frameon=False, loc="lower right", fontsize=8, ncol=3)
    ax = axes[1]
    for k, (name, events) in enumerate(events_by.items()):
        for a, b in events:
            ax.plot([a, max(b, a + pd.Timedelta("2h"))], [k, k], color=COLOURS[name], lw=7, solid_capstyle="butt")
    for a, b in windows:
        ax.axvspan(a, b, color="#d6336c", alpha=0.15)
    ax.set_yticks(range(len(events_by)), list(events_by))
    ax.set_ylim(-0.6, len(events_by) - 0.4)
    ax.set_title("Alarms raised by each detector", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "anomaly_timeline.png", dpi=150)

    # ---------- chart 2: zoom on each labelled event ----------
    fig, axes = plt.subplots(1, len(labels), figsize=(14, 3.6), sharey=True)
    for ax, (a, b), l in zip(axes, windows, labels):
        seg = s[a - pd.Timedelta("12h"): b]
        ax.plot(seg.index, seg.values, color="#4a4a4a", lw=0.8)
        ax.axvspan(a, b, color="#d6336c", alpha=0.12)
        ax.axvline(l, color="#d6336c", lw=1.2, ls=":")
        for k, (name, events) in enumerate(events_by.items()):
            starts = [max(e[0], a) for e in events if e[0] <= b and e[1] >= a]
            if starts:
                ax.axvline(min(starts), color=COLOURS[name], lw=2, alpha=0.9, label=name)
        ax.set_title(l.strftime("Labelled anomaly %d %b %Y"), fontsize=10)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("temperature")
    handles = [plt.Line2D([], [], color=c, lw=2) for c in COLOURS.values()] + \
              [plt.Line2D([], [], color="#d6336c", ls=":", lw=1.2)]
    fig.legend(handles, [f"first alarm: {n}" for n in COLOURS] + ["labelled anomaly"], loc="lower center", ncol=4,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(ASSETS / "anomaly_zoom.png", dpi=150)


if __name__ == "__main__":
    main()

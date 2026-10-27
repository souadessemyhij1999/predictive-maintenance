"""
Part 1 - Predict machine failures on the AI4I 2020 milling dataset.

Steps
  1. Hold out 20% of runs as an untouched test set (stratified, same failure rate).
  2. Compare a maintenance rule of thumb and three ML models with 5-fold cross-validation.
  3. Pick the alarm threshold that minimises expected maintenance cost (configs/costs.yaml).
  4. Evaluate once on the test set: cost vs. today's strategies, failures caught per failure mode.
  5. Explain the model: which signals drive the risk, and where in the operating range failures happen.

    python src/failure_prediction.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import ENGINEERED_FEATURES, FAILURE_MODES, RAW_FEATURES, load_ai4i

ROOT = Path(__file__).resolve().parent.parent
RESULTS, ASSETS, MODELS = ROOT / "results", ROOT / "assets", ROOT / "models"
SEED = 42
BLUE, GREY, ORANGE, RED, GREEN = "#1c69d4", "#9aa5b8", "#f08c00", "#d6336c", "#2e9e44"

LABELS = {"type": "Product quality (L/M/H)", "air_temp_k": "Air temperature", "process_temp_k": "Process temperature",
          "speed_rpm": "Rotational speed", "torque_nm": "Torque", "tool_wear_min": "Tool wear",
          "temp_diff_k": "Temperature difference*", "power_w": "Power*", "strain": "Strain (wear x torque)*"}


def make_pipeline(model, features, scale=False):
    numeric = [f for f in features if f != "type"]
    prep = ColumnTransformer([
        ("type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
        ("num", StandardScaler() if scale else "passthrough", numeric),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def candidates():
    all_features = RAW_FEATURES + ENGINEERED_FEATURES
    return {
        "Logistic regression": (make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced"),
                                              all_features, scale=True), all_features),
        "Random forest": (make_pipeline(RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                               class_weight="balanced_subsample", n_jobs=-1,
                                                               random_state=SEED), all_features), all_features),
        "Gradient boosting (raw sensors only)": (make_pipeline(HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, random_state=SEED), RAW_FEATURES), RAW_FEATURES),
        "Gradient boosting": (make_pipeline(HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, random_state=SEED), all_features), all_features),
    }


def tool_wear_rule(df, limit=200):
    """Today's simple rule: change the tool after a fixed number of minutes."""
    return (df["tool_wear_min"] >= limit).astype(int)


def cost(y, alarm, c):
    tp = int(((alarm == 1) & (y == 1)).sum())
    fp = int(((alarm == 1) & (y == 0)).sum())
    fn = int(((alarm == 0) & (y == 1)).sum())
    return tp * c["planned_intervention"] + fp * c["false_alarm"] + fn * c["unplanned_failure"], tp, fp, fn


def best_threshold(y, proba, c):
    grid = np.round(np.arange(0.01, 0.99, 0.01), 2)
    costs = [cost(y, (proba >= t).astype(int), c)[0] for t in grid]
    return float(grid[int(np.argmin(costs))]), grid, np.array(costs)


def best_hybrid(y, proba, wear, c, threshold):
    """Model alarm OR a planned tool change after `limit` minutes. Tool-wear failures happen at a random
    point between ~200 and 240 min, so the sensors can't foresee them - a planned change can."""
    options = {None: cost(y, (proba >= threshold).astype(int), c)[0]}
    for limit in range(190, 255, 5):
        options[limit] = cost(y, ((proba >= threshold) | (wear >= limit)).astype(int), c)[0]
    best = min(options, key=options.get)
    return best, options


def scores(y, proba, alarm):
    return {
        "PR-AUC": average_precision_score(y, proba) if proba is not None else np.nan,
        "ROC-AUC": roc_auc_score(y, proba) if proba is not None else np.nan,
        "Recall": recall_score(y, alarm), "Precision": precision_score(y, alarm, zero_division=0),
        "F1": f1_score(y, alarm),
    }


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def main():
    for d in (RESULTS, ASSETS, MODELS):
        d.mkdir(exist_ok=True)
    costs = yaml.safe_load(open(ROOT / "configs" / "costs.yaml"))

    df = load_ai4i()
    train, test = train_test_split(df, test_size=0.2, stratify=df["failure"], random_state=SEED)
    y_train, y_test = train["failure"].values, test["failure"].values
    print(f"Train: {len(train):,} runs ({y_train.sum()} failures) | Test: {len(test):,} runs ({y_test.sum()} failures)")

    # ---------- 1. model comparison with 5-fold cross-validation on the training set ----------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof, rows = {}, []
    rule = tool_wear_rule(train)
    rows.append({"model": "Rule of thumb: change tool at 200 min", **scores(y_train, None, rule)})
    for name, (pipe, feats) in candidates().items():
        proba = cross_val_predict(pipe, train[feats], y_train, cv=cv, method="predict_proba")[:, 1]
        oof[name] = proba
        rows.append({"model": name, **scores(y_train, proba, (proba >= 0.5).astype(int))})
        print(f"  {name:40s} PR-AUC {rows[-1]['PR-AUC']:.3f}")
    comparison = pd.DataFrame(rows).round(3)
    comparison.to_csv(RESULTS / "model_comparison_cv.csv", index=False)

    best_name = max(oof, key=lambda n: average_precision_score(y_train, oof[n]))
    pipe, feats = candidates()[best_name]
    print(f"Best model (cross-validated PR-AUC): {best_name}")

    # ---------- 2. cost-optimal threshold, chosen on cross-validated predictions ----------
    threshold, grid, grid_costs = best_threshold(y_train, oof[best_name], costs)
    tool_limit, hybrid_options = best_hybrid(y_train, oof[best_name], train["tool_wear_min"].values, costs, threshold)
    print(f"Cost-optimal threshold: {threshold:.2f} | planned tool change: {tool_limit or 'not worth it'} min")
    pd.DataFrame([{"tool_change_limit_min": k or "none", "cv_cost_eur": v} for k, v in hybrid_options.items()]) \
        .to_csv(RESULTS / "tool_change_search_cv.csv", index=False)

    # ---------- 3. final model, evaluated once on the untouched test set ----------
    pipe.fit(train[feats], y_train)
    proba_test = pipe.predict_proba(test[feats])[:, 1]
    model_alarm = (proba_test >= threshold).astype(int)
    tool_due = (test["tool_wear_min"].values >= tool_limit) if tool_limit else np.zeros(len(test), dtype=bool)
    alarm = (model_alarm.astype(bool) | tool_due).astype(int)  # recommended strategy
    test_scores = scores(y_test, proba_test, model_alarm)

    strategies = {
        "Run to failure (no prediction)": np.zeros(len(test), dtype=int),
        "Rule of thumb: change tool at 200 min": tool_wear_rule(test).values,
        "ML model, default threshold 0.50": (proba_test >= 0.5).astype(int),
        f"ML model, cost-optimal threshold {threshold:.2f}": model_alarm,
    }
    if tool_limit:
        strategies[f"ML model + planned tool change at {tool_limit} min"] = alarm
    cost_rows = []
    for name, a in strategies.items():
        total, tp, fp, fn = cost(y_test, a, costs)
        cost_rows.append({"strategy": name, "failures_caught": tp, "failures_missed": fn, "false_alarms": fp,
                          "total_cost_eur": total, "cost_per_1000_runs_eur": round(total / len(test) * 1000)})
    cost_df = pd.DataFrame(cost_rows)
    base = cost_df.loc[0, "total_cost_eur"]
    cost_df["saving_vs_run_to_failure"] = (1 - cost_df["total_cost_eur"] / base).round(3)
    cost_df.to_csv(RESULTS / "cost_comparison_test.csv", index=False)

    # Sensitivity: which strategy wins as the cost of an unplanned breakdown changes?
    sens_rows, breakeven = [], None
    if tool_limit:
        _, tp_m, fp_m, fn_m = cost(y_test, model_alarm, costs)
        _, tp_h, fp_h, fn_h = cost(y_test, alarm, costs)
        extra = (fp_h - fp_m) * costs["false_alarm"] + (tp_h - tp_m) * costs["planned_intervention"]
        breakeven = round(extra / max(fn_m - fn_h, 1))  # breakdown cost at which both strategies cost the same
        for c_fail in (5000, 10000, 20000, 50000, 100000):
            c2 = {**costs, "unplanned_failure": c_fail}
            m, h = cost(y_test, model_alarm, c2)[0], cost(y_test, alarm, c2)[0]
            sens_rows.append({"unplanned_failure_cost_eur": c_fail, "model_only_eur": m, "model_plus_tool_change_eur": h,
                              "cheaper": "model + tool change" if h < m else "model only"})
        pd.DataFrame(sens_rows).to_csv(RESULTS / "cost_sensitivity_test.csv", index=False)
        print(f"Break-even breakdown cost (test set): EUR {breakeven:,}")

    # Recall per failure mode. Only runs recorded as a machine failure count: in AI4I most "random
    # failure" (RNF) flags sit on runs labelled as no machine failure, so they are not real failures here.
    mode_rows = []
    for code, label in FAILURE_MODES.items():
        mask = (test[code].values == 1) & (y_test == 1)
        row = {"failure_mode": label, "code": code, "failures_in_test": int(mask.sum())}
        for tag, a in (("model", model_alarm), ("model_plus_tool_change", alarm)):
            caught = int((a[mask] == 1).sum())
            row[f"caught_{tag}"] = caught
            row[f"recall_{tag}"] = round(caught / mask.sum(), 3) if mask.sum() else np.nan
        mode_rows.append(row)
    modes = pd.DataFrame(mode_rows)
    modes.to_csv(RESULTS / "failure_mode_recall_test.csv", index=False)

    risk = test[["UDI", "product_id", "type", *[f for f in RAW_FEATURES if f != "type"], "failure"]].copy()
    risk["failure_risk"] = proba_test.round(4)
    risk["model_alarm"] = model_alarm
    risk["tool_change_due"] = tool_due.astype(int)
    risk["alarm"] = alarm
    risk["outcome"] = np.select([(alarm == 1) & (y_test == 1), (alarm == 1) & (y_test == 0), (alarm == 0) & (y_test == 1)],
                                ["caught", "false alarm", "missed"], "ok")
    risk.sort_values("failure_risk", ascending=False).to_csv(RESULTS / "test_risk_scores.csv", index=False)

    # Retrain on all data for the shipped model; the threshold stays the one chosen on training data
    pipe_all, _ = candidates()[best_name]
    pipe_all.fit(df[feats], df["failure"])
    joblib.dump(pipe_all, MODELS / "failure_model.joblib")
    card = {"model": best_name, "features": feats, "threshold": threshold, "tool_change_limit_min": tool_limit,
            "costs_eur": costs,
            "test_scores": {k: round(float(v), 3) for k, v in test_scores.items()},
            "breakeven_breakdown_cost_eur": breakeven,
            "trained_on": "AI4I 2020, all 10,000 runs (evaluation used an 80/20 split)"}
    (MODELS / "model_card.json").write_text(json.dumps(card, indent=2))

    # ---------- 4. explainability ----------
    imp = permutation_importance(pipe, test[feats], y_test, scoring="average_precision", n_repeats=20,
                                 random_state=SEED, n_jobs=-1)
    importance = pd.DataFrame({"feature": feats, "importance": imp.importances_mean, "std": imp.importances_std})
    importance = importance.sort_values("importance", ascending=False)
    importance.round(4).to_csv(RESULTS / "feature_importance_test.csv", index=False)

    # ---------- charts ----------
    # PR curves (cross-validated)
    fig, ax = plt.subplots(figsize=(7, 5.6))
    colours = {"Logistic regression": GREY, "Random forest": ORANGE, "Gradient boosting (raw sensors only)": "#7c8ba1",
               "Gradient boosting": BLUE}
    for name, proba in oof.items():
        p, r, _ = precision_recall_curve(y_train, proba)
        ax.plot(r, p, color=colours[name], lw=2.4 if name == best_name else 1.4,
                ls="--" if "raw" in name else "-", label=f"{name} (PR-AUC {average_precision_score(y_train, proba):.2f})")
    rr = comparison.iloc[0]
    ax.scatter([rr["Recall"]], [rr["Precision"]], color=RED, zorder=5, label="Rule: change tool at 200 min")
    ax.set(xlabel="Recall: share of failures caught", ylabel="Precision: share of alarms that are real",
           title="Precision vs. recall (5-fold cross-validation)", xlim=(0, 1.02), ylim=(0, 1.05))
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(ASSETS / "pr_curves.png", dpi=150)

    # Cost curve
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(grid, grid_costs / len(train) * 1000, color=BLUE, lw=2)
    ax.axvline(threshold, color=ORANGE, ls="--")
    ax.annotate(f"cost-optimal\nthreshold {threshold:.2f}", (threshold, grid_costs.min() / len(train) * 1000),
                textcoords="offset points", xytext=(12, 25), fontsize=9, color=ORANGE)
    ax.set(xlabel="Alarm threshold (predicted failure probability)", ylabel="Expected cost per 1,000 runs (EUR)",
           title="Choosing the alarm threshold by cost")
    style(ax)
    fig.tight_layout()
    fig.savefig(ASSETS / "cost_curve.png", dpi=150)

    # Cost comparison bar
    fig, ax = plt.subplots(figsize=(7.6, 0.75 * len(cost_df) + 1))
    labels = [s.replace("ML model, ", "ML model,\n").replace("ML model + ", "ML model +\n") for s in cost_df["strategy"]]
    vals = cost_df["cost_per_1000_runs_eur"]
    palette = [GREY, GREY, "#b9cff0", "#8fb3e8", BLUE][-len(cost_df):] if len(cost_df) == 5 else [GREY, GREY, "#8fb3e8", BLUE]
    bars = ax.barh(labels, vals, color=palette)
    ax.invert_yaxis()
    for b, v, s in zip(bars, vals, cost_df["saving_vs_run_to_failure"]):
        ax.text(v + vals.max() * 0.01, b.get_y() + b.get_height() / 2,
                f"EUR {v:,.0f}" + (f"  ({s:.0%} saved)" if s > 0 else ""), va="center", fontsize=9)
    ax.set(xlabel="Maintenance cost per 1,000 runs, test set (EUR, thousands)", xlim=(0, vals.max() * 1.38),
           title="Maintenance strategies compared (illustrative costs)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k"))
    style(ax)
    fig.tight_layout()
    fig.savefig(ASSETS / "cost_comparison.png", dpi=150)

    # Confusion matrix
    cm = confusion_matrix(y_test, alarm)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    names = [["Correctly no alarm", "False alarm"], ["Missed failure", "Failure caught"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{names[i][j]}", ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["No alarm", "Alarm"])
    ax.set_yticks([0, 1], ["No failure", "Failure"])
    ax.set(xlabel="Recommended strategy", ylabel="Reality", title=f"Test set ({len(test):,} runs)")
    fig.tight_layout()
    fig.savefig(ASSETS / "confusion_matrix.png", dpi=150)

    # Feature importance
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh([LABELS[f] for f in importance["feature"]], importance["importance"], xerr=importance["std"],
            color=[BLUE if f in ENGINEERED_FEATURES else GREY for f in importance["feature"]])
    ax.invert_yaxis()
    ax.set(xlabel="Drop in PR-AUC when the feature is shuffled", title="What drives the failure risk? (* = engineered)")
    style(ax)
    fig.tight_layout()
    fig.savefig(ASSETS / "feature_importance.png", dpi=150)

    # Failure map: risk across torque x tool wear for a typical L-type run
    torque = np.linspace(df["torque_nm"].min(), df["torque_nm"].max(), 140)
    wear = np.linspace(0, df["tool_wear_min"].max(), 120)
    T, W = np.meshgrid(torque, wear)
    med = df[RAW_FEATURES].drop(columns="type").median()
    grid_df = pd.DataFrame({"type": "L", "air_temp_k": med["air_temp_k"], "process_temp_k": med["process_temp_k"],
                            "torque_nm": T.ravel(), "tool_wear_min": W.ravel()})
    # keep power realistic: speed falls as torque rises (fitted on the data)
    slope, intercept = np.polyfit(df["torque_nm"], df["speed_rpm"], 1)
    grid_df["speed_rpm"] = intercept + slope * grid_df["torque_nm"]
    grid_df = grid_df.pipe(lambda d: d.assign(temp_diff_k=d.process_temp_k - d.air_temp_k,
                                              power_w=d.torque_nm * d.speed_rpm * 2 * np.pi / 60,
                                              strain=d.tool_wear_min * d.torque_nm))
    Z = pipe.predict_proba(grid_df[feats])[:, 1].reshape(T.shape)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.contourf(T, W, Z, levels=np.linspace(0, 1, 11), cmap="RdYlGn_r")
    fails = df[(df["failure"] == 1) & (df["type"] == "L")]
    ax.scatter(fails["torque_nm"], fails["tool_wear_min"], s=8, c="black", alpha=0.6, label="actual failures (type L)")
    fig.colorbar(im, ax=ax, label="predicted failure risk")
    ax.set(xlabel="Torque (Nm)", ylabel="Tool wear (min)", title="Failure risk map: low-quality (L) parts")
    ax.legend(frameon=True, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "failure_map.png", dpi=150)

    # ---------- summary ----------
    print("\nCross-validated comparison (training set):\n" + comparison.to_string(index=False))
    print("\nTest set:", {k: round(float(v), 3) for k, v in test_scores.items()})
    print("\nCost comparison (test set):\n" + cost_df.to_string(index=False))
    print("\nFailures caught per mode (test set):\n" + modes.to_string(index=False))
    print("\nPermutation importance:\n" + importance.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

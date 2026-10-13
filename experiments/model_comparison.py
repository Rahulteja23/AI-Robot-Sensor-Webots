"""
model_comparison.py
===================
Random Forest (original model) vs SVM (additional model), plus the dynamic speed profile.

    python experiments/model_comparison.py                 # offline held-out test + plots of the newest run
    python experiments/model_comparison.py path/to/run.csv

PART 1  Held-out test. The robot is simulated offline, every (features, Navigator-action) training sample is
        recorded, the first 70 % are used for training and the last 30 % (later in time) for testing.
        Metrics: accuracy, macro-F1 and recall of the rare, safety-relevant class STOP.
PART 2  From a Webots / simulation CSV in results/:
          * rolling agreement of RF and SVM with the Navigator
          * speed profile: planned speed (v_nav) vs executed speed, with the moving-obstacle modes
            (YIELD / BOOST / BACKOFF / DODGE) shaded
Outputs go next to the CSV (results/).
"""
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ai"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def part1(out_dir):
    import headless_validation as hv
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    X, Y = [], []
    for seed in (1, 2, 3):
        r = hv.run_trial(seed, use_ml=True)
        X += r["brain"].ml._X
        Y += r["brain"].ml._y
    X, Y = np.array(X), np.array([str(y) for y in Y])
    cut = int(len(X) * 0.7)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "SVM (RBF, C=1)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")),
    }
    res = {}
    for name, m in models.items():
        m.fit(X[:cut], Y[:cut])
        p = m.predict(X[cut:])
        stop = Y[cut:] == "STOP"
        res[name] = dict(
            accuracy=float(accuracy_score(Y[cut:], p)),
            macro_f1=float(f1_score(Y[cut:], p, average="macro", zero_division=0)),
            stop_recall=float(recall_score(Y[cut:], p, labels=["STOP"], average=None, zero_division=0)[0]) if stop.any() else None)
    print(f"held-out test: {cut} training / {len(X) - cut} test samples (chronological split)")
    for k, v in res.items():
        sr = "n/a (no STOP samples in the test part)" if v["stop_recall"] is None else f"{v['stop_recall']:.2f}"
        print(f"  {k:<16} accuracy {v['accuracy']:.3f}   macro-F1 {v['macro_f1']:.3f}   STOP recall {sr}")
    json.dump(dict(train=cut, test=len(X) - cut, models=res), open(os.path.join(out_dir, "model_comparison.json"), "w"), indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    labels = ["accuracy", "macro-F1", "STOP recall"]
    w = 0.36
    for i, (k, v) in enumerate(res.items()):
        vals = [v["accuracy"], v["macro_f1"], v["stop_recall"] or 0.0]
        b = ax.bar(np.arange(3) + (i - 0.5) * w, vals, w, label=k, color=["#2e7d32", "#1565c0"][i])
        for xi, val in zip(np.arange(3) + (i - 0.5) * w, vals):
            ax.text(xi, val + 0.01, f"{val:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.set_ylim(0, 1.12)
    ax.set_title("Random Forest vs SVM - held-out test (imitating the Navigator)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "model_comparison_rf_vs_svm.png"), dpi=140); plt.close()


def part2(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    base = os.path.splitext(csv_path)[0]
    t = df["sim_time_s"]
    nav = df["action"].astype(str).str.replace(r"\(.*\)", "", regex=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax = axes[0]
    for col, lab, c in (("ml_action", "Random Forest", "#2e7d32"), ("svm_action", "SVM", "#1565c0")):
        if col in df:
            agree = (df[col].astype(str) == nav).astype(float)
            ax.plot(t, agree.rolling(60, min_periods=10).mean() * 100, label=lab, color=c)
    ax.set_ylabel("agreement with Navigator (%)\n(rolling 60 steps)"); ax.set_ylim(0, 105); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    ax.set_title("Online agreement of the two ML models with the Navigator")

    ax = axes[1]
    if "v_nav" in df and "v_cmd" in df:
        ax.plot(t, pd.to_numeric(df["v_nav"], errors="coerce"), color="#9e9e9e", lw=1, label="planned speed (Navigator)")
        ax.plot(t, df["v_cmd"], color="#d84315", lw=1.4, label="executed speed")
        colors = {"YIELD": "#4fc3f7", "BOOST": "#ffb300", "BACKOFF": "#ff7043", "DODGE": "#ab47bc"}
        if "dyn_mode" in df:
            for mode, c in colors.items():
                m = (df["dyn_mode"] == mode).to_numpy()
                ax.fill_between(t, -0.08, 0.14, where=m, color=c, alpha=0.35, label=mode, step="mid")
    ax.set_ylim(-0.08, 0.14); ax.set_xlabel("time (s)"); ax.set_ylabel("speed (m/s)")
    ax.set_title("Dynamic speed control around moving cars"); ax.legend(ncol=3, fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = base + "_dynamic_speed_and_models.png"
    plt.savefig(out, dpi=140); plt.close()
    print("saved", out)


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else (sorted(glob.glob(os.path.join(ROOT, "results", "simulation_run_*.csv"))) or [None])[-1]
    out_dir = os.path.dirname(csv) if csv else os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    part1(out_dir)
    if csv:
        part2(csv)
    else:
        print("no results/simulation_run_*.csv found - run the Webots simulation first for part 2")

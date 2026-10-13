"""
analyze_results.py
==================
Post-simulation CSV analysis script.
Run this OUTSIDE Webots after a simulation run to visualise results.

Usage
-----
    python experiments/analyze_results.py                     # auto-picks latest CSV
    python experiments/analyze_results.py results/run_XYZ.csv # specific file

Outputs (saved next to the CSV)
--------------------------------
  * path_plot.png      - robot XY trajectory coloured by action type
  * sensor_plot.png    - IR front/left/right readings over time
  * risk_timeline.png  - fused risk level and ML confidence over time
  * action_pie.png     - pie chart of action distribution
  * summary_report.txt - text summary statistics
"""

import sys
import os
import glob

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "ai"))
import scene_config as scene

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install pandas matplotlib numpy")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Locate the CSV file
# ─────────────────────────────────────────────────────────────────────────────

def find_csv(path_hint=None):
    if path_hint and os.path.isfile(path_hint):
        return path_hint
    # Auto-discover: look for most recent run in results/
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    pattern     = os.path.join(results_dir, "simulation_run_*.csv")
    candidates  = sorted(glob.glob(pattern))
    if not candidates:
        print(f"No CSVs found in {results_dir}. Run the Webots simulation first.")
        sys.exit(1)
    latest = candidates[-1]
    print(f"[Analyze] Using latest CSV: {latest}")
    return latest


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

ACTION_COLORS = {
    "FORWARD"  : "#4CAF50",
    "LEFT"     : "#2196F3",
    "RIGHT"    : "#FF9800",
    "STOP"     : "#F44336",
}

def action_color(action_str):
    for k, c in ACTION_COLORS.items():
        if k in str(action_str):
            return c
    return "#9E9E9E"

RISK_MAP_NUM = {"low": 0, "medium": 1, "critical": 2}

def risk_to_num(risk_str):
    return RISK_MAP_NUM.get(str(risk_str).lower(), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze(csv_path: str):
    df = pd.read_csv(csv_path)
    # the goal of THIS run (random mission); old CSVs without the columns use the fixed goal
    GOAL = (float(df['goal_x'].iloc[-1]), float(df['goal_y'].iloc[-1])) if 'goal_x' in df else (0.0, 0.0)
    out_dir = os.path.dirname(csv_path)
    base    = os.path.splitext(os.path.basename(csv_path))[0]

    print(f"[Analyze] Loaded {len(df)} rows x {len(df.columns)} columns.")
    print(f"[Analyze] Simulation duration: {df['sim_time_s'].max():.1f}s")
    print(f"[Analyze] Steps recorded: {df['step'].max()}")

    # ── 1. Path trajectory plot: one panel per floor, drawn on the hospital floor plan ──────────────────────
    import plan_draw as pdw
    fl_col = df["floor"] if "floor" in df else pd.Series(1, index=df.index)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.6))
    last_floor = int(fl_col.iloc[-1]) - 1
    for f, ax in enumerate(axes):
        pdw.draw_floor(ax, f, lanes=True)
        part = df[fl_col == f + 1]
        if len(part):
            ax.scatter(part["pos_x"], part["pos_y"], c=[action_color(a) for a in part["action"]], s=7, alpha=0.9, linewidths=0, zorder=6)
            ax.plot(part["pos_x"].iloc[0], part["pos_y"].iloc[0], "g^", ms=11, zorder=7)
            ax.plot(part["pos_x"].iloc[-1], part["pos_y"].iloc[-1], "rs", ms=9, zorder=7)
        if f == last_floor:
            ax.plot(*GOAL, "k*", ms=16, zorder=8)
    patches = [mpatches.Patch(color=c, label=k) for k, c in ACTION_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=9)
    fig.suptitle("Robot path per floor (coloured by action; green triangle = where the robot entered the floor, "
                 "red square = where it left it, star = goal; dashed = lanes of people / trolleys)", fontsize=11, fontweight="bold")
    out = os.path.join(out_dir, f"{base}_path_plot.png")
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"[Analyze] Saved: {out}")

    # ── 2. IR sensor readings over time ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["sim_time_s"], df["front_val"], label="Front (ps0,ps7)", color="#E53935")
    ax.plot(df["sim_time_s"], df["left_val"],  label="Left  (ps5,ps6)", color="#1E88E5")
    ax.plot(df["sim_time_s"], df["right_val"], label="Right (ps1,ps2)", color="#43A047")
    ax.axhline(250, color="orange", linestyle="--", linewidth=1.2, label="Obstacle threshold (250)")
    ax.set_title("IR Proximity Sensor Readings Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Simulation time (s)"); ax.set_ylabel("IR Value (raw)")
    ax.legend(); ax.grid(True, alpha=0.3)
    out = os.path.join(out_dir, f"{base}_sensor_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Analyze] Saved: {out}")

    # ── 3. Risk timeline ─────────────────────────────────────────────────────
    df["risk_num"] = df["fused_risk"].apply(risk_to_num)
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()
    ax1.fill_between(df["sim_time_s"], df["risk_num"], alpha=0.35, color="#EF5350", label="Fused Risk")
    ax1.plot(df["sim_time_s"], df["risk_num"], color="#EF5350", linewidth=1.2)
    ax2.plot(df["sim_time_s"], df["confidence"], color="#7E57C2", linewidth=1.4, label="ML Confidence")
    ax1.set_yticks([0, 1, 2]); ax1.set_yticklabels(["low", "medium", "critical"])
    ax1.set_xlabel("Simulation time (s)"); ax1.set_ylabel("Fused Risk Level", color="#EF5350")
    ax2.set_ylabel("Confidence", color="#7E57C2"); ax2.set_ylim(0, 1.05)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Fused Risk Level and ML Confidence Over Time", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    out = os.path.join(out_dir, f"{base}_risk_timeline.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Analyze] Saved: {out}")

    # ── 4. Action distribution pie chart ─────────────────────────────────────
    def base_action(a):
        for k in ACTION_COLORS:
            if k in str(a):
                return k
        return "OTHER"
    df["base_action"] = df["action"].apply(base_action)
    counts = df["base_action"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    wedge_colors = [ACTION_COLORS.get(k, "#9E9E9E") for k in counts.index]
    ax.pie(counts.values, labels=counts.index, colors=wedge_colors,
           autopct="%1.1f%%", startangle=140,
           textprops={"fontsize": 12})
    ax.set_title("Action Distribution", fontsize=13, fontweight="bold")
    out = os.path.join(out_dir, f"{base}_action_pie.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Analyze] Saved: {out}")

    # ── 5. Text summary ───────────────────────────────────────────────────────
    ml_rows     = df[df["action"].str.contains("ML",       na=False)]
    heur_rows   = df[df["action"].str.contains("HEURISTIC", na=False)]

    summary = [
        "=" * 60,
        "SIMULATION SUMMARY REPORT",
        "=" * 60,
        f"CSV file        : {csv_path}",
        f"Total steps     : {len(df)}",
        f"Sim duration    : {df['sim_time_s'].max():.2f} s",
        "",
        "-- Navigation --",
        f"Goal            : ({GOAL[0]:+.3f}, {GOAL[1]:+.3f})   (success if final goal dist < {scene.GOAL_STOP} m)",
        f"Start position  : ({df['pos_x'].iloc[0]:.3f}, {df['pos_y'].iloc[0]:.3f})",
        f"End position    : ({df['pos_x'].iloc[-1]:.3f}, {df['pos_y'].iloc[-1]:.3f})",
        f"Final goal dist : {df['distance_to_goal'].iloc[-1]:.3f} m",
        "",
        "-- Actions --",
    ]
    for k, v in counts.items():
        pct = 100 * v / len(df)
        summary.append(f"  {k:<10}: {v:>5} steps ({pct:.1f}%)")
    summary += [
        "",
        "-- ML vs Heuristic --",
        f"  ML decisions       : {len(ml_rows)} ({100*len(ml_rows)/len(df):.1f}%)",
        f"  Heuristic decisions: {len(heur_rows)} ({100*len(heur_rows)/len(df):.1f}%)",
        "",
        "-- Risk Distribution --",
    ]
    for risk in ["low", "medium", "critical"]:
        cnt = (df["fused_risk"] == risk).sum()
        summary.append(f"  {risk:<10}: {cnt:>5} steps ({100*cnt/len(df):.1f}%)")
    summary += [
        "",
        "-- Sensor Stats --",
        f"  Max front IR     : {df['front_val'].max():.1f}",
        f"  Max left  IR     : {df['left_val'].max():.1f}",
        f"  Max right IR     : {df['right_val'].max():.1f}",
        f"  Mean confidence  : {df['confidence'].mean():.3f}",
        "=" * 60,
    ]

    report_path = os.path.join(out_dir, f"{base}_summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary))
    print(f"[Analyze] Saved: {report_path}")
    print("\n".join(summary))


if __name__ == "__main__":
    hint = sys.argv[1] if len(sys.argv) > 1 else None
    analyze(find_csv(hint))

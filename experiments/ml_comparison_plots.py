"""
ml_comparison_plots.py
=======================
Generates a rich comparison dashboard:
  - Before ML (heuristic phase, steps 1-135)
  - After ML   (ML-active phase, steps 136-1535)

Produces 6 high-quality charts saved to results/.
"""

import csv
import os
import collections
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────
import sys, glob
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "ai"))
import scene_config as sc
import collections
import plan_draw as pdw
_CANDIDATES = sorted(glob.glob(os.path.join(_ROOT, "results", "simulation_run_*.csv")))
if len(sys.argv) > 1:
    CSV_PATH = sys.argv[1]
elif _CANDIDATES:
    CSV_PATH = _CANDIDATES[-1]
else:
    sys.exit("No results/simulation_run_*.csv found - run the Webots simulation first.")
print("[plots] using", CSV_PATH)
OUT_DIR     = os.path.dirname(os.path.abspath(CSV_PATH))
ML_STEP     = None  # auto-detected below: first step where ML took over

PALETTE = {
    "heuristic_bg" : "#1A237E",   # deep indigo (heuristic phase)
    "ml_bg"        : "#1B5E20",   # deep green  (ML phase)
    "forward"      : "#66BB6A",   # green
    "right"        : "#FFA726",   # orange
    "left"         : "#42A5F5",   # blue
    "stop"         : "#EF5350",   # red
    "front"        : "#EF5350",
    "left_ir"      : "#42A5F5",
    "right_ir"     : "#66BB6A",
    "conf"         : "#CE93D8",
    "risk"         : "#FF7043",
    "threshold"    : "#FFD54F",
    "dark_bg"      : "#0D0D1A",
    "panel_bg"     : "#12122A",
    "text"         : "#E8E8FF",
    "grid"         : "#2A2A4A",
}

# ── Load CSV ────────────────────────────────────────────────────────────────
with open(CSV_PATH, encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

def fv(r, k, default=0.0):
    try: return float(r.get(k, default) or default)
    except: return default

steps       = [int(r['step']) for r in rows]
sim_times   = [fv(r,'sim_time_s') for r in rows]
pos_x       = [fv(r,'pos_x') for r in rows]
pos_y       = [fv(r,'pos_y') for r in rows]
front_vals  = [fv(r,'front_val') for r in rows]
left_vals   = [fv(r,'left_val') for r in rows]
right_vals  = [fv(r,'right_val') for r in rows]
confidence  = [fv(r,'confidence') for r in rows]
actions     = [r.get('action','') for r in rows]
fused_risk  = [r.get('fused_risk','low') for r in rows]
dist_goal   = [fv(r,'distance_to_goal') for r in rows]
GOAL = (float(rows[-1]['goal_x']), float(rows[-1]['goal_y'])) if 'goal_x' in rows[-1] else (0.0, 0.0)   # goal of THIS run
max_prox    = [fv(r,'max_proximity') for r in rows]

_ml_first = [int(r['step']) for r in rows if '(ML)' in r.get('action', '')]
ML_STEP = _ml_first[0] if _ml_first else int(rows[-1]['step']) + 1
print(f"[plots] ML first active at step {ML_STEP}")

RISK_NUM = {"low":0, "medium":1, "critical":2}
risk_nums = [RISK_NUM.get(r, 0) for r in fused_risk]

# Split into phases
def phase(data, ml_step=ML_STEP, before=True):
    if before:
        return [(s,v) for s,v in zip(steps, data) if s < ml_step]
    else:
        return [(s,v) for s,v in zip(steps, data) if s >= ml_step]

def xs(phase_data): return [p[0] for p in phase_data]
def ys(phase_data): return [p[1] for p in phase_data]

h_steps  = [s for s in steps if s <  ML_STEP]
ml_steps = [s for s in steps if s >= ML_STEP]
h_idx    = [i for i,s in enumerate(steps) if s <  ML_STEP]
ml_idx   = [i for i,s in enumerate(steps) if s >= ML_STEP]

def arr(data, idx): return [data[i] for i in idx]

# Action classification helper
def base_action(a):
    for k in ('FORWARD','LEFT','RIGHT','STOP'):
        if k in a: return k
    return 'OTHER'

h_actions  = [base_action(actions[i]) for i in h_idx]
ml_actions = [base_action(actions[i]) for i in ml_idx]

def action_counts(alist):
    c = collections.Counter(alist)
    return {k: c.get(k,0) for k in ['FORWARD','LEFT','RIGHT','STOP']}

h_ac  = action_counts(h_actions)
ml_ac = action_counts(ml_actions)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Grand Dashboard (3x2 grid)
# ──────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'text.color'       : PALETTE['text'],
    'axes.facecolor'   : PALETTE['panel_bg'],
    'figure.facecolor' : PALETTE['dark_bg'],
    'axes.edgecolor'   : PALETTE['grid'],
    'axes.labelcolor'  : PALETTE['text'],
    'xtick.color'      : PALETTE['text'],
    'ytick.color'      : PALETTE['text'],
    'grid.color'       : PALETTE['grid'],
    'axes.titlecolor'  : PALETTE['text'],
})

fig = plt.figure(figsize=(20, 14), facecolor=PALETTE['dark_bg'])
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.36,
                        left=0.07, right=0.97, top=0.91, bottom=0.06)

title = fig.suptitle(
    "AI Robot Sensor Webots — Before vs After ML Model Comparison Dashboard",
    fontsize=17, fontweight='bold', color=PALETTE['text'], y=0.97
)

# helper to shade phases
def shade_phases(ax, ml_boundary=ML_STEP, xdata=steps):
    ax.axvline(ml_boundary, color='#FFD54F', linewidth=1.8, linestyle='--', alpha=0.9)
    ax.axvspan(xdata[0], ml_boundary, alpha=0.06, color='#3949AB')
    ax.axvspan(ml_boundary, xdata[-1], alpha=0.06, color='#2E7D32')
    ax.text(ml_boundary * 0.5, ax.get_ylim()[1]*0.92, 'HEURISTIC',
            ha='center', va='top', fontsize=7.5, color='#90CAF9', alpha=0.85)
    ax.text((xdata[-1] + ml_boundary)/2, ax.get_ylim()[1]*0.92, 'ML ACTIVE',
            ha='center', va='top', fontsize=7.5, color='#A5D6A7', alpha=0.85)

# ── Panel 1: IR Sensor readings over time ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(steps, front_vals, color=PALETTE['front'],   linewidth=1.0, alpha=0.85, label='Front IR')
ax1.plot(steps, left_vals,  color=PALETTE['left_ir'], linewidth=1.0, alpha=0.85, label='Left IR')
ax1.plot(steps, right_vals, color=PALETTE['right_ir'],linewidth=1.0, alpha=0.85, label='Right IR')
ax1.axhline(250, color=PALETTE['threshold'], linewidth=1.5, linestyle=':', label='Obstacle threshold (250)')
ax1.set_title('IR Proximity Sensors Over Simulation Steps', fontsize=11, fontweight='bold')
ax1.set_xlabel('Step'); ax1.set_ylabel('IR Raw Value')
ax1.legend(fontsize=8, loc='upper right'); ax1.grid(True, alpha=0.25)
ax1.set_xlim(steps[0], steps[-1])
shade_phases(ax1)

# ── Panel 2: Action donut charts (Before / After) ────────────────────────────
ACT_COLORS = [PALETTE['forward'], PALETTE['right'], PALETTE['left'], PALETTE['stop']]
ACT_LABELS = ['FORWARD','RIGHT','LEFT','STOP']

ax2a = fig.add_subplot(gs[0, 2])
h_vals  = [h_ac[k] for k in ACT_LABELS]
ml_vals = [ml_ac[k] for k in ACT_LABELS]

# Stack bar chart showing action distribution
x     = np.arange(2)
bar_w = 0.5
totals = [sum(h_vals), sum(ml_vals)]
bottoms = [0, 0]
bars_h  = []
for idx, (k, c) in enumerate(zip(ACT_LABELS, ACT_COLORS)):
    hv  = h_vals[idx]  / totals[0] * 100
    mlv = ml_vals[idx] / totals[1] * 100
    b = ax2a.bar(x, [hv, mlv], bar_w, bottom=bottoms, color=c, label=k, alpha=0.9)
    bars_h.append(b)
    bottoms[0] += hv; bottoms[1] += mlv
ax2a.set_xticks(x); ax2a.set_xticklabels(['Heuristic\n(178 steps)', 'ML Active\n(1356 steps)'], fontsize=8)
ax2a.set_ylabel('% of steps'); ax2a.set_ylim(0,105)
ax2a.set_title('Action Distribution\nBefore vs After ML', fontsize=10, fontweight='bold')
ax2a.legend(fontsize=7, loc='upper right')
ax2a.grid(axis='y', alpha=0.25)

# ── Panel 3: ML Confidence over time ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(steps, confidence, color=PALETTE['conf'], linewidth=0.9, alpha=0.9)
ax3.fill_between(steps, confidence, alpha=0.25, color=PALETTE['conf'])
ax3.axhline(0.65, color='#FFD54F', linewidth=1.4, linestyle='--', label='ML override threshold (0.65)')
ax3.set_title('ML Fusion Confidence Over Simulation Steps', fontsize=11, fontweight='bold')
ax3.set_xlabel('Step'); ax3.set_ylabel('Confidence Score')
ax3.set_ylim(0, 1.05); ax3.legend(fontsize=8, loc='lower right')
ax3.grid(True, alpha=0.25); ax3.set_xlim(steps[0], steps[-1])
shade_phases(ax3)

# ── Panel 4: Obstacle encounters bar ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
h_obs_count  = sum(1 for v in arr(front_vals, h_idx)  if v > 250)
ml_obs_count = sum(1 for v in arr(front_vals, ml_idx) if v > 250)
h_obs_rate   = h_obs_count  / len(h_idx)  * 100
ml_obs_rate  = ml_obs_count / len(ml_idx) * 100

cats = ['Heuristic', 'ML Active']
vals = [h_obs_rate, ml_obs_rate]
colors_bar = ['#3949AB', '#2E7D32']
bars = ax4.bar(cats, vals, color=colors_bar, alpha=0.88, width=0.45, edgecolor='#555577')
for bar, v in zip(bars, vals):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3,
             f'{v:.1f}%', ha='center', va='bottom', fontsize=9, color=PALETTE['text'])
ax4.set_title('Obstacle Encounter Rate\n(front IR > 250)', fontsize=10, fontweight='bold')
ax4.set_ylabel('% of steps with obstacle'); ax4.grid(axis='y', alpha=0.25)
ax4.set_ylim(0, max(vals)*1.35)

# ── Panel 5: Robot XY path, coloured by phase (floor where the robot spent most steps) ──────────────────────
ax5 = fig.add_subplot(gs[2, :2])
_floors = [int(float(r.get('floor', 1) or 1)) for r in rows]
_main = collections.Counter(_floors).most_common(1)[0][0]
pdw.draw_floor(ax5, _main - 1, lanes=False, names=False, title=False)
_on = [i for i, f_ in enumerate(_floors) if f_ == _main]
hx = arr(pos_x, [i for i in h_idx if i in set(_on)]); hy = arr(pos_y, [i for i in h_idx if i in set(_on)])
mx = arr(pos_x, [i for i in ml_idx if i in set(_on)]); my = arr(pos_y, [i for i in ml_idx if i in set(_on)])
ax5.scatter(hx, hy, c='#7986CB', s=4, alpha=0.8, label='Heuristic phase', linewidths=0, zorder=6)
ax5.scatter(mx, my, c='#66BB6A', s=4, alpha=0.8, label='ML Active phase',  linewidths=0, zorder=6)
ax5.plot(pos_x[_on[0]],  pos_y[_on[0]],  'b^', ms=10, label='Enter floor', zorder=7)
ax5.plot(pos_x[_on[-1]], pos_y[_on[-1]], 'rs', ms=9, label='Leave floor',   zorder=7)
ax5.set_title(f'Robot XY Trajectory on {pdw.hp.FLOOR_LABELS[_main - 1]} (Blue=Heuristic, Green=ML)', fontsize=11, fontweight='bold')
ax5.legend(fontsize=8, loc='lower left')

# ── Panel 6: Comparison Stats Table ──────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

h_max_prox = arr(max_prox, h_idx)
ml_max_prox = arr(max_prox, ml_idx)
h_fwd_pct  = round(h_ac['FORWARD'] / len(h_idx) * 100, 1)
ml_fwd_pct = round(ml_ac['FORWARD'] / len(ml_idx) * 100, 1)

table_data = [
    ['Metric',              'Heuristic',          'ML Active'],
    ['Steps',               str(len(h_idx)),      str(len(ml_idx))],
    ['Duration',            f'{len(h_idx)*0.064:.1f}s', f'{len(ml_idx)*0.064:.1f}s'],
    ['% FORWARD',           f'{h_fwd_pct}%',      f'{ml_fwd_pct}%'],
    ['Obstacle encounters', str(h_obs_count),      str(ml_obs_count)],
    ['Obstacle rate',       f'{h_obs_rate:.1f}%', f'{ml_obs_rate:.1f}%'],
    ['Mean confidence',     f'{statistics.mean(arr(confidence,h_idx)):.3f}',
                            f'{statistics.mean(arr(confidence,ml_idx)):.3f}'],
    ['Max IR (front)',      f'{max(h_max_prox):.0f}',  f'{max(ml_max_prox):.0f}'],
    ['Mean IR (front)',     f'{statistics.mean(arr(front_vals,h_idx)):.1f}',
                            f'{statistics.mean(arr(front_vals,ml_idx)):.1f}'],
]

tbl = ax6.table(
    cellText   = table_data[1:],
    colLabels  = table_data[0],
    loc        = 'center',
    cellLoc    = 'center',
)
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
tbl.scale(1.0, 1.6)

header_color = '#283593'
h_col        = '#1A237E'
ml_col       = '#1B5E20'
row_bg       = '#12122A'

for (r,c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#3A3A6A')
    cell.set_text_props(color=PALETTE['text'])
    if r == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color='white', fontweight='bold')
    elif c == 1:
        cell.set_facecolor(h_col)
    elif c == 2:
        cell.set_facecolor(ml_col)
    else:
        cell.set_facecolor(row_bg)

ax6.set_title('Key Metrics Comparison', fontsize=10, fontweight='bold', pad=12)

out_dashboard = os.path.join(OUT_DIR, 'ml_comparison_dashboard.png')
plt.savefig(out_dashboard, dpi=160, bbox_inches='tight', facecolor=PALETTE['dark_bg'])
plt.close()
print("Saved dashboard:", out_dashboard)


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Max Proximity Rolling Average (Before vs After)
# ──────────────────────────────────────────────────────────────────────────────

def rolling_avg(data, window=15):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result

fig2, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=PALETTE['dark_bg'])
fig2.suptitle('Deep Dive: Heuristic vs ML Phase Analysis',
              fontsize=15, fontweight='bold', color=PALETTE['text'], y=0.98)

for ax in axes.flat:
    ax.set_facecolor(PALETTE['panel_bg'])
    ax.tick_params(colors=PALETTE['text'])
    for spine in ax.spines.values(): spine.set_edgecolor(PALETTE['grid'])

# ── 2a: Max proximity rolling avg per phase ──────────────────────────────────
ax2a = axes[0, 0]
h_mp  = arr(max_prox, h_idx)
ml_mp = arr(max_prox, ml_idx)
ax2a.plot(range(len(h_mp)),  rolling_avg(h_mp,  15), color='#7986CB', linewidth=2.0, label='Heuristic (rolling avg)')
ax2a.plot(range(len(ml_mp)), rolling_avg(ml_mp, 15), color='#66BB6A', linewidth=2.0, label='ML Active (rolling avg)')
ax2a.fill_between(range(len(h_mp)),  rolling_avg(h_mp,  15), alpha=0.2, color='#7986CB')
ax2a.fill_between(range(len(ml_mp)), rolling_avg(ml_mp, 15), alpha=0.2, color='#66BB6A')
ax2a.axhline(250, color=PALETTE['threshold'], linestyle='--', linewidth=1.2, label='Obstacle threshold')
ax2a.set_title('Max IR Proximity (Rolling Avg, w=15)', fontsize=10, fontweight='bold', color=PALETTE['text'])
ax2a.set_xlabel('Steps within phase', color=PALETTE['text']); ax2a.set_ylabel('IR Value', color=PALETTE['text'])
ax2a.legend(fontsize=8); ax2a.grid(True, alpha=0.22)

# ── 2b: Distance to goal over full run ───────────────────────────────────────
ax2b = axes[0, 1]
ax2b.plot(steps, dist_goal, color='#EF9A9A', linewidth=1.2, alpha=0.9)
ax2b.axvline(ML_STEP, color='#FFD54F', linewidth=2, linestyle='--', label=f'ML starts (step {ML_STEP})')
ax2b.fill_between(steps, dist_goal, alpha=0.2, color='#EF9A9A')
ax2b.axhline(0.15, color='#A5D6A7', linewidth=1.4, linestyle=':', label='Goal threshold (0.15m)')
ax2b.set_title('Distance to Goal Over Time', fontsize=10, fontweight='bold', color=PALETTE['text'])
ax2b.set_xlabel('Step', color=PALETTE['text']); ax2b.set_ylabel('Distance (m)', color=PALETTE['text'])
ax2b.legend(fontsize=8); ax2b.grid(True, alpha=0.22); ax2b.set_xlim(steps[0], steps[-1])
ax2b.text(ML_STEP*0.45, max(dist_goal)*0.85, 'HEURISTIC', ha='center', fontsize=8, color='#90CAF9')
ax2b.text((steps[-1]+ML_STEP)*0.5, max(dist_goal)*0.85, 'ML ACTIVE', ha='center', fontsize=8, color='#A5D6A7')

# ── 2c: Confidence distribution histogram ────────────────────────────────────
ax2c = axes[1, 0]
h_conf_arr  = arr(confidence, h_idx)
ml_conf_arr = arr(confidence, ml_idx)
bins = np.linspace(0, 1, 25)
ax2c.hist(h_conf_arr,  bins=bins, alpha=0.7, color='#7986CB', label='Heuristic', density=True)
ax2c.hist(ml_conf_arr, bins=bins, alpha=0.7, color='#66BB6A', label='ML Active',  density=True)
ax2c.axvline(0.65, color=PALETTE['threshold'], linewidth=1.5, linestyle='--', label='Override threshold')
ax2c.set_title('Confidence Score Distribution', fontsize=10, fontweight='bold', color=PALETTE['text'])
ax2c.set_xlabel('Confidence', color=PALETTE['text']); ax2c.set_ylabel('Density', color=PALETTE['text'])
ax2c.legend(fontsize=8); ax2c.grid(True, alpha=0.22)

# ── 2d: Action proportions grouped bar ───────────────────────────────────────
ax2d = axes[1, 1]
act_keys = ['FORWARD', 'RIGHT', 'LEFT', 'STOP']
act_cols_list = [PALETTE['forward'], PALETTE['right'], PALETTE['left'], PALETTE['stop']]
x = np.arange(len(act_keys))
w = 0.35
h_pcts  = [h_ac[k]/max(len(h_idx),1)*100  for k in act_keys]
ml_pcts = [ml_ac[k]/max(len(ml_idx),1)*100 for k in act_keys]

b1 = ax2d.bar(x - w/2, h_pcts,  w, label='Heuristic', color='#3949AB', alpha=0.88)
b2 = ax2d.bar(x + w/2, ml_pcts, w, label='ML Active',  color='#2E7D32', alpha=0.88)
for bar, v in list(zip(b1,h_pcts)) + list(zip(b2,ml_pcts)):
    if v > 1:
        ax2d.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                  f'{v:.1f}%', ha='center', va='bottom', fontsize=7.5, color=PALETTE['text'])
ax2d.set_xticks(x); ax2d.set_xticklabels(act_keys, color=PALETTE['text'])
ax2d.set_title('Action Proportion Comparison (%)', fontsize=10, fontweight='bold', color=PALETTE['text'])
ax2d.set_ylabel('% of phase steps', color=PALETTE['text']); ax2d.set_ylim(0, 100)
ax2d.legend(fontsize=9); ax2d.grid(axis='y', alpha=0.22)

out_deep = os.path.join(OUT_DIR, 'ml_deep_dive.png')
plt.savefig(out_deep, dpi=160, bbox_inches='tight', facecolor=PALETTE['dark_bg'])
plt.close()
print("Saved deep dive:", out_deep)

print("\nAll plots generated successfully!")

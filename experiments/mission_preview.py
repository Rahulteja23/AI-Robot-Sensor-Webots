"""
mission_preview.py
==================
Shows where random hospital missions start (blue) and end (green) on the three floors.

    python experiments/mission_preview.py [n_missions]      ->  results/mission_distribution.png

A mission = "deliver from <room, floor> to <room, floor>" (60 % across floors via a lift). Points are random positions
inside the rooms, clear of walls and furniture, and a route must exist (ai/mission.py). The same seed always gives
the same mission.
"""
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ai"))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hospital as hp
import mission as ms
import plan_draw as pdw

n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
M = [ms.random_mission(s) for s in range(1, n + 1)]
cross = sum(m["floor_s"] != m["floor_g"] for m in M)

fig, axes = plt.subplots(1, 3, figsize=(20, 5.6))
for f, ax in enumerate(axes):
    pdw.draw_floor(ax, f, lanes=False, names=True, fontsize=5.5)
    ax.scatter([m["start"][0] for m in M if m["floor_s"] == f], [m["start"][1] for m in M if m["floor_s"] == f],
               s=16, c="#1e5eff", zorder=6, label="random START")
    ax.scatter([m["goal"][0] for m in M if m["floor_g"] == f], [m["goal"][1] for m in M if m["floor_g"] == f],
               s=16, c="#1db954", zorder=6, label="random GOAL")
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle(f"{n} random hospital missions: {cross} cross floors (lift), route length median "
             f"{statistics.median(m['route_length'] for m in M):.1f} m, max {max(m['route_length'] for m in M):.1f} m", fontsize=11)
plt.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(ROOT, "results", "mission_distribution.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=120)
print("saved", out)

"""Drawing helpers: hospital floor plans (used by the analysis plots, GIFs and previews)."""
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ai"))

import matplotlib.patches as mp

import hospital as hp


def _rect(cx, cy, yaw, L, W, **kw):
    c, s = math.cos(yaw), math.sin(yaw)
    return mp.Rectangle((cx - (L / 2) * c + (W / 2) * s, cy - (L / 2) * s - (W / 2) * c), L, W, angle=math.degrees(yaw), **kw)


def draw_floor(ax, f, lanes=True, names=True, title=True, fontsize=6.5):
    """Draw floor f (index) in floor-local coordinates on ax."""
    fl = hp.FLOORS[f]
    ax.set_xlim(hp.X0 - 0.08, hp.X1 + 0.08)
    ax.set_ylim(hp.Y0 - 0.08, hp.Y1 + 0.08)
    ax.set_aspect("equal")
    ax.add_patch(mp.Rectangle((hp.X0, hp.Y0), hp.X1 - hp.X0, hp.Y1 - hp.Y0, fc="#e9e9ee", ec="none", zorder=0))
    for r in fl.rooms:
        x0, x1, ya, yb = r.rect
        ax.add_patch(mp.Rectangle((x0, ya), x1 - x0, yb - ya, fc=hp.ROOM_COLORS[r.kind], ec="none", zorder=0.5))
        if names:
            ax.text((x0 + x1) / 2, (ya + yb) / 2 + (0.45 * r.side), r.name, ha="center", va="center", fontsize=fontsize,
                    color="#333", zorder=2, weight="bold")
    ax.add_patch(mp.Rectangle((hp.X0, -hp.HW), hp.X1 - hp.X0, 2 * hp.HW, fc="#d6d6dc", ec="none", zorder=0.6))
    ax.add_patch(mp.Rectangle((-hp.HW, hp.Y0), 2 * hp.HW, hp.Y1 - hp.Y0, fc="#d6d6dc", ec="none", zorder=0.6))
    for lf in fl.lifts:
        ax.add_patch(mp.Rectangle((-hp.HW, min(lf.side * hp.LIFT_Y, lf.side * hp.Y1)), 2 * hp.HW, abs(hp.Y1 - hp.LIFT_Y),
                                  fc="#c5d8f2", ec="none", zorder=0.7))
        ax.text(lf.pad[0], lf.pad[1], "LIFT", ha="center", va="center", fontsize=fontsize - 1.5, color="#1a4f9c", zorder=2)
    for w in fl.walls:
        ax.add_patch(_rect(w["x"], w["y"], w["yaw"], w["L"], w["W"], fc="#3a3a44", ec="none", zorder=3))
    for o in fl.furniture:
        ax.add_patch(_rect(o["x"], o["y"], o["yaw"], o["size"][0], o["size"][1], fc=o["color"], ec="#555", lw=0.4, zorder=3))
    if lanes:
        for a in fl.agents:
            ax.plot([a["p0"][0], a["p1"][0]], [a["p0"][1], a["p1"][1]], ls="--", lw=1.2, color=a["color"], zorder=2.5)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(fl.label, fontsize=9)

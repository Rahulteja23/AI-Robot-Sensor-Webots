"""
headless_validation.py
======================
Offline test of the COMPLETE robot brain (ai/robot_brain.py) in the 3-floor hospital - no Webots needed.

What is simulated
  * the real floor plans (walls, doors, furniture) of ai/hospital.py and ALL moving people / trolleys of all three floors
    (same DynamicMover code as the Webots supervisor)
  * 24 long-range + 8 IR distance sensors by ray casting (with noise), differential-drive kinematics with wheel slip,
    GPS + IMU with noise
  * lift rides: the robot waits 3 s in the cabin and appears on the other floor (as the Webots controller does)

What is measured per run:  success (goal reached)  |  collisions  |  time  |  path length  |  closest approach

    python experiments/headless_validation.py                  # 30 random missions
    python experiments/headless_validation.py --runs 100       # more runs
    python experiments/headless_validation.py --no-yield       # stress: nobody ever brakes for the robot
    python experiments/headless_validation.py --fixed          # the demo mission only (Pharmacy -> Operating Theatre 1)
    python experiments/headless_validation.py --plot --gif     # trajectory figure + animation of the first run

NOTE: an approximation of Webots physics.  It validates the navigation LOGIC and the building layout;
the final proof is of course the Webots run.
"""

import argparse
import math
import os
import random
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ai"))
sys.path.insert(0, os.path.dirname(__file__))

import scene_config as sc                       # noqa: E402
import hospital as hp                           # noqa: E402
import mission as ms                            # noqa: E402
import route_planner as rp                      # noqa: E402
from navigator import (RS_COUNT, RS_ANGLE, RS_RADIUS, RS_MAX,       # noqa: E402
                       PS_ANGLES, PS_RADIUS, _PS_TABLE)
from robot_brain import RobotBrain              # noqa: E402

CTRL_DT = 0.064          # controller step  (2 x basicTimeStep 32 ms)
PHYS_DT = 0.032          # physics / supervisor step
WHEEL_R, AXLE = 0.0205, 0.052
LIFT_RIDE_S = 3.0        # seconds the robot waits inside the lift


# ═════════════════════════════════════════════════════════════════════════════
# Ray casting
# ═════════════════════════════════════════════════════════════════════════════

def ray_rect(ox, oy, dx, dy, cx, cy, yaw, L, W):
    """Distance along ray (ox,oy)+t*(dx,dy) to an oriented rectangle, or None."""
    c, s = math.cos(yaw), math.sin(yaw)
    rx, ry = ox - cx, oy - cy
    lox, loy = rx * c + ry * s, -rx * s + ry * c
    ldx, ldy = dx * c + dy * s, -dx * s + dy * c
    tmin, tmax = 0.0, 1e9
    for o, d, h in ((lox, ldx, L / 2.0), (loy, ldy, W / 2.0)):
        if abs(d) < 1e-12:
            if abs(o) > h:
                return None
        else:
            t1, t2 = (-h - o) / d, (h - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin, tmax = max(tmin, t1), min(tmax, t2)
            if tmin > tmax:
                return None
    return tmin


def ps_value(d):
    """distance (m) -> raw E-puck IR value."""
    if d is None or d >= 0.1:
        return 0.0
    for (d0, v0), (d1, v1) in zip(_PS_TABLE, _PS_TABLE[1:]):
        if d0 <= d <= d1:
            return v0 + (d - d0) / (d1 - d0) * (v1 - v0)
    return 4095.0


# ═════════════════════════════════════════════════════════════════════════════
# World + robot (one floor at a time; the other floors' people keep moving)
# ═════════════════════════════════════════════════════════════════════════════

class HeadlessWorld:
    def __init__(self, seed, static, movers, start, start_yaw=None, noise=True, other_movers=()):
        """static: footprints [(cx, cy, yaw, L, W), ...] of the current floor (walls included);
        movers: DynamicMovers of the current floor; other_movers: movers of the other floors (just kept moving)."""
        self.rng = random.Random(seed)
        self.noise = noise
        self.static = list(static)
        self.movers = list(movers)
        self.other_movers = list(other_movers)
        self.x, self.y = start[0] + self.rng.uniform(-0.01, 0.01), start[1] + self.rng.uniform(-0.01, 0.01)
        self.yaw = sc.wrap((self.rng.uniform(-math.pi, math.pi) if start_yaw is None else start_yaw)
                           + (0.0 if start_yaw is None else self.rng.uniform(-0.05, 0.05)))
        self.k_left = self.rng.gauss(1.0, 0.02) if noise else 1.0     # wheel slip / gain error
        self.k_right = self.rng.gauss(1.0, 0.02) if noise else 1.0
        self.t = 0.0
        self.collisions = 0
        self._in_collision = False
        self.min_clearance = 9.0

    # -- geometry ----------------------------------------------------------
    def obstacles(self):
        obs = list(self.static)
        for m in self.movers:
            obs.append((m.x, m.y, m.yaw, m.L, m.W))
        return obs

    def clearance(self):
        """Smallest gap between the robot body and anything (m); <0 = collision."""
        best = 9.0
        for cx, cy, yaw, L, W in self.obstacles():
            if abs(cx - self.x) > 1.5 or abs(cy - self.y) > 1.5:
                continue
            best = min(best, sc.rect_distance(self.x, self.y, cx, cy, yaw, L, W) - sc.ROBOT_RADIUS)
        return best

    def change_floor(self, static, movers, other_movers, pos, yaw):
        """Lift ride finished: the robot appears on another floor."""
        self.static, self.movers, self.other_movers = list(static), list(movers), list(other_movers)
        self.x, self.y, self.yaw = pos[0], pos[1], yaw
        self._in_collision = False

    # -- sensors ------------------------------------------------------------
    def sense(self):
        obs = [o for o in self.obstacles() if math.hypot(o[0] - self.x, o[1] - self.y) < 0.75 + 0.5 * max(o[3], o[4])]
        rng = self.rng

        def cast(angle, radius, max_range):
            a = self.yaw + angle
            dx, dy = math.cos(a), math.sin(a)
            ox = self.x + radius * math.cos(a)
            oy = self.y + radius * math.sin(a)
            best = 1e9
            for cx, cy, yaw, L, W in obs:
                t = ray_rect(ox, oy, dx, dy, cx, cy, yaw, L, W)
                if t is not None and t < best:
                    best = t
            return best if best < max_range else None

        rs = []
        for k in range(RS_COUNT):
            d = cast(k * RS_ANGLE, RS_RADIUS, RS_MAX)
            if d is None:
                rs.append(RS_MAX)
            else:
                if self.noise:
                    d += rng.gauss(0.0, 0.003)
                rs.append(max(0.0, min(RS_MAX, d)))
        ps = []
        for j in range(8):
            d = cast(PS_ANGLES[j], PS_RADIUS, 0.1)
            v = ps_value(d)
            if self.noise and v > 0:
                v *= rng.gauss(1.0, 0.04)
            ps.append(max(0.0, v))

        n = 0.001 if self.noise else 0.0
        yaw_noise = rng.gauss(0.0, 0.004) if self.noise else 0.0
        return {
            "distance": ps,
            "range": rs,
            "camera": {},
            "position": (self.x + rng.gauss(0, n), self.y + rng.gauss(0, n), 0.0),
            "heading": sc.wrap(self.yaw + yaw_noise),
        }

    # -- dynamics ---------------------------------------------------------------
    def advance(self, wl, wr, dt):
        """Advance the world by dt using wheel speeds wl, wr (rad/s)."""
        n = max(1, int(round(dt / PHYS_DT)))
        h = dt / n
        for _ in range(n):
            v = WHEEL_R * (self.k_left * wl + self.k_right * wr) / 2.0
            w = WHEEL_R * (self.k_right * wr - self.k_left * wl) / AXLE
            self.yaw = sc.wrap(self.yaw + w * h)
            self.x += v * math.cos(self.yaw) * h
            self.y += v * math.sin(self.yaw) * h
            for m in self.movers:
                m.step(h, (self.x, self.y))
            for m in self.other_movers:
                m.step(h, None)
            self.t += h
            c = self.clearance()
            self.min_clearance = min(self.min_clearance, c)
            if c < 0.0:
                if not self._in_collision:
                    self.collisions += 1
                self._in_collision = True
            else:
                self._in_collision = False


# ═════════════════════════════════════════════════════════════════════════════
# One complete mission
# ═════════════════════════════════════════════════════════════════════════════

def run_trial(seed, yield_enabled=True, noise=True, log_dir=None, record=False,
              max_time=600.0, use_ml=True, use_motion_prediction=True, verbose=False,
              use_dynamic_safety=True, use_svm=True, random_mission=None, mission=None):
    """random_mission: True / False (demo mission) / None (None -> scene_config.RANDOM_MISSION).
    A random mission is drawn from the trial seed, so every run is reproducible."""
    if mission is None:
        use_random = sc.RANDOM_MISSION if random_mission is None else random_mission
        mission = ms.random_mission(seed) if use_random else ms.fixed_mission()
    legs = mission["legs"]
    rng = random.Random(seed * 7919 + 13)
    movers = [fl.make_movers(phases=[rng.random() for _ in fl.agents], yield_enabled=yield_enabled) for fl in hp.FLOORS]

    f0 = mission["floor_s"]
    world = HeadlessWorld(seed, hp.FLOORS[f0].footprints(), movers[f0], mission["start"], mission["yaw"], noise,
                          other_movers=[m for f in range(hp.N_FLOORS) if f != f0 for m in movers[f]])
    brain = RobotBrain(CTRL_DT, results_dir=log_dir, verbose=verbose, use_ml=use_ml,
                       use_motion_prediction=use_motion_prediction,
                       use_dynamic_safety=use_dynamic_safety, use_svm=use_svm)

    def begin(li):
        lg = legs[li]
        brain.start_leg(lg["floor"], rp.RouteFollower(lg["floor"], lg["path"], lg["end"], lg["stop"]),
                        final=(li == len(legs) - 1), leg_index=li)
    begin(0)

    path, frames = [], []
    steps, li, result = 0, 0, None
    leg_times, rides = [], 0
    while world.t < max_time:
        data = world.sense()
        out = brain.step(world.t, data)
        steps += 1
        if record and steps % 4 == 0:
            path.append((world.t, brain.floor, world.x, world.y, [[(m.x, m.y, m.yaw) for m in mv] for mv in movers],
                         out.get("mode", ""), out.get("dist", 0.0), world.yaw, world.collisions, out.get("v", 0.0)))
        if out["done"]:
            result = "success"
            leg_times.append(round(world.t, 1))
            break
        if out.get("leg_done"):                                     # ride the lift to the next floor
            lg = legs[li]
            for _ in range(int(LIFT_RIDE_S / CTRL_DT)):
                world.advance(0.0, 0.0, CTRL_DT)
            leg_times.append(round(world.t, 1))
            li += 1
            rides += 1
            nf = legs[li]["floor"]
            lf = lg["lift"]
            world.change_floor(hp.FLOORS[nf].footprints(), movers[nf],
                               [m for f in range(hp.N_FLOORS) if f != nf for m in movers[f]], lf.pad, lf.yaw_out)
            begin(li)
            continue
        world.advance(out["left"], out["right"], CTRL_DT)
    brain.close()
    if result is None:
        result = "timeout"
    ok = result == "success" and world.collisions == 0
    allm = [m for mv in movers for m in mv]
    return dict(seed=seed, ok=ok, result=result, mission=mission, collisions=world.collisions, time=world.t,
                path_len=brain.path_length, min_clear=world.min_clearance, brakes=sum(m.brake_events for m in allm),
                escapes=brain.navigator.escape_count, rides=rides, leg_times=leg_times, replans=sum(
                    l_.get("replans", 0) for l_ in legs), path=path, world=world, brain=brain)


# ═════════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════════

def plot_run(res, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    import plan_draw as pdw

    m_ = res["mission"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.4))
    for f, ax in enumerate(axes):
        pdw.draw_floor(ax, f)
    for t, f, x, y, mv, mode, dist, yaw, nc, v in res["path"]:
        pass
    for f in range(3):
        pts = [(p[2], p[3], p[0]) for p in res["path"] if p[1] == f]
        if pts:
            sc_ = axes[f].scatter([p[0] for p in pts], [p[1] for p in pts], c=[p[2] for p in pts], cmap="viridis", s=7, zorder=6)
    axes[m_["floor_s"]].add_patch(mp.Circle(m_["start"], 0.09, fc="#3366ff", alpha=0.55, zorder=5))
    axes[m_["floor_s"]].text(*m_["start"], "START", ha="center", va="center", fontsize=7, fontweight="bold", zorder=7)
    axes[m_["floor_g"]].add_patch(mp.Circle(m_["goal"], 0.09, fc="#22cc44", alpha=0.6, zorder=5))
    axes[m_["floor_g"]].text(*m_["goal"], "GOAL", ha="center", va="center", fontsize=7, fontweight="bold", zorder=7)
    fig.suptitle(f"Seed {res['seed']}: {m_['start_name']}  ->  {m_['goal_name']}   |   {res['result'].upper()}, "
                 f"{res['collisions']} collisions, {res['time']:.0f} s, path {res['path_len']:.1f} m, lift rides {res['rides']}",
                 fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_png, dpi=110)
    plt.close()


def make_gif(res, out_gif, fps=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    from matplotlib.animation import FuncAnimation, PillowWriter
    import plan_draw as pdw

    m_ = res["mission"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), dpi=62)
    fig.patch.set_facecolor("#14141f")
    for f, ax in enumerate(axes):
        pdw.draw_floor(ax, f, lanes=False, names=True, fontsize=5)
        ax.set_title(hp.FLOOR_LABELS[f], color="w", fontsize=9)
    axes[m_["floor_s"]].add_patch(mp.Circle(m_["start"], 0.09, fc="#3366ff", alpha=0.6, zorder=5))
    axes[m_["floor_g"]].add_patch(mp.Circle(m_["goal"], 0.09, fc="#22cc44", alpha=0.7, zorder=5))
    dyn = []
    for f in range(3):
        row = []
        for a in hp.FLOORS[f].agents:
            row.append(axes[f].add_patch(pdw._rect(0, 0, 0, a["size"][0], a["size"][1], fc=a["color"], ec="w", lw=1.0, zorder=4)))
        dyn.append(row)
    trails = [ax.plot([], [], "-", color="#1e88e5", lw=2, zorder=5)[0] for ax in axes]
    bodies = [ax.add_patch(mp.Circle((0, 0), sc.ROBOT_RADIUS * 1.6, fc="#ffeb3b", ec="k", lw=1.2, zorder=7, visible=False)) for ax in axes]
    noses = [ax.plot([], [], "-", color="k", lw=2, zorder=8)[0] for ax in axes]
    txt = fig.text(0.5, 0.905, "", ha="center", color="w", fontsize=9, family="monospace")
    fig.suptitle(f"{m_['start_name']}  ->  {m_['goal_name']}", color="w", fontsize=10, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    frames = res["path"]
    xs = [[], [], []]
    ys = [[], [], []]

    def upd(i):
        t, f, x, y, mv, mode, dist, yaw, nc, v = frames[min(i, len(frames) - 1)]
        for k in range(3):
            bodies[k].set_visible(k == f)
            noses[k].set_data([], [])
        xs[f].append(x); ys[f].append(y); trails[f].set_data(xs[f], ys[f])
        bodies[f].center = (x, y)
        r = sc.ROBOT_RADIUS * 1.6
        noses[f].set_data([x, x + r * math.cos(yaw)], [y, y + r * math.sin(yaw)])
        for k in range(3):
            for patch, a, (mx, my, myaw) in zip(dyn[k], hp.FLOORS[k].agents, mv[k]):
                L, W = a["size"][0], a["size"][1]
                c, s_ = math.cos(myaw), math.sin(myaw)
                patch.set_xy((mx - (L / 2) * c + (W / 2) * s_, my - (L / 2) * s_ - (W / 2) * c))
                patch.angle = math.degrees(myaw)
        txt.set_text(f"t={t:6.1f}s  {hp.FLOOR_LABELS[f]:<17} mode {mode:<8} speed {v:+.3f} m/s  collisions {nc}")
        return ()
    n = len(frames)
    anim = FuncAnimation(fig, upd, frames=n + 12, interval=1000 // fps, blit=False, repeat=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--no-yield", action="store_true", help="people / trolleys never brake (stress test)")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--no-motion-prediction", action="store_true")
    ap.add_argument("--no-safety", action="store_true", help="disable the moving-obstacle safety layer")
    ap.add_argument("--fixed", action="store_true", help="the demo mission instead of a random mission per run")
    ap.add_argument("--plot", action="store_true", help="save trajectory figure of the first run")
    ap.add_argument("--gif", action="store_true", help="save an animated GIF of the first run")
    ap.add_argument("--csv", action="store_true", help="write the first run's CSV into results/headless_validation/")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "results", "headless_validation")
    os.makedirs(out_dir, exist_ok=True)
    tag = "nobody brakes (stress test)" if args.no_yield else "people brake for obstacles ahead (but the trolleys never do)"
    print(f"Headless validation: {args.runs} runs, {tag}, noise={'off' if args.no_noise else 'on'}")
    print("-" * 100)
    print(f"{'seed':>5} {'result':>8} {'coll':>5} {'time(s)':>8} {'path(m)':>8} {'gap(cm)':>8} {'rides':>5} {'esc':>4}  mission")
    n_ok, rows, t0, first = 0, [], time.time(), None
    for i in range(args.runs):
        seed = args.seed0 + i
        want_extra = (i == 0 and (args.plot or args.csv or args.gif))
        res = run_trial(seed, yield_enabled=not args.no_yield, noise=not args.no_noise,
                        log_dir=out_dir if (i == 0 and args.csv) else None, record=want_extra,
                        use_ml=not args.no_ml, random_mission=False if args.fixed else None,
                        use_motion_prediction=not args.no_motion_prediction, use_dynamic_safety=not args.no_safety)
        if i == 0:
            first = res
        n_ok += res["ok"]
        rows.append(res)
        m = res["mission"]
        print(f"{seed:>5} {res['result']:>8} {res['collisions']:>5} {res['time']:>8.1f} {res['path_len']:>8.2f} "
              f"{res['min_clear'] * 100:>8.1f} {res['rides']:>5} {res['escapes']:>4}  {m['start_room']} (F{m['floor_s'] + 1}) -> "
              f"{m['goal_room']} (F{m['floor_g'] + 1})", flush=True)
    print("-" * 100)
    ts = [r["time"] for r in rows if r["result"] == "success"]
    print(f"SUCCESS (goal reached AND zero collisions): {n_ok}/{args.runs}")
    print(f"total collisions: {sum(r['collisions'] for r in rows)}   "
          f"timeouts: {sum(r['result'] == 'timeout' for r in rows)}   "
          f"worst clearance: {min(r['min_clear'] for r in rows) * 100:.1f} cm")
    if ts:
        print(f"time to goal: mean {sum(ts) / len(ts):.0f} s, max {max(ts):.0f} s   (wall time {time.time() - t0:.0f} s)")
    if args.plot and first is not None:
        png = os.path.join(out_dir, "validation_trajectory.png")
        plot_run(first, png)
        print("saved", png)
    if args.gif and first is not None:
        g = os.path.join(out_dir, "mission_animation.gif")
        make_gif(first, g)
        print("saved", g)
    return 0 if n_ok == args.runs else 1


if __name__ == "__main__":
    sys.exit(main())

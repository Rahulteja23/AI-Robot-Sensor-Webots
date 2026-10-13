"""
perpendicular_test.py
=====================
T-BONE TEST: a car that NEVER brakes drives across the robot's path at 90 degrees.

Scene  : empty 3 m x 3 m arena. The robot drives from (-0.6, 0) to (0.9, 0) along +x.
         A car on the lane x = x_c (perpendicular to the robot's heading) is timed to reach
         the robot's path at  t_robot_arrives + delta  seconds.
         delta = 0 is an exact side impact on the robot.
Sweeps : car speed, arrival offset delta, car direction (from the left / from the right).
Counts : collisions (body overlap), goal reached, closest approach.

    python experiments/perpendicular_test.py                  # full sweep
    python experiments/perpendicular_test.py --baseline       # same sweep WITHOUT dynamic safety
    python experiments/perpendicular_test.py --gif            # animation of one dodge
"""
import argparse
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ai"))
sys.path.insert(0, os.path.dirname(__file__))

import scene_config as sc                       # noqa: E402
import headless_validation as hv                # noqa: E402

# an empty 3 m x 3 m test arena (four walls) - the T-bone situation without any building around it
ARENA = [(0.0, 1.52, 0.0, 3.08, 0.04), (0.0, -1.52, 0.0, 3.08, 0.04), (1.52, 0.0, 0.0, 0.04, 3.08), (-1.52, 0.0, 0.0, 0.04, 3.08)]
from robot_brain import RobotBrain              # noqa: E402

START, GOAL_T = (-0.6, 0.0), (0.9, 0.0)
X_CROSS = 0.15                                   # where the car's lane crosses the robot's path
SPEEDS = (0.05, 0.08, 0.12, 0.16)
DELTAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)  # car arrival time minus robot arrival time (s)
V_ROBOT_NOMINAL = 0.08


def make_scene(v_car, delta, from_left, seed):
    """The car waits at the end of its lane and departs so that it reaches the robot's path
    at (robot arrival time + delta)."""
    t_robot = (X_CROSS - START[0]) / V_ROBOT_NOMINAL
    t_car = max(1.0, t_robot + delta)
    half = 1.30                                  # lane runs from -half to +half (arena wall is at 1.5)
    t_drive = half / v_car                       # time to travel from lane end to the crossing point
    t_wait = max(0.0, t_car - t_drive)
    if from_left:
        p0, p1 = (X_CROSS, -half), (X_CROSS, half)
    else:
        p0, p1 = (X_CROSS, half), (X_CROSS, -half)
    cfg = dict(def_name="AMBUSH_CAR", kind="car", p0=p0, p1=p1, speed=v_car, pause=99.0,
               phase=0.0, size=(sc.CAR_L, sc.CAR_W, sc.CAR_H), color=(0.9, 0.1, 0.1))
    car = sc.DynamicMover(cfg, yield_enabled=False)
    car.pause_left = t_wait                      # parked until departure time
    car.target_yaw = car.yaw
    world = hv.HeadlessWorld(seed, ARENA, [car], START, 0.0)
    return world


def run_case(v_car, delta, from_left, seed, brain_kwargs, record=False, max_time=60.0):
    world = make_scene(v_car, delta, from_left, seed)
    brain = RobotBrain(hv.CTRL_DT, results_dir=None, goal=GOAL_T, verbose=False, **brain_kwargs)
    steps, path = 0, []
    reached = False
    while world.t < max_time:
        out = brain.step(world.t, world.sense())
        steps += 1
        if record and steps % 2 == 0:
            m = world.movers[0]
            path.append((world.t, world.x, world.y, (m.x, m.y, m.yaw), out.get("mode", ""), out.get("v", 0.0),
                         world.yaw, world.collisions))
        if out["done"]:
            reached = True
            break
        world.advance(out["left"], out["right"], hv.CTRL_DT)
    brain.close()
    return dict(collisions=world.collisions, reached=reached, min_clear=world.min_clearance,
                time=world.t, path=path, world=world)


def make_gif(v_car, delta, from_left, out_gif, brain_kwargs, seed=11):
    """Animation of one T-bone encounter: the robot (yellow) dodges the car (red) that never brakes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    from matplotlib.animation import FuncAnimation, PillowWriter

    res = run_case(v_car, delta, from_left, seed, brain_kwargs, record=True)
    frames = res["path"]
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=80)
    fig.patch.set_facecolor("#14141f")
    ax.set_facecolor("#d8d8dc")
    ax.set_xlim(-0.75, 1.05); ax.set_ylim(-0.55, 0.55); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(mp.Rectangle((X_CROSS - 0.14, -0.55), 0.28, 1.10, fc="#6b6b75", ec="none"))     # the car's road
    ax.add_patch(mp.Circle(START, 0.05, fc="#3366ff", alpha=0.5))
    ax.add_patch(mp.Circle(GOAL_T, 0.05, fc="#22cc44", alpha=0.6))
    ax.text(*START, "START", ha="center", va="center", fontsize=7, color="w")
    ax.text(*GOAL_T, "GOAL", ha="center", va="center", fontsize=7, color="w")
    car = ax.add_patch(mp.Rectangle((0, 0), sc.CAR_L, sc.CAR_W, fc="#e53935", ec="w", lw=1.4, zorder=4))
    body = ax.add_patch(mp.Circle((0, 0), sc.ROBOT_RADIUS * 1.5, fc="#ffeb3b", ec="k", lw=1.2, zorder=6))
    nose, = ax.plot([], [], "-", color="k", lw=2, zorder=7)
    trail, = ax.plot([], [], "-", color="#1e88e5", lw=2, zorder=5)
    txt = ax.text(0.5, 1.02, "", transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="w", family="monospace")
    ax.set_title(f"T-bone test: car never brakes, {v_car:.2f} m/s, arrives {delta:+.1f} s vs robot", color="w", fontsize=10, pad=22)
    fig.subplots_adjust(top=0.86, bottom=0.03, left=0.03, right=0.97)
    xs, ys = [], []

    def upd(i):
        t, x, y, (mx, my, myaw), mode, v, yaw, ncol = frames[min(i, len(frames) - 1)]
        xs.append(x); ys.append(y); trail.set_data(xs, ys)
        body.center = (x, y)
        r = sc.ROBOT_RADIUS * 1.5
        nose.set_data([x, x + r * math.cos(yaw)], [y, y + r * math.sin(yaw)])
        c, s_ = math.cos(myaw), math.sin(myaw)
        car.set_xy((mx - (sc.CAR_L / 2) * c + (sc.CAR_W / 2) * s_, my - (sc.CAR_L / 2) * s_ - (sc.CAR_W / 2) * c))
        car.angle = math.degrees(myaw)
        txt.set_text(f"t={t:5.1f}s  robot mode {mode:<8} speed {v:+.3f} m/s  collisions {ncol}")
        return ()
    anim = FuncAnimation(fig, upd, frames=len(frames) + 10, interval=70, repeat=False)
    anim.save(out_gif, writer=PillowWriter(fps=14))
    plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml", action="store_true", help="run with Random Forest + SVM enabled (slower)")
    ap.add_argument("--baseline", action="store_true", help="disable the dynamic-safety layer")
    ap.add_argument("--reps", type=int, default=2, help="noise seeds per case")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    kw = dict(use_ml=args.ml)
    if args.baseline:
        kw["use_dynamic_safety"] = False
    label = "WITHOUT dynamic safety (previous version)" if args.baseline else "WITH dynamic safety"
    print(f"T-bone test, car never brakes - {label}")
    print("-" * 74)
    print(f"{'car speed':>10} | " + " ".join(f"{d:+.1f}s".rjust(7) for d in DELTAS) + " | collisions")
    tot_cases = tot_coll = tot_goal = 0
    worst = 9.0
    for v in SPEEDS:
        cells, vcoll = [], 0
        for d in DELTAS:
            c = 0
            for side in (True, False):
                for r in range(args.reps):
                    res = run_case(v, d, side, seed=100 + r, brain_kwargs=kw)
                    c += res["collisions"] > 0
                    tot_cases += 1
                    tot_goal += res["reached"]
                    worst = min(worst, res["min_clear"])
            vcoll += c
            cells.append(f"{c}/{2 * args.reps}".rjust(7))
        tot_coll += vcoll
        print(f"{v:>7.2f} m/s | " + " ".join(cells) + f" | {vcoll}")
    print("-" * 74)
    print(f"cases with a collision: {tot_coll}/{tot_cases}   goal reached: {tot_goal}/{tot_cases}   "
          f"closest approach: {worst * 100:.1f} cm")
    print("(each cell = collisions / runs; 0 collision = robot dodged the car)")
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    import json
    with open(os.path.join(out_dir, "tbone_baseline.json" if args.baseline else "tbone_result.json"), "w") as fh:
        json.dump(dict(label=label, cases=tot_cases, collisions=tot_coll, goal_reached=tot_goal,
                       closest_approach_cm=round(worst * 100, 1)), fh, indent=2)
    if args.gif:
        # pick an encounter where the robot has to react: car arrives at the same moment as the robot
        g = os.path.join(out_dir, "tbone_dodge.gif")
        r = make_gif(0.14, 0.0, True, g, kw)
        print(f"saved {g}   (collisions in that run: {r['collisions']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

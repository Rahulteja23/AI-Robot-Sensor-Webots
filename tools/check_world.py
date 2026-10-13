"""
check_world.py  -  debug / validation tool for the hospital model AND the generated Webots world file.

    python tools/check_world.py            # prints a PASS / FAIL report, exit code 1 on any FAIL

Checks
  A  building model (ai/hospital.py)
       * no piece of furniture overlaps a wall or another piece of furniture
       * every doorway and every lift door has free space of at least the robot's width
       * every room is reachable from every other room of its floor and from both lifts (route planner)
       * every moving person / trolley lane keeps the lift lobbies free and does not run into furniture
       * every random-mission start / goal sample is valid (200 missions)
  B  world file (worlds/ai_sensor_world.wbt)
       * braces / brackets balanced, node names unique, every DEF the controllers look up exists
       * every wall, furniture item and moving agent of the model is in the file at the right place, orientation and size
       * the robot has supervisor TRUE and the 24 range sensors, GPS and IMU
"""
import math
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ai"))
import scene_config as sc          # noqa: E402
import hospital as hp              # noqa: E402
import route_planner as rp         # noqa: E402
import mission as ms               # noqa: E402
from navigator import RS_COUNT     # noqa: E402

FAILS = []


def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


def corners(cx, cy, yaw, L, W):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + dx * c - dy * s, cy + dx * s + dy * c) for dx, dy in ((L / 2, W / 2), (L / 2, -W / 2), (-L / 2, W / 2), (-L / 2, -W / 2))]


def overlap(a, b, tol=0.002):
    """True if two oriented rectangles overlap by more than tol (corner / centre sampling + edge sampling)."""
    for p, q in ((a, b), (b, a)):
        pts = corners(*p) + [(p[0], p[1])]
        for k in range(4):                                  # midpoints of the edges as well
            x0, y0 = pts[k]; x1, y1 = pts[(k + 1) % 4]
            pts.append(((x0 + x1) / 2, (y0 + y1) / 2))
        for (x, y) in pts:
            if sc.rect_distance(x, y, *q) < -1e-9 or (sc.rect_distance(x, y, *q) == 0.0 and
                                                        sc.rect_distance(x, y, q[0], q[1], q[2], q[3] - 2 * tol, q[4] - 2 * tol) == 0.0):
                return True
    return False


def part_a():
    print("A. Building model (ai/hospital.py)")
    for fl in hp.FLOORS:
        fp_f = [(o["name"], (o["x"], o["y"], o["yaw"], o["size"][0], o["size"][1])) for o in fl.furniture]
        fp_w = [(f"wall{k}", (w["x"], w["y"], w["yaw"], w["L"], w["W"])) for k, w in enumerate(fl.walls)]
        bad = []
        for n, r in fp_f:
            for m, q in fp_w:
                if overlap(r, q):
                    bad.append((n, m))
        for i in range(len(fp_f)):
            for j in range(i + 1, len(fp_f)):
                if overlap(fp_f[i][1], fp_f[j][1]):
                    bad.append((fp_f[i][0], fp_f[j][0]))
        check(not bad, f"{fl.label}: no furniture overlaps a wall or other furniture" + (f"  -> {bad[:4]}" if bad else ""))
        g = rp.grid(fl.index)
        door_pts = [(r.name, r.door) for r in fl.rooms] + [(lf.name, lf.door) for lf in fl.lifts]
        low = [(n, round(g.clearance(*p), 3)) for n, p in door_pts if g.clearance(*p) < sc.ROBOT_RADIUS + 0.03]
        check(not low, f"{fl.label}: every doorway is wide enough for the robot (>= {sc.ROBOT_RADIUS + 0.03:.3f} m from the posts)" + (f" -> {low}" if low else ""))
        unreachable = []
        pts = [(r.name, r.center) for r in fl.rooms] + [(lf.name, lf.pad) for lf in fl.lifts]
        for na, a in pts:
            for nb, b in pts:
                if na < nb and rp.plan_path(fl.index, a, b) is None:
                    unreachable.append((na, nb))
        check(not unreachable, f"{fl.label}: all {len(pts)} rooms/lifts reachable from each other" + (f" -> {unreachable[:4]}" if unreachable else ""))
        lobby = []
        for a in fl.agents:
            for lf in fl.lifts:
                for p in (a["p0"], a["p1"]):
                    d = math.hypot(p[0] - lf.exit[0], p[1] - lf.exit[1])
                    # the agent turns around at the end of its lane (swept radius = half its diagonal): the robot, which
                    # appears at the lift exit, must stay clear of that circle by its own radius plus a margin
                    if d < 0.5 * math.hypot(a["size"][0], a["size"][1]) + sc.ROBOT_RADIUS + 0.06:
                        lobby.append((a["def_name"], lf.name, round(d, 2)))
        check(not lobby, f"{fl.label}: the people / trolley lanes keep the lift lobbies free" + (f" -> {lobby}" if lobby else ""))
        hit = []
        for a in fl.agents:
            for k in range(41):
                t = k / 40
                x, y = a["p0"][0] + t * (a["p1"][0] - a["p0"][0]), a["p0"][1] + t * (a["p1"][1] - a["p0"][1])
                for n, r in fp_f + fp_w:
                    if sc.rect_distance(x, y, *r) < 0.5 * max(a["size"][0], a["size"][1]) + 0.015:
                        hit.append((a["def_name"], n))
        check(not hit, f"{fl.label}: moving people / trolleys never run into furniture or walls" + (f" -> {sorted(set(hit))[:4]}" if hit else ""))
    bad = []
    for seed in range(1, 201):
        m = ms.random_mission(seed)
        for key, fl_i in (("start", m["floor_s"]), ("goal", m["floor_g"])):
            if hp.static_clearance(fl_i, *m[key]) < ms.CLEAR_POINT - 2e-3:      # coordinates are stored with 1 mm resolution
                bad.append((seed, key))
        if not m["random"]:
            bad.append((seed, "fallback"))
    check(not bad, "200 random missions: every start / goal point is clear of obstacles, none fell back" + (f" -> {bad[:4]}" if bad else ""))


def part_b():
    print("B. World file (worlds/ai_sensor_world.wbt)")
    path = os.path.join(ROOT, "worlds", "ai_sensor_world.wbt")
    if not os.path.exists(path):
        check(False, "world file exists (run python tools/generate_world.py)")
        return
    s = open(path, encoding="utf-8").read()
    body = "\n".join(l for l in s.splitlines() if not l.strip().startswith("#"))
    check(body.count("{") == body.count("}") and body.count("[") == body.count("]"), "braces and brackets are balanced")
    names = re.findall(r'^\s*name "([^"]+)"', s, re.M)
    dup = sorted({n for n in names if names.count(n) > 1})
    check(not dup, f"all {len(names)} node names are unique" + (f" -> {dup[:5]}" if dup else ""))
    defs = set(re.findall(r"DEF (\w+)", s))
    need = {"EPUCK", "VIEW", "START_PAD", "GOAL_PAD", "GOAL_FLAG"} | {a["def_name"] for fl in hp.FLOORS for a in fl.agents}
    check(need <= defs, "every DEF used by the controllers exists" + (f" -> missing {sorted(need - defs)}" if not need <= defs else ""))
    check("supervisor TRUE" in s.split("DEF EPUCK")[1].split("turretSlot")[0], "the E-puck has supervisor TRUE (needed to place the mission and ride lifts)")
    rs = len(re.findall(r'name "rs\d+"', s))
    check(rs == RS_COUNT and 'name "gps"' in s and 'name "inertial unit"' in s, f"robot has {rs}/{RS_COUNT} range sensors, GPS and IMU")

    # every model object is in the file at the right place
    blocks = re.findall(r'(?:DEF \w+ )?Solid \{\s*translation ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)\s*rotation 0 0 1 ([-\d.e]+)(.*?)name "([^"]+)"(.*?)\n\}', s, re.S)
    by_name = {}
    for tx, ty, tz, yaw, kids, name, tail in blocks:
        m = re.search(r"boundingObject Transform \{ translation 0 0 [-\d.e]+ children \[ Box \{ size ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)", tail)
        by_name[name] = (float(tx), float(ty), float(tz), float(yaw), tuple(map(float, m.groups())) if m else None)
    missing, wrong = [], []
    for fl in hp.FLOORS:
        ox, oy, z0 = fl.origin
        items = [(f"f{fl.index + 1}_wall{k}", w["x"], w["y"], w["yaw"], (w["L"], w["W"], w["H"])) for k, w in enumerate(fl.walls)]
        items += [(o["name"], o["x"], o["y"], o["yaw"], o["size"]) for o in fl.furniture]
        items += [(a["def_name"].lower(), None, None, None, a["size"]) for a in fl.agents]
        for name, x, y, yaw, size in items:
            if name not in by_name:
                missing.append(name)
                continue
            tx, ty, tz, tyaw, bound = by_name[name]
            if x is not None and (abs(tx - (ox + x)) > 2e-3 or abs(ty - (oy + y)) > 2e-3 or abs(tz - z0) > 2e-3 or abs(tyaw - yaw) > 2e-3):
                wrong.append(name)
            if bound is None or any(abs(a - b) > 2e-3 for a, b in zip(bound, size)):
                wrong.append(name + "(size)")
    n_items = sum(len(fl.walls) + len(fl.furniture) + len(fl.agents) for fl in hp.FLOORS)
    check(not missing and not wrong, f"all {n_items} walls / furniture / agents of the model are in the file with the right pose and size"
          + (f" -> missing {missing[:3]}, wrong {wrong[:3]}" if missing or wrong else ""))
    zs = sorted({round(v[2], 3) for k, v in by_name.items() if k.startswith("f") and "_wall" in k})
    check(zs == [round(hp.plate_origin(i)[2], 3) for i in range(hp.N_FLOORS)], f"the three floors stand at heights {zs} m")


if __name__ == "__main__":
    part_a()
    part_b()
    print()
    print("RESULT:", "ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED")
    sys.exit(1 if FAILS else 0)

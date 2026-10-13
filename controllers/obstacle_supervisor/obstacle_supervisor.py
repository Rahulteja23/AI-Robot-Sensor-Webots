"""
obstacle_supervisor.py
======================
Webots Supervisor controller of the hospital world.

  1. Moves every nurse and medicine trolley of all three floors back and forth along its lane (they turn around at the
     ends and brake when the robot is directly in front, unless TROLLEY_NEVER_BRAKES is set in ai/hospital.py).
  2. Watches the robot (DEF EPUCK), works out on which floor it is, and counts collisions geometrically (robot body vs
     walls, furniture and moving people / trolleys of THAT floor).
  3. Shows a live HUD, drops a colour-fading trail, prints a MISSION REPORT when the goal is reached (or on timeout),
     writes results/mission_result.json and pauses the simulation.

The mission (random START / GOAL rooms) is chosen by the robot controller; the supervisor learns its id from the
robot's status text and re-creates the identical mission (same seed -> same mission).
"""
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "ai"))

from controller import Supervisor                      # noqa: E402
import scene_config as sc                              # noqa: E402
import hospital as hp                                  # noqa: E402
import mission as ms                                   # noqa: E402

MISSION_TIMEOUT = 900.0     # s of simulated time
PAUSE_AFTER_GOAL = 3.0      # s

SHOW_HUD   = True           # live text overlay in the 3D view
SHOW_TRAIL = True           # coloured dots behind the robot
TRAIL_EVERY_S, TRAIL_MIN_STEP, TRAIL_MAX = 0.5, 0.025, 700


def _lerp(c0, c1, f):
    return tuple(c0[i] + (c1[i] - c0[i]) * f for i in range(3))


def hud(sup, lines):
    """Draw up to 6 lines of text in the top-left corner of the 3D view: (text, size, 0xRRGGBB colour)."""
    try:
        for i, (text, size, colr) in enumerate(lines):
            sup.setLabel(i, text, 0.015, 0.02 + i * 0.045, size, colr, 0.0, "Arial")
    except Exception:                                  # noqa: BLE001
        pass


def add_trail_dot(sup, wx, wy, wz, frac):
    """Drop one small coloured disc on the floor (blue at the start -> green at the goal)."""
    r, g, b = _lerp((0.15, 0.45, 1.0), (0.1, 0.9, 0.3), max(0.0, min(1.0, frac)))
    node = (f"Solid {{ translation {wx:.3f} {wy:.3f} {wz + 0.004:.3f} children [ Shape {{ "
            f"appearance PBRAppearance {{ baseColor {r:.2f} {g:.2f} {b:.2f} emissiveColor {r:.2f} {g:.2f} {b:.2f} "
            f"roughness 1 metalness 0 }} geometry Cylinder {{ height 0.002 radius 0.008 }} }} ] }}")
    sup.getRoot().getField("children").importMFNodeFromString(-1, node)


def main():
    sup = Supervisor()
    dt_ms = int(sup.getBasicTimeStep())
    dt = dt_ms / 1000.0

    robot = sup.getFromDef("EPUCK")
    if robot is None:
        print("[Supervisor] ERROR: robot DEF EPUCK not found in the world file.")
        return
    try:
        status_field = robot.getField("customData")      # live status published by the robot controller
    except Exception:                                    # noqa: BLE001
        status_field = None

    movers, fields = [], []                               # movers[f] = [DynamicMover ...] ; fields[f] = [(translation, rotation)]
    for fl in hp.FLOORS:
        row_m, row_f = [], []
        for a in fl.agents:
            node = sup.getFromDef(a["def_name"])
            if node is None:
                print(f"[Supervisor] ERROR: {a['def_name']} not found in the world file (regenerate it with tools/generate_world.py).")
                return
            row_m.append(sc.DynamicMover(a))
            row_f.append((node.getField("translation"), node.getField("rotation")))
        movers.append(row_m)
        fields.append(row_f)
    footprints = [fl.footprints() for fl in hp.FLOORS]

    print("=" * 70)
    print(f"[Supervisor] {sum(len(r) for r in movers)} moving people/trolleys on {hp.N_FLOORS} floors. Waiting for the robot's mission ...")
    print("=" * 70)

    mis = None                 # the mission, re-created from the id the robot publishes
    t_mis = 0.0
    collisions, in_collision = 0, False
    min_clear = 9.0
    path_len, last_w = 0.0, None
    floors_seen = []
    reached_at, report_done = None, False
    t, next_print, next_hud, next_trail = 0.0, 20.0, 0.0, 0.0
    trail_n, last_trail, trail_ok = 0, None, SHOW_TRAIL
    route_total = 1.0

    while sup.step(dt_ms) != -1:
        t += dt
        p = robot.getPosition()
        floor, rx, ry = hp.to_local(p[0], p[1])           # the robot's floor and floor-local position

        # ---- move people / trolleys on every floor (they brake only for the robot on THEIR floor) ----
        for f in range(hp.N_FLOORS):
            ox, oy, z0 = hp.plate_origin(f)
            for mv, (ft, fr) in zip(movers[f], fields[f]):
                mv.step(dt, (rx, ry) if f == floor else None)
                x, y, yaw = mv.pose()
                ft.setSFVec3f([ox + x, oy + y, z0])
                fr.setSFRotation([0.0, 0.0, 1.0, yaw])

        # ---- wait until the robot has published its mission ----
        status = ""
        if status_field is not None:
            try:
                status = status_field.getSFString()
            except Exception:                              # noqa: BLE001
                status = ""
        parts = (status.split("|") + [""] * 7)[:7]
        if mis is None:
            if parts[4]:
                try:
                    mis = ms.mission_from_id(parts[4])
                except ValueError:
                    mis = None
            if mis is None:
                if SHOW_HUD and t >= next_hud:
                    next_hud = t + 0.5
                    hud(sup, [("AI Hospital Robot - waiting for the mission ...", 0.07, 0xFFFFFF)])
                continue
            t_mis = t
            route_total = max(mis["route_length"], 0.1)
            next_print = t + 20.0
            print("=" * 70)
            print(f"[Supervisor] MISSION  FROM {mis['start_name']}   TO {mis['goal_name']}")
            print(f"[Supervisor] route {mis['route_length']:.1f} m in {len(mis['legs'])} leg(s), seed {mis['seed']}")
            print("=" * 70)
            last_w = None
            continue

        # ---- bookkeeping ----
        if last_w is not None and abs(p[2] - last_w[2]) < 0.2:       # ignore the jump caused by a lift ride
            path_len += math.hypot(p[0] - last_w[0], p[1] - last_w[1])
        last_w = p
        if not floors_seen or floors_seen[-1] != floor:
            floors_seen.append(floor)

        clear = 9.0
        for cx, cy, yaw, L, W in footprints[floor]:
            if abs(cx - rx) < 0.6 and abs(cy - ry) < 0.6:
                clear = min(clear, sc.rect_distance(rx, ry, cx, cy, yaw, L, W) - sc.ROBOT_RADIUS)
        for mv in movers[floor]:
            clear = min(clear, sc.rect_distance(rx, ry, mv.x, mv.y, mv.yaw, mv.L, mv.W) - sc.ROBOT_RADIUS)
        min_clear = min(min_clear, clear)
        if clear < -0.002:
            if not in_collision:
                collisions += 1
                print(f"[Supervisor] !! COLLISION #{collisions} at t={t - t_mis:.1f}s on {hp.FLOOR_LABELS[floor]}  robot=({rx:+.2f},{ry:+.2f})")
            in_collision = True
        else:
            in_collision = False

        gx, gy = mis["goal"]
        d_goal = math.hypot(rx - gx, ry - gy) if floor == mis["floor_g"] else float("inf")
        prog = max(0.0, min(1.0, path_len / route_total))

        # ---- visuals: trail + live HUD ----
        if trail_ok and reached_at is None and t >= next_trail and trail_n < TRAIL_MAX:
            next_trail = t + TRAIL_EVERY_S
            if last_trail is None or math.hypot(p[0] - last_trail[0], p[1] - last_trail[1]) > TRAIL_MIN_STEP:
                try:
                    add_trail_dot(sup, p[0], p[1], hp.plate_origin(floor)[2], prog)
                    trail_n += 1
                    last_trail = p
                except Exception as e:                      # noqa: BLE001
                    trail_ok = False
                    print("[Supervisor] trail disabled:", e)
        if SHOW_HUD and t >= next_hud and not report_done:
            next_hud = t + 0.5
            bar = "#" * int(prog * 20) + "-" * (20 - int(prog * 20))
            ok_col = 0x66FF66 if collisions == 0 else 0xFF5555
            mode_col = {"BOOST": 0xFFC04D, "YIELD": 0x66CCFF, "BACKOFF": 0xFF9966, "DODGE": 0xFF9966, "LIFT": 0xC8A0FF}.get(parts[0], 0xDDDDDD)
            robot_line = (f"Robot: {parts[0] or '-':<8} speed {float(parts[1] or 0):+.3f} m/s   RF {parts[2] or '-'}   SVM {parts[3] or '-'}"
                          if status else "Robot: starting ...")
            hud(sup, [
                (f"FROM {mis['start_name']}", 0.055, 0xFFFFFF),
                (f"TO   {mis['goal_name']}", 0.055, 0xFFFFFF),
                (f"{hp.FLOOR_LABELS[floor]}   Time {t - t_mis:6.1f} s   Progress [{bar}] {prog * 100:3.0f} %", 0.055, 0x66CCFF),
                (f"Collisions {collisions}    Closest approach {max(min_clear, 0) * 100:4.1f} cm", 0.055, ok_col),
                (robot_line, 0.055, mode_col),
            ])

        if t >= next_print and reached_at is None:
            print(f"[Supervisor] t={t - t_mis:5.1f}s  {hp.FLOOR_LABELS[floor]:<17} robot=({rx:+.2f},{ry:+.2f})  "
                  f"collisions={collisions}  min_gap={min_clear * 100:.1f} cm")
            next_print += 20.0

        # ---- mission end ----
        if reached_at is None and d_goal < sc.GOAL_STOP + 0.02:
            reached_at = t
        timeout = t - t_mis > MISSION_TIMEOUT
        if (reached_at is not None and t - reached_at >= PAUSE_AFTER_GOAL) or timeout:
            if not report_done:
                report_done = True
                success = reached_at is not None
                verdict = ("MISSION SUCCESS - goal reached without collision" if success and collisions == 0
                           else "GOAL REACHED but %d collision(s)" % collisions if success
                           else "MISSION FAILED - timeout")
                dur = (reached_at - t_mis) if reached_at else float("nan")
                print("\n" + "=" * 70)
                print("  MISSION REPORT")
                print("=" * 70)
                print(f"  Result            : {verdict}")
                print(f"  From              : {mis['start_name']}")
                print(f"  To                : {mis['goal_name']}   (seed {mis['seed']})")
                print(f"  Time to goal      : {dur:.1f} s (simulated)")
                print(f"  Path length       : {path_len:.2f} m   (planned route {mis['route_length']:.2f} m)")
                print(f"  Floors visited    : {' -> '.join(str(f + 1) for f in floors_seen)}")
                print(f"  Collisions        : {collisions}")
                print(f"  Closest approach  : {min_clear * 100:.1f} cm (body to obstacle)")
                print(f"  Braking events    : {sum(mv.brake_events for row in movers for mv in row)} (people / trolleys stopping for the robot)")
                print("=" * 70)
                if SHOW_HUD:
                    good = success and collisions == 0
                    hud(sup, [
                        ("MISSION SUCCESS" if good else "GOAL REACHED - WITH COLLISION" if success else "MISSION FAILED (timeout)",
                         0.11, 0x66FF66 if good else 0xFF5555),
                        (f"{mis['start_name']}  ->  {mis['goal_name']}", 0.05, 0xFFFFFF),
                        (f"Time {dur if success else t - t_mis:.1f} s   Path {path_len:.2f} m   Collisions {collisions}", 0.055, 0xFFFFFF),
                        (f"Closest approach {max(min_clear, 0) * 100:.1f} cm", 0.055, 0xDDDDDD),
                        ("", 0.05, 0xFFFFFF),
                    ])
                res = dict(result=verdict, success=bool(success and collisions == 0),
                           mission=dict(start=mis["start_name"], goal=mis["goal_name"], seed=mis["seed"]),
                           time_to_goal_s=(round(dur, 1) if success else None), path_length_m=round(path_len, 3),
                           planned_route_m=mis["route_length"], floors=[f + 1 for f in floors_seen],
                           collisions=collisions, min_clearance_cm=round(min_clear * 100, 2),
                           timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
                try:
                    os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)
                    with open(os.path.join(_ROOT, "results", "mission_result.json"), "w") as fh:
                        json.dump(res, fh, indent=2)
                except OSError as e:
                    print("[Supervisor] could not write mission_result.json:", e)
                sup.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)


if __name__ == "__main__":
    main()

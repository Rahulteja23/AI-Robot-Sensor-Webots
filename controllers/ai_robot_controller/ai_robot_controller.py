"""
ai_robot_controller.py
======================
Webots controller of the E-puck = a hospital delivery robot (scale 1:8).
Mission: drive from a START room to a GOAL room - possibly on another floor (it rides the lift) - without colliding with
walls, furniture, people or trolleys.

This file does I/O and the mission bookkeeping:
    pick mission -> place robot / pads -> [ read devices -> RobotBrain.step() -> set wheel speeds ] -> lift ride -> ...
All decision making lives in ai/robot_brain.py (shared with the offline tests).

Pipeline inside RobotBrain
--------------------------
  8 IR sensors + 24 range sensors + camera + GPS + IMU
      -> Perception -> SensorFusion
      -> RouteFollower   global route through corridors and doors (A* on the known floor plan)
      -> Navigator       local planner: follows the route, avoids obstacles
      -> DynamicSafety   tracks MOVING people / trolleys, predicts them, changes speed / direction
      -> MLPathPlanner (Random Forest) + SVMPlanner (SVM): online-learned second opinions
      -> wheel speeds                       -> DataLogger (CSV in results/)

All positions inside the software are FLOOR-LOCAL; the world has three floor plates (see ai/hospital.py), this file
converts GPS (world) <-> local.  Device names: motors "left wheel motor" / "right wheel motor", "ps0".."ps7" (IR),
"rs0".."rs23" (range ring), "camera", "gps", "inertial unit".
"""
import math
import os
import sys

_CTRL_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CTRL_DIR, "..", ".."))
_AI_DIR       = os.path.join(_PROJECT_ROOT, "ai")
_RESULTS_DIR  = os.path.join(_PROJECT_ROOT, "results")
for _p in (_CTRL_DIR, _AI_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from controller import Supervisor                 # noqa: E402  (provided by Webots)

from sensor_manager import SensorManager          # noqa: E402
from robot_brain import RobotBrain                # noqa: E402
import scene_config as sc                         # noqa: E402
import hospital as hp                             # noqa: E402
import route_planner as rp                        # noqa: E402
import mission as mission_mod                     # noqa: E402

TIMESTEP_MULT   = 2        # controller step = 2 x basicTimeStep = 64 ms
LOG_EVERY_STEPS = 30       # console line every ~2 s
LIFT_RIDE_S     = 3.0      # seconds spent in the lift


def place_robot(robot, floor, xy, yaw):
    """Teleport the robot onto floor `floor` at floor-local position xy.  Needs supervisor TRUE on the E-puck."""
    me = robot.getSelf()
    wx, wy = hp.to_world(floor, *xy)
    z0 = hp.plate_origin(floor)[2]
    me.getField("translation").setSFVec3f([wx, wy, z0])
    me.getField("rotation").setSFRotation([0.0, 0.0, 1.0, yaw])
    me.resetPhysics()


def focus_camera(robot, floor):
    """Point the 3D view at floor `floor` (harmless if the viewpoint cannot be moved)."""
    if not sc.AUTO_CAMERA:
        return
    try:
        view = robot.getFromDef("VIEW")
        pos, rot = hp.floor_camera(floor)
        view.getField("position").setSFVec3f(list(pos))
        view.getField("orientation").setSFRotation(list(rot))
    except Exception:                                         # noqa: BLE001
        pass


def place_mission(robot, m: dict) -> bool:
    """Put the robot on its START point and move the blue START pad, the green GOAL pad and the goal flag."""
    try:
        place_robot(robot, m["floor_s"], m["start"], m["yaw"])
        zs, zg = hp.plate_origin(m["floor_s"])[2], hp.plate_origin(m["floor_g"])[2]
        sx, sy = hp.to_world(m["floor_s"], *m["start"])
        gx, gy = hp.to_world(m["floor_g"], *m["goal"])
        pad = sc.GOAL_PAD
        for def_name, pos in (("START_PAD", (sx, sy, zs + 0.0015)), ("GOAL_PAD", (gx, gy, zg + 0.0015)),
                              ("GOAL_FLAG", (gx + pad / 2 - 0.015, gy + pad / 2 - 0.015, zg))):
            node = robot.getFromDef(def_name)
            if node is not None:
                node.getField("translation").setSFVec3f(list(pos))
        focus_camera(robot, m["floor_s"])
        return True
    except Exception as e:                                    # noqa: BLE001
        print(f"[INIT] WARNING: could not place the mission ({e}). "
              "Open worlds/ai_sensor_world.wbt from this project (the robot needs supervisor TRUE).")
        return False


def status_text(m: dict, mode="START", v=0.0, rf="-", svm="-", floor=0, leg=0) -> str:
    """Live status for the supervisor: robot mode / speed / model decisions, mission id, floor and leg."""
    return f"{mode}|{v:.3f}|{rf}|{svm}|{m['seed']}|{floor}|{leg}"


def main() -> None:
    print("=" * 74)
    print("  AI Hospital Delivery Robot - route planning, obstacle avoidance, lifts")
    print("=" * 74)

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep()) * TIMESTEP_MULT
    print(f"[INIT] basicTimeStep={robot.getBasicTimeStep():.0f} ms -> controller step {timestep} ms")

    left = robot.getDevice("left wheel motor")
    right = robot.getDevice("right wheel motor")
    for mt in (left, right):
        mt.setPosition(float("inf"))
        mt.setVelocity(0.0)

    # ---- mission: random START / GOAL rooms (or the demo mission from scene_config) ----
    m = mission_mod.get_mission()
    legs = m["legs"]
    if not place_mission(robot, m) and m["random"]:
        print("[INIT] Falling back to the demo mission (it starts where the robot stands in the world file).")
        m = mission_mod.fixed_mission()
        legs = m["legs"]
    for _ in range(3):                                        # let the new pose / pads take effect
        robot.step(timestep)
    try:
        robot.setCustomData(status_text(m, floor=m["floor_s"] + 1, leg=1))   # the supervisor reads the mission id from here
    except Exception:                                         # noqa: BLE001
        pass

    sm = SensorManager(robot, timestep)
    sm.init_all()
    brain = RobotBrain(timestep / 1000.0, results_dir=_RESULTS_DIR, verbose=True)

    def begin(li):
        lg = legs[li]
        brain.start_leg(lg["floor"], rp.RouteFollower(lg["floor"], lg["path"], lg["end"], lg["stop"]),
                        final=(li == len(legs) - 1), leg_index=li)
    begin(0)
    li = 0

    print(f"[INIT] {'RANDOM' if m['random'] else 'DEMO'} mission"
          + (f" (seed {m['seed']} - set MISSION_SEED = {m['seed']} in ai/scene_config.py to repeat it)" if m["random"] else ""))
    print(f"[INIT] FROM  {m['start_name']}  at ({m['start'][0]:+.2f}, {m['start'][1]:+.2f}), heading {math.degrees(m['yaw']):+.0f} deg")
    print(f"[INIT] TO    {m['goal_name']}  at ({m['goal'][0]:+.2f}, {m['goal'][1]:+.2f})")
    print(f"[INIT] route {m['route_length']:.1f} m in {len(legs)} leg(s)" + ("  (rides " + legs[0]["lift"].name + ")" if legs[0]["lift"] else ""))
    print("-" * 74)

    def local_data():
        d = sm.read_all()
        wx, wy, wz = d["position"]
        ox, oy, z0 = hp.plate_origin(brain.floor)
        d["position"] = (wx - ox, wy - oy, wz - z0)
        return d

    step = 0
    try:
        while robot.step(timestep) != -1:
            step += 1
            t = step * timestep / 1000.0
            out = brain.step(t, local_data())
            left.setVelocity(out["left"])
            right.setVelocity(out["right"])

            if step % 4 == 0:                                  # live status for the supervisor's HUD
                try:
                    robot.setCustomData(status_text(m, out["mode"], out["v"], out.get("ml_action") or "-",
                                                    out.get("svm_action") or "-", brain.floor + 1, li + 1))
                except Exception:                              # noqa: BLE001
                    pass

            if out["done"]:
                print(f"\n[GOAL] {m['goal_name']} reached at t={t:.1f} s (distance {out['dist']:.3f} m). Robot stopped.")
                break

            if out.get("leg_done"):                            # standing in the lift: ride to the next floor
                lg = legs[li]
                lf, nf = lg["lift"], legs[li + 1]["floor"]
                print(f"[LIFT] t={t:6.1f}s  {lf.name} on {hp.FLOOR_LABELS[lg['floor']]}: riding to {hp.FLOOR_LABELS[nf]} ...")
                left.setVelocity(0.0)
                right.setVelocity(0.0)
                try:
                    robot.setCustomData(status_text(m, "LIFT", 0.0, "-", "-", lg["floor"] + 1, li + 1))
                except Exception:                              # noqa: BLE001
                    pass
                for _ in range(int(LIFT_RIDE_S * 1000 / timestep)):
                    if robot.step(timestep) == -1:
                        return
                place_robot(robot, nf, lf.pad, lf.yaw_out)
                focus_camera(robot, nf)
                for _ in range(3):
                    robot.step(timestep)
                li += 1
                begin(li)
                continue

            if step % LOG_EVERY_STEPS == 0:
                nav = out.get("nav") or {}
                dyn = out.get("dyn") or {}
                x, y, _ = local_data()["position"]
                print(f"[t={t:6.1f}s] F{brain.floor + 1} pos=({x:+.2f},{y:+.2f})  to go={out['dist']:.2f} m  "
                      f"mode={out['mode']:<8} v={out['v']:+.3f} m/s (planned {nav.get('v_nav', 0):.3f})  "
                      f"moving obstacles tracked={dyn.get('n_moving', 0)}  RF={out.get('ml_action') or '-':<7} "
                      f"SVM={out.get('svm_action') or '-':<7}")

        # Goal reached: keep the robot stopped; the supervisor prints the mission report.
        while robot.step(timestep) != -1:
            left.setVelocity(0.0)
            right.setVelocity(0.0)
    finally:
        brain.close()
        print("[END] Random Forest :", ", ".join(f"{k}={v}" for k, v in brain.ml.stats().items()))
        print("[END] SVM           :", ", ".join(f"{k}={v}" for k, v in brain.svm.stats().items()))
        print("[END]", brain.model_report())
        if brain.logger:
            print(f"[END] CSV saved to: {brain.logger.filepath}")


if __name__ == "__main__":
    main()

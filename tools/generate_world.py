"""
generate_world.py
=================
Writes  worlds/ai_sensor_world.wbt  (the 3-floor hospital) from  ai/hospital.py.

    python tools/generate_world.py

Edit ai/hospital.py (rooms, furniture, people, lifts) and run this again - the Webots world, the supervisor,
the controller and the offline tests always stay consistent.

World layout: three floor plates side by side (x = -4.7, 0, +4.7) at heights 0, 0.45, 0.9 m, so that all floors are
visible at once.  Each plate is 4.0 m x 3.0 m with 14 cm high cut-away walls.  Scale 1:8.
"""
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ai"))
import scene_config as sc                       # noqa: E402
import hospital as hp                           # noqa: E402
import mission as ms                            # noqa: E402
from navigator import RS_COUNT, RS_RADIUS, RS_MAX   # noqa: E402

SENSOR_Z = 0.03     # height of the long-range sensor ring, relative to the robot origin
SLAB_T = 0.05


def f(v):
    return f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) > 1e-9 else "0"


def col(c):
    return f"{f(c[0])} {f(c[1])} {f(c[2])}"


def app(color, rough=0.85, metal=0.0, emissive=None):
    e = f" emissiveColor {col(emissive)}" if emissive else ""
    return f"PBRAppearance {{ baseColor {col(color)} roughness {rough} metalness {metal}{e} }}"


def box(size, color, **kw):
    return f"Shape {{ appearance {app(color, **kw)} geometry Box {{ size {f(size[0])} {f(size[1])} {f(size[2])} }} }}"


def cyl(radius, height, color, **kw):
    return f"Shape {{ appearance {app(color, **kw)} geometry Cylinder {{ height {f(height)} radius {f(radius)} }} }}"


def sph(radius, color, **kw):
    return f"Shape {{ appearance {app(color, **kw)} geometry Sphere {{ radius {f(radius)} subdivision 2 }} }}"


def tr(x, y, z, child, rot=None):
    r = f" rotation {rot}" if rot else ""
    return f"Transform {{ translation {f(x)} {f(y)} {f(z)}{r} children [ {child} ] }}"


def solid(name, x, y, z, yaw, children, bound=None, def_name=None):
    """bound = (L, W, H): a box bounding object standing on the solid's origin (so that the range sensors see it)."""
    d = f"DEF {def_name} " if def_name else ""
    b = ""
    if bound:
        b = (f"\n  boundingObject Transform {{ translation 0 0 {f(bound[2] / 2)} "
             f"children [ Box {{ size {f(bound[0])} {f(bound[1])} {f(bound[2])} }} ] }}")
    kids = "\n".join("    " + c for c in children)
    return (f"{d}Solid {{\n  translation {f(x)} {f(y)} {f(z)}\n  rotation 0 0 1 {f(yaw)}\n  children [\n{kids}\n  ]\n"
            f"  name \"{name}\"{b}\n}}\n")


# ═════════════════════════════════════════════════════════════════════════════
# furniture models (all dimensions in metres, 1:8 scale)
# ═════════════════════════════════════════════════════════════════════════════
def furniture_children(kind, L, W, H, c):
    dark = (0.35, 0.37, 0.42)
    if kind in ("bed", "cot"):
        blanket = (0.45, 0.68, 0.88) if kind == "bed" else (0.95, 0.65, 0.75)
        return [tr(0, 0, 0.02, box((L, W, 0.03), (0.55, 0.58, 0.62), metal=0.4)),
                tr(0, 0, 0.045, box((L * 0.94, W * 0.88, 0.025), c)),
                tr(L * 0.12, 0, 0.060, box((L * 0.62, W * 0.86, 0.012), blanket)),
                tr(-L * 0.38, 0, 0.062, box((L * 0.16, W * 0.6, 0.02), (1, 1, 1))),
                tr(-L / 2 + 0.005, 0, 0.045, box((0.01, W, 0.05), (0.5, 0.5, 0.55)))]
    if kind == "chair":
        leg = lambda sx, sy: tr(sx * (L / 2 - 0.006), sy * (W / 2 - 0.006), 0.014, box((0.008, 0.008, 0.028), dark))
        return [tr(0, 0, 0.034, box((L, W, 0.012), c)), tr(-L / 2 + 0.004, 0, 0.056, box((0.008, W, 0.04), c)),
                leg(1, 1), leg(1, -1), leg(-1, 1), leg(-1, -1)]
    if kind in ("table", "bigtable", "desk", "bench", "counter"):
        leg = lambda sx, sy: tr(sx * (L / 2 - 0.01), sy * (W / 2 - 0.01), (H - 0.012) / 2, box((0.012, 0.012, H - 0.012), dark))
        return [tr(0, 0, H - 0.006, box((L, W, 0.012), c)), leg(1, 1), leg(1, -1), leg(-1, 1), leg(-1, -1)]
    if kind == "shelf":
        return [tr(0, 0, H / 2, box((L, W, H), c))] + [tr(0, 0, z, box((L * 1.0, W * 1.25, 0.004), dark)) for z in (0.03, 0.06, 0.09)]
    if kind in ("cabinet", "fridge", "box"):
        return [tr(0, 0, H / 2, box((L, W, H), c)), tr(0, 0, H + 0.002, box((L * 1.04, W * 1.04, 0.004), dark))]
    if kind == "iv":
        return [tr(0, 0, 0.003, cyl(0.012, 0.006, dark)), tr(0, 0, 0.08, cyl(0.003, 0.16, c, metal=0.5)),
                tr(0.0, 0, 0.145, box((0.02, 0.006, 0.03), (0.8, 0.9, 1.0), rough=0.3))]
    if kind == "monitor":
        return [tr(0, 0, 0.015, box((L, W, 0.03), dark)), tr(0, W / 2 - 0.004, 0.058, box((L * 0.9, 0.008, 0.04), (0.05, 0.25, 0.2), emissive=(0.1, 0.8, 0.5)))]
    if kind == "or_table":
        return [tr(0, 0, 0.05, box((L, W, 0.012), (0.75, 0.88, 0.88), metal=0.5)), tr(0, 0, 0.026, cyl(0.02, 0.045, dark)),
                tr(0, 0, 0.004, box((0.08, 0.06, 0.008), dark))]
    if kind == "lamp":
        return [tr(0, 0, 0.003, cyl(0.016, 0.006, dark)), tr(0, 0, 0.08, cyl(0.004, 0.16, dark)),
                tr(0, 0, 0.152, cyl(0.02, 0.008, (1.0, 0.98, 0.85), emissive=(0.9, 0.85, 0.5)))]
    if kind == "xray":
        return [tr(0, 0, 0.02, box((L, W, 0.04), (0.85, 0.85, 0.88))), tr(-L * 0.3, 0, 0.09, box((0.04, 0.10, 0.10), c)),
                tr(0.0, 0, 0.13, box((L * 0.9, 0.06, 0.025), (0.7, 0.72, 0.78)))]
    if kind == "sofa":
        return [tr(0, 0, 0.02, box((L, W, 0.04), c)), tr(0, -W / 2 + 0.008, 0.045, box((L, 0.016, 0.05), c))]
    if kind == "plant":
        return [tr(0, 0, 0.0125, cyl(0.016, 0.025, (0.55, 0.35, 0.25))), tr(0, 0, 0.055, sph(0.026, c))]
    if kind == "wheelchair":
        wheel = lambda sy: tr(0, sy * (W / 2), 0.02, cyl(0.02, 0.006, dark), rot="1 0 0 1.5708")
        return [tr(0, 0, 0.035, box((L * 0.6, W * 0.8, 0.012), c)), tr(-L * 0.25, 0, 0.06, box((0.008, W * 0.8, 0.04), c)),
                wheel(1), wheel(-1)]
    if kind == "wallbench":
        return [tr(0, 0, H / 2, box((L, W, H), c))]
    return [tr(0, 0, H / 2, box((L, W, H), c))]


def agent_children(kind, L, W, H, c):
    if kind == "nurse":
        return [tr(0, 0, 0.065, cyl(0.022, 0.13, c)), tr(0, 0, 0.155, sph(0.017, (0.95, 0.80, 0.68))),
                tr(0, 0, 0.176, box((0.03, 0.03, 0.008), (1, 1, 1)))]
    # medicine trolley
    wheel = lambda sx, sy: tr(sx * (L / 2 - 0.012), sy * (W / 2 - 0.004), 0.008, cyl(0.008, 0.006, (0.1, 0.1, 0.1)), rot="1 0 0 1.5708")
    return [tr(0, 0, 0.03, box((L, W, 0.012), (0.75, 0.75, 0.8), metal=0.5)), tr(0, 0, 0.07, box((L, W, 0.012), (0.75, 0.75, 0.8), metal=0.5)),
            tr(L / 2 - 0.006, 0, 0.05, box((0.008, W, 0.04), c)),
            tr(-L * 0.15, 0, 0.083, box((0.03, 0.03, 0.014), (0.9, 0.2, 0.2))), tr(L * 0.12, 0.01, 0.083, box((0.025, 0.02, 0.014), (0.2, 0.5, 0.9))),
            wheel(1, 1), wheel(1, -1), wheel(-1, 1), wheel(-1, -1)]


def digit(d, color=(1, 1, 1)):
    seg = {"a": (0, 0.11, "h"), "b": (0.045, 0.055, "v"), "c": (0.045, -0.055, "v"), "d": (0, -0.11, "h"),
           "e": (-0.045, -0.055, "v"), "f": (-0.045, 0.055, "v"), "g": (0, 0, "h")}
    on = {1: "bc", 2: "abged", 3: "abgcd"}[d]
    out = []
    for s_ in on:
        x, z, o = seg[s_]
        size = (0.07, 0.004, 0.016) if o == "h" else (0.016, 0.004, 0.07)
        out.append(tr(x, 0, z, box(size, color, emissive=(0.9, 0.9, 0.9))))
    return out


# ═════════════════════════════════════════════════════════════════════════════
def floor_block(fl):
    out = [f"\n# ======================= {fl.label} =======================\n"]
    ox, oy, z0 = fl.origin
    # slab (the floor the robot drives on)
    out.append(solid(f"floor{fl.index + 1}_slab", ox, oy, z0 - SLAB_T, 0.0,
                     [tr(0, 0, SLAB_T / 2, box((4.20, 3.20, SLAB_T), (0.80, 0.82, 0.86), rough=0.6))], bound=(4.20, 3.20, SLAB_T)))
    # supports
    if fl.index > 0:
        h = z0 + 0.01
        for sx in (-1.9, 0.0, 1.9):
            for sy in (-1.4, 1.4):
                out.append(solid(f"floor{fl.index + 1}_col_{sx}_{sy}", ox + sx, oy + sy, -0.06, 0.0,
                                 [tr(0, 0, (z0 - SLAB_T + 0.06) / 2, box((0.09, 0.09, z0 - SLAB_T + 0.06), (0.62, 0.64, 0.68)))]))
    # coloured room floors + corridors (flat, visual only)
    z = z0 + 0.0012
    tiles = []
    for r in fl.rooms:
        x0, x1, ya, yb = r.rect
        tiles.append(tr((x0 + x1) / 2, (ya + yb) / 2, 0, box((x1 - x0, yb - ya, 0.0024), hp.ROOM_COLORS[r.kind], rough=0.5)))
    tiles.append(tr(0, 0, 0.0004, box((4.0, 2 * hp.HW, 0.0016), (0.80, 0.82, 0.88), rough=0.45)))
    tiles.append(tr(0, 0, 0.0006, box((2 * hp.HW, 3.0, 0.0016), (0.80, 0.82, 0.88), rough=0.45)))
    for lf in fl.lifts:
        tiles.append(tr(0, lf.side * (hp.LIFT_Y + hp.Y1) / 2, 0.0012, box((2 * hp.HW, abs(hp.Y1 - hp.LIFT_Y), 0.0016), (0.70, 0.80, 0.95), rough=0.4)))
        tiles.append(tr(lf.pad[0], lf.pad[1], 0.003, cyl(0.075, 0.003, (0.35, 0.60, 0.95), emissive=(0.1, 0.25, 0.5))))
        # door frame posts + a coloured sign above each lift door
        for sx in (-1, 1):
            tiles.append(tr(sx * (hp.DOOR_W / 2 + 0.008), lf.side * hp.LIFT_Y, 0.09, box((0.016, 0.034, 0.18), (0.25, 0.35, 0.60))))
        tiles.append(tr(0, lf.side * hp.LIFT_Y, 0.17, box((hp.DOOR_W + 0.04, 0.034, 0.02), (0.2, 0.5, 0.95), emissive=(0.1, 0.3, 0.6))))
    out.append(solid(f"floor{fl.index + 1}_tiles", ox, oy, z, 0.0, tiles))
    # walls
    for k, w in enumerate(fl.walls):
        out.append(solid(f"f{fl.index + 1}_wall{k}", ox + w["x"], oy + w["y"], z0, w["yaw"],
                         [tr(0, 0, w["H"] / 2, box((w["L"], w["W"], w["H"]), (0.93, 0.93, 0.95), rough=0.7))], bound=(w["L"], w["W"], w["H"])))
    # furniture
    for o in fl.furniture:
        L, W, H = o["size"]
        out.append(solid(o["name"], ox + o["x"], oy + o["y"], z0, o["yaw"], furniture_children(o["kind"], L, W, H, o["color"]), bound=(L, W, H)))
    # people / trolleys (moved by the supervisor)
    for a in fl.agents:
        m = hp.sc.DynamicMover(a)
        L, W, H = a["size"]
        out.append(solid(a["def_name"].lower(), ox + m.x, oy + m.y, z0, m.yaw, agent_children(a["kind"], L, W, H, a["color"]),
                         bound=(L, W, H), def_name=a["def_name"]))
    # floor number sign south of the plate
    panel = [tr(0, 0, 0.17, box((0.46, 0.02, 0.30), (0.10, 0.18, 0.40)))]
    panel += [tr(0, -0.012, 0.17, c) for c in digit(fl.index + 1)]
    out.append(solid(f"floor{fl.index + 1}_sign", ox, oy - 1.78, z0, 0.0, panel))
    return "".join(out)


def robot_sensors():
    out = ""
    for k in range(RS_COUNT):
        a = k * 2.0 * math.pi / RS_COUNT
        out += f'''    DistanceSensor {{
      translation {f(RS_RADIUS * math.cos(a))} {f(RS_RADIUS * math.sin(a))} {SENSOR_Z}
      rotation 0 0 1 {f(a)}
      name "rs{k}"
      type "generic"
      numberOfRays 1
      aperture 0
      lookupTable [ 0 0 0, {RS_MAX} {RS_MAX} 0.004 ]
    }}
'''
    return out


HEADER = '''#VRML_SIM R2025a utf8

# -----------------------------------------------------------------------------
# AI-Enhanced Intelligent Sensor System - Autonomous hospital delivery robot (E-puck, scale 1:8)
# AUTO-GENERATED by tools/generate_world.py from ai/hospital.py
# (edit hospital.py and re-run the generator instead of editing this file)
#
#   * 3 floors (floor plates side by side, increasing height), 8 rooms per floor, 2 lifts per floor
#   * STATIC obstacles : walls, beds, desks, shelves, IV stands, X-ray machine, wheelchairs ...
#   * DYNAMIC obstacles: a nurse and a medicine trolley per floor, moved by controllers/obstacle_supervisor
#   * E-puck: 8 IR sensors + camera (built in) + GPS + IMU + 24 range sensors (rs0..rs23)
#   * Mission: START room -> GOAL room (random, possibly on another floor; the robot rides the lifts).  The blue and
#     green pads below are only the initial positions of the demo mission, the controller moves them at start-up.
# -----------------------------------------------------------------------------

EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/robots/gctronic/e-puck/protos/E-puck.proto"

WorldInfo {
  info [
    "AI-Enhanced Intelligent Sensor System for an Autonomous Hospital Robot"
    "3-floor hospital, static + dynamic obstacles, multi-floor missions with lifts"
  ]
  title "AI Robot - 3-Floor Hospital"
  basicTimeStep 32
  coordinateSystem "ENU"
}

DEF VIEW Viewpoint {
  orientation %(vp_rot)s
  position %(vp_pos)s
  fieldOfView 1.1
}

TexturedBackground {
}
TexturedBackgroundLight {
}
'''


def main():
    pos, rot = hp.overview_camera()
    w = HEADER % {"vp_rot": " ".join(f(v) for v in rot), "vp_pos": " ".join(f(v) for v in pos)}
    # ground
    w += solid("ground", 0.0, 0.0, -0.085, 0.0, [tr(0, 0, 0, box((19.0, 7.0, 0.05), (0.45, 0.55, 0.42), rough=1.0))])
    for fl in hp.FLOORS:
        w += floor_block(fl)

    m = ms.fixed_mission()
    sx, sy = hp.to_world(m["floor_s"], *m["start"])
    gx, gy = hp.to_world(m["floor_g"], *m["goal"])
    zs, zg = hp.plate_origin(m["floor_s"])[2], hp.plate_origin(m["floor_g"])[2]
    w += "\n# ---- start / goal markers (moved by the robot controller at start-up) ----\n"
    w += solid("start_pad", sx, sy, zs + 0.0015, 0.0, [tr(0, 0, 0, cyl(sc.GOAL_PAD / 2, 0.003, (0.20, 0.40, 0.90), emissive=(0.05, 0.1, 0.4)))], def_name="START_PAD")
    w += solid("goal_pad", gx, gy, zg + 0.0015, 0.0, [tr(0, 0, 0, cyl(sc.GOAL_PAD / 2, 0.003, (0.15, 0.80, 0.25), emissive=(0.05, 0.35, 0.1)))], def_name="GOAL_PAD")
    w += solid("goal_flag", gx + sc.GOAL_PAD / 2 - 0.015, gy + sc.GOAL_PAD / 2 - 0.015, zg, 0.0,
               [tr(0, 0, 0.09, cyl(0.004, 0.18, (0.85, 0.85, 0.85), metal=0.6)), tr(0.03, 0, 0.15, box((0.06, 0.002, 0.04), (0.9, 0.1, 0.1)))],
               def_name="GOAL_FLAG")

    w += f'''
# ---- E-puck robot ----
DEF EPUCK E-puck {{
  translation {f(sx)} {f(sy)} {f(zs)}
  rotation 0 0 1 {f(m["yaw"])}
  name "e-puck"
  controller "ai_robot_controller"
  controllerArgs []
  supervisor TRUE
  turretSlot [
    GPS {{
      name "gps"
    }}
    InertialUnit {{
      name "inertial unit"
    }}
{robot_sensors()}  ]
}}

# ---- Supervisor: animates the people / trolleys, counts collisions, HUD, mission report ----
Robot {{
  name "obstacle_supervisor"
  controller "obstacle_supervisor"
  supervisor TRUE
}}
'''
    out = os.path.join(ROOT, "worlds", "ai_sensor_world.wbt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(w)
    n_solid = w.count("Solid {")
    print(f"World written: {out}  ({len(w.splitlines())} lines, {n_solid} solids, "
          f"{sum(len(fl.walls) for fl in hp.FLOORS)} walls, {sum(len(fl.furniture) for fl in hp.FLOORS)} furniture items, "
          f"{sum(len(fl.agents) for fl in hp.FLOORS)} moving agents)")


if __name__ == "__main__":
    main()

"""
hospital.py
===========
The 3-FLOOR HOSPITAL (single source of truth for the world file, the robot, the planner and every test).

Scale 1:8  ->  the 7.4 cm E-puck is a 0.6 m hospital delivery robot, a corridor is 0.40 m (3.2 m real),
a door 0.22 m (1.8 m real), a bed 0.26 x 0.12 m (2.1 x 1.0 m real), a nurse is 0.05 m wide and 0.20 m tall.

Every floor has the same skeleton (4.0 m x 3.0 m, floor-local coordinates, x east, y north, origin = centre):

                  y = +1.5 +-------+-------+---+---+-------+-------+
                           | NW1   | NW2   | L |   | NE1   | NE2   |      rooms open onto the corridors
                           |       |       | I | C | (doors face the main corridor)
                  y = +0.2 +--[ ]--+--[ ]--+ F | R +--[ ]--+--[ ]--+
                           =========== main corridor (0.4 m) ======      L = lift cabin (north + south end of the
                  y = -0.2 +--[ ]--+--[ ]--+ T | O +--[ ]--+--[ ]--+      cross corridor), CR = cross corridor
                           | SW1   | SW2   |   | S | SE1   | SE2   |
                  y = -1.5 +-------+-------+---+---+-------+-------+
                          x=-2.0                x=0                x=+2.0

  * 8 rooms per floor, each with one door (0.22 m) onto a corridor, furniture and a name
  * 2 lifts per floor (north and south end of the cross corridor); the robot rides them between floors
  * 2 moving people / trolleys per floor: a nurse in the main corridor, a medicine trolley in the cross corridor
    (set TROLLEY_NEVER_BRAKES = True for a trolley that never brakes -> perpendicular-collision situations at the crossing)

In the Webots world the three floors are three plates side by side at increasing height
(x offset PLATE_DX, height Z_STEP) so that all floors are visible at once.  Positions inside the robot software are
FLOOR-LOCAL; to_world() / to_local() convert.
"""

import math

import scene_config as sc

# ── placement of the floor plates in the world ───────────────────────────────
N_FLOORS   = 3
PLATE_DX   = 4.7           # x distance between the centres of two floor plates
Z_STEP     = 0.45          # height difference between two floors (visual)
# True: the medicine trolleys NEVER brake for the robot (stress test / perpendicular-collision demo).
# False (default): like real staff, a trolley stops when somebody is directly in front of it.
TROLLEY_NEVER_BRAKES = False
FLOOR_LABELS = ["Floor 1 (Ground)", "Floor 2", "Floor 3"]

# ── building dimensions ──────────────────────────────────────────────────────
X0, X1, Y0, Y1 = -2.0, 2.0, -1.5, 1.5        # interior of a floor
HW       = 0.20             # half width of the corridors (free space)
WALL_T   = 0.03
WALL_H   = 0.14
OUTER_T  = 0.04
DOOR_W   = 0.22
LIFT_Y   = 1.13             # y of the front wall of the north lift cabin (south one: -1.13)
LIFT_PAD = 1.32             # |y| of the lift pad
DOOR_X   = (-1.56, -0.66, 0.66, 1.56)        # x of the room doors (north and south rows)
ROOM_X   = ((-2.0, -1.115), (-1.085, -0.23), (0.23, 1.085), (1.115, 2.0))   # interior x range of the 4 rooms per row


def plate_origin(f):
    """World position (x, y, z) of the origin of floor f."""
    return (-PLATE_DX + f * PLATE_DX, 0.0, f * Z_STEP)


def to_world(f, x, y):
    ox, oy, _ = plate_origin(f)
    return ox + x, oy + y


def to_local(wx, wy):
    """World (x, y) -> (floor index, local x, local y).  The floor is the plate whose x range contains wx."""
    f = max(0, min(N_FLOORS - 1, int(round((wx + PLATE_DX) / PLATE_DX))))
    ox, oy, _ = plate_origin(f)
    return f, wx - ox, wy - oy


# ═════════════════════════════════════════════════════════════════════════════
# furniture catalogue: kind -> (length along its x axis, width, height, colour)
# ═════════════════════════════════════════════════════════════════════════════
KINDS = {
    "bed":        (0.26, 0.12, 0.06, (0.93, 0.93, 0.96)),
    "cot":        (0.20, 0.10, 0.06, (0.95, 0.80, 0.85)),
    "cabinet":    (0.05, 0.05, 0.07, (0.75, 0.75, 0.78)),
    "chair":      (0.045, 0.045, 0.06, (0.25, 0.45, 0.70)),
    "table":      (0.12, 0.08, 0.055, (0.80, 0.70, 0.55)),
    "bigtable":   (0.22, 0.10, 0.055, (0.60, 0.45, 0.30)),
    "desk":       (0.16, 0.07, 0.06, (0.55, 0.40, 0.28)),
    "shelf":      (0.22, 0.04, 0.12, (0.65, 0.65, 0.70)),
    "iv":         (0.025, 0.025, 0.16, (0.85, 0.85, 0.90)),
    "monitor":    (0.06, 0.05, 0.08, (0.20, 0.25, 0.30)),
    "or_table":   (0.24, 0.09, 0.06, (0.70, 0.85, 0.85)),
    "lamp":       (0.04, 0.04, 0.16, (0.95, 0.95, 0.80)),
    "xray":       (0.20, 0.14, 0.14, (0.85, 0.85, 0.88)),
    "bench":      (0.28, 0.07, 0.05, (0.55, 0.60, 0.65)),
    "counter":    (0.34, 0.07, 0.07, (0.40, 0.55, 0.70)),
    "sofa":       (0.22, 0.07, 0.05, (0.35, 0.50, 0.55)),
    "fridge":     (0.07, 0.07, 0.12, (0.90, 0.92, 0.95)),
    "plant":      (0.04, 0.04, 0.09, (0.20, 0.55, 0.25)),
    "wheelchair": (0.065, 0.05, 0.07, (0.25, 0.25, 0.30)),
    "wallbench":  (0.25, 0.035, 0.045, (0.55, 0.60, 0.65)),
    "box":        (0.06, 0.06, 0.06, (0.70, 0.55, 0.35)),
}

# room templates: list of (kind, u, v, yaw_local) ; u = sideways from the door axis, v = distance from the door wall
# into the room (0 .. 1.27).  yaw_local 0: long axis along u, pi/2: long axis along v.  The strip |u| < 0.2 in front of
# the door and a central lane are kept free.
_P = math.pi / 2
TEMPLATES = {
    "ward":      [("bed", -0.29, 0.80, _P), ("bed", 0.29, 0.80, _P), ("cabinet", -0.38, 1.08, 0), ("cabinet", 0.38, 1.08, 0),
                  ("chair", -0.16, 1.18, 0), ("iv", -0.21, 0.58, 0), ("plant", 0.36, 0.38, 0)],
    "icu":       [("bed", -0.29, 0.75, _P), ("bed", 0.29, 0.75, _P), ("monitor", -0.37, 0.45, 0), ("monitor", 0.37, 0.45, 0),
                  ("desk", 0.0, 1.20, 0), ("iv", -0.21, 1.00, 0), ("iv", 0.21, 1.00, 0)],
    "or":        [("or_table", 0.0, 0.85, _P), ("lamp", -0.30, 0.85, 0), ("lamp", 0.30, 0.85, 0), ("table", -0.33, 1.15, 0),
                  ("table", 0.33, 1.15, 0), ("cabinet", -0.38, 0.40, 0)],
    "emergency": [("bed", -0.30, 0.55, _P), ("bed", -0.30, 0.95, _P), ("bed", 0.30, 0.75, _P), ("iv", 0.18, 0.55, 0),
                  ("cabinet", 0.38, 1.15, 0)],
    "pharmacy":  [("counter", 0.0, 0.95, 0), ("shelf", -0.40, 0.75, _P), ("shelf", 0.40, 0.75, _P), ("chair", 0.0, 1.15, 0),
                  ("fridge", -0.37, 1.18, 0)],
    "lab":       [("bench", -0.15, 0.70, 0), ("bench", 0.15, 1.00, 0), ("fridge", -0.38, 1.18, 0), ("chair", 0.30, 0.60, 0),
                  ("cabinet", 0.38, 1.18, 0)],
    "radiology": [("xray", 0.05, 1.05, 0), ("bed", -0.30, 0.60, _P), ("cabinet", 0.38, 0.45, 0), ("chair", -0.33, 1.15, 0)],
    "cafeteria": [("table", -0.25, 0.70, 0), ("table", 0.25, 0.70, 0), ("table", -0.25, 1.10, 0), ("table", 0.25, 1.10, 0),
                  ("chair", -0.25, 0.58, 0), ("chair", 0.25, 0.58, 0), ("chair", -0.25, 0.98, 0), ("chair", 0.25, 0.98, 0)],
    "clinic":    [("desk", 0.0, 1.15, 0), ("bed", 0.30, 0.65, _P), ("chair", -0.15, 1.00, 0), ("cabinet", -0.38, 0.55, 0)],
    "store":     [("shelf", -0.40, 0.70, _P), ("shelf", 0.40, 0.70, _P), ("box", -0.30, 1.15, 0), ("box", 0.28, 1.15, 0),
                  ("box", 0.0, 1.20, 0)],
    "waiting":   [("chair", -0.36, 0.50, 0), ("chair", -0.36, 0.62, 0), ("chair", 0.36, 0.50, 0), ("chair", 0.36, 0.62, 0),
                  ("sofa", 0.0, 1.20, 0), ("plant", -0.38, 1.20, 0)],
    "nurse":     [("desk", 0.0, 0.95, 0), ("chair", 0.0, 1.10, 0), ("cabinet", -0.38, 0.55, 0), ("cabinet", 0.38, 0.55, 0),
                  ("shelf", -0.40, 1.00, _P)],
    "peds":      [("cot", -0.30, 0.80, _P), ("cot", 0.30, 0.80, _P), ("table", 0.0, 1.15, 0), ("plant", 0.38, 0.40, 0)],
    "recovery":  [("bed", -0.29, 0.60, _P), ("bed", 0.29, 0.60, _P), ("bed", 0.0, 1.05, 0), ("monitor", -0.37, 1.15, 0)],
    "office":    [("desk", 0.0, 1.10, 0), ("chair", 0.0, 0.95, 0), ("shelf", -0.40, 0.75, _P), ("sofa", 0.28, 0.55, _P)],
    "meeting":   [("bigtable", 0.0, 0.85, 0), ("chair", -0.15, 0.70, 0), ("chair", 0.15, 0.70, 0), ("chair", -0.15, 1.00, 0),
                  ("chair", 0.15, 1.00, 0), ("shelf", -0.40, 1.15, _P)],
}

ROOM_COLORS = {"ward": (0.80, 0.92, 0.85), "icu": (0.75, 0.88, 0.95), "or": (0.70, 0.90, 0.90), "emergency": (0.95, 0.80, 0.78),
               "pharmacy": (0.85, 0.88, 0.97), "lab": (0.90, 0.90, 0.80), "radiology": (0.82, 0.82, 0.92),
               "cafeteria": (0.97, 0.90, 0.75), "clinic": (0.88, 0.95, 0.88), "store": (0.85, 0.82, 0.78),
               "waiting": (0.92, 0.88, 0.92), "nurse": (0.88, 0.92, 0.80), "peds": (0.98, 0.88, 0.90),
               "recovery": (0.82, 0.92, 0.82), "office": (0.90, 0.87, 0.80), "meeting": (0.85, 0.87, 0.90)}

# rooms of every floor: (slot, name, kind); slot order = NW1 NW2 NE1 NE2 SW1 SW2 SE1 SE2
ROOM_PLAN = [
    [("NW1", "Emergency", "emergency"), ("NW2", "Pharmacy", "pharmacy"), ("NE1", "Laboratory", "lab"), ("NE2", "Radiology", "radiology"),
     ("SW1", "Cafeteria", "cafeteria"), ("SW2", "Outpatient Clinic", "clinic"), ("SE1", "Store Room", "store"), ("SE2", "Waiting Hall", "waiting")],
    [("NW1", "Ward 201", "ward"), ("NW2", "Ward 202", "ward"), ("NE1", "ICU", "icu"), ("NE2", "Ward 203", "ward"),
     ("SW1", "Nurse Station", "nurse"), ("SW2", "Ward 204", "ward"), ("SE1", "Pediatrics", "peds"), ("SE2", "Ward 205", "ward")],
    [("NW1", "Operating Theatre 1", "or"), ("NW2", "Operating Theatre 2", "or"), ("NE1", "Recovery", "recovery"), ("NE2", "Surgeon Office", "office"),
     ("SW1", "Sterilization", "lab"), ("SW2", "Ward 301", "ward"), ("SE1", "Ward 302", "ward"), ("SE2", "Meeting Room", "meeting")],
]


class Room:
    def __init__(self, floor, slot, name, kind):
        self.floor, self.slot, self.name, self.kind = floor, slot, name, kind
        self.side = 1 if slot[0] == "N" else -1                          # +1 north row, -1 south row
        col = {"W1": 0, "W2": 1, "E1": 2, "E2": 3}[slot[1:]]
        self.x0, self.x1 = ROOM_X[col]
        self.xd = DOOR_X[col]                                             # door centre x
        self.y_in = self.side * (HW + WALL_T)                             # y of the room wall inside face (|y| = 0.23)
        self.y_far = self.side * Y1
        self.door = (self.xd, self.side * (HW + WALL_T / 2))              # centre of the doorway
        self.entry = (self.xd, self.side * (HW + WALL_T + 0.17))          # a point just inside the room
        self.center = (self.xd, self.side * (HW + WALL_T + 0.62))
        self.rect = (self.x0, self.x1, min(self.y_in, self.y_far), max(self.y_in, self.y_far))   # interior (x0, x1, ymin, ymax)

    def local_to_floor(self, u, v, yaw=0.0):
        """Room-local (u, v, yaw) -> floor coordinates; south rooms are the mirror image of north rooms."""
        x = self.xd + u
        y = self.side * (HW + WALL_T + v)
        return x, y, (yaw if self.side > 0 else -yaw)

    def __repr__(self):
        return f"Room({self.floor + 1}:{self.name})"


class Lift:
    def __init__(self, name, side):
        self.name, self.side = name, side
        self.pad = (0.0, side * LIFT_PAD)                  # robot stands here while riding
        self.door = (0.0, side * LIFT_Y)                   # doorway
        self.exit = (0.0, side * (LIFT_Y - 0.20))          # first point outside the cabin
        self.yaw_out = -math.pi / 2 if side > 0 else math.pi / 2     # heading when leaving the cabin


class Floor:
    def __init__(self, index):
        self.index = index
        self.label = FLOOR_LABELS[index]
        self.origin = plate_origin(index)
        self.bounds = (X0, X1, Y0, Y1)
        self.walls = []            # dicts: x y yaw L W H
        self.furniture = []        # dicts: name kind x y yaw size color
        self.agents = []           # dicts: def_name kind p0 p1 speed pause phase size color yield
        self.rooms = []
        self.lifts = [Lift("Lift A (north)", +1), Lift("Lift B (south)", -1)]
        self._build_walls()
        self._build_rooms()
        self._build_corridor_items()
        self._build_agents()

    # ---- walls --------------------------------------------------------------
    def _wall_h(self, xa, xb, y, t=WALL_T):
        if xb - xa > 1e-6:
            self.walls.append(dict(x=(xa + xb) / 2, y=y, yaw=0.0, L=xb - xa, W=t, H=WALL_H))

    def _wall_v(self, x, ya, yb, t=WALL_T):
        if yb - ya > 1e-6:
            self.walls.append(dict(x=x, y=(ya + yb) / 2, yaw=0.0, L=t, W=yb - ya, H=WALL_H))

    def _build_walls(self):
        # outer walls (outside the interior rectangle)
        self._wall_h(X0 - OUTER_T, X1 + OUTER_T, Y1 + OUTER_T / 2, OUTER_T)
        self._wall_h(X0 - OUTER_T, X1 + OUTER_T, Y0 - OUTER_T / 2, OUTER_T)
        self._wall_v(X0 - OUTER_T / 2, Y0, Y1, OUTER_T)
        self._wall_v(X1 + OUTER_T / 2, Y0, Y1, OUTER_T)
        # main-corridor walls (y = +-0.215) with a doorway in front of every room; the cross corridor is open
        for s in (+1, -1):
            y = s * (HW + WALL_T / 2)
            for (xa, xb), doors in (((X0, -HW), (DOOR_X[0], DOOR_X[1])), ((HW, X1), (DOOR_X[2], DOOR_X[3]))):
                cur = xa
                for xd in doors:
                    self._wall_h(cur, xd - DOOR_W / 2, y)
                    cur = xd + DOOR_W / 2
                self._wall_h(cur, xb, y)
        # cross-corridor walls (x = +-0.215) and the dividers between neighbouring rooms (x = +-1.1)
        for s in (+1, -1):
            ya, yb = (HW + WALL_T, Y1) if s > 0 else (Y0, -(HW + WALL_T))
            for x in (-(HW + WALL_T / 2), HW + WALL_T / 2, -1.1, 1.1):
                self._wall_v(x, ya, yb)
        # lift cabins: front wall with a doorway
        for s in (+1, -1):
            y = s * LIFT_Y
            self._wall_h(-(HW + WALL_T), -DOOR_W / 2, y)
            self._wall_h(DOOR_W / 2, HW + WALL_T, y)

    # ---- rooms + furniture ----------------------------------------------------
    def _build_rooms(self):
        for slot, name, kind in ROOM_PLAN[self.index]:
            room = Room(self.index, slot, name, kind)
            room.kind = kind
            self.rooms.append(room)
            for k, (fk, u, v, yaw) in enumerate(TEMPLATES[kind]):
                L, W, H, color = KINDS[fk]
                x, y, yw = room.local_to_floor(u, v, yaw)
                self.furniture.append(dict(name=f"f{self.index + 1}_{slot}_{fk}{k}", kind=fk, x=x, y=y, yaw=yw,
                                           size=(L, W, H), color=color, room=name))

    def _build_corridor_items(self):
        L, W, H, c = KINDS["wallbench"]
        items = [("wallbench", -1.11, +(HW - W / 2), 0.0), ("wallbench", 1.11, -(HW - W / 2), 0.0)]
        for fk, x, y, yaw in items:
            L, W, H, c = KINDS[fk]
            self.furniture.append(dict(name=f"f{self.index + 1}_corr_{fk}{len(self.furniture)}", kind=fk, x=x, y=y, yaw=yaw,
                                       size=(L, W, H), color=c, room="corridor"))
        for fk, x, y in (("wheelchair", -1.80, -0.14), ("plant", 1.82, 0.16)) if self.index != 1 else (("wheelchair", 1.80, 0.14), ("plant", -1.82, -0.16)):
            L, W, H, c = KINDS[fk]
            self.furniture.append(dict(name=f"f{self.index + 1}_corr_{fk}{len(self.furniture)}", kind=fk, x=x, y=y, yaw=0.0,
                                       size=(L, W, H), color=c, room="corridor"))

    # ---- moving people / trolleys ----------------------------------------------
    def _build_agents(self):
        f = self.index
        self.agents = [
            # a nurse patrols the main corridor (a careful person: stops when somebody is directly in front)
            dict(def_name=f"AGENT_F{f + 1}_1", kind="nurse", p0=(-1.85, 0.0), p1=(1.85, 0.0), speed=0.045, pause=3.0,
                 phase=(0.15 + 0.3 * f) % 1.0, size=(0.05, 0.05, 0.20), color=(0.15, 0.55, 0.75), **{"yield": True}),
            # a medicine trolley runs along the cross corridor and crosses the main corridor (TROLLEY_NEVER_BRAKES: it never brakes).
            # Its lane stops 0.4 m short of the lifts so that the lift lobby stays free (the robot appears there)
            dict(def_name=f"AGENT_F{f + 1}_2", kind="trolley", p0=(0.0, -0.72), p1=(0.0, 0.72), speed=0.06, pause=2.0,
                 phase=(0.55 + 0.2 * f) % 1.0, size=(0.12, 0.06, 0.09), color=(0.85, 0.35, 0.15), **{"yield": not TROLLEY_NEVER_BRAKES}),
        ]

    # ---- queries ------------------------------------------------------------------
    def footprints(self):
        """[(cx, cy, yaw, L, W), ...] of every wall and every piece of furniture (everything static)."""
        fp = [(w["x"], w["y"], w["yaw"], w["L"], w["W"]) for w in self.walls]
        fp += [(o["x"], o["y"], o["yaw"], o["size"][0], o["size"][1]) for o in self.furniture]
        return fp

    def room(self, name):
        return next(r for r in self.rooms if r.name == name)

    def make_movers(self, phases=None, yield_enabled=True):
        """Fresh DynamicMover objects for this floor's agents."""
        return [sc.DynamicMover(a, phase=(None if phases is None else phases[k]), yield_enabled=yield_enabled)
                for k, a in enumerate(self.agents)]


FLOORS = [Floor(i) for i in range(N_FLOORS)]


def all_rooms():
    return [r for f in FLOORS for r in f.rooms]


def find_room(floor_index, name):
    return FLOORS[floor_index].room(name)


def inside(x, y, tol=0.0):
    return X0 - tol <= x <= X1 + tol and Y0 - tol <= y <= Y1 + tol


def static_clearance(floor, x, y):
    """Distance from (x, y) to the nearest wall / furniture of that floor (0 inside)."""
    return min(sc.rect_distance(x, y, *fp) for fp in FLOORS[floor].footprints())


def lane_distance(floor, x, y):
    best = 9.0
    for a in FLOORS[floor].agents:
        (ax, ay), (bx, by) = a["p0"], a["p1"]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        u = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
        best = min(best, math.hypot(x - (ax + u * dx), y - (ay + u * dy)))
    return best


# ── camera helpers (Webots ENU viewpoint: the camera looks along its +x axis, y = left, z = up) ───────────────
def look_at(pos, target):
    """Axis-angle orientation (ax, ay, az, angle) of a Webots viewpoint at `pos` looking at `target`."""
    fx, fy, fz = (target[i] - pos[i] for i in range(3))
    n = math.sqrt(fx * fx + fy * fy + fz * fz)
    fx, fy, fz = fx / n, fy / n, fz / n
    lx, ly = -fy, fx
    ln = math.hypot(lx, ly)
    lx, ly, lz = lx / ln, ly / ln, 0.0
    ux, uy, uz = fy * lz - fz * ly, fz * lx - fx * lz, fx * ly - fy * lx
    R = [[fx, lx, ux], [fy, ly, uy], [fz, lz, uz]]
    tr = R[0][0] + R[1][1] + R[2][2]
    ang = math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0)))
    ax = (R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1])
    an = math.sqrt(sum(a * a for a in ax))
    return (ax[0] / an, ax[1] / an, ax[2] / an, ang)


def floor_camera(f):
    """(position, orientation) of a camera that looks at floor f from the south, above."""
    ox, oy, z0 = plate_origin(f)
    pos = (ox, oy - 3.0, z0 + 3.2)
    return pos, look_at(pos, (ox, oy, z0 + 0.1))


def overview_camera():
    pos = (0.0, -9.8, 7.4)
    return pos, look_at(pos, (0.0, 0.0, 0.5))

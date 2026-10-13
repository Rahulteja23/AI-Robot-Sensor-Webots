"""
scene_config.py
===============
Generic constants and geometry helpers shared by every module (robot, tests, world generator).
The BUILDING itself (3 hospital floors, rooms, furniture, moving people) is described in ai/hospital.py.

Coordinate system : Webots ENU  (x = East, y = North, z = Up), metres.   Robot heading: yaw 0 = +x, counter-clockwise.
All positions used by the robot, the planner and the tests are FLOOR-LOCAL (x, y inside one floor, see hospital.py);
the world file places the three floors side by side, hospital.py converts between local and world coordinates.

Scale: the hospital is modelled at 1:8, so the 7.4 cm E-puck is a 0.6 m delivery robot.
"""

import math

# ── Robot / mission constants ────────────────────────────────────────────────
ROBOT_RADIUS  = 0.037        # e-puck body radius
GOAL_STOP     = 0.08         # robot stops when closer than this to its goal (m)
GOAL_PAD      = 0.18         # size of the drawn start / goal pads (m)

# RANDOM_MISSION = True : START room, GOAL room and the exact points are chosen at random on every run
#                         (ai/mission.py guarantees they are valid and reachable, possibly on different floors)
# MISSION_SEED          : None = a new random mission every run, an int = the same mission every run
#                         (the seed of every run is printed and saved in results/mission_result.json)
# RANDOM_MISSION = False: the demo mission defined in ai/mission.py (Pharmacy, ground floor -> Operating Theatre 1, floor 3)
RANDOM_MISSION = True
MISSION_SEED   = None
AUTO_CAMERA    = True        # the 3D view jumps to the floor the robot is on (you can still move the camera by hand)

# ── Size of a toy car (kept for the perpendicular-collision test) ─────────────
CAR_L, CAR_W, CAR_H = 0.20, 0.09, 0.07

# ── Moving objects brake for anything directly ahead (like a careful person) ──
CAR_YIELD_LOOKAHEAD = 0.18   # m ahead of the front
CAR_YIELD_MARGIN    = 0.03   # extra side margin (on top of robot radius)


# ═════════════════════════════════════════════════════════════════════════════
# Geometry helpers (pure python, no Webots)
# ═════════════════════════════════════════════════════════════════════════════

def wrap(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def rect_distance(px, py, cx, cy, yaw, length, width) -> float:
    """Distance from point (px,py) to an oriented rectangle (0 if inside)."""
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = px - cx, py - cy
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    qx = max(abs(lx) - length / 2.0, 0.0)
    qy = max(abs(ly) - width / 2.0, 0.0)
    return math.hypot(qx, qy)


# ═════════════════════════════════════════════════════════════════════════════
# Dynamic obstacle motion model (used by the supervisor AND the headless test)
# ═════════════════════════════════════════════════════════════════════════════

class DynamicMover:
    """
    Shuttles an object along a straight lane, turning around at both ends.
    The object brakes when the robot is directly in front of it
    (cars stop for obstacles, just like in real traffic).
    """

    TURN_RATE = 2.0   # rad/s while turning around at the end of the lane

    def __init__(self, cfg: dict, phase: float = None, yield_enabled: bool = True):
        self.cfg = cfg
        self.L, self.W = cfg["size"][0], cfg["size"][1]
        x0, y0 = cfg["p0"]
        x1, y1 = cfg["p1"]
        self.x0, self.y0 = x0, y0
        self.lane_len = math.hypot(x1 - x0, y1 - y0)
        self.lane_yaw = math.atan2(y1 - y0, x1 - x0)
        self.speed = cfg["speed"]
        self.pause = cfg.get("pause", 1.0)
        self.yield_enabled = yield_enabled and cfg.get("yield", True)

        ph = cfg.get("phase", 0.0) if phase is None else phase
        self.s = max(0.0, min(1.0, ph)) * self.lane_len
        self.forward = True
        self.yaw = self.lane_yaw
        self.target_yaw = self.lane_yaw
        self.pause_left = 0.0
        self.braking = False
        self.brake_events = 0
        self._was_braking = False

    # -- pose ---------------------------------------------------------------
    @property
    def x(self):
        return self.x0 + math.cos(self.lane_yaw) * self.s

    @property
    def y(self):
        return self.y0 + math.sin(self.lane_yaw) * self.s

    def pose(self):
        return self.x, self.y, self.yaw

    # -- yielding -------------------------------------------------------------
    def _should_brake(self, robot_xy) -> bool:
        if robot_xy is None or not self.yield_enabled:
            return False
        rx, ry = robot_xy
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = rx - self.x, ry - self.y
        fx = dx * c + dy * s          # along the car's heading
        fy = -dx * s + dy * c         # sideways
        ahead = fx - self.L / 2.0
        side_ok = abs(fy) <= self.W / 2.0 + ROBOT_RADIUS + CAR_YIELD_MARGIN
        if 0.0 <= ahead <= CAR_YIELD_LOOKAHEAD and side_ok:
            return True
        # anything very close from any direction
        d = rect_distance(rx, ry, self.x, self.y, self.yaw, self.L, self.W)
        return d <= ROBOT_RADIUS + CAR_YIELD_MARGIN

    # -- time step ----------------------------------------------------------------
    def step(self, dt: float, robot_xy=None):
        self.braking = self._should_brake(robot_xy)
        if self.braking and not self._was_braking:
            self.brake_events += 1
        self._was_braking = self.braking
        if self.braking:
            return

        if self.pause_left > 0.0:                       # turning around at lane end
            self.pause_left -= dt
            err = wrap(self.target_yaw - self.yaw)
            step = self.TURN_RATE * dt
            self.yaw = self.target_yaw if abs(err) <= step else self.yaw + math.copysign(step, err)
            if self.pause_left <= 0.0:
                self.yaw = self.target_yaw
            return

        self.s += (self.speed * dt) * (1.0 if self.forward else -1.0)
        if self.s >= self.lane_len:
            self.s = self.lane_len
            self.forward = False
            self.target_yaw = wrap(self.lane_yaw + math.pi)
            self.pause_left = self.pause
        elif self.s <= 0.0:
            self.s = 0.0
            self.forward = True
            self.target_yaw = self.lane_yaw
            self.pause_left = self.pause

"""
navigator.py
============
Goal-seeking + obstacle-avoiding local navigator for the E-puck.

WHY THIS FILE EXISTS
--------------------
The earlier controller only reacted to obstacles - it had no idea where the
goal was, so the robot wandered around the arena.  This navigator fuses

    * 24 long-range distance sensors  ("rs0".."rs23", 360 deg, 0.5 m range)
    * 8  E-puck IR proximity sensors  ("ps0".."ps7",  ~7 cm range)
    * GPS position + IMU heading      (where am I, where is the goal)

into a polar obstacle map and picks, every step, the free direction that is
closest to the direction of the goal.  It is a *reactive* planner: it needs no
map of the world, so it also works with moving cars.

Algorithm per step
------------------
1. Convert all sensor hits to points (distance, angle) in the robot frame.
2. For 36 candidate directions (every 10 deg) compute the free travel
   distance of a "safety circle" (robot radius + margin) along that direction.
3. Cost of a direction = angular distance to the goal bearing
                         + penalty for short free distance
                         + hysteresis (avoid dithering)
4. Rotate toward the cheapest direction; drive forward only when facing it,
   at a speed that shrinks as the obstacle ahead gets closer.
5. If the robot is not making progress for a long time -> ESCAPE mode
   (follow the most open direction for a few seconds).
6. If a nearby object is approaching while the robot waits, it is treated as
   closer than it is (simple moving-obstacle prediction).
7. EVADE: if anything comes within ~3 cm of the body (e.g. a car that drives
   toward the robot) the robot moves straight away from it - forward or in
   REVERSE, whichever is quicker - using the full 360 degree sensor ring.
"""

import math
from collections import deque

from scene_config import wrap, ROBOT_RADIUS

# ── Sensor geometry (must match worlds/ai_sensor_world.wbt) ──────────────────
RS_COUNT    = 24                 # long-range sensors, every 15 deg
RS_ANGLE    = math.radians(15.0)
RS_RADIUS   = 0.045              # mounting radius from robot centre
RS_MAX      = 0.50               # max range (m)

PS_ANGLES   = [math.radians(a) for a in
               (-17.2, -45.8, -90.0, -151.4, 151.4, 90.0, 45.8, 17.2)]   # ps0..ps7
PS_RADIUS   = 0.035

# E-puck IR lookup table (distance m -> raw value), from E-puck.proto
_PS_TABLE = [(0.000, 4095.0), (0.005, 2133.33), (0.010, 1465.73), (0.015, 601.46),
             (0.020, 383.84), (0.030, 234.93), (0.040, 158.03), (0.050, 120.0),
             (0.060, 104.09), (0.070, 67.19), (0.100, 0.0)]


def ps_to_distance(value: float):
    """Raw E-puck IR value -> distance in metres (None if nothing detected)."""
    if value < 80.0:
        return None
    if value >= _PS_TABLE[0][1]:
        return 0.0
    for (d0, v0), (d1, v1) in zip(_PS_TABLE, _PS_TABLE[1:]):
        if v1 <= value <= v0:
            f = (v0 - value) / (v0 - v1) if v0 != v1 else 0.0
            return d0 + f * (d1 - d0)
    return None


# ── Tunable parameters ───────────────────────────────────────────────────────
V_MAX       = 0.090      # m/s   forward cruise speed
W_MAX       = 2.6        # rad/s max turn rate
K_W         = 2.8        # P-gain on heading error
W_CORR      = ROBOT_RADIUS + 0.022   # safety-circle radius (m); narrow hospital corridors need a snug circle
D_LOOK      = 0.45       # look-ahead (m of free travel)
D_STOP      = 0.02       # free travel below which the robot stops
D_SLOW      = 0.25       # free travel above which the robot goes full speed
D_BLOCK     = 0.12       # a direction with less free travel counts as blocked
D_PREF      = 0.30       # preferred free travel
TURN_ONLY   = math.radians(55.0)     # larger heading error -> rotate in place
STUCK_TIME  = 40.0       # s without progress -> escape
ESCAPE_TIME = 6.0        # s
T_PRED      = 2.0        # s look-ahead for approaching obstacles
R_PANIC     = ROBOT_RADIUS + 0.028   # anything closer than this (centre distance) -> EVADE
V_EVADE     = 0.075      # m/s forward evasion speed
V_REVERSE   = 0.060      # m/s reverse speed


class Navigator:

    def __init__(self, use_motion_prediction: bool = True):
        self.use_motion_prediction = use_motion_prediction
        self.arrive_slow = True              # slow down when close to the goal
        self._prev_dir_abs = None
        self._prev_rs = None
        self._dyn = [0.0] * RS_COUNT
        self._last_v = 0.0
        self._last_w = 0.0

        self._best_goal_dist = None
        self._t_best = 0.0
        self._escape_until = -1.0
        self._escape_dir_abs = None
        self.escape_count = 0

    # ------------------------------------------------------------------ utils
    def _build_points(self, rs, ps_raw, dt):
        """Polar obstacle points [(r, theta)] in the robot frame (centre = origin)."""
        # moving-obstacle estimate: only when the robot is (almost) still,
        # so that any range decrease is caused by the obstacle itself.
        approach = [0.0] * RS_COUNT
        if self.use_motion_prediction and self._prev_rs is not None and dt > 0:
            still = abs(self._last_v) < 0.02 and abs(self._last_w) < 0.35
            for k in range(RS_COUNT):
                if still and rs[k] < RS_MAX - 0.01 and self._prev_rs[k] < RS_MAX - 0.01:
                    rate = (self._prev_rs[k] - rs[k]) / dt        # >0 : getting closer
                    self._dyn[k] = 0.6 * self._dyn[k] + 0.4 * rate
                else:
                    self._dyn[k] *= 0.5
                if self._dyn[k] > 0.025:
                    approach[k] = min(self._dyn[k], 0.15)
        self._prev_rs = list(rs)

        pts = []
        for k, d in enumerate(rs):
            if d < RS_MAX - 0.01:
                r = RS_RADIUS + d
                r = max(0.5 * ROBOT_RADIUS, r - T_PRED * approach[k])
                pts.append((r, wrap(k * RS_ANGLE)))
        for j, v in enumerate(ps_raw):
            d = ps_to_distance(v)
            if d is not None:
                pts.append((PS_RADIUS + d, PS_ANGLES[j]))
        return pts

    @staticmethod
    def _free_travel(cart_pts, phi):
        """Free centre-travel along direction phi before the safety circle hits a point."""
        cp, sp = math.cos(phi), math.sin(phi)
        best = D_LOOK
        for px, py in cart_pts:
            along = px * cp + py * sp
            if along <= 0.0:
                continue
            lat = abs(-px * sp + py * cp)
            if lat >= W_CORR:
                continue
            travel = along - math.sqrt(W_CORR * W_CORR - lat * lat)
            if travel < best:
                best = max(travel, 0.0)
        return best

    # ------------------------------------------------------------------- main
    def plan(self, t, dt, pose, goal, rs, ps_raw, progress=None):
        """
        Parameters
        ----------
        t      : simulation time (s)
        dt     : controller time step (s)
        pose   : (x, y, yaw)
        goal   : (gx, gy)
        rs     : 24 long-range distances (m), RS_MAX if nothing seen
        ps_raw : 8 raw E-puck IR values

        progress : optional measure that DECREASES while the robot makes headway (e.g. remaining route length); the
                   stuck monitor uses it instead of the distance to `goal` when the goal is a moving look-ahead point
        Returns dict: v, w, mode, action, phi, c0, c_left, c_right, bearing, dist
        """
        x, y, yaw = pose
        gx, gy = goal
        dist = math.hypot(gx - x, gy - y)
        bearing = wrap(math.atan2(gy - y, gx - x) - yaw)         # goal direction, robot frame

        pts = self._build_points(rs, ps_raw, dt)
        cart = [(r * math.cos(th), r * math.sin(th)) for r, th in pts]

        # ---- free travel in 36 directions ----
        cands = [wrap(math.radians(d)) for d in range(-180, 180, 10)]
        free = [self._free_travel(cart, phi) for phi in cands]
        c0 = self._free_travel(cart, 0.0)
        c_left = min((f for p, f in zip(cands, free) if math.radians(30) <= p <= math.radians(90)), default=D_LOOK)
        c_right = min((f for p, f in zip(cands, free) if -math.radians(90) <= p <= -math.radians(30)), default=D_LOOK)

        # ---- progress / stuck monitor ----
        prog = dist if progress is None else progress
        if self._best_goal_dist is None or prog < self._best_goal_dist - 0.05:
            self._best_goal_dist = prog
            self._t_best = t
        mode = "GO"
        if t < self._escape_until:
            mode = "ESCAPE"
        elif t - self._t_best > STUCK_TIME:
            self._escape_until = t + ESCAPE_TIME
            self._escape_dir_abs = None
            self._t_best = t + ESCAPE_TIME
            self._best_goal_dist = prog
            self.escape_count += 1
            mode = "ESCAPE"

        # ---- choose direction ----
        best_i, best_cost = 0, 1e9
        for i, (phi, f) in enumerate(zip(cands, free)):
            if mode == "ESCAPE":
                cost = -3.0 * f
                if self._escape_dir_abs is not None:
                    cost += 0.5 * abs(wrap(yaw + phi - self._escape_dir_abs))
                else:
                    cost += 0.2 * abs(phi)
            else:
                cost = abs(wrap(phi - bearing))
                if f < D_PREF:
                    cost += 6.0 * (D_PREF - f)
                if f < D_BLOCK:
                    cost += 2.0
                if self._prev_dir_abs is not None:
                    cost += 0.35 * abs(wrap(yaw + phi - self._prev_dir_abs))
            if cost < best_cost:
                best_cost, best_i = cost, i
        phi_star, c_star = cands[best_i], free[best_i]
        if mode == "ESCAPE" and self._escape_dir_abs is None:
            self._escape_dir_abs = wrap(yaw + phi_star)
        self._prev_dir_abs = wrap(yaw + phi_star)

        # ---- EVADE: something is dangerously close -> move straight away from it ----
        evade_phi = None
        close = [(r, th) for r, th in pts if r < R_PANIC]
        if close:
            sx = sy = 0.0
            for r, th in close:
                wgt = 1.0 / max(r - ROBOT_RADIUS, 0.004)
                sx -= wgt * math.cos(th)
                sy -= wgt * math.sin(th)
            if math.hypot(sx, sy) > 1e-6:
                phi_rep = math.atan2(sy, sx)
                ok = [(abs(wrap(phi - phi_rep)), phi) for phi, f in zip(cands, free) if f >= 0.05]
                if ok:
                    evade_phi = min(ok)[1]
        if evade_phi is not None:
            mode = "EVADE"
            phi_star = evade_phi
            c_star = free[cands.index(evade_phi)] if evade_phi in cands else c0
            self._prev_dir_abs = wrap(yaw + phi_star)

        # ---- velocity command ----
        err = phi_star
        reverse = False
        if mode == "EVADE":
            if abs(err) > math.radians(100):                 # quicker to back away than to turn around
                reverse = True
                err_b = wrap(err - math.pi)
                w = max(-W_MAX, min(W_MAX, K_W * err_b))
                v = -V_REVERSE * max(0.0, math.cos(err_b)) ** 2
            else:
                w = max(-W_MAX, min(W_MAX, K_W * err))
                v = V_EVADE * max(0.0, math.cos(err)) ** 2
        elif abs(err) > math.radians(110) and c0 < D_BLOCK and free[cands.index(wrap(math.radians(180)))] > 0.10:
            # front blocked and the best way out is behind us -> reverse out
            reverse = True
            err_b = wrap(err - math.pi)
            w = max(-W_MAX, min(W_MAX, K_W * err_b))
            v = -V_REVERSE * max(0.0, math.cos(err_b)) ** 2
            mode = "REVERSE"
        else:
            w = max(-W_MAX, min(W_MAX, K_W * err))
            if abs(err) > TURN_ONLY:
                v = 0.0
            else:
                v = V_MAX * max(0.0, math.cos(err)) ** 2
            v *= max(0.0, min(1.0, (c0 - D_STOP) / (D_SLOW - D_STOP)))     # slow down for obstacle ahead
            v *= max(0.0, min(1.0, (c_star - D_STOP) / (D_SLOW - D_STOP)))
            if self.arrive_slow:
                v *= max(0.35, min(1.0, dist / 0.30))                  # gentle arrival

        # ---- label ----
        deviated = abs(wrap(phi_star - bearing)) > math.radians(25)
        if mode not in ("ESCAPE", "EVADE", "REVERSE"):
            if v < 0.01 and abs(w) < 0.3:
                mode = "WAIT"
            elif deviated:
                mode = "AVOID"
        if v < -0.01:
            action = "STOP"          # reversing is logged as STOP so older plots keep working
        elif v < 0.01 and abs(w) < 0.3:
            action = "STOP"
        elif w > 0.3:
            action = "LEFT"
        elif w < -0.3:
            action = "RIGHT"
        else:
            action = "FORWARD"

        self._last_v, self._last_w = v, w
        return dict(v=v, w=w, mode=mode, action=action, phi=phi_star,
                    c0=c0, c_left=c_left, c_right=c_right, c_star=c_star,
                    bearing=bearing, dist=dist,
                    min_range=min((r for r, _ in pts), default=RS_MAX + RS_RADIUS) - ROBOT_RADIUS)

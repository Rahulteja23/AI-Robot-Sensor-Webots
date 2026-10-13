"""
motion_predictor.py
===================
Handles MOVING obstacles (cars) that drive toward the robot - including a car that
hits the robot from the SIDE (perpendicular / T-bone).

Why it is needed
----------------
The Navigator only reasons about the corridor in front of the robot and about what
obstacles are doing *right now*.  A car approaching from the side is outside that
corridor until it is almost touching.  This module adds PREDICTION and DYNAMIC SPEED
CONTROL on top of the Navigator (the Navigator is not replaced).

WHEN IT ACTS: only for movers at least V_ENGAGE (0.085 m/s) fast.  Slower ones (hospital staff 0.045 m/s, trolleys 0.06 m/s)
are left to the reactive Navigator: in 0.4 m corridors its simple "stop / step aside" behaviour was measurably safer
and faster than cautious predictive manoeuvres (see README, section 3).  Fast objects (T-bone situations, toy cars at
0.10-0.28 m/s) are exactly where last-moment reactions fail, and where this layer reduced collisions from 28 to 1.

Pipeline (every control step)
-----------------------------
1. DynamicTracker
     * keeps a free-space map in world coordinates (cells that a range ray passed through)
     * a ray hit on a cell (and the space behind it) that was recently free  ->  that surface
       MOVED (a car); a parked car or wall never does this, whatever the robot's own motion
     * groups moving hits into clusters, tracks them and estimates their velocity
2. DynamicSafety
     * rolls the Navigator's command forward for 3 s against the PREDICTED car positions
     * if no collision is predicted -> the command passes unchanged (no behaviour change)
     * otherwise it scores ~30 candidate (speed, turn-rate) commands - brake, creep, REVERSE,
       DODGE and ACCELERATE (up to 0.12 m/s) - and chooses the safest one closest to the plan
     * (optional, OFF by default: ENABLE_WATCH) creep-and-watch while the speed of a brand-new
       moving object is unknown.  Tests showed it keeps the robot lingering inside car lanes, which
       is more dangerous than driving on, so the robot instead decides immediately and re-plans
       every control step (~15 times per second) as the estimate improves

Output modes (CSV column dyn_mode):
    CLEAR   nothing moving nearby / no predicted collision
    WATCH   (disabled by default) creep while a new object's speed is measured
    YIELD   robot slows or stops to let the car pass
    BOOST   robot speeds up to clear the car's path
    BACKOFF robot reverses out of the car's path
    DODGE   robot steers away
"""

import math
from collections import deque

from scene_config import ROBOT_RADIUS
from navigator import RS_COUNT, RS_ANGLE, RS_RADIUS, RS_MAX

# ── free-space map ───────────────────────────────────────────────────────────
CELL         = 0.03      # m
FREE_MARGIN  = 0.05      # cells closer than this to a hit are NOT marked free
FREE_STEP    = 0.024     # sampling step along a ray
T_FREE       = 3.0       # s a "free" observation stays valid
EDGE_JUMP    = 0.05      # m range discontinuity that marks an object edge
BEHIND       = 0.035     # m: the space BEHIND a moving hit must also have been seen free

# ── tracking ─────────────────────────────────────────────────────────────────
GATE_BASE    = 0.18      # m association gate
SPLIT_DIST   = 0.22      # m: a new cluster this close to a known car is part of that car
TRACK_TTL    = 0.6       # s a lost track is kept (coasting)
V_MIN        = 0.02      # m/s below this a mover is treated as static
V_MAX_CAR    = 0.40      # m/s sanity clamp
SIDEWAYS_BELOW = 0.05    # m/s: measured speed below this -> also test sideways motion (aperture problem)
SIDEWAYS_SPEED = 0.07    # m/s assumed sideways speed
HIST_S       = 0.6       # s regression window
MIN_SAMPLES  = 4
SLOW_EGO     = 0.045     # m/s: samples taken while the robot is this slow have no 'sliding' bias
MIN_SLOW     = 4
SLOW_SPAN    = 0.20      # s of slow samples needed before the velocity counts as 'good'

# ── car body model used for prediction ───────────────────────────────────────
BODY_BACK    = 0.14      # m behind the visible point
BODY_FRONT   = 0.06      # m in front of the visible point
CAR_HALF_W   = 0.04      # m   (people / trolleys / small cars)

# ── safety layer ─────────────────────────────────────────────────────────────
HORIZON      = 3.0       # s
DT_ROLL      = 0.15      # s
MARGIN_DYN   = 0.025     # m extra clearance to a moving object
D_DYN        = ROBOT_RADIUS + CAR_HALF_W + MARGIN_DYN
R_STATIC     = ROBOT_RADIUS + 0.025      # centre-to-scan-point distance counted as a hit
WHEEL_R, AXLE, MAX_WHEEL = 0.0205, 0.052, 6.28      # E-puck drive limits


def limit_vw(v, w):
    """Scale (v, w) so neither wheel exceeds its maximum angular speed."""
    m = max(abs(v - w * AXLE / 2.0), abs(v + w * AXLE / 2.0)) / WHEEL_R
    if m > MAX_WHEEL:
        k = MAX_WHEEL / m
        return v * k, w * k
    return v, w


V_CANDS      = (-0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12)
W_CANDS      = (-1.6, -0.7, 0.0, 0.7, 1.6)
TURN_RATE    = 2.4                       # rad/s turn-on-the-spot used by the 'turn then go' manoeuvres
TURN_ANGLES  = (-2.36, -1.57, -0.79, 0.79, 1.57, 2.36)   # +-135, +-90, +-45 deg
VEER_ANGLES  = (-0.52, -0.26, 0.26, 0.52)                # +-30, +-15 deg: step aside while driving, then go straight
VEER_RATE    = 1.0                                       # rad/s while veering
V_ENGAGE     = 0.085     # m/s: the predictive layer only handles movers at least this fast. Slower ones (people, trolleys
                         # in a hospital) are left to the reactive Navigator, which copes with them better in narrow corridors.
ENABLE_WATCH = False     # creep-and-watch while a new moving object's speed is unknown
THREAT_USES_CONE = False # detect threats against the full uncertainty cone (False: best estimate only)
HOLD_RANGE   = 0.45      # m  an unknown mover closer than this -> creep and watch
V_HOLD       = 0.02      # m/s


def _cell(x, y):
    return (int(math.floor(x / CELL)), int(math.floor(y / CELL)))


def _bb(seg, pad=0.30):
    """A predicted body segment plus its bounding box (grown by `pad`): far-away robots skip the exact distance test."""
    ax, ay, bx, by = seg
    return (ax, ay, bx, by, min(ax, bx) - pad, max(ax, bx) + pad, min(ay, by) - pad, max(ay, by) + pad)


def _pt_seg(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    u = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + u * dx), py - (ay + u * dy))


class Track:
    """One moving object.  Several velocity HYPOTHESES are kept because, with 15-degree ray spacing,
    the sideways speed of a car is hard to measure while its closing speed is measured precisely."""
    _next_id = 1

    def __init__(self, t, c, pts, ego_speed=0.0):
        self.id = Track._next_id
        Track._next_id += 1
        self.c = c
        self.pts = pts
        self.hist = deque(maxlen=14)          # (t, cx, cy, ego_speed, rho, rx, ry, ex, ey)
        self.last_t = t
        self.born_t = t
        self.v_slow = None                    # (vx, vy) measured while the robot was slow  -> 'good'
        self.v_all = None                     # (vx, vy) from all samples
        self.v_rad = None                     # (vx, vy) closing-speed hypothesis along the line of sight
        self.good = False

    # -- main (best-guess) velocity -------------------------------------------------
    @property
    def v(self):
        for h in (self.v_slow, self.v_all, self.v_rad):
            if h is not None:
                return h
        return (0.0, 0.0)

    @property
    def vx(self):
        return self.v[0]

    @property
    def vy(self):
        return self.v[1]

    @property
    def valid(self):
        return self.v_slow is not None or self.v_all is not None

    @property
    def speed(self):
        return math.hypot(*self.v)

    @property
    def moving(self):
        if not self.valid:
            return False
        return any(h is not None and math.hypot(*h) >= V_MIN for h in (self.v_slow, self.v_all, self.v_rad))

    def pos_at(self, tau):
        return self.c[0] + self.vx * tau, self.c[1] + self.vy * tau

    def _sideways(self):
        """APERTURE PROBLEM: an object sliding ALONG the surface the robot is looking at barely changes any
        range reading, so its sideways speed is invisible.  A free-space violation proves the object moved;
        if the measured speed is nevertheless tiny, also test "moving sideways" in both directions."""
        if self.speed >= SIDEWAYS_BELOW or not self.hist:
            return []
        ux, uy = self.hist[-1][5], self.hist[-1][6]            # line of sight robot -> object
        tx, ty = -uy * SIDEWAYS_SPEED, ux * SIDEWAYS_SPEED
        return [(tx, ty, math.radians(10.0)), (-tx, -ty, math.radians(10.0))]

    def central(self):
        """Best single estimate of the velocity (vx, vy)."""
        return self.v

    def hypotheses(self):
        """[(vx, vy, half_cone_rad), ...] velocities the car may plausibly have.
        Once the speed was measured while the robot stood (nearly) still, that clean value is trusted
        (small cone).  Before that, the closing-speed and centroid hypotheses are both kept."""
        out = self._sideways()
        if self.v_slow is not None:
            return out + [(self.v_slow[0], self.v_slow[1], math.radians(18.0))]
        if self.v_all is not None:
            out.append((self.v_all[0], self.v_all[1], math.radians(35.0)))
        if self.v_rad is not None and math.hypot(*self.v_rad) >= V_MIN:
            out.append((self.v_rad[0], self.v_rad[1], math.radians(35.0)))
        return out

    def segment_at(self, tau, vx, vy, rot=0.0):
        """Predicted body of the car as a line segment (x0, y0, x1, y1); velocity rotated by `rot`."""
        c, s_ = math.cos(rot), math.sin(rot)
        rvx, rvy = vx * c - vy * s_, vx * s_ + vy * c
        cx, cy = self.c[0] + rvx * tau, self.c[1] + rvy * tau
        sp = math.hypot(rvx, rvy)
        if sp < V_MIN:
            return cx, cy, cx, cy
        ux, uy = rvx / sp, rvy / sp
        return cx - BODY_BACK * ux, cy - BODY_BACK * uy, cx + BODY_FRONT * ux, cy + BODY_FRONT * uy


# ═════════════════════════════════════════════════════════════════════════════
class DynamicTracker:
    """Finds moving objects in the range-ring scan and estimates their velocity."""

    def __init__(self):
        self.free = {}
        self.tracks = []
        self._frame = 0
        self.static_pts = []          # scan points that are NOT moving (world frame)
        self.static_segs = []         # surfaces between neighbouring static hits
        self.dyn_rays = 0
        self._prev_rs = None

    # ---------------------------------------------------------------- update
    def update(self, t, pose, rs, ego_vec):
        x, y, yaw = pose
        ego_speed = math.hypot(*ego_vec)
        self._frame += 1

        # pass 1: hit points and the free-space-violation test (uses PAST frames only)
        hits, dyn = {}, set()
        rays = []
        for k, d in enumerate(rs):
            a = yaw + k * RS_ANGLE
            ca, sa = math.cos(a), math.sin(a)
            ox, oy = x + RS_RADIUS * ca, y + RS_RADIUS * sa
            hit = d < RS_MAX - 0.01
            rays.append((ox, oy, ca, sa, d, hit))
            if hit:
                px, py = ox + d * ca, oy + d * sa
                hits[k] = (px, py)
                st = self.free.get(_cell(px, py))
                if st is not None and t - st <= T_FREE:
                    # a real mover also drove into space that was free BEHIND its surface;
                    # a static surface has solid material there (this removes grazing-ray artefacts)
                    sb = self.free.get(_cell(px + BEHIND * ca, py + BEHIND * sa))
                    if sb is not None and t - sb <= T_FREE:
                        dyn.add(k)

        # pass 2: mark free space along every ray.
        # A ray at a DEPTH DISCONTINUITY (a neighbouring ray, or its own previous reading, was much
        # closer) probably just grazed the edge of an object - it must not claim the space beyond that
        # closer reading is free, otherwise the object's own edge would later look like a mover.
        prev = self._prev_rs
        for k, (ox, oy, ca, sa, d, hit) in enumerate(rays):
            r_eff = d if hit else RS_MAX
            lim = r_eff
            for nb in ((k - 1) % RS_COUNT, (k + 1) % RS_COUNT):
                if rs[nb] < r_eff - EDGE_JUMP:
                    lim = min(lim, rs[nb])
            if prev is not None and prev[k] < r_eff - EDGE_JUMP:
                lim = min(lim, prev[k])
            length = lim - FREE_MARGIN
            s = 0.02
            while s < length:
                self.free[_cell(ox + s * ca, oy + s * sa)] = t
                s += FREE_STEP
        if self._frame % 120 == 0:
            self.free = {c: st for c, st in self.free.items() if t - st <= T_FREE}

        self._prev_rs = list(rs)
        self.dyn_rays = len(dyn)
        self.dyn_idx = dyn            # exposed for diagnostics
        self.static_pts = [p for k, p in hits.items() if k not in dyn]
        # connect neighbouring hits that lie on the same surface -> a flat car side is seen as a LINE,
        # not as isolated points a robot could slip between
        self.static_segs = []
        for k in range(RS_COUNT):
            k2 = (k + 1) % RS_COUNT
            if k in hits and k2 in hits and k not in dyn and k2 not in dyn:
                (x0, y0), (x1, y1) = hits[k], hits[k2]
                if abs(rs[k] - rs[k2]) < 0.08 and math.hypot(x1 - x0, y1 - y0) < 0.16:
                    self.static_segs.append((x0, y0, x1, y1))

        # cluster the moving hits (neighbouring rays, close in space)
        clusters, cur = [], []
        for k in sorted(dyn):
            if cur:
                k0 = cur[-1]
                (x0, y0), (x1, y1) = hits[k0], hits[k]
                rng = min(rs[k0], rs[k])
                near = (k - k0) % RS_COUNT == 1 or (k - k0) == 1
                if near and math.hypot(x1 - x0, y1 - y0) < 0.05 + 0.30 * (rng + RS_RADIUS):
                    cur.append(k)
                    continue
                clusters.append(cur)
                cur = []
            cur.append(k)
        if cur:
            clusters.append(cur)
        if len(clusters) > 1 and (clusters[0][0] == 0 and clusters[-1][-1] == RS_COUNT - 1):
            clusters[0] = clusters[-1] + clusters[0]       # ring wrap-around
            clusters.pop()

        # associate with tracks
        pts_list = [[hits[k] for k in cl] for cl in clusters]
        cents = [(sum(p[0] for p in pl) / len(pl), sum(p[1] for p in pl) / len(pl)) for pl in pts_list]
        used = set()
        for tr in self.tracks:
            tau = t - tr.last_t
            px, py = tr.pos_at(tau) if tr.valid else tr.c
            best, bj = 1e9, None
            for j, c in enumerate(cents):
                if j in used:
                    continue
                dd = math.hypot(c[0] - px, c[1] - py)
                if dd < best:
                    best, bj = dd, j
            if bj is not None and best < GATE_BASE + tr.speed * 0.3:
                used.add(bj)
                self._update_track(tr, t, cents[bj], pts_list[bj], ego_vec, (x, y))
        for j, c in enumerate(cents):
            if j in used:
                continue
            # a cluster that merely SPLIT off an existing car (same car, different rays) must not become a new "unknown" object
            owner = None
            for tr in self.tracks:
                if t - tr.last_t <= 0.4 and any(math.hypot(c[0] - px, c[1] - py) < SPLIT_DIST for px, py in tr.pts):
                    owner = tr
                    break
            if owner is not None:
                owner.pts = owner.pts + pts_list[j]
                owner.last_t = t
                continue
            if True:
                tr = Track(t, c, pts_list[j], ego_speed)
                self._add_sample(tr, t, c, pts_list[j], ego_vec, (x, y))
                self.tracks.append(tr)

        # drop stale tracks
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= TRACK_TTL]
        return self.tracks

    # ------------------------------------------------------------ one track
    @staticmethod
    def _add_sample(tr, t, c, pts, ego_vec, rob):
        # nearest hit point of the cluster -> line of sight + range
        rx0, ry0 = rob
        best = min(pts, key=lambda p: (p[0] - rx0) ** 2 + (p[1] - ry0) ** 2)
        rho = math.hypot(best[0] - rx0, best[1] - ry0)
        ux, uy = ((best[0] - rx0) / rho, (best[1] - ry0) / rho) if rho > 1e-6 else (1.0, 0.0)
        tr.hist.append((t, c[0], c[1], math.hypot(*ego_vec), rho, ux, uy, ego_vec[0], ego_vec[1]))

    @staticmethod
    def _slope(ts, vals):
        n = len(ts)
        mt, mv = sum(ts) / n, sum(vals) / n
        den = sum((a - mt) ** 2 for a in ts)
        return None if den < 1e-9 else sum((a - mt) * (b - mv) for a, b in zip(ts, vals)) / den

    @staticmethod
    def _clamp(vx, vy):
        sp = math.hypot(vx, vy)
        return (vx, vy) if sp <= V_MAX_CAR else (vx * V_MAX_CAR / sp, vy * V_MAX_CAR / sp)

    def _update_track(self, tr, t, c, pts, ego_vec, rob):
        tr.c, tr.pts, tr.last_t = c, pts, t
        self._add_sample(tr, t, c, pts, ego_vec, rob)
        h = [e for e in tr.hist if t - e[0] <= HIST_S]
        if len(h) < MIN_SAMPLES or h[-1][0] - h[0][0] < 0.18:
            return
        ts = [e[0] for e in h]

        # (a) centroid velocity from ALL samples (biased by sliding while the robot moves)
        vx, vy = self._slope(ts, [e[1] for e in h]), self._slope(ts, [e[2] for e in h])
        if vx is not None:
            vx, vy = self._clamp(vx, vy)
            tr.v_all = (vx, vy) if tr.v_all is None else (tr.v_all[0] + 0.5 * (vx - tr.v_all[0]),
                                                           tr.v_all[1] + 0.5 * (vy - tr.v_all[1]))

        # (b) radial hypothesis: range-rate of the nearest point, corrected for the robot's own motion
        rdot = self._slope(ts, [e[4] for e in h])
        if rdot is not None:
            last = h[-1]
            vr = rdot + last[7] * last[5] + last[8] * last[6]       # car velocity along the line of sight
            vr = max(-V_MAX_CAR, min(V_MAX_CAR, vr))
            tr.v_rad = (vr * last[5], vr * last[6])

        # (c) clean centroid velocity from the samples taken while the robot was (almost) stopped
        slow = [e for e in h if e[3] < SLOW_EGO]
        if len(slow) >= MIN_SLOW and slow[-1][0] - slow[0][0] >= SLOW_SPAN:
            st = [e[0] for e in slow]
            sx, sy = self._slope(st, [e[1] for e in slow]), self._slope(st, [e[2] for e in slow])
            if sx is not None:
                sx, sy = self._clamp(sx, sy)
                if tr.v_slow is None:
                    tr.v_slow = (sx, sy)
                else:
                    tr.v_slow = (tr.v_slow[0] + 0.6 * (sx - tr.v_slow[0]), tr.v_slow[1] + 0.6 * (sy - tr.v_slow[1]))
                tr.good = True


# ═════════════════════════════════════════════════════════════════════════════
class DynamicSafety:
    """Predictive speed / direction governor.  Wraps the Navigator's (v, w) command."""

    def __init__(self):
        self.tracker = DynamicTracker()
        self._prev = (0.0, 0.0)
        self.hold_time = 0.0
        self.last_info = dict(mode="CLEAR", ttc=None, level=0, n_moving=0, min_dist=None)

    # --------------------------------------------------------------- rollout
    @staticmethod
    def _rollout(plan, pose, goal, segs, static_pts, static_segs=()):
        """Simulate a two-phase plan (v1, w1, t1, v2, w2) for HORIZON seconds: (v1, w1) for t1 seconds,
        then (v2, w2).  A plain command has t1 = 0.  Returns (t_collide, d_min_dyn, static_hit, goal_dist, d_static)."""
        v1, w1, t1, v2, w2 = plan
        x, y, th = pose
        n = int(HORIZON / DT_ROLL)
        t_col, d_min, st_hit = None, 9.0, False
        st_min = 9.0
        for k in range(1, n + 1):
            v, w = (v1, w1) if k * DT_ROLL <= t1 else (v2, w2)
            th += w * DT_ROLL
            x += v * math.cos(th) * DT_ROLL
            y += v * math.sin(th) * DT_ROLL
            for (ax, ay, bx, by, bx0, bx1, by0, by1) in segs[k]:
                if x < bx0 or x > bx1 or y < by0 or y > by1:
                    continue                                     # farther than 0.3 m: cannot matter
                d = _pt_seg(x, y, ax, ay, bx, by)
                if d < d_min:
                    d_min = d
                if t_col is None and d < D_DYN:
                    t_col = k * DT_ROLL
            for (px, py) in static_pts:
                if abs(px - x) < 0.14 and abs(py - y) < 0.14:
                    d = math.hypot(px - x, py - y)
                    if d < st_min:
                        st_min = d
            for (ax, ay, bx, by) in static_segs:
                if min(ax, bx) - 0.14 < x < max(ax, bx) + 0.14 and min(ay, by) - 0.14 < y < max(ay, by) + 0.14:
                    d = _pt_seg(x, y, ax, ay, bx, by)
                    if d < st_min:
                        st_min = d
            if st_min < R_STATIC:
                st_hit = True
        return t_col, d_min, st_hit, math.hypot(goal[0] - x, goal[1] - y), st_min

    # ---------------------------------------------------------------- filter
    def filter(self, t, dt, pose, goal, rs, v_nav, w_nav, ego_vec):
        """
        Returns (v, w, info).  v, w equal (v_nav, w_nav) unless a moving object threatens the robot.
        """
        x, y, yaw = pose
        tracks = self.tracker.update(t, pose, rs, ego_vec)
        movers = [tr for tr in tracks if tr.moving and tr.speed >= V_ENGAGE and t - tr.last_t < 0.35]
        known_moving = any(tr.good and tr.moving and t - tr.last_t < 0.5 for tr in tracks)
        unknown = [] if known_moving else [tr for tr in tracks if (not tr.valid or not tr.good) and t - tr.last_t < 0.2 and
                   min(math.hypot(px - x, py - y) for px, py in tr.pts) < HOLD_RANGE]

        info = dict(mode="CLEAR", ttc=None, level=0, n_moving=len(movers), min_dist=None)

        # --- unknown fast-changing object near: creep & watch while its velocity is measured
        if ENABLE_WATCH and unknown and self.hold_time < 2.5:
            self.hold_time += dt
            v = max(-V_HOLD, min(V_HOLD, v_nav))
            w = w_nav * (abs(v) / max(abs(v_nav), 1e-3)) if abs(v_nav) > V_HOLD else w_nav
            info.update(mode="WATCH", level=1)
            self._prev = (v, w)
            self.last_info = info
            return v, w, info
        if not unknown:
            self.hold_time = max(0.0, self.hold_time - 0.5 * dt)

        if not movers:
            self._prev = (v_nav, w_nav)
            self.last_info = info
            return v_nav, w_nav, info

        # --- predicted body segments of every mover at every roll-out step
        n = int(HORIZON / DT_ROLL)
        segs = [[] for _ in range(n + 1)]          # every velocity hypothesis, each with its direction cone
        segs_c = [[] for _ in range(n + 1)]        # the best single estimate only
        for k in range(1, n + 1):
            tau = k * DT_ROLL + 0.0
            for tr in movers:
                lag = t - tr.last_t
                for (hx, hy, th) in tr.hypotheses():
                    for rot in (-th, 0.0, th):
                        segs[k].append(_bb(tr.segment_at(tau + lag, hx, hy, rot)))
                cx, cy = tr.central()
                segs_c[k].append(_bb(tr.segment_at(tau + lag, cx, cy, 0.0)))
        static_pts = self.tracker.static_pts
        static_segs = self.tracker.static_segs

        # does the plan collide?  (the plan itself is rolled out as it will really be executed)
        v_nav, w_nav = limit_vw(v_nav, w_nav)
        t_col, d_min, st_hit, _, _ = self._rollout((0.0, 0.0, 0.0, v_nav, w_nav), pose, goal,
                                                   segs if THREAT_USES_CONE else segs_c, static_pts, static_segs)
        info["min_dist"] = d_min
        if t_col is None:
            info["level"] = 1 if d_min < D_DYN + 0.10 else 0
            self._prev = (v_nav, w_nav)
            self.last_info = info
            return v_nav, w_nav, info

        # --- threat: score candidate commands
        info["ttc"] = t_col
        info["level"] = 2
        best, best_cost = None, 1e9
        # candidate plans: (v1, w1, t1, v2, w2)  = (v1, w1) for t1 seconds, then (v2, w2)
        cands = [(0.0, 0.0, 0.0, v_nav, w_nav)]
        for v in V_CANDS:
            for w in W_CANDS:
                cv, cw = limit_vw(v, w)
                cands.append((0.0, 0.0, 0.0, cv, cw))
        for ang in TURN_ANGLES:                      # "turn on the spot, then drive away"  (step out of a lane)
            for v2 in (0.06, 0.12):
                wt = math.copysign(TURN_RATE, ang)
                cands.append((0.0, wt, abs(ang) / TURN_RATE, v2, 0.0))
        for ang in VEER_ANGLES:                      # "veer a little, then straight on"  (pass a person in a narrow corridor)
            for v2 in (0.045, 0.09):
                wt = math.copysign(VEER_RATE, ang)
                cands.append((0.05, wt, abs(ang) / VEER_RATE, v2, 0.0))
        best, best_cost = None, 1e9
        for plan in cands:
            tc, dm, sh, gend, stm = self._rollout(plan, pose, goal, segs, static_pts, static_segs)   # vs full uncertainty cone
            if tc is None:                                           # safe against the whole cone -> safe against its centre
                tcc, dmc = None, dm
            else:
                tcc, dmc, _, _, _ = self._rollout(plan, pose, goal, segs_c, static_pts)              # vs best estimate only
            v, w = (plan[0], plan[1]) if plan[2] > 0.0 else (plan[3], plan[4])                       # the command executed now
            cost = 3.0 * abs(v - v_nav) + 0.3 * abs(w - w_nav)
            cost += 1.5 * abs(v - self._prev[0]) + 0.2 * abs(w - self._prev[1])
            cost += 1.0 * gend
            if tc is not None:
                cost += 100.0 + 30.0 * (HORIZON - tc)
            if tcc is not None:                                    # hits even the best-estimate path: much worse
                cost += 500.0 + 120.0 * (HORIZON - tcc)
            cost += 30.0 * max(0.0, 0.25 - dmc) + 10.0 * max(0.0, 0.25 - dm)     # keep well clear of the car's predicted path
            if sh:
                cost += 1500.0          # a parked car / wall is a CERTAIN obstacle; a prediction about a mover is not
            cost += 12.0 * max(0.0, 0.11 - stm)            # do not hug parked cars / walls while dodging
            if v < 0.0:
                cost += 2.5                                     # reversing is the last resort
            if cost < best_cost:
                best_cost, best = cost, (v, w)
        v, w = best

        if v > v_nav + 0.015 and v > 0.02:
            mode = "BOOST"
        elif v < -0.01:
            mode = "BACKOFF"
        elif v < v_nav - 0.015:
            mode = "YIELD"
        elif abs(w - w_nav) > 0.5:
            mode = "DODGE"
        else:
            mode = "YIELD"
        info["mode"] = mode
        self._prev = (v, w)
        self.last_info = info
        return v, w, info

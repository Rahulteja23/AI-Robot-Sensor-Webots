"""
route_planner.py
================
GLOBAL route planning inside the hospital.  A hospital delivery robot knows the floor plan (walls, doors, lifts)
but NOT where people and trolleys are: those are handled by the local Navigator + DynamicSafety layers.

  plan_path(floor, a, b)      A* on a 2.5 cm grid of the floor's walls + furniture (inflated by the robot radius plus a
                              margin).  Cells close to obstacles cost more, so routes run down the middle of corridors
                              and through the middle of doors.  Returns a dense list of points.
  plan_mission(mission)       -> list of LEGS:  walk on a floor [-> ride a lift] [-> walk on the next floor]
  RouteFollower               gives the Navigator a target point a short distance ahead on the route (pure pursuit with a
                              line-of-sight check), tracks progress and asks for a re-plan when the robot is pushed off.

Pure Python, no Webots needed.
"""
import heapq
import math

import hospital as hp
import scene_config as sc

CELL        = 0.025
NX          = int(round((hp.X1 - hp.X0) / CELL))
NY          = int(round((hp.Y1 - hp.Y0) / CELL))
PLAN_CLEAR  = sc.ROBOT_RADIUS + 0.030       # A*: cells closer than this to a wall / furniture are blocked
LOS_CLEAR   = 0.078                         # a straight shortcut must keep this distance to every obstacle
PREF_CLEAR  = 0.16                          # cells closer than this to an obstacle cost more
LOOKAHEAD   = 0.42                          # m along the route
SPACING     = 0.03                          # m between the points of the dense path
INF         = 1e9


class FloorGrid:
    """Obstacle mask, distance-to-obstacle field and blocked mask for one floor."""

    def __init__(self, f):
        self.f = f
        reach = PREF_CLEAR + 0.05                    # distances are only needed up to this value
        self.dist = [[reach] * NY for _ in range(NX)]
        for (cx, cy, yaw, L, W) in hp.FLOORS[f].footprints():       # exact distance to every wall / piece of furniture
            r = 0.5 * math.hypot(L, W) + reach
            i0 = max(0, int((cx - r - hp.X0) / CELL))
            i1 = min(NX - 1, int((cx + r - hp.X0) / CELL) + 1)
            j0 = max(0, int((cy - r - hp.Y0) / CELL))
            j1 = min(NY - 1, int((cy + r - hp.Y0) / CELL) + 1)
            for i in range(i0, i1 + 1):
                x = hp.X0 + (i + 0.5) * CELL
                row = self.dist[i]
                for j in range(j0, j1 + 1):
                    d = sc.rect_distance(x, hp.Y0 + (j + 0.5) * CELL, cx, cy, yaw, L, W)
                    if d < row[j]:
                        row[j] = d
        self.blocked = [[self.dist[i][j] < PLAN_CLEAR for j in range(NY)] for i in range(NX)]

    # ---- helpers -----------------------------------------------------------------
    @staticmethod
    def cell(p):
        return (min(NX - 1, max(0, int((p[0] - hp.X0) / CELL))), min(NY - 1, max(0, int((p[1] - hp.Y0) / CELL))))

    @staticmethod
    def centre(c):
        return hp.X0 + (c[0] + 0.5) * CELL, hp.Y0 + (c[1] + 0.5) * CELL

    def clearance(self, x, y):
        i, j = self.cell((x, y))
        return self.dist[i][j]

    def los(self, a, b, min_clear=LOS_CLEAR):
        """True if the straight segment a -> b keeps `min_clear` metres to every obstacle."""
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (0.5 * CELL)))
        for k in range(n + 1):
            t = k / n
            if self.clearance(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])) < min_clear:
                return False
        return True

    def nearest_free(self, p, max_r=0.25):
        c = self.cell(p)
        if not self.blocked[c[0]][c[1]]:
            return c
        best, bc = INF, None
        r = int(max_r / CELL)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                i, j = c[0] + di, c[1] + dj
                if 0 <= i < NX and 0 <= j < NY and not self.blocked[i][j]:
                    dd = di * di + dj * dj
                    if dd < best:
                        best, bc = dd, (i, j)
        return bc


_GRIDS = {}


def grid(f):
    if f not in _GRIDS:
        _GRIDS[f] = FloorGrid(f)
    return _GRIDS[f]


# ═════════════════════════════════════════════════════════════════════════════
def plan_path(f, a, b):
    """Route on floor f from point a to point b.  Returns dict(path=[(x, y) ...], length=m) or None."""
    g = grid(f)
    s, t = g.nearest_free(a), g.nearest_free(b)
    if s is None or t is None:
        return None
    openq = [(0.0, 0.0, s)]
    best = {s: 0.0}
    parent = {}
    while openq:
        _, d, c = heapq.heappop(openq)
        if c == t:
            break
        if d > best.get(c, INF):
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                n = (c[0] + di, c[1] + dj)
                if not (0 <= n[0] < NX and 0 <= n[1] < NY) or g.blocked[n[0]][n[1]]:
                    continue
                if di and dj and (g.blocked[c[0] + di][c[1]] or g.blocked[c[0]][c[1] + dj]):
                    continue
                step = 1.4142 if di and dj else 1.0
                clr = g.dist[n[0]][n[1]]
                pen = 1.0 + 3.0 * max(0.0, (PREF_CLEAR - clr) / PREF_CLEAR)          # keep to the middle of corridors / doors
                nd = d + step * pen
                if nd < best.get(n, INF):
                    best[n] = nd
                    parent[n] = c
                    heapq.heappush(openq, (nd + math.hypot(t[0] - n[0], t[1] - n[1]), nd, n))
    else:
        return None
    if t not in best:
        return None
    cells = [t]
    while cells[-1] != s:
        cells.append(parent[cells[-1]])
    cells.reverse()
    pts = [g.centre(c) for c in cells]
    pts[0], pts[-1] = tuple(a), tuple(b)
    pts = _string_pull(g, pts)
    dense = _resample(pts)
    length = sum(math.hypot(dense[k + 1][0] - dense[k][0], dense[k + 1][1] - dense[k][1]) for k in range(len(dense) - 1))
    return dict(path=dense, length=length)


def _string_pull(g, pts):
    """Remove waypoints that are not needed: keep a straight shortcut whenever it keeps LOS_CLEAR to all obstacles."""
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not g.los(pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def _resample(pts):
    dense = [pts[0]]
    for k in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[k], pts[k + 1]
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / SPACING))
        for m in range(1, n + 1):
            dense.append((x0 + (x1 - x0) * m / n, y0 + (y1 - y0) * m / n))
    return dense


# ═════════════════════════════════════════════════════════════════════════════
def plan_mission(m):
    """
    Legs of a mission.  m: dict with floor_s, start, floor_g, goal  (floor-local points).
    Leg: dict(floor, start, end, path, length, stop, lift, lift_to)
      lift    : the Lift object to ride at the END of this leg (or None);   lift_to : floor index after the ride
    Returns the list of legs, or None if the mission is impossible.
    """
    fs, fg = m["floor_s"], m["floor_g"]
    if fs == fg:
        r = plan_path(fs, m["start"], m["goal"])
        if r is None:
            return None
        return [dict(floor=fs, start=m["start"], end=m["goal"], path=r["path"], length=r["length"], stop=sc.GOAL_STOP,
                     lift=None, lift_to=None)]
    best = None
    for k, lf in enumerate(hp.FLOORS[fs].lifts):
        a = plan_path(fs, m["start"], lf.pad)
        b = plan_path(fg, lf.pad, m["goal"])
        if a is None or b is None:
            continue
        tot = a["length"] + b["length"]
        if best is None or tot < best[0]:
            best = (tot, lf, a, b)
    if best is None:
        return None
    _, lf, a, b = best
    return [dict(floor=fs, start=m["start"], end=lf.pad, path=a["path"], length=a["length"], stop=0.05, lift=lf, lift_to=fg),
            dict(floor=fg, start=lf.pad, end=m["goal"], path=b["path"], length=b["length"], stop=sc.GOAL_STOP, lift=None, lift_to=None)]


# ═════════════════════════════════════════════════════════════════════════════
class RouteFollower:
    """Gives the local Navigator a point a short distance ahead on the planned route."""

    def __init__(self, floor, path, end, stop):
        self.floor, self.path, self.end, self.stop = floor, path, tuple(end), stop
        self.g = grid(floor)
        self.idx = 0
        self.replans = 0

    def progress(self, x, y):
        lo, hi = self.idx, min(len(self.path), self.idx + 80)
        best, bi = INF, self.idx
        for k in range(lo, hi):
            d = (self.path[k][0] - x) ** 2 + (self.path[k][1] - y) ** 2
            if d < best:
                best, bi = d, k
        self.idx = bi
        return math.sqrt(best)

    def target(self, x, y):
        off = self.progress(x, y)
        n_look = int(LOOKAHEAD / SPACING)
        last = len(self.path) - 1
        for j in range(min(last, self.idx + n_look), self.idx, -1):
            if self.g.los((x, y), self.path[j], 0.06):
                return self.path[j], off
        return self.path[min(last, self.idx + 1)], off

    def remaining(self, x, y):
        return math.hypot(self.end[0] - x, self.end[1] - y)

    def arrived(self, x, y):
        return self.remaining(x, y) < self.stop

    def replan(self, x, y):
        r = plan_path(self.floor, (x, y), self.end)
        if r is not None:
            self.path, self.idx = r["path"], 0
            self.replans += 1
        return r is not None

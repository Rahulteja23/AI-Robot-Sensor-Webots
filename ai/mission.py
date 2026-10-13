"""
mission.py
==========
Random hospital delivery missions:  "take something from <START room, floor a> to <GOAL room, floor b>".

    from mission import random_mission, get_mission, mission_from_id
    m = random_mission(seed=None)       # seed None -> new random mission, int -> reproducible
    m["start_name"], m["goal_name"], m["start"], m["goal"], m["floor_s"], m["floor_g"], m["legs"]

How a mission is drawn
  * START room and GOAL room: random rooms, 60 % of the missions on DIFFERENT floors (the robot rides a lift)
  * START point and GOAL point: random points inside those rooms, at least 0.10 m from every wall / piece of furniture
  * the start heading is random (the robot calibrates its heading at start-up)
  * a route must exist (route_planner.plan_mission) and, on one floor, START and GOAL must be >= 1.2 m apart
People and trolleys move in the corridors only, so no mission starts or ends in their lane.
Pure Python, no Webots needed.
"""
import math
import random

import hospital as hp
import route_planner as rp
import scene_config as sc

CLEAR_POINT   = 0.10        # START / GOAL point to the nearest wall or furniture
CROSS_FLOOR_P = 0.60        # probability that START and GOAL are on different floors
MIN_SAME_FLOOR_SEP = 1.2


def _sample_point(rng, room):
    """A random point inside the room that is clear of furniture and walls (None if none was found)."""
    x0, x1, ya, yb = room.rect
    for _ in range(200):
        x = rng.uniform(x0 + 0.10, x1 - 0.10)
        y = rng.uniform(ya + 0.10, yb - 0.10)
        if hp.static_clearance(room.floor, x, y) >= CLEAR_POINT:
            return round(x, 3), round(y, 3)
    return None


def _describe(room):
    return f"{room.name} ({hp.FLOOR_LABELS[room.floor]})"


def _finish(m):
    legs = rp.plan_mission(m)
    if legs is None:
        return None
    m["legs"] = legs
    m["route_length"] = round(sum(l["length"] for l in legs), 2)
    return m


def build_mission(room_s, room_g, p_start, p_goal, yaw, seed, is_random):
    m = dict(start=p_start, floor_s=room_s.floor, start_room=room_s.name, start_name=_describe(room_s), yaw=round(yaw, 3),
             goal=p_goal, floor_g=room_g.floor, goal_room=room_g.name, goal_name=_describe(room_g),
             seed=seed, random=is_random)
    m["straight"] = round(math.hypot(p_goal[0] - p_start[0], p_goal[1] - p_start[1]), 2) if room_s.floor == room_g.floor else None
    return _finish(m)


def fixed_mission():
    """The demo mission: Pharmacy (ground floor) -> Operating Theatre 1 (floor 3)."""
    rs, rg = hp.find_room(0, "Pharmacy"), hp.find_room(2, "Operating Theatre 1")
    return build_mission(rs, rg, rs.center, rg.center, -math.pi / 2 if rs.side > 0 else math.pi / 2, "fixed", False)


def random_mission(seed=None):
    """Draw a valid random mission.  seed=None -> fresh random seed (returned in the result)."""
    if seed is None:
        seed = random.SystemRandom().randrange(1, 10 ** 6)
    rng = random.Random(seed)
    rooms = hp.all_rooms()
    for _ in range(300):
        rs = rng.choice(rooms)
        if rng.random() < CROSS_FLOOR_P:
            rg = rng.choice([r for r in rooms if r.floor != rs.floor])
        else:
            rg = rng.choice([r for r in rooms if r.floor == rs.floor and r is not rs])
        a, b = _sample_point(rng, rs), _sample_point(rng, rg)
        if a is None or b is None:
            continue
        if rs.floor == rg.floor and math.hypot(a[0] - b[0], a[1] - b[1]) < MIN_SAME_FLOOR_SEP:
            continue
        m = build_mission(rs, rg, a, b, rng.uniform(-math.pi, math.pi), seed, True)
        if m is not None:
            return m
    return fixed_mission()                                # cannot happen with the shipped building


def get_mission(seed=None):
    """What the controller / tests use: random or the demo mission according to scene_config.RANDOM_MISSION."""
    if not sc.RANDOM_MISSION:
        return fixed_mission()
    return random_mission(sc.MISSION_SEED if seed is None else seed)


def mission_from_id(mid):
    """Re-create a mission from the id the robot controller publishes ('fixed' or the integer seed)."""
    return fixed_mission() if str(mid) == "fixed" else random_mission(int(mid))

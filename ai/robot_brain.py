"""
robot_brain.py
==============
The complete decision pipeline of the robot, independent of Webots.

    sensors -> Perception -> SensorFusion -> Navigator -> DynamicSafety -> ML advisors -> wheels
                                                      (moving cars)   (RF + SVM)     \\-> DataLogger (CSV)

The Webots controller (controllers/ai_robot_controller) only does I/O:
it reads the devices, calls  RobotBrain.step(),  and applies the wheel speeds.
The very same class is used by experiments/headless_validation.py, so the
offline test exercises exactly the code that runs inside the simulator.

Start-up sequence
-----------------
1. CALIBRATE_TURN  : turn left in place for ~0.8 s and check the IMU yaw
                     increases (verifies the heading sign convention).
2. CALIBRATE_DRIVE : drive straight ~1.2 s and compare GPS course with IMU yaw
                     (verifies / corrects a constant heading offset).
3. NAVIGATE        : goal seeking with obstacle avoidance.
4. DONE            : goal reached -> motors stop, CSV closed.
"""

import math

from scene_config import GOAL_STOP, wrap
from perception import Perception
from sensor_fusion import SensorFusion
from navigator import Navigator
from ml_path_planner import MLPathPlanner
from svm_planner import SVMPlanner
from motion_predictor import DynamicSafety, limit_vw
from data_logger import DataLogger

WHEEL_RADIUS = 0.0205      # m
AXLE_LENGTH  = 0.052       # m
MAX_WHEEL    = 6.28        # rad/s

CAL_TURN_STEPS  = 12
CAL_DRIVE_STEPS = 18


def vw_to_wheels(v: float, w: float):
    """(forward m/s, turn rad/s) -> (left, right) wheel angular speed in rad/s."""
    left = (v - w * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
    right = (v + w * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
    m = max(abs(left), abs(right))
    if m > MAX_WHEEL:
        left, right = left * MAX_WHEEL / m, right * MAX_WHEEL / m
    return left, right


def _zones(ps):
    """Simple directional zone flags from the IR sensors (same idea as SensorManager)."""
    thr = 500.0
    return {
        "front": ps[7] > thr or ps[0] > thr,
        "right": ps[0] > thr or ps[1] > thr or ps[2] > thr,
        "left": ps[5] > thr or ps[6] > thr or ps[7] > thr,
        "rear": ps[3] > thr or ps[4] > thr,
        "critical_front": ps[7] > 1500.0 or ps[0] > 1500.0,
        "max_value": max(ps),
        "max_sensor_idx": ps.index(max(ps)),
    }


class RobotBrain:

    def __init__(self, timestep_s: float, results_dir: str = None,
                 goal=None, verbose: bool = True, use_ml: bool = True,
                 use_motion_prediction: bool = True,
                 use_dynamic_safety: bool = True, use_svm: bool = True):
        self.dt = timestep_s
        self.goal = goal
        self.route = None                 # RouteFollower of the current leg (None: steer straight to `goal`)
        self.final_leg = True
        self.leg_done = False
        self.floor = 0
        self.leg_index = 0
        self.target = goal
        self._use_mp = use_motion_prediction
        self._last_replan = -99.0
        self.verbose = verbose
        self.use_ml = use_ml
        self.use_dynamic_safety = use_dynamic_safety
        self.use_svm = use_svm

        self.perception = Perception()
        self.fusion = SensorFusion()
        self.navigator = Navigator(use_motion_prediction=use_motion_prediction)
        self.ml = MLPathPlanner(train_after_steps=120, retrain_interval=150,
                                confidence_threshold=0.70)           # Random Forest (original model)
        self.svm = SVMPlanner(train_after_steps=120, retrain_interval=150,
                              confidence_threshold=0.62)             # SVM (additional model)
        self.safety = DynamicSafety()
        self._last_v = 0.0
        self.ensemble_stats = dict(steps=0, rf_active=0, rf_agree=0, svm_active=0, svm_agree=0,
                                   both_active=0, rf_svm_agree=0)
        self.logger = DataLogger(results_dir) if results_dir else None

        self.phase = "CAL_TURN"
        self.step_count = 0
        self.yaw_sign = 1.0
        self.yaw_bias = 0.0
        self._cal = {}
        self.done = False
        self.path_length = 0.0
        self._last_xy = None
        self.last = {}

    # ------------------------------------------------------------ mission legs
    def start_leg(self, floor, route, final, leg_index=0):
        """Begin a new leg of the mission on `floor` (after a lift ride the robot is on a new floor).
        route: RouteFollower.  Perception, calibration and the ML models are kept, the floor-specific state is reset."""
        self.floor, self.route, self.final_leg, self.leg_index = floor, route, final, leg_index
        self.goal = route.end
        self.done = False
        self.leg_done = False
        self.navigator = Navigator(use_motion_prediction=self._use_mp)
        self.safety = DynamicSafety()
        self._last_v = 0.0
        self._last_xy = None
        self._last_replan = -99.0

    # ------------------------------------------------------------------ utils
    def heading(self, yaw_raw: float) -> float:
        return wrap(self.yaw_sign * yaw_raw + self.yaw_bias)

    def _say(self, msg):
        if self.verbose:
            print(msg)

    # ----------------------------------------------------------- calibration
    def _calibrate(self, pos, yaw_raw, min_range):
        if self.phase == "CAL_TURN":
            c = self._cal
            c.setdefault("n", 0)
            c.setdefault("yaw0", yaw_raw)
            c["n"] += 1
            if c["n"] >= CAL_TURN_STEPS:
                dyaw = wrap(yaw_raw - c["yaw0"])
                if dyaw < -0.15:
                    self.yaw_sign = -1.0
                    self._say("[CAL] IMU yaw decreases when turning left -> sign flipped.")
                elif dyaw > 0.15:
                    self._say(f"[CAL] IMU yaw sign OK (turned {math.degrees(dyaw):+.0f} deg).")
                else:
                    self._say("[CAL] WARNING: IMU yaw did not change during turn test; keeping defaults.")
                self.phase = "CAL_DRIVE"
                self._cal = {}
            return 0.0, 1.0          # v, w  (turn left in place)

        # CAL_DRIVE
        c = self._cal
        c.setdefault("n", 0)
        c["n"] += 1
        if c["n"] == 3:
            c["p0"] = (pos[0], pos[1])
        if c["n"] >= CAL_DRIVE_STEPS or min_range < 0.10:
            p0 = c.get("p0")
            if p0 is not None:
                dx, dy = pos[0] - p0[0], pos[1] - p0[1]
                if math.hypot(dx, dy) > 0.02:
                    course = math.atan2(dy, dx)
                    err = wrap(course - self.heading(yaw_raw))
                    if abs(err) > 0.25:
                        self.yaw_bias = err
                        self._say(f"[CAL] Heading offset of {math.degrees(err):+.1f} deg corrected.")
                    else:
                        self._say(f"[CAL] GPS course matches IMU heading (err {math.degrees(err):+.1f} deg).")
            self.phase = "NAVIGATE"
            self._say("[CAL] Calibration done -> NAVIGATE")
            return 0.0, 0.0
        return 0.05, 0.0

    # ------------------------------------------------------------------- step
    def step(self, t: float, data: dict) -> dict:
        """
        data keys: distance (8 raw IR), range (24 m), camera (dict), position (x,y,z), heading (yaw)
        returns : dict(left, right, v, w, action, source, mode, done, ...)
        """
        self.step_count += 1
        ps = list(data["distance"])
        rs = list(data["range"])
        x, y, z = data["position"]
        yaw_raw = data["heading"]

        if self._last_xy is not None:
            self.path_length += math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._last_xy = (x, y)

        # perception + fusion (AI sensing layer)
        dist_perc = self.perception.process_distance(ps)
        cam_perc = self.perception.process_camera(data.get("camera", {}))
        fused = self.fusion.fuse(dist_perc, cam_perc, _zones(ps))

        gx, gy = self.goal
        dist_goal = math.hypot(x - gx, y - gy)
        min_range = min(rs) if rs else 0.5

        # ---- leg / goal reached ----
        stop = self.route.stop if self.route is not None else GOAL_STOP
        if self.phase == "NAVIGATE" and dist_goal < stop:
            if self.final_leg:
                if not self.done:
                    self.done = True
                    self._log(t, data, dist_goal, dist_perc, cam_perc, fused, "STOP", "GOAL", 0.0, 0.0,
                              0.0, 0.0, None, None, 0.0, "ARRIVED", 0.0, min_range)
                    if self.logger:
                        self.logger.close()
                return dict(left=0.0, right=0.0, v=0.0, w=0.0, action="STOP", source="GOAL",
                            mode="ARRIVED", done=True, leg_done=False, dist=dist_goal)
            self.leg_done = True                              # e.g. standing on the lift pad: the lift takes over
            return dict(left=0.0, right=0.0, v=0.0, w=0.0, action="STOP", source="GOAL",
                        mode="LEG_DONE", done=False, leg_done=True, dist=dist_goal)
        if self.leg_done:
            return dict(left=0.0, right=0.0, v=0.0, w=0.0, action="STOP", source="GOAL",
                        mode="LEG_DONE", done=False, leg_done=True, dist=dist_goal)

        # ---- calibration phases ----
        if self.phase != "NAVIGATE":
            v, w = self._calibrate((x, y), yaw_raw, min_range)
            left, right = vw_to_wheels(v, w)
            self._log(t, data, dist_goal, dist_perc, cam_perc, fused, "CALIBRATE", "HEURISTIC",
                      left, right, v, w, None, None, 0.0, self.phase, 0.0, min_range)
            return dict(left=left, right=right, v=v, w=w, action="CALIBRATE", source="HEURISTIC",
                        mode=self.phase, done=False, leg_done=False, dist=dist_goal)

        # ---- navigation ----
        yaw = self.heading(yaw_raw)
        target, progress = self.goal, None
        if self.route is not None:                           # follow the planned route: steer to a point ahead on it
            target, off = self.route.target(x, y)
            if off > 0.30 and t - self._last_replan > 2.0:    # pushed off the route (avoiding a person ...) -> re-plan
                self._last_replan = t
                if self.route.replan(x, y):
                    target, off = self.route.target(x, y)
            progress = (len(self.route.path) - 1 - self.route.idx) * 0.03
            self.navigator.arrive_slow = self.route.remaining(x, y) < 0.40
        self.target = target
        nav = self.navigator.plan(t, self.dt, (x, y, yaw), target, rs, ps, progress)
        h_action = nav["action"]

        # Risk level from the 360 deg range ring (body-to-obstacle gap, in metres).
        # The 8 IR sensors alone only trigger on near-contact, which a robot that
        # avoids obstacles never reaches; the range ring gives early warning.
        gap = nav["min_range"]
        rank = {"low": 0, "medium": 1, "critical": 2}
        range_risk = "critical" if gap < 0.04 else "medium" if gap < 0.12 else "low"
        if rank[range_risk] > rank.get(fused["fused_risk"], 0):
            fused["fused_risk"] = range_risk
            fused["confidence"] = max(fused.get("confidence", 0.5), 0.85)
        if rank[range_risk] > rank.get(dist_perc.get("risk_level", "low"), 0):
            dist_perc["risk_level"] = range_risk

        # ---- moving obstacles: predict, then adapt speed / direction (accelerate, brake, reverse, dodge) ----
        v_nav, w_nav = nav["v"], nav["w"]
        dyn = dict(mode="CLEAR", ttc=None, level=0, n_moving=0, min_dist=None)
        v_cmd, w_cmd = v_nav, w_nav
        if self.use_dynamic_safety:
            v_cmd, w_cmd, dyn = self.safety.filter(t, self.dt, (x, y, yaw), target, rs,
                                                   v_nav, w_nav,
                                                   (self._last_v * math.cos(yaw), self._last_v * math.sin(yaw)))

        # ---- two ML models (Random Forest + SVM) = second opinion ----
        feats = dict(bearing=nav["bearing"], c0=nav["c0"], c_left=nav["c_left"],
                     c_right=nav["c_right"], dist=nav["dist"],
                     ttc=dyn["ttc"], threat_level=dyn["level"])
        ml_action, ml_conf, source = None, 0.0, "HEURISTIC"
        svm_action, svm_conf, ens_action = None, 0.0, h_action
        speed_scale = 1.0
        if self.use_ml:
            args = dict(ps_readings=ps, fused_state=fused, cam_perception=cam_perc,
                        heuristic_action=h_action, extra=feats)
            rf = self.ml.decide(**args)
            ml_action, ml_conf = rf["raw_action"], rf["ml_conf"]
            rf_ok = ml_action is not None and ml_conf >= self.ml.confidence_threshold
            sv_ok = False
            if self.use_svm:
                sv = self.svm.decide(**args)
                svm_action, svm_conf = sv["raw_action"], sv["ml_conf"]
                sv_ok = svm_action is not None and svm_conf >= self.svm.confidence_threshold

            # weighted vote: Navigator (1.0) + Random Forest (its confidence) + SVM (its confidence)
            votes = {h_action: 1.0}
            if rf_ok:
                votes[ml_action] = votes.get(ml_action, 0.0) + ml_conf
            if sv_ok:
                votes[svm_action] = votes.get(svm_action, 0.0) + svm_conf
            ens_action = max(votes, key=votes.get)

            es = self.ensemble_stats
            es["steps"] += 1
            if rf_ok:
                es["rf_active"] += 1
                es["rf_agree"] += ml_action == h_action
            if sv_ok:
                es["svm_active"] += 1
                es["svm_agree"] += svm_action == h_action
            if rf_ok and sv_ok:
                es["both_active"] += 1
                es["rf_svm_agree"] += ml_action == svm_action

            if (rf_ok and ml_action == h_action) or (sv_ok and svm_action == h_action):
                source = "ML"
            # ML can only make the robot MORE careful, and only when BOTH models agree: they both predict STOP
            # while the Navigator drives on.  (A single vote is not enough - lingering in a car's lane because one
            # model hesitated is more dangerous than driving on.)
            if h_action != "STOP" and nav["c0"] < 0.35 and rf_ok and sv_ok and ml_action == "STOP" and svm_action == "STOP":
                source = "ML"
                speed_scale = 0.6

        if dyn["mode"] in ("BOOST", "BACKOFF", "DODGE"):
            speed_scale = 1.0               # an evasive manoeuvre is never slowed down by the ML advisors
        v = v_cmd * speed_scale if v_cmd > 0 else v_cmd
        w = w_cmd
        self._last_v = v
        left, right = vw_to_wheels(v, w)

        # label of what the robot really does now
        mode = nav["mode"] if dyn["mode"] == "CLEAR" else dyn["mode"]
        if v < -0.01 or (abs(v) < 0.01 and abs(w) < 0.3):
            h_action_log = "STOP"
        elif w > 0.3:
            h_action_log = "LEFT"
        elif w < -0.3:
            h_action_log = "RIGHT"
        else:
            h_action_log = "FORWARD"
        h_action = h_action_log if dyn["mode"] != "CLEAR" else h_action
        nav = dict(nav, mode=mode, ens=ens_action, dyn=dyn, v_nav=v_nav,
                   svm_action=svm_action, svm_conf=svm_conf)

        self._log(t, data, dist_goal, dist_perc, cam_perc, fused, h_action, source, left, right,
                  v, w, nav, ml_action, ml_conf, nav["mode"], speed_scale, min_range)
        self.last = dict(left=left, right=right, v=v, w=w, action=h_action, source=source,
                         mode=nav["mode"], done=False, dist=dist_goal, nav=nav, fused=fused,
                         dyn=dyn, svm_action=svm_action, svm_conf=svm_conf, ensemble_action=ens_action,
                         dist_perc=dist_perc, cam_perc=cam_perc, ml_action=ml_action, ml_conf=ml_conf)
        return self.last

    # ---------------------------------------------------------------- logging
    def _log(self, t, data, dist_goal, dist_perc, cam_perc, fused, action, source,
             left, right, v, w, nav, ml_action, ml_conf, mode, speed_scale, min_range):
        if not self.logger:
            return
        ps = data["distance"]
        x, y, z = data["position"]
        row = {
            "step": self.step_count, "sim_time_s": t,
            "pos_x": x, "pos_y": y, "pos_z": z,
            "ps0": ps[0], "ps1": ps[1], "ps2": ps[2], "ps3": ps[3],
            "ps4": ps[4], "ps5": ps[5], "ps6": ps[6], "ps7": ps[7],
            "front_val": max(ps[0], ps[7]), "left_val": max(ps[5], ps[6]), "right_val": max(ps[1], ps[2]),
            "mean_proximity": dist_perc.get("mean_proximity", ""),
            "max_proximity": dist_perc.get("max_proximity", ""),
            "std_deviation": dist_perc.get("std_deviation", ""),
            "risk_level": dist_perc.get("risk_level", ""),
            "cam_obstacle_detected": cam_perc.get("obstacle_detected", False),
            "cam_dominant_color": cam_perc.get("dominant_color", ""),
            "cam_edge_density": cam_perc.get("edge_density", 0.0),
            "fused_risk": fused.get("fused_risk", ""),
            "confidence": fused.get("confidence", ""),
            "camera_confirmed": fused.get("camera_confirmed", False),
            "action": f"{action}({source})",
            "left_speed": left, "right_speed": right,
            "distance_to_goal": dist_goal,
            "goal_bearing_deg": math.degrees(nav["bearing"]) if nav else "",
            "min_clearance_m": min_range,
            "nav_mode": mode,
            "v_cmd": v, "w_cmd": w,
            "ml_action": ml_action if ml_action else "",
            "ml_conf": ml_conf,
            "front_free_m": nav["c0"] if nav else "",
            "svm_action": nav.get("svm_action") or "" if nav else "",
            "svm_conf": nav.get("svm_conf", 0.0) if nav else "",
            "ensemble_action": nav.get("ens", "") if nav else "",
            "dyn_mode": nav["dyn"]["mode"] if nav else "",
            "dyn_ttc_s": nav["dyn"]["ttc"] if nav and nav["dyn"]["ttc"] is not None else "",
            "dyn_movers": nav["dyn"]["n_moving"] if nav else "",
            "v_nav": nav["v_nav"] if nav else "",
            "speed_scale": speed_scale,
            "goal_x": self.goal[0], "goal_y": self.goal[1],
            "floor": self.floor + 1, "leg": self.leg_index + 1,
        }
        self.logger.log(row)

    def model_report(self) -> str:
        """One-paragraph summary of the two ML models (printed at the end of a run)."""
        es = self.ensemble_stats

        def pct(a, b):
            return f"{100.0 * a / b:.0f}%" if b else "n/a"
        return (f"Random Forest: confident on {es['rf_active']}/{es['steps']} steps, agrees with the Navigator on "
                f"{pct(es['rf_agree'], es['rf_active'])} | SVM: confident on {es['svm_active']}/{es['steps']} steps, "
                f"agrees on {pct(es['svm_agree'], es['svm_active'])} | RF vs SVM agree on "
                f"{pct(es['rf_svm_agree'], es['both_active'])} of {es['both_active']} steps where both are confident")

    def close(self):
        if self.logger:
            self.logger.close()

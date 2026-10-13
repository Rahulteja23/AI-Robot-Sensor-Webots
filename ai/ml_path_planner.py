"""
ml_path_planner.py
==================
PHASE 4/5 - ML-based Path Planner
------------------------------------
Trains a Random Forest classifier online during the simulation to predict
the best robot action (FORWARD, LEFT, RIGHT, STOP) from sensor features.

Two operating modes
--------------------
1. HEURISTIC PHASE (first N steps):
   Uses the rule-based obstacle avoidance logic to collect labeled data.
   Each step is stored as a (features, label) training sample.

2. ML PHASE (after N steps):
   Fits a RandomForestClassifier on the collected data.
   Predictions override the heuristic where confidence is high enough.
   Falls back to heuristic if ML confidence is below CONFIDENCE_THRESHOLD.

Feature vector (19 features)
------------------------------
  ps0..ps7        - raw IR proximity values (normalized 0-1)
  fused_risk_enc  - encoded fused risk (0=low, 1=medium, 2=critical)
  confidence      - fusion confidence score
  cam_obstacle    - 0 or 1
  edge_density    - 0.0-1.0
  goal_bearing    - angle to the goal in the robot frame / pi      (-1..1)
  front_free      - free travel straight ahead / 0.45 m             (0..1)
  left_free       - free travel on the left  / 0.45 m               (0..1)
  right_free      - free travel on the right / 0.45 m               (0..1)
  goal_dist       - distance to goal / 3 m                          (0..1)
  ttc_norm        - predicted time-to-collision with a MOVING car / 3 s (1 = none)
  threat_level    - moving-obstacle threat 0 / 0.5 / 1

ROLE OF THE ML MODEL (see robot_brain.py)
-----------------------------------------
The Random Forest learns online to imitate the sensor-based Navigator and acts
as a *second opinion*.  Its output can only make the robot MORE cautious
(e.g. predicted STOP -> reduce speed); it can never override a safety decision.
"""

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("[MLPathPlanner] WARNING: scikit-learn not found. "
          "ML planner will stay in heuristic mode. "
          "Install with: pip install scikit-learn")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ACTIONS = ["FORWARD", "LEFT", "RIGHT", "STOP"]
IR_MAX  = 4096.0   # E-puck IR sensor max raw value (for normalisation)

RISK_MAP = {"low": 0, "medium": 1, "critical": 2}


class MLPathPlanner:
    """
    Online-learning path planner that bootstraps from heuristic labels,
    then gradually takes over navigation with a trained Random Forest.
    """

    def __init__(self,
                 train_after_steps: int = 80,
                 retrain_interval: int  = 50,
                 confidence_threshold: float = 0.65,
                 n_estimators: int = 60):
        """
        Parameters
        ----------
        train_after_steps    : int   – number of heuristic steps before first training
        retrain_interval     : int   – retrain the model every N steps after initial fit
        confidence_threshold : float – min ML confidence to override heuristic decision
        n_estimators         : int   – number of trees in the Random Forest
        """
        self.train_after_steps    = train_after_steps
        self.retrain_interval     = retrain_interval
        self.confidence_threshold = confidence_threshold
        self.n_estimators         = n_estimators

        self._X: list = []          # feature vectors
        self._y: list = []          # action labels
        self._model   = None        # fitted RandomForestClassifier
        self._step    = 0           # call counter
        self._ml_active = False     # True once model is first trained
        self._predictions_made = 0  # stats counter
        self._ml_overrides     = 0  # times ML disagreed with heuristic

        if _SKLEARN_AVAILABLE:
            print(f"[MLPathPlanner] Initialised. Will train after {train_after_steps} "
                  f"heuristic steps, retrain every {retrain_interval} steps.")
        else:
            print("[MLPathPlanner] Running in heuristic-only mode (scikit-learn unavailable).")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def decide(self,
               ps_readings: list,
               fused_state: dict,
               cam_perception: dict,
               heuristic_action: str,
               extra: dict = None) -> dict:
        """
        Given current sensor state and the heuristic recommendation,
        return the final action (possibly overridden by ML).

        Parameters
        ----------
        ps_readings      : list[float]  – raw IR values [ps0..ps7]
        fused_state      : dict         – output of SensorFusion.fuse()
        cam_perception   : dict         – output of Perception.process_camera()
        heuristic_action : str          – one of ACTIONS from rule-based logic

        Returns
        -------
        dict:
            "action"      : str   – chosen action
            "source"      : str   – "ML" or "HEURISTIC"
            "ml_conf"     : float – ML prediction confidence (0 if heuristic)
        """
        self._step += 1

        # 1. Build feature vector from current state
        features = self._build_features(ps_readings, fused_state, cam_perception, extra)

        # 2. Store heuristic label for training
        label = heuristic_action if heuristic_action in ACTIONS else "FORWARD"
        self._X.append(features)
        self._y.append(label)
        if len(self._X) > self.BUFFER_MAX:   # keep only recent samples -> fast re-training
            del self._X[:-self.BUFFER_MAX]
            del self._y[:-self.BUFFER_MAX]

        # 3. Trigger (re)training if enough data
        result = self._maybe_retrain()
        if result:
            print(result)

        # 4. Ask the ML model for a prediction (if available)
        ml_action = None
        ml_conf   = 0.0

        if self._ml_active and _SKLEARN_AVAILABLE and self._model is not None:
            ml_action, ml_conf = self._predict(features)
            self._predictions_made += 1

        # 5. Decide: use ML if confidence above threshold, else fallback
        if ml_action and ml_conf >= self.confidence_threshold:
            if ml_action != heuristic_action:
                self._ml_overrides += 1
            return {
                "action"  : ml_action,
                "source"  : "ML",
                "ml_conf" : round(ml_conf, 3),
                "raw_action": ml_action,
            }

        return {
            "action"  : heuristic_action,
            "source"  : "HEURISTIC",
            "ml_conf" : round(ml_conf, 3),
            "raw_action": ml_action,
        }

    def stats(self) -> dict:
        """Return summary statistics of the planner."""
        return {
            "total_steps"        : self._step,
            "training_samples"   : len(self._X),
            "ml_active"          : self._ml_active,
            "predictions_made"   : self._predictions_made,
            "ml_overrides"       : self._ml_overrides,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_features(self, ps_readings, fused_state, cam_perception, extra=None) -> list:
        """
        Construct the 19-element feature vector.
        All values are normalised to [0, 1] for the classifier.
        """
        # IR readings (8 sensors), normalised
        ps_norm = [min(v / IR_MAX, 1.0) for v in (ps_readings or [0] * 8)]
        if len(ps_norm) < 8:
            ps_norm += [0.0] * (8 - len(ps_norm))

        # Fused risk encoded as ordinal
        risk_str  = fused_state.get("fused_risk", "low")
        risk_enc  = RISK_MAP.get(risk_str, 0) / 2.0   # normalise to 0-1

        # Confidence already in 0-1
        confidence = float(fused_state.get("confidence", 0.5))

        # Camera features
        cam_obs    = 1.0 if cam_perception.get("obstacle_detected", False) else 0.0
        edge_dens  = float(cam_perception.get("edge_density", 0.0))

        # Navigation features (goal bearing + free space)
        ex = extra or {}
        bearing = float(ex.get("bearing", 0.0)) / 3.14159265
        front_f = min(float(ex.get("c0", 0.45)) / 0.45, 1.0)
        left_f = min(float(ex.get("c_left", 0.45)) / 0.45, 1.0)
        right_f = min(float(ex.get("c_right", 0.45)) / 0.45, 1.0)
        gdist = min(float(ex.get("dist", 3.0)) / 3.0, 1.0)

        ttc = ex.get("ttc")
        ttc_n = 1.0 if ttc is None else max(0.0, min(float(ttc) / 3.0, 1.0))
        threat = float(ex.get("threat_level", 0)) / 2.0

        return ps_norm[:8] + [risk_enc, confidence, cam_obs, edge_dens,
                              bearing, front_f, left_f, right_f, gdist, ttc_n, threat]

    def _maybe_retrain(self):
        """Train / retrain the model if conditions are met. Returns log string or None."""
        if not _SKLEARN_AVAILABLE:
            return None

        n = len(self._X)

        # Initial training trigger
        if not self._ml_active and n >= self.train_after_steps:
            return self._fit_model("INITIAL")

        # Periodic retraining (counted in planner steps, NOT buffer length,
        # because the buffer is capped at 3000 samples)
        if self._ml_active and (self._step % self.retrain_interval == 0):
            return self._fit_model("RETRAIN")

        return None

    # --- hooks that SVMPlanner (svm_planner.py) overrides --------------------
    MODEL_NAME = "MLPathPlanner"
    BUFFER_MAX = 3000

    def _make_model(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=8,
            random_state=42,
            n_jobs=1,
        )

    def _predict(self, features):
        """-> (action, confidence 0..1)"""
        proba = self._model.predict_proba([features])[0]
        best_idx = int(np.argmax(proba))
        return self._model.classes_[best_idx], float(proba[best_idx])

    def _top_features(self, names, k=4):
        imp = getattr(self._model, "feature_importances_", None)
        if imp is None:
            return ""
        top = sorted(zip(names, imp), key=lambda t: -t[1])[:k]
        return "top features: " + ", ".join(f"{n}={v:.3f}" for n, v in top)

    def _fit_model(self, reason: str) -> str:
        """Fit the classifier and activate ML mode."""
        X = np.array(self._X)
        y = np.array(self._y)

        # Need at least 2 unique classes to train
        if len(set(y)) < 2:
            return f"[{self.MODEL_NAME}] Skipping {reason}: need >= 2 action classes in data."

        self._model = self._make_model()
        self._model.fit(X, y)
        self._ml_active = True

        feat_names = [f"ps{i}" for i in range(8)] + ["risk_enc", "confidence", "cam_obs", "edge_dens",
                                                     "goal_bearing", "front_free", "left_free", "right_free",
                                                     "goal_dist", "ttc_norm", "threat_level"]
        return (f"[{self.MODEL_NAME}] {reason} fit on {len(X)} samples | "
                f"classes={[str(c) for c in self._model.classes_]} | {self._top_features(feat_names)}")

"""
svm_planner.py
==============
Support Vector Machine (SVM) planner - the SECOND machine-learning model of the robot,
running next to the existing Random Forest (ml_path_planner.py).

How it learns (same online scheme as the Random Forest)
------------------------------------------------------
* every control step the Navigator's decision (FORWARD / LEFT / RIGHT / STOP) is stored together
  with the 19 sensor features as a labelled training sample
* after `train_after_steps` samples the SVM is fitted; it is re-fitted every `retrain_interval` steps
* afterwards it predicts an action for every step together with a confidence value

Model
-----
    StandardScaler  ->  SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced")

* the scaler is essential for an SVM (features have very different ranges)
* class_weight="balanced": STOP / LEFT / RIGHT are rare compared with FORWARD
* the SVM has no probabilities, so the confidence is derived from the decision-function MARGIN
  between the best and the second-best class:   conf = 1 / (1 + exp(-2.5 * margin))

Why two models?
---------------
Random Forest = many axis-aligned decision trees (robust, handles raw sensor thresholds well).
SVM           = one smooth margin-based boundary in feature space (good on the continuous
                goal-bearing / free-space / time-to-collision features).
RobotBrain combines both in a weighted vote (see robot_brain.py) and compares them in the logs.
"""

import math

from ml_path_planner import MLPathPlanner, _SKLEARN_AVAILABLE, np

if _SKLEARN_AVAILABLE:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline


class SVMPlanner(MLPathPlanner):

    MODEL_NAME = "SVMPlanner"
    BUFFER_MAX = 1500           # an RBF-SVM trains in O(n^2..n^3): keep the buffer small

    def __init__(self, train_after_steps: int = 120, retrain_interval: int = 150,
                 confidence_threshold: float = 0.62, C: float = 1.0):
        super().__init__(train_after_steps=train_after_steps, retrain_interval=retrain_interval,
                         confidence_threshold=confidence_threshold, n_estimators=0)
        self.C = C

    # ---- hooks used by MLPathPlanner -------------------------------------------------
    def _make_model(self):
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=self.C, gamma="scale", class_weight="balanced",
                decision_function_shape="ovr"),
        )

    def _predict(self, features):
        scores = np.ravel(self._model.decision_function([features]))
        classes = self._model.classes_
        if len(classes) == 2:                        # binary: a single signed score
            s = float(scores[0])
            return classes[1 if s > 0 else 0], 1.0 / (1.0 + math.exp(-2.5 * abs(s)))
        order = np.argsort(scores)[::-1]
        margin = float(scores[order[0]] - scores[order[1]])
        return classes[int(order[0])], 1.0 / (1.0 + math.exp(-2.5 * margin))

    def _top_features(self, names, k=4):
        svc = self._model.steps[-1][1]
        return f"support vectors={int(sum(svc.n_support_))}"

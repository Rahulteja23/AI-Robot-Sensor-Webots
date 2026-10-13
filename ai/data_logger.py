"""
data_logger.py
==============
PHASE 4/5 - CSV Data Logger
----------------------------
Captures every simulation step full sensor state, AI perception output,
fused risk assessment, and final robot action to a timestamped CSV file
inside the results/ directory.
"""

import csv
import os
import time


class DataLogger:
    """
    CSV data logger for the robot simulation.

    Usage
    -----
    logger = DataLogger(results_dir="path/to/results")
    logger.log(row_dict)
    logger.close()
    """

    COLUMNS = [
        "step", "sim_time_s",
        "pos_x", "pos_y", "pos_z",
        "ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7",
        "front_val", "left_val", "right_val",
        "mean_proximity", "max_proximity", "std_deviation", "risk_level",
        "cam_obstacle_detected", "cam_dominant_color", "cam_edge_density",
        "fused_risk", "confidence", "camera_confirmed",
        "action", "left_speed", "right_speed",
        "distance_to_goal",
        # --- added for goal-seeking navigation ---
        "goal_bearing_deg", "min_clearance_m", "nav_mode",
        "v_cmd", "w_cmd", "ml_action", "ml_conf", "front_free_m",
        # --- Random Forest vs SVM + moving-obstacle safety layer ---
        "svm_action", "svm_conf", "ensemble_action",
        "dyn_mode", "dyn_ttc_s", "dyn_movers", "v_nav", "speed_scale",
        # --- mission (random start / goal): goal position of this run ---
        "goal_x", "goal_y", "floor", "leg",
    ]

    def __init__(self, results_dir: str):
        os.makedirs(results_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._filepath = os.path.join(results_dir, f"simulation_run_{ts}.csv")
        self._file = open(self._filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self.COLUMNS,
            extrasaction="ignore",
        )
        self._writer.writeheader()
        self._file.flush()
        self._rows_written = 0
        print(f"[DataLogger] CSV log opened -> {self._filepath}")

    def log(self, row: dict) -> None:
        """Write one simulation step as a CSV row."""
        clean = {col: "" for col in self.COLUMNS}
        clean.update(row)
        float_cols = {
            "sim_time_s", "pos_x", "pos_y", "pos_z",
            "ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7",
            "front_val", "left_val", "right_val",
            "mean_proximity", "max_proximity", "std_deviation",
            "cam_edge_density", "confidence",
            "left_speed", "right_speed", "distance_to_goal",
            "goal_bearing_deg", "min_clearance_m", "v_cmd", "w_cmd", "ml_conf", "front_free_m",
            "svm_conf", "dyn_ttc_s", "v_nav", "speed_scale", "goal_x", "goal_y",
        }
        for col in float_cols:
            if clean[col] != "":
                try:
                    clean[col] = round(float(clean[col]), 4)
                except (ValueError, TypeError):
                    pass
        self._writer.writerow(clean)
        self._rows_written += 1
        if self._rows_written % 50 == 0:
            self._file.flush()

    def close(self) -> None:
        """Flush remaining data and close the file handle."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()
            print(f"[DataLogger] Closed. {self._rows_written} rows written -> {self._filepath}")

    @property
    def filepath(self) -> str:
        return self._filepath

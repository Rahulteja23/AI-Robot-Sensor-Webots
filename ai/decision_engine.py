"""
decision_engine.py
==================
PHASE 4 - Full Decision Engine Implementation
----------------------------------------------
Maps the fused environment state to motor commands.
Replaces the Phase 4 skeleton with a complete rule-based engine
that is designed to work alongside the MLPathPlanner.

Decision priority
-----------------
  1. CRITICAL risk front  -> stop + turn to best side
  2. CRITICAL risk side   -> turn away
  3. MEDIUM risk front    -> slow turn
  4. CLEAR path           -> move forward at cruise speed
  5. Low confidence       -> reduce speed (cautious)
"""

MAX_SPEED     = 6.28
CRUISE_SPEED  = 3.0
SLOW_SPEED    = 1.5
TURN_SPEED    = 2.0


class DecisionEngine:
    """
    Maps perceived environment state to motor commands.
    Works with both heuristic logic and the MLPathPlanner.
    """

    def __init__(self):
        print("[DecisionEngine] Full Phase 4 Decision Engine initialised.")

    def decide(self, fused_state: dict, ps_readings: list = None) -> dict:
        """
        Parameters
        ----------
        fused_state : dict  - output from SensorFusion.fuse()
        ps_readings : list  - raw [ps0..ps7] values for directional checks

        Returns
        -------
        dict:
            "action"      : str   - "FORWARD" | "LEFT" | "RIGHT" | "STOP"
            "left_speed"  : float - left wheel velocity (rad/s)
            "right_speed" : float - right wheel velocity (rad/s)
            "reason"      : str   - human-readable explanation
        """
        risk        = fused_state.get("fused_risk", "low")
        confidence  = fused_state.get("confidence", 0.5)
        zones       = fused_state.get("zones", {})

        ps = ps_readings or [0] * 8

        # Aggregate directional IR values
        front_val = max(ps[0], ps[7])
        left_val  = max(ps[5], ps[6])
        right_val = max(ps[1], ps[2])
        THRESHOLD = 250.0

        front_blocked = front_val > THRESHOLD
        left_blocked  = left_val  > THRESHOLD
        right_blocked = right_val > THRESHOLD

        # --- Decision tree ---
        if risk == "critical":
            if front_blocked:
                # Turn toward more open side
                if left_val <= right_val:
                    action, ls, rs, reason = "LEFT", -TURN_SPEED * 0.5, TURN_SPEED * 0.5, "CRIT-FRONT: turn left (more space)"
                else:
                    action, ls, rs, reason = "RIGHT", TURN_SPEED * 0.5, -TURN_SPEED * 0.5, "CRIT-FRONT: turn right (more space)"
            elif left_blocked:
                action, ls, rs, reason = "RIGHT", TURN_SPEED * 0.5, -TURN_SPEED * 0.5, "CRIT-LEFT: turn right"
            elif right_blocked:
                action, ls, rs, reason = "LEFT", -TURN_SPEED * 0.5, TURN_SPEED * 0.5, "CRIT-RIGHT: turn left"
            else:
                # Rear sensors only - slow down
                action, ls, rs, reason = "FORWARD", SLOW_SPEED, SLOW_SPEED, "CRIT-REAR: slow forward"

        elif risk == "medium":
            if front_blocked:
                if left_val <= right_val:
                    action, ls, rs, reason = "LEFT", -SLOW_SPEED * 0.5, SLOW_SPEED * 0.5, "MED-FRONT: cautious left"
                else:
                    action, ls, rs, reason = "RIGHT", SLOW_SPEED * 0.5, -SLOW_SPEED * 0.5, "MED-FRONT: cautious right"
            elif left_blocked:
                action, ls, rs, reason = "RIGHT", CRUISE_SPEED * 0.5, -CRUISE_SPEED * 0.5, "MED-LEFT: nudge right"
            elif right_blocked:
                action, ls, rs, reason = "LEFT", -CRUISE_SPEED * 0.5, CRUISE_SPEED * 0.5, "MED-RIGHT: nudge left"
            else:
                spd = SLOW_SPEED if confidence < 0.6 else CRUISE_SPEED
                action, ls, rs, reason = "FORWARD", spd, spd, f"MED: forward (conf={confidence:.2f})"

        else:  # low risk
            spd = CRUISE_SPEED if confidence >= 0.7 else SLOW_SPEED
            action, ls, rs, reason = "FORWARD", spd, spd, f"CLEAR path (conf={confidence:.2f})"

        return {
            "action"      : action,
            "left_speed"  : round(ls, 3),
            "right_speed" : round(rs, 3),
            "reason"      : reason,
        }

    @staticmethod
    def action_to_speeds(action: str) -> tuple:
        """Convert an action string to (left_speed, right_speed) tuple."""
        mapping = {
            "FORWARD" : ( CRUISE_SPEED,  CRUISE_SPEED),
            "LEFT"    : (-TURN_SPEED * 0.5,  TURN_SPEED * 0.5),
            "RIGHT"   : ( TURN_SPEED * 0.5, -TURN_SPEED * 0.5),
            "STOP"    : (0.0, 0.0),
        }
        return mapping.get(action, (CRUISE_SPEED, CRUISE_SPEED))

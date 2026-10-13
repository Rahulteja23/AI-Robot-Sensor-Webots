"""
sensor_fusion.py
======================
PHASE 3 – AI Sensor Fusion
----------------------------
Combines distance sensor arrays and camera perception to produce
a unified, high-confidence environmental risk map.
"""

class SensorFusion:
    """
    Fuses distance sensor readings and camera features to produce
    a unified environment map used by the DecisionEngine.
    """

    def __init__(self):
        print("[SensorFusion] AI Sensor Fusion module initialised (Phase 3).")

    def fuse(self, distance_perception: dict, camera_perception: dict, raw_zones: dict) -> dict:
        """
        Fuses inputs from multiple modalities to build a robust state representation.
        
        Parameters
        ----------
        distance_perception : dict – output of Perception.process_distance()
        camera_perception   : dict – output of Perception.process_camera()
        raw_zones           : dict – basic directional zones from SensorManager
        
        Returns
        -------
        dict with fused risk assessment and directional cues.
        """
        # Base risk comes from distance sensors (IR is reliable at close range)
        base_risk = distance_perception.get("risk_level", "low")
        
        # Camera adds long-range or texture-based context
        cam_obstacle = camera_perception.get("obstacle_detected", False)
        edge_density = camera_perception.get("edge_density", 0.0)
        
        fused_risk = base_risk
        confidence = 0.5  # default medium confidence
        
        # AI FUSION RULES:
        if base_risk == "critical":
            if cam_obstacle:
                confidence = 0.95  # Both modalities strongly agree!
            else:
                confidence = 0.70  # IR says critical, camera misses it (maybe bad lighting?)
                
        elif base_risk == "medium":
            if cam_obstacle:
                fused_risk = "critical" # Camera elevates the risk (sees object texture clearly)
                confidence = 0.85
            else:
                confidence = 0.80
                
        else: # base_risk == "low"
            if cam_obstacle and edge_density > 0.08:
                fused_risk = "medium"   # Camera sees something far away before IR detects it
                confidence = 0.60
            else:
                confidence = 0.90       # Both agree the path is clear

        return {
            "fused_risk": fused_risk,
            "confidence": confidence,
            "camera_confirmed": cam_obstacle,
            "zones": raw_zones
        }

"""
perception.py
===================
PHASE 3 – AI Perception Pipeline
----------------------------------
Processes raw camera images and distance readings to extract
semantic features using Computer Vision and statistical concepts.
"""

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
    print("[Perception] WARNING: OpenCV (cv2) not found. Camera perception will be limited. Please install opencv-python.")

class Perception:
    """
    Processes raw camera images and distance readings to extract
    semantic features: obstacle type, edge density, and statistical risk.
    """

    def __init__(self):
        print("[Perception] AI Perception module initialised (Phase 3).")

    def process_camera(self, camera_data: dict) -> dict:
        """
        Convert raw BGRA camera image to high-level features.
        
        Returns dict:
            "obstacle_detected" : bool  - True if camera identifies an object
            "dominant_color"    : str   - "brown_wood", "grey_floor", etc.
            "edge_density"      : float - ratio of edge pixels (proxy for texture/closeness)
        """
        result = {
            "obstacle_detected": False,
            "dominant_color": "unknown",
            "edge_density": 0.0
        }

        if not camera_data or "raw" not in camera_data:
            return result

        raw = camera_data["raw"]
        w = camera_data["width"]
        h = camera_data["height"]

        if cv2 is not None and isinstance(raw, bytes):
            # Convert raw bytes to numpy array (Webots camera is BGRA)
            img = np.frombuffer(raw, np.uint8).reshape((h, w, 4))
            
            # Convert BGRA to BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # 1. Edge Detection
            # High edge density usually means a textured object (like the wooden box) is close
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)
            result["edge_density"] = round(float(edge_density), 3)
            
            # 2. Color Classification (Simple AI heuristic)
            # The wooden boxes are brown/dark. The floor is grey.
            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            mean_h = np.mean(hsv[:, :, 0])
            mean_s = np.mean(hsv[:, :, 1])
            mean_v = np.mean(hsv[:, :, 2])
            
            # Toy cars / coloured blocks are strongly saturated, floor + road are grey.
            # Look only at the lower 60 % of the image (below the horizon).
            low = hsv[int(h * 0.4):, :, :]
            coloured_fraction = float(np.mean(low[:, :, 1] > 110))

            # Heuristic for brown wood: hue is in the warm range (10-30), saturation is noticeable
            if mean_s > 40 and 10 < mean_h < 30:
                result["dominant_color"] = "brown_wood"
                result["obstacle_detected"] = True
            elif coloured_fraction > 0.12:
                result["dominant_color"] = "coloured_car"
                result["obstacle_detected"] = True
            else:
                result["dominant_color"] = "grey_floor"
                
            # If edge density is high, we likely see the box texture even if color varies
            if edge_density > 0.05:
                result["obstacle_detected"] = True

        return result

    def process_distance(self, readings: list) -> dict:
        """
        Classify proximity readings into statistical risk levels.
        This smooths out raw sensor data and extracts patterns.
        """
        if not readings:
            return {"risk_level": "low", "max_proximity": 0.0}

        arr = np.array(readings)
        mean_dist = np.mean(arr)
        max_dist = np.max(arr)
        std_dist = np.std(arr)
        
        # Map raw maximums to a symbolic risk level
        risk_level = "low"
        if max_dist > 1500:
            risk_level = "critical"
        elif max_dist > 500:
            risk_level = "medium"
            
        return {
            "mean_proximity": round(float(mean_dist), 2),
            "max_proximity": round(float(max_dist), 2),
            "std_deviation": round(float(std_dist), 2),
            "risk_level": risk_level
        }

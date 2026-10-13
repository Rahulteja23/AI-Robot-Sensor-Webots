"""
sensor_manager.py
=================
PHASE 2 – Full Sensor System Implementation
--------------------------------------------
Manages all sensors on the E-puck robot in Webots R2025a.

Sensor inventory (verified from official E-puck PROTO + Webots C source):
  ┌──────────────────────┬───────────────────┬────────────────────────────┐
  │ Sensor               │ Webots device name│ Returns                    │
  ├──────────────────────┼───────────────────┼────────────────────────────┤
  │ IR proximity × 8     │ "ps0" … "ps7"     │ float 0–4095 (higher=near) │
  │ Camera               │ "camera"          │ BGRA bytes, 52×39 px       │
  │ GPS                  │ "gps"             │ (x, y, z) metres           │
  │ InertialUnit (IMU)   │ "inertial unit"   │ (roll, pitch, yaw) radians │
  └──────────────────────┴───────────────────┴────────────────────────────┘

Usage:
    sm = SensorManager(robot, timestep)
    sm.init_all()
    while robot.step(timestep) != -1:
        data = sm.read_all()
        # data["distance"]  → list of 8 floats
        # data["camera"]    → dict with raw/width/height/mean_brightness
        # data["position"]  → (x, y, z)
        # data["heading"]   → yaw in radians
"""

from __future__ import annotations
import math
from typing import Optional


# ── Sensor device names (verified from E-puck PROTO & Webots C controller) ───
DISTANCE_SENSOR_NAMES = ["ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"]
CAMERA_NAME           = "camera"
GPS_NAME              = "gps"
IMU_NAME              = "inertial unit"

# ── E-puck IR sensor physical positions (degrees from front, counter-clockwise)
# Used for display / logging only. Layout verified from e-puck documentation.
#   ps0 = 10°  (front-right)
#   ps1 = 45°  (right)
#   ps2 = 90°  (right)
#   ps3 = 150° (rear-right)
#   ps4 = 210° (rear-left)
#   ps5 = 270° (left)
#   ps6 = 315° (left)
#   ps7 = 350° (front-left)
SENSOR_ANGLES_DEG = [10, 45, 90, 150, 210, 270, 315, 350]

# ── Threshold values (from E-puck lookupTable in E-puckDistanceSensor.proto) ─
#   0 cm → 4095   (maximum, obstacle touching)
#   7 cm → 67     (effective sensing limit)
DISTANCE_MAX_VALUE   = 4095.0   # raw units – closest measurable distance
DISTANCE_NEAR_THRESH = 500.0    # raw units – "near" obstacle warning
DISTANCE_STOP_THRESH = 1500.0   # raw units – "critical" stop zone


class SensorManager:
    """
    Central manager for all E-puck robot sensors.

    Initialise once before the loop, read every timestep.
    """

    def __init__(self, robot, timestep: int):
        """
        Parameters
        ----------
        robot    : controller.Robot – active Webots Robot instance
        timestep : int              – simulation timestep in ms
        """
        self.robot    = robot
        self.timestep = timestep

        # Internal device handles (set in init_* methods)
        self._distance_sensors: list = []
        self._range_sensors: list = []     # 24 long-range sensors rs0..rs23
        self._camera              = None
        self._gps                 = None
        self._imu                 = None

        # Cached last readings (so read_all() always returns something valid)
        self._last_distance: list[float] = [0.0] * 8
        self._last_range:    list[float] = [0.5] * 24
        self._last_camera:   dict        = {}
        self._last_position: tuple       = (0.0, 0.0, 0.0)
        self._last_heading:  float       = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # Initialisation
    # ══════════════════════════════════════════════════════════════════════════

    def init_distance_sensors(self) -> None:
        """
        Enable all 8 E-puck infrared proximity sensors.

        Device names: "ps0" through "ps7"  (verified from C controller source)
        Lookup table range: 0 (far) → 4095 (very close, ~0 cm)
        Effective sensing limit: ~7 cm (value ~67 at 7 cm)
        """
        self._distance_sensors = []
        for name in DISTANCE_SENSOR_NAMES:
            sensor = self.robot.getDevice(name)
            if sensor is None:
                print(f"[SensorManager] WARNING: Distance sensor '{name}' not found!")
                continue
            sensor.enable(self.timestep)
            self._distance_sensors.append(sensor)

        print(f"[SensorManager] Distance sensors enabled: "
              f"{len(self._distance_sensors)}/8  ({DISTANCE_SENSOR_NAMES})")

    def init_range_sensors(self, count: int = 24) -> None:
        """
        Enable the 24 long-range sensors "rs0".."rs23" (360 deg ring, 0.5 m range)
        that the world file adds to the E-puck.  They return DISTANCE IN METRES.
        """
        self._range_sensors = []
        for k in range(count):
            sensor = self.robot.getDevice(f"rs{k}")
            if sensor is None:
                print(f"[SensorManager] WARNING: range sensor 'rs{k}' not found "
                      "- regenerate the world with tools/generate_world.py")
                continue
            sensor.enable(self.timestep)
            self._range_sensors.append(sensor)
        print(f"[SensorManager] Long-range sensors enabled: {len(self._range_sensors)}/{count}")

    def init_camera(self) -> None:
        """
        Enable the E-puck onboard camera.

        Device name : "camera"  (verified from Webots Python sample)
        Resolution  : 52 × 39 px (default)
        Format      : BGRA (Blue-Green-Red-Alpha), 4 bytes per pixel
        Field of view: 0.84 radians (~48°)
        """
        self._camera = self.robot.getDevice(CAMERA_NAME)
        if self._camera is None:
            print(f"[SensorManager] WARNING: Camera '{CAMERA_NAME}' not found!")
            return
        self._camera.enable(self.timestep)
        w = self._camera.getWidth()
        h = self._camera.getHeight()
        print(f"[SensorManager] Camera enabled: '{CAMERA_NAME}'  "
              f"resolution={w}×{h} px  format=BGRA")

    def init_gps(self) -> None:
        """
        Enable the GPS sensor for absolute position tracking.

        Device name : "gps"  (added to world file via turretSlot)
        Returns     : (x, y, z) in metres  (Webots NUE coordinate system)
        Note        : In NUE coordinates, Y is "up". x and z are floor plane.
        """
        self._gps = self.robot.getDevice(GPS_NAME)
        if self._gps is None:
            print(f"[SensorManager] WARNING: GPS '{GPS_NAME}' not found! "
                  "Check that turretSlot contains a GPS node in the .wbt file.")
            return
        self._gps.enable(self.timestep)
        print(f"[SensorManager] GPS enabled: '{GPS_NAME}'")

    def init_imu(self) -> None:
        """
        Enable the InertialUnit (IMU) for orientation / heading.

        Device name : "inertial unit"  (added to world via turretSlot)
        Returns     : (roll, pitch, yaw) in radians
        yaw = 0       → robot facing +X axis
        yaw = π/2     → robot facing +Z axis (left)
        yaw = ±π      → robot facing -X axis
        """
        self._imu = self.robot.getDevice(IMU_NAME)
        if self._imu is None:
            print(f"[SensorManager] WARNING: InertialUnit '{IMU_NAME}' not found! "
                  "Check that turretSlot contains an InertialUnit node.")
            return
        self._imu.enable(self.timestep)
        print(f"[SensorManager] InertialUnit (IMU) enabled: '{IMU_NAME}'")

    def init_all(self) -> None:
        """
        Enable all sensors in the correct order.
        Call this once BEFORE the main simulation loop.
        """
        print("[SensorManager] Initialising all sensors …")
        self.init_distance_sensors()
        self.init_range_sensors()
        self.init_camera()
        self.init_gps()
        self.init_imu()
        print("[SensorManager] All sensors initialised ✓")
        print(f"  Active: distance={bool(self._distance_sensors)}, "
              f"camera={self._camera is not None}, "
              f"gps={self._gps is not None}, "
              f"imu={self._imu is not None}")

    # ══════════════════════════════════════════════════════════════════════════
    # Reading
    # ══════════════════════════════════════════════════════════════════════════

    def read_distance_sensors(self) -> list[float]:
        """
        Read all 8 IR proximity sensors.

        Returns
        -------
        list[float] : 8 values, each 0–4095.
                      Higher value = obstacle is CLOSER.
                      Ordering: ps0 (front-right) → ps7 (front-left).
        """
        values = []
        for sensor in self._distance_sensors:
            values.append(sensor.getValue())
        self._last_distance = values if values else self._last_distance
        return self._last_distance

    def read_range_sensors(self) -> list:
        """24 long-range distances in metres (0.5 = nothing in range)."""
        if not self._range_sensors:
            return self._last_range
        vals = []
        for sensor in self._range_sensors:
            v = sensor.getValue()
            vals.append(0.5 if (v != v or v > 0.5) else max(0.0, v))   # NaN / overflow -> 'nothing'
        self._last_range = vals
        return vals

    def read_camera(self) -> dict:
        """
        Read camera image and compute basic statistics.

        Returns
        -------
        dict:
            "raw"             : bytes  – raw BGRA pixel data (52×39×4 bytes)
            "width"           : int    – image width in pixels
            "height"          : int    – image height in pixels
            "mean_brightness" : float  – average pixel intensity (0–255)
            "mean_red"        : float  – average red channel value
            "mean_green"      : float  – average green channel value
            "mean_blue"       : float  – average blue channel value
        """
        if self._camera is None:
            return self._last_camera

        raw = self._camera.getImage()    # bytes: BGRA format, row-major
        w   = self._camera.getWidth()
        h   = self._camera.getHeight()

        # Compute per-channel averages from raw BGRA bytes
        total_pixels = w * h
        sum_b = sum_g = sum_r = 0
        for i in range(total_pixels):
            base = i * 4
            # BGRA order: byte 0=Blue, 1=Green, 2=Red, 3=Alpha
            sum_b += raw[base]
            sum_g += raw[base + 1]
            sum_r += raw[base + 2]

        mean_b = sum_b / total_pixels
        mean_g = sum_g / total_pixels
        mean_r = sum_r / total_pixels
        mean_brightness = (mean_r + mean_g + mean_b) / 3.0

        result = {
            "raw"             : raw,
            "width"           : w,
            "height"          : h,
            "mean_brightness" : round(mean_brightness, 2),
            "mean_red"        : round(mean_r, 2),
            "mean_green"      : round(mean_g, 2),
            "mean_blue"       : round(mean_b, 2),
        }
        self._last_camera = result
        return result

    def read_position(self) -> tuple[float, float, float]:
        """
        Read GPS position.

        Returns
        -------
        tuple (x, y, z) in metres.
        World uses the ENU coordinate system: x = East, y = North, z = Up.
        The robot's forward direction is typically +X (when yaw=0).
        """
        if self._gps is None:
            return self._last_position

        pos = self._gps.getValues()        # returns list [x, y, z]
        self._last_position = (round(pos[0], 4),
                               round(pos[1], 4),
                               round(pos[2], 4))
        return self._last_position

    def read_heading(self) -> float:
        """
        Read heading (yaw) from the InertialUnit.

        Returns
        -------
        float : yaw angle in radians.  Range: -π to +π.
                0 = facing +X, π/2 = facing +Z, ±π = facing -X.
        """
        if self._imu is None:
            return self._last_heading

        rpy = self._imu.getRollPitchYaw()   # returns [roll, pitch, yaw]
        self._last_heading = round(rpy[2], 4)
        return self._last_heading

    def read_all(self) -> dict:
        """
        Read ALL sensors in one call. Use this inside the main loop.

        Returns
        -------
        dict:
            "distance"      : list[float]         – 8 IR values (ps0..ps7)
            "camera"        : dict                – camera image + statistics
            "position"      : tuple (x, y, z)    – GPS metres
            "heading"       : float               – yaw in radians (IMU)
            "obstacle_zones": dict                – pre-computed zone flags
        """
        dist    = self.read_distance_sensors()
        rng     = self.read_range_sensors()
        cam     = self.read_camera()
        pos     = self.read_position()
        heading = self.read_heading()
        zones   = self._compute_obstacle_zones(dist)

        return {
            "distance"      : dist,
            "range"         : rng,
            "camera"        : cam,
            "position"      : pos,
            "heading"       : heading,
            "obstacle_zones": zones,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_obstacle_zones(self, dist: list[float]) -> dict:
        """
        Classify distance sensor readings into three directional zones.

        E-puck sensor layout (top view, robot facing right →):
            ps7 (350°)  ps0 (10°)   ← FRONT sensors
            ps6 (315°)  ps1 (45°)   ← FRONT-SIDE sensors
            ps5 (270°)  ps2 (90°)   ← SIDE sensors
            ps4 (210°)  ps3 (150°)  ← REAR sensors

        Zone grouping:
            FRONT  : ps7, ps0            (indices 7, 0)
            RIGHT  : ps0, ps1, ps2       (indices 0, 1, 2)
            LEFT   : ps5, ps6, ps7       (indices 5, 6, 7)
            REAR   : ps3, ps4            (indices 3, 4)
        """
        if len(dist) < 8:
            return {"front": False, "left": False, "right": False, "rear": False}

        thresh = DISTANCE_NEAR_THRESH

        zone_front = (dist[7] > thresh or dist[0] > thresh)
        zone_right = (dist[0] > thresh or dist[1] > thresh or dist[2] > thresh)
        zone_left  = (dist[5] > thresh or dist[6] > thresh or dist[7] > thresh)
        zone_rear  = (dist[3] > thresh or dist[4] > thresh)

        # Critical = stop-threshold check (closer than NEAR)
        critical_front = (dist[7] > DISTANCE_STOP_THRESH or
                          dist[0] > DISTANCE_STOP_THRESH)

        return {
            "front"          : zone_front,
            "right"          : zone_right,
            "left"           : zone_left,
            "rear"           : zone_rear,
            "critical_front" : critical_front,
            "max_value"      : max(dist),
            "max_sensor_idx" : dist.index(max(dist)),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Logging helpers
    # ══════════════════════════════════════════════════════════════════════════

    def print_distance_bar(self, dist: list[float]) -> None:
        """
        Print a compact ASCII bar chart of all 8 sensor values.
        Useful for quick visual inspection of proximity readings.
        """
        bar_max = 20
        print("  IR sensors: ", end="")
        for i, v in enumerate(dist):
            filled = int((v / DISTANCE_MAX_VALUE) * bar_max)
            bar = "█" * filled + "░" * (bar_max - filled)
            print(f"ps{i}[{bar}]{v:5.0f}", end="  ")
        print()

    def print_summary(self, data: dict, step: int) -> None:
        """Print a one-line sensor summary for a given simulation step."""
        dist  = data.get("distance", [0.0]*8)
        pos   = data.get("position", (0, 0, 0))
        head  = data.get("heading", 0.0)
        cam   = data.get("camera", {})
        zones = data.get("obstacle_zones", {})
        bright = cam.get("mean_brightness", 0.0)
        head_deg = math.degrees(head)

        zone_str = " ".join(
            k.upper() for k, v in zones.items()
            if isinstance(v, bool) and v
        ) or "CLEAR"

        print(f"[{step:5d}] "
              f"pos=({pos[0]:+.3f},{pos[1]:+.3f})m  "
              f"yaw={head_deg:+6.1f}°  "
              f"IR_max={max(dist):5.0f}  "
              f"cam_bright={bright:5.1f}  "
              f"zones=[{zone_str}]")

    def __repr__(self) -> str:
        return (f"SensorManager(timestep={self.timestep}ms, "
                f"ds={len(self._distance_sensors)}/8, "
                f"cam={self._camera is not None}, "
                f"gps={self._gps is not None}, "
                f"imu={self._imu is not None})")

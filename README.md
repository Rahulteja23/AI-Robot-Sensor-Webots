# AI-Enhanced Intelligent Sensor System - Autonomous Hospital Delivery Robot
### B.Tech Project | Webots R2025a | Python 3.10

A delivery robot (an E-puck, scale 1:8 = a 0.6 m robot) drives through a **3-floor hospital** - from any room to any room,
across floors by lift - without colliding with walls, furniture, nurses or medicine trolleys.

Floor plans: `results/sample_offline_run/hospital_floor_plans.png`

---

## 1. The hospital

* **3 floors**, each 4.0 m x 3.0 m (real size 32 m x 24 m). The floors are three plates side by side at increasing height
  (0 / 0.45 / 0.90 m), so all floors are visible at once. The 3D view jumps to the floor the robot is on.
* **Per floor:** a main corridor and a cross corridor (0.40 m wide), **8 rooms** with doors (0.22 m), two **lifts** at the
  ends of the cross corridor, 14 cm high cut-away walls, furniture, a floor-number sign.
* **Floor 1 (Ground):** Emergency, Pharmacy, Laboratory, Radiology, Cafeteria, Outpatient Clinic, Store Room, Waiting Hall
* **Floor 2:** Wards 201-205, ICU, Nurse Station, Pediatrics
* **Floor 3:** Operating Theatres 1-2, Recovery, Surgeon Office, Sterilization, Wards 301-302, Meeting Room
* **Static obstacles:** 84 walls and 150 pieces of furniture (beds, cots, IV stands, monitors, operating tables and lamps,
  X-ray machine, counters, shelves, fridges, benches, chairs, tables, wheelchairs, plants ...)
* **Moving obstacles:** on every floor a **nurse** patrols the main corridor and a **medicine trolley** runs along the cross
  corridor. Like real staff they stop when somebody is directly in front of them.
  (`TROLLEY_NEVER_BRAKES = True` in `ai/hospital.py` makes the trolleys never brake: perpendicular-collision situations.)
* **Lifts:** the robot drives into a lift cabin, waits 3 s and appears in the lift of the target floor.
  (A lift is a teleport - the simulator does not move a physical cabin.)

## 2. Missions (random start and goal)

Every run picks a random mission "take something from **room A, floor a** to **room B, floor b**" (`ai/mission.py`):
60 % of the missions are across floors. The exact START and GOAL points are random clear spots inside the rooms and the
start heading is random. A route must exist.

* The seed is printed at start-up and saved in `results/mission_result.json`. To **repeat** a mission set
  `MISSION_SEED = <seed>` in `ai/scene_config.py`.
* `RANDOM_MISSION = False` selects the demo mission: Pharmacy (floor 1) -> Operating Theatre 1 (floor 3).
* How it works in Webots: the robot controller picks the mission, teleports the robot to its start, moves the blue/green
  pads and the flag, and publishes the mission id to the supervisor (HUD, collision counting, report).
  The E-puck therefore has `supervisor TRUE` in the world file.

## 3. Run it in Webots

1. Webots -> **File -> Open World...** -> `worlds/ai_sensor_world.wbt`
2. (once) `pip install -r requirements.txt` into the Python that Webots uses (Tools -> Preferences -> Python command).
3. Press **Play** (use fast-forward: a cross-floor trip takes 1-2 minutes of simulated time).

In the 3D view: a live **HUD** (start room -> goal room, floor, time, progress bar, collisions, closest approach, the robot's
mode / speed and the Random Forest / SVM decisions), a colour-fading **trail**, a **MISSION SUCCESS** banner at the end.
`SHOW_HUD`, `SHOW_TRAIL` (supervisor file) and `AUTO_CAMERA` (`ai/scene_config.py`) switch these off.

Console output (from the offline stand-in run):

```
[INIT] RANDOM mission (seed 12 - set MISSION_SEED = 12 in ai/scene_config.py to repeat it)
[INIT] FROM  Ward 205 (Floor 2)  at (+1.47, -1.11), heading +36 deg
[INIT] TO    Surgeon Office (Floor 3)  at (+1.77, +1.07)
[INIT] route 7.4 m in 2 leg(s)  (rides Lift B (south))
[LIFT] t=  69.0s  Lift B (south) on Floor 2: riding to Floor 3 ...
  MISSION REPORT   Result : MISSION SUCCESS - goal reached without collision
  Time to goal 115.4 s | Path 7.34 m (planned 7.43 m) | Floors 2 -> 3 | Collisions 0 | Closest approach 2.2 cm
```

Files written to `results/`: `simulation_run_<timestamp>.csv` (every step, incl. floor and leg) and `mission_result.json`.
Graphs for the report (run after the simulation, outside Webots):

```bash
python experiments/analyze_results.py        # path per floor on the floor plan, sensors, risk, actions
python experiments/ml_comparison_plots.py    # heuristic vs ML dashboard
python experiments/model_comparison.py       # Random Forest vs SVM + speed profile
```

## 4. How the robot works

```
 24 range sensors (360 deg, 0.5 m)  8 IR sensors  camera  GPS  IMU
        |
 Perception -> SensorFusion (risk level + confidence)
        |
 RouteFollower  global route (A* on the known floor plan: through corridors, doors, lifts)
        |
 Navigator      local planner: follows the route, avoids obstacles it senses
        |
 DynamicSafety  tracks FAST moving objects, predicts them, changes speed / direction (see below)
        |
 Random Forest + SVM   online-learned second opinions, combined in a weighted vote
        |
 wheel speeds  ->  DataLogger (CSV)
```

* **A hospital robot knows its floor plan but not where people are.** The route (`ai/route_planner.py`) is planned on
  the walls and furniture; people and trolleys are handled by the local layers.
* **Dynamic safety layer** (`ai/motion_predictor.py`): detects what *moves* (a range-ray hit on space that was recently free),
  estimates its velocity, simulates ~55 candidate commands (brake, reverse, accelerate, steer, veer, turn on the spot) against
  the predicted paths and executes the safest. **It acts only on objects faster than 0.085 m/s** (see section 6 for why).
* **Two ML models** (`ai/ml_path_planner.py`, `ai/svm_planner.py`) learn online from the Navigator and may only make the
  robot **more careful, and only when both agree on STOP**.

| held-out test on the sample run (chronological 70/30 split, 3965 samples) | accuracy | macro-F1 |
|---|---|---|
| Random Forest (100 trees, depth 8) | 0.864 | 0.746 |
| SVM (RBF, C=1, balanced classes) | 0.851 | 0.555 |

(The test part of this run contains no STOP samples, so STOP recall cannot be given.)

## 5. Test results (offline simulation with sensor noise and wheel slip - NOT Webots physics)

| Test | Result |
|---|---|
| `tools/check_world.py`: layout + world file (overlaps, door widths, reachability, lift lobbies, 240 objects match the file) | 14 / 14 checks pass |
| **Random missions**, normal traffic - fresh seeds 2001-2120 | **120 / 120** reach the goal, 0 collisions, mean 71 s |
| **Random missions, stress: nobody ever brakes** - fresh seeds 2001-2180 | **180 / 180**, 0 collisions (closest approach 0.2 cm) |
| Random missions with Random Forest + SVM on - fresh seeds 2001-2045 | **45 / 45**, 0 collisions |
| **Perpendicular-car test** in an open 3 m arena (fast cars 0.05-0.16 m/s that never brake, 112 runs) | **28 collisions without the safety layer -> 1 with it** |
| same, wider sweep up to 0.28 m/s (156 runs) | 42 -> 1 |
| Ablation on hospital missions (`experiments/benchmark.py`, 20 runs each, 4 configurations) | 20 / 20 in all four |
| Real controller + supervisor files on a stand-in for Webots (random cross-floor mission, ML on) | success, 0 collisions |

Seeds 1-50 and 1001-1060 were used while debugging; the numbers above use seeds that were never used for tuning.

```bash
python tools/check_world.py                                   # validate the building and the world file
python experiments/headless_validation.py --runs 30 --plot --gif    # random missions (--no-yield: stress, --fixed: demo mission)
python experiments/perpendicular_test.py --gif                # fast-car dodge test (--baseline: without the safety layer)
python experiments/benchmark.py 20                            # ablation
python experiments/mission_preview.py                         # where random missions start / end
```
`results/sample_offline_run/` holds example outputs. They come from the real controller and supervisor code running against a
**simulated stand-in for Webots**, not from a Webots recording. Your own Webots run creates the real files.

## 6. What debugging found (and fixed)

| Problem found | Cause | Fix |
|---|---|---|
| Robot reversed for 30 s in a corridor, 373 s missions | safety circle (7.2 cm) left only a 2.6 cm window to pass a 6 cm trolley in a 0.4 m corridor | snugger safety circle (5.9 cm) and EVADE radius; missions 2-3x faster |
| Collisions right after leaving a lift | the trolley lane ended 0.18 m from the lift door; the turning trolley hit the robot as it appeared | lanes end 0.4 m short of the lifts; `check_world.py` now checks this |
| Predictive layer made the hospital *worse* in stress (52/60 vs 59/60 without it) | it was tuned for fast toy cars in an open arena; in narrow corridors its cautious moves caused contacts | it now acts only on movers >= 0.085 m/s; hospital stress 60/60, T-bone test kept |
| 85 % false "moving object" alarms on parked cars | a ray skimming an object's edge marked the edge as free space | edge rays do not claim free space (earlier phase) |
| Safety layer chose to hit a parked car rather than a moving object | static hit was priced lower than a predicted dynamic hit | a predicted static hit costs more than any dynamic one |

## 7. Limits - please read

* **Not run in real Webots.** Everything Webots-specific is untested there: the generated world file (checked for structure
  and against the model, not loaded), `getSelf()` / `resetPhysics()` / teleporting the robot, moving the pads and the
  viewpoint, `setCustomData`, `setLabel`, `importMFNodeFromString` (trail). If the robot cannot be placed it falls back to
  the demo mission and prints a warning; HUD, trail and camera follow are decoration and fail silently.
* The e-puck on a plate at height 0.45 / 0.90 m relies on the slab's collision box; if it slides, check the friction.
* People are simple figures, doors are always open, lifts are teleports, people/trolleys never enter rooms.
* Objects faster than ~0.2 m/s arriving right behind the robot can still hit it (sensed only ~45 cm away).
* Offline runs take 1.5-5 s each; a Webots run is real-time (fast-forward helps).

## 8. Project structure

```
worlds/ai_sensor_world.wbt           generated world (do not edit by hand)
controllers/ai_robot_controller/     robot controller: mission, route legs, lifts, device I/O
controllers/obstacle_supervisor/     moves people/trolleys, counts collisions, HUD, trail, mission report
ai/hospital.py                       THE BUILDING: floors, walls, rooms, furniture, people, lifts (single source of truth)
ai/scene_config.py                   robot constants, RANDOM_MISSION / MISSION_SEED / AUTO_CAMERA, DynamicMover
ai/mission.py                        random missions     ai/route_planner.py  A* routes, route follower
ai/navigator.py                      local planner       ai/motion_predictor.py  moving-obstacle tracker + safety layer
ai/ml_path_planner.py  svm_planner.py   Random Forest + SVM      ai/robot_brain.py   full pipeline
ai/perception.py  sensor_fusion.py  decision_engine.py  data_logger.py
tools/generate_world.py              writes the .wbt from hospital.py      tools/check_world.py   validation / debugging
experiments/                         headless_validation, perpendicular_test, benchmark, model_comparison, plots
results/                             CSV + plots of your runs;   sample_offline_run/ = examples
```

## 9. Changing the building

Edit `ai/hospital.py` (rooms, furniture templates, people, lifts, `TROLLEY_NEVER_BRAKES`), then
`python tools/generate_world.py` and `python tools/check_world.py`, and re-open the world in Webots.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `WARNING: range sensor 'rs0' not found` / `could not place the mission` | An old world file is open. Open `worlds/ai_sensor_world.wbt` from this project or re-run `tools/generate_world.py`. |
| `ModuleNotFoundError: numpy / cv2 / sklearn` | `pip install -r requirements.txt` for the Python selected in Webots. OpenCV and scikit-learn are optional: without scikit-learn the ML models stay inactive and the robot still navigates. |
| `AGENT_F1_1 not found` | Regenerate the world (`python tools/generate_world.py`). |
| No HUD / trail / camera jump | Decoration only; the mission still runs. |

Device names (E-puck): `left wheel motor`, `right wheel motor`, `ps0..ps7`, `rs0..rs23`, `camera`, `gps`, `inertial unit`.
"# AI-Robot-Sensor-Webots" 

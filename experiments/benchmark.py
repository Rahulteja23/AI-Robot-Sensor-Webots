"""
benchmark.py
============
Ablation study of the navigation stack on the full traffic scene (randomised car phases, sensor noise,
wheel slip).  The question: what does each part contribute?

    python experiments/benchmark.py            # 12 runs per configuration, ML models off (fast)
    python experiments/benchmark.py 20 --ml    # 20 runs, Random Forest + SVM on (slow)
    python experiments/benchmark.py 12 --fixed # the old fixed START / GOAL instead of random ones

  A  full system, people and trolleys brake for the robot (normal)
  B  A without the moving-obstacle safety layer (the previous version of the robot)
  C  STRESS: nobody ever brakes, full system
  D  STRESS: nobody ever brakes, WITHOUT the safety layer

The perpendicular (T-bone) test lives in experiments/perpendicular_test.py.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import headless_validation as hv

args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if args else 12
USE_ML = "--ml" in sys.argv
RANDOM = False if "--fixed" in sys.argv else None      # None -> scene_config.RANDOM_MISSION (random START/GOAL)
CONFIGS = [
    ("A full system",                  dict(yield_enabled=True,  use_dynamic_safety=True)),
    ("B without safety layer",   dict(yield_enabled=True,  use_dynamic_safety=False)),
    ("C stress: nobody brakes",        dict(yield_enabled=False, use_dynamic_safety=True)),
    ("D stress, without layer",  dict(yield_enabled=False, use_dynamic_safety=False)),
]
summary = {}
print(f"{N} runs per configuration, ML models {'ON' if USE_ML else 'off'}, "
      f"{'fixed' if RANDOM is False else 'random'} START/GOAL per run")
print(f"{'configuration':<32}{'success':>9}{'collisions':>12}{'mean t (s)':>12}{'closest (cm)':>14}")
for name, kw in CONFIGS:
    rs = [hv.run_trial(s, use_ml=USE_ML, random_mission=RANDOM, **kw) for s in range(1, N + 1)]
    ok = sum(r["ok"] for r in rs)
    ts = [r["time"] for r in rs if r["result"] == "success"]
    summary[name] = dict(runs=N, success=ok, collisions=sum(r["collisions"] for r in rs),
                         mean_time_s=round(sum(ts) / max(len(ts), 1), 1),
                         closest_cm=round(min(r["min_clear"] for r in rs) * 100, 1))
    s = summary[name]
    print(f"{name:<32}{ok:>6}/{N:<3}{s['collisions']:>12}{s['mean_time_s']:>12}{s['closest_cm']:>14}")
out = os.path.join(hv.ROOT, "results", "benchmark_summary.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(summary, open(out, "w"), indent=2)
print("saved", out)

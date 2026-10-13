import csv, collections, statistics, sys

import os, glob
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_c = sorted(glob.glob(os.path.join(_ROOT, "results", "simulation_run_*.csv")))
csv_path = sys.argv[1] if len(sys.argv) > 1 else (_c[-1] if _c else None)
if csv_path is None:
    raise SystemExit("No CSV found in results/")

with open(csv_path, encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

total = len(rows)
print("Total rows:", total)

# Find first ML row
first_ml = None
for i, r in enumerate(rows):
    if '(ML)' in r.get('action', ''):
        first_ml = int(r['step'])
        first_ml_idx = i
        break

print("First ML step:", first_ml)
last_step = rows[-1]['step']
print("Last step:", last_step)

heur_rows = [r for r in rows if '(HEURISTIC)' in r.get('action','')]
ml_rows   = [r for r in rows if '(ML)' in r.get('action','')]
print("Heuristic rows:", len(heur_rows))
print("ML rows:", len(ml_rows))

def count_actions(rlist):
    c = collections.Counter()
    for r in rlist:
        a = r['action'].split('(')[0]
        c[a] += 1
    return dict(c)

print("Heuristic actions:", count_actions(heur_rows))
print("ML actions:", count_actions(ml_rows))

h_obs = sum(1 for r in heur_rows if float(r.get('front_val','0') or 0) > 250)
m_obs = sum(1 for r in ml_rows if float(r.get('front_val','0') or 0) > 250)
print("Obstacle events HEURISTIC:", h_obs, " ML:", m_obs)

h_conf = [float(r['confidence']) for r in heur_rows if r.get('confidence')]
m_conf = [float(r['confidence']) for r in ml_rows if r.get('confidence')]
print("Mean confidence HEURISTIC:", round(statistics.mean(h_conf),4), " ML:", round(statistics.mean(m_conf),4))

print("Final goal distance:", rows[-1]['distance_to_goal'])
print("Final position:", rows[-1]['pos_x'], rows[-1]['pos_y'])
print("Sim duration:", rows[-1]['sim_time_s'], "s")

"""
evaluate_obstacle.py
---------------------
Offline comparison on the obstacle task (sim/obstacle_env.py), in the
spirit of evaluate.py: RL policy vs the classical planner, on seeds never
used in training (training used 100000+ / 20000+ / 30000+ seeds).

    python3 evaluate_obstacle.py                          # writes logs/eval_obstacle_summary.json
    python3 evaluate_obstacle.py --policies logs/a.npz logs/b.npz --seeds 7000 100
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from integration.policy_io import load_policy
from sim.obstacle_baseline import ObstaclePotentialFieldPlanner
from sim.obstacle_env import OrcaObstacleNavEnv

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def _episode(args):
    path, seed = args
    if path == "baseline":
        planner = ObstaclePotentialFieldPlanner()
        act = planner.act
    else:
        pol = load_policy(path)[0]
        act = pol.forward
    env = OrcaObstacleNavEnv(seed=seed)
    obs, _ = env.reset()
    done, R = False, 0.0
    while not done:
        obs, r, done, info = env.step(act(obs))
        R += r
    return dict(goal_frac=info["goals_reached"] / info["n_goals"], collided=bool(info["collided"]),
                reward=R, steps=len(env.log), shield=sum(s["overridden"] for s in env.log),
                min_clearance=float(info["min_clearance"]))


def evaluate(path, seeds, pool):
    eps = pool.map(_episode, [(path, s) for s in seeds])
    g = np.array([e["goal_frac"] for e in eps])
    return dict(n=len(eps), goal_fraction=float(g.mean()), all_goals_rate=float((g == 1).mean()),
                collision_rate=float(np.mean([e["collided"] for e in eps])),
                mean_reward=float(np.mean([e["reward"] for e in eps])),
                mean_shield_steps=float(np.mean([e["shield"] for e in eps])),
                median_min_clearance_m=float(np.median([e["min_clearance"] for e in eps])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="*", default=[os.path.join(LOG_DIR, "trained_policy_obstacle.npz"),
                                                      os.path.join(LOG_DIR, "trained_policy_obstacle_bc.npz")])
    ap.add_argument("--seeds", nargs=2, type=int, default=[7000, 100], metavar=("FIRST", "COUNT"))
    ap.add_argument("--out", default=os.path.join(LOG_DIR, "eval_obstacle_summary.json"))
    args = ap.parse_args()
    seeds = range(args.seeds[0], args.seeds[0] + args.seeds[1])
    res = {}
    with mp.get_context("fork").Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for p in ["baseline"] + args.policies:
            name = "classical_planner" if p == "baseline" else os.path.basename(p)
            res[name] = evaluate(p, seeds, pool)
            print(f"{name:40s} {json.dumps({k: round(v, 3) for k, v in res[name].items()})}", flush=True)
    res["seeds"] = [args.seeds[0], args.seeds[1]]
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", args.out)

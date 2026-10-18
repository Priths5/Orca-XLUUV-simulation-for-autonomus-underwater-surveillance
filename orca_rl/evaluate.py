"""
evaluate.py
------------
Runs the trained RL policy AND the classical baseline planner across a set
of evaluation seeds, and writes:
  - logs/eval_trajectories.json  (per-step trajectory logs for BOTH methods,
    consumed directly by dashboard.html for live playback)
  - logs/eval_summary.json       (aggregate metrics table)

Usage:
    python3 evaluate.py                                  # random goal each seed
    python3 evaluate.py --n_seeds 10
    python3 evaluate.py --goal_x 3000 --goal_y 1000       # fixed destination,
                                                            # same goal every seed -
                                                            # use this to test specific
                                                            # scenarios (due-east run,
                                                            # steep diagonal, short vs
                                                            # long range, etc.)
"""

import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sim.orca_env import OrcaNavEnv
from sim.baseline_planner import ClassicalPotentialFieldPlanner
from rl.policy_network import MLPPolicy

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


class NumpyEncoder(json.JSONEncoder):
    """Safety net for any numpy scalar/array types that slip through -
    orca_env.py's info dict is already cast to native Python types, but this
    catches anything added later without needing to remember to cast it."""
    def default(self, o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def load_trained_policy(path, obs_dim=12, act_dim=3):
    data = np.load(path)
    theta = data["theta"]
    sizes = list(data["sizes"])
    hidden = tuple(sizes[1:-1])
    policy = MLPPolicy(obs_dim, act_dim, hidden=hidden)
    policy.set_flat_params(theta)
    return policy


def run_episode_rl(policy, seed, fixed_goal):
    env = OrcaNavEnv(seed=seed, fixed_goal=fixed_goal)
    obs, _ = env.reset()
    done = False
    total_r = 0.0
    while not done:
        action = policy.forward(obs)
        obs, r, done, info = env.step(action)
        total_r += r
    return total_r, env.log, env.goal.tolist()


def run_episode_baseline(seed, fixed_goal):
    env = OrcaNavEnv(seed=seed, fixed_goal=fixed_goal)
    obs, _ = env.reset()
    planner = ClassicalPotentialFieldPlanner(env.params)
    done = False
    total_r = 0.0
    while not done:
        action = planner.act(obs, env.ocean, env.est_x, env.est_y, env.goal, env.uncertainty)
        obs, r, done, info = env.step(action)
        total_r += r
    return total_r, env.log, env.goal.tolist()


def summarize(logs, total_r):
    final = logs[-1]
    dist = float(np.hypot(final["goal_x"] - final["est_x"], final["goal_y"] - final["est_y"]))
    true_dist = float(np.hypot(final["goal_x"] - final["true_x"], final["goal_y"] - final["true_y"]))
    max_unc = float(max(s["uncertainty"] for s in logs))
    mean_risk = float(np.mean([s["risk"] for s in logs]))
    total_overrides = int(sum(1 for s in logs if s["overridden"]))
    return dict(
        total_reward=float(total_r),
        reached_goal=bool(final["reached_goal"]),
        steps=len(logs),
        final_localization_error_est_to_goal=dist,
        final_true_position_error=true_dist,
        max_uncertainty=max_unc,
        mean_risk_exposure=mean_risk,
        final_battery_frac=float(final["battery_frac"]),
        safety_overrides=total_overrides,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--policy_path", type=str,
                     default=os.path.join(LOG_DIR, "trained_policy.npz"))
    ap.add_argument("--dashboard_seed", type=int, default=1,
                     help="Which single seed's full trajectory to export for the live dashboard")
    ap.add_argument("--goal_x", type=float, default=None,
                     help="Fixed goal x-coordinate in meters (relative to start at 0,0). "
                          "If set, --goal_y must also be set, and every seed uses this "
                          "same destination instead of a randomized one.")
    ap.add_argument("--goal_y", type=float, default=None,
                     help="Fixed goal y-coordinate in meters (relative to start at 0,0).")
    args = ap.parse_args()

    if (args.goal_x is None) != (args.goal_y is None):
        raise SystemExit("--goal_x and --goal_y must both be provided together, or neither.")
    fixed_goal = (args.goal_x, args.goal_y) if args.goal_x is not None else None

    os.makedirs(LOG_DIR, exist_ok=True)

    rl_summaries, base_summaries = [], []
    dashboard_export = {}

    have_trained_policy = os.path.exists(args.policy_path)
    if have_trained_policy:
        policy = load_trained_policy(args.policy_path)
    else:
        print(f"WARNING: {args.policy_path} not found - run train.py first. "
              f"Skipping RL evaluation, baseline-only results will be produced.")

    for seed in range(args.n_seeds):
        base_r, base_log, base_goal = run_episode_baseline(seed, fixed_goal)
        base_summaries.append(summarize(base_log, base_r))

        if have_trained_policy:
            rl_r, rl_log, rl_goal = run_episode_rl(policy, seed, fixed_goal)
            rl_summaries.append(summarize(rl_log, rl_r))

        if seed == args.dashboard_seed:
            dashboard_export["baseline"] = {"log": base_log, "goal": base_goal}
            if have_trained_policy:
                dashboard_export["rl"] = {"log": rl_log, "goal": rl_goal}

    def agg(summaries, key):
        return float(np.mean([s[key] for s in summaries])) if summaries else None

    summary_report = {
        "n_seeds": args.n_seeds,
        "fixed_goal": fixed_goal,
        "baseline": {
            "mean_reward": agg(base_summaries, "total_reward"),
            "goal_reach_rate": agg(base_summaries, "reached_goal"),
            "mean_final_localization_error": agg(base_summaries, "final_localization_error_est_to_goal"),
            "mean_max_uncertainty": agg(base_summaries, "max_uncertainty"),
            "mean_risk_exposure": agg(base_summaries, "mean_risk_exposure"),
            "mean_final_battery_frac": agg(base_summaries, "final_battery_frac"),
            "mean_safety_overrides": agg(base_summaries, "safety_overrides"),
            "per_seed": base_summaries,
        },
        "rl": {
            "mean_reward": agg(rl_summaries, "total_reward"),
            "goal_reach_rate": agg(rl_summaries, "reached_goal"),
            "mean_final_localization_error": agg(rl_summaries, "final_localization_error_est_to_goal"),
            "mean_max_uncertainty": agg(rl_summaries, "max_uncertainty"),
            "mean_risk_exposure": agg(rl_summaries, "mean_risk_exposure"),
            "mean_final_battery_frac": agg(rl_summaries, "final_battery_frac"),
            "mean_safety_overrides": agg(rl_summaries, "safety_overrides"),
            "per_seed": rl_summaries,
        } if have_trained_policy else None,
    }

    with open(os.path.join(LOG_DIR, "eval_summary.json"), "w") as f:
        json.dump(summary_report, f, indent=2, cls=NumpyEncoder)

    with open(os.path.join(LOG_DIR, "eval_trajectories.json"), "w") as f:
        json.dump(dashboard_export, f, cls=NumpyEncoder)

    print("Wrote logs/eval_summary.json and logs/eval_trajectories.json")
    print(json.dumps(summary_report["baseline"], indent=2, default=str)[:800])

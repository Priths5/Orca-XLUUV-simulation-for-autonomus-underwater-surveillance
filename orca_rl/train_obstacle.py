"""
train_obstacle.py
------------------
Trains the obstacle-aware, goal-directed Orca policy that drives the
Stonefish Orca (logs/trained_policy_obstacle.npz).

Same methodology as train.py: MLPPolicy (2 x 64 tanh hidden layers, tanh
output), Evolution Strategies (rl/es_trainer.py), the reward terms of
orca_env.py and the safety shield in the loop. What changes is the task
(sim/obstacle_env.py: Stonefish-scale obstacles, goal chain, sonar-only
perception) and therefore the observation layout ("obstacle_v1", 23-d).

ES from a random network learned this task too slowly (8 % of goals after
50 generations), so by default the same network is first warm-started by
behaviour cloning the classical planner (sim/obstacle_baseline.py, one
DAgger round), and ES then optimises the actual reward from there with a
small perturbation scale. --no-bc reproduces pure ES.

    python3 train_obstacle.py                      # BC warm start + 200 ES generations
    python3 train_obstacle.py --generations 20     # quick check
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from rl.es_trainer import train_es
from rl.policy_network import MLPPolicy
from sim.obstacle_baseline import ObstaclePotentialFieldPlanner
from sim.obstacle_env import OrcaObstacleNavEnv

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def env_factory(seed):
    return OrcaObstacleNavEnv(seed=seed)


# ---------------------------------------------------------------- BC warm start
def _rollout(args):
    """one episode; actions from the planner, or from `theta` mixed with the
    planner (DAgger). Every visited observation is labelled by the planner."""
    seed, theta, sizes, beta = args
    rng = np.random.default_rng(seed)
    planner = ObstaclePotentialFieldPlanner()
    pol = None
    if theta is not None:
        pol = MLPPolicy(sizes[0], sizes[-1], hidden=tuple(sizes[1:-1]))
        pol.set_flat_params(theta)
    env = OrcaObstacleNavEnv(seed=seed)
    obs, _ = env.reset()
    X, Y = [], []
    done = False
    while not done:
        target = planner.act(obs)
        X.append(obs)
        Y.append(target)
        a = target if (pol is None or rng.random() < beta) else pol.forward(obs)
        obs, _, done, _ = env.step(a)
    return np.array(X), np.array(Y)


def _fit(policy, X, Y, epochs=60, lr=1e-3, batch=256, seed=0):
    """MSE regression of the tanh MLP onto the planner's actions (Adam)."""
    rng = np.random.default_rng(seed)
    Y = np.clip(Y, -0.97, 0.97)
    W, B = policy.weights, policy.biases
    params = W + B
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    step = 0
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        tot = 0.0
        for k in range(0, len(X), batch):
            b = idx[k:k + batch]
            acts = [X[b]]
            for i, (w, bb) in enumerate(zip(W, B)):
                acts.append(np.tanh(acts[-1] @ w + bb))
            err = acts[-1] - Y[b]
            tot += float((err ** 2).sum())
            g = 2 * err / len(b) * (1 - acts[-1] ** 2)
            gW, gB = [None] * len(W), [None] * len(B)
            for i in range(len(W) - 1, -1, -1):
                gW[i] = acts[i].T @ g
                gB[i] = g.sum(0)
                if i:
                    g = (g @ W[i].T) * (1 - acts[i] ** 2)
            step += 1
            for j, (p, gr) in enumerate(zip(params, gW + gB)):
                m[j] = 0.9 * m[j] + 0.1 * gr
                v[j] = 0.999 * v[j] + 0.001 * gr * gr
                p -= lr * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"[BC] epoch {ep:3d}  mse {tot / len(X) / 3:.4f}", flush=True)
    return policy


def behaviour_clone(sizes, workers, n_episodes=240, dagger_rounds=1):
    import multiprocessing as mp
    policy = MLPPolicy(sizes[0], sizes[-1], hidden=tuple(sizes[1:-1]), seed=3)
    with mp.get_context("fork").Pool(workers) as pool:
        data = pool.map(_rollout, [(20000 + i, None, sizes, 1.0) for i in range(n_episodes)])
        X = np.concatenate([d[0] for d in data]); Y = np.concatenate([d[1] for d in data])
        print(f"[BC] {len(X)} planner samples", flush=True)
        _fit(policy, X, Y)
        for r in range(dagger_rounds):
            th = policy.get_flat_params()
            data = pool.map(_rollout, [(30000 + 1000 * r + i, th, sizes, 0.3) for i in range(n_episodes)])
            X = np.concatenate([X] + [d[0] for d in data]); Y = np.concatenate([Y] + [d[1] for d in data])
            print(f"[BC] DAgger round {r + 1}: {len(X)} samples", flush=True)
            _fit(policy, X, Y, epochs=40)
    return policy


def save_policy(path, theta, sizes, extra):
    np.savez(path, theta=theta, sizes=np.array(sizes), obs_layout=np.array("obstacle_v1"),
             params_class=np.array("StonefishOrcaParams"), meta=np.array(json.dumps(extra)))


def evaluate(policy, seeds):
    goals, coll, steps = [], [], []
    for s in seeds:
        env = OrcaObstacleNavEnv(seed=s)
        obs, _ = env.reset()
        done = False
        while not done:
            obs, r, done, info = env.step(policy.forward(obs))
        goals.append(info["goals_reached"] / info["n_goals"])
        coll.append(info["collided"])
        steps.append(len(env.log))
    return dict(goal_fraction=float(np.mean(goals)), collision_rate=float(np.mean(coll)),
                all_goals_rate=float(np.mean([g == 1.0 for g in goals])), mean_steps=float(np.mean(steps)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=None, help="default 0.03 after BC, 0.1 from scratch")
    ap.add_argument("--lr", type=float, default=None, help="default 0.01 after BC, 0.03 from scratch")
    ap.add_argument("--no-bc", action="store_true", help="pure ES from a random network")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--init", type=str, default=None, help="warm-start from a policy .npz")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    probe = OrcaObstacleNavEnv(seed=0)
    obs_dim = probe.obs_dim
    hidden = (64, 64)
    sizes = [obs_dim, *hidden, 3]
    init_theta = np.load(args.init)["theta"] if args.init else None
    bc_eval = None
    if init_theta is None and not args.no_bc:
        bc = behaviour_clone(sizes, args.workers)
        bc_eval = evaluate(bc, range(5000, 5040))
        print(f"[BC] warm-start policy, held-out eval (40 seeds): {bc_eval}", flush=True)
        save_policy(os.path.join(OUT_DIR, "trained_policy_obstacle_bc.npz"), bc.get_flat_params(), sizes,
                    dict(stage="behaviour_cloning", heldout_eval=bc_eval))
        init_theta = bc.get_flat_params()
    warm = init_theta is not None
    sigma = args.sigma if args.sigma is not None else (0.03 if warm else 0.1)
    lr = args.lr if args.lr is not None else (0.01 if warm else 0.03)
    ckpt = os.path.join(OUT_DIR, "trained_policy_obstacle_ckpt.npz")
    t0 = time.time()

    def on_gen(gen, theta, history):
        if gen % 10 == 0 or gen == args.generations - 1:
            save_policy(ckpt, theta, sizes, dict(generation=gen))
            with open(os.path.join(OUT_DIR, "training_history_obstacle.json"), "w") as f:
                json.dump(history, f)

    policy, history = train_es(env_factory, obs_dim=obs_dim, act_dim=3,
                               n_generations=args.generations, pop_size=args.pop,
                               n_eval_seeds=args.seeds, sigma=sigma, lr=lr,
                               seed=7, hidden=hidden, log_every=5, verbose=True,
                               n_workers=args.workers, common_seeds=True,
                               init_theta=init_theta, on_generation=on_gen)
    elapsed = time.time() - t0
    ev = evaluate(policy, range(5000, 5040))
    print(f"TRAINING COMPLETE in {elapsed:.0f}s; held-out eval (40 seeds): {ev}")
    save_policy(os.path.join(OUT_DIR, "trained_policy_obstacle.npz"), policy.get_flat_params(), sizes,
                dict(generations=args.generations, pop=args.pop, seeds=args.seeds, sigma=sigma,
                     lr=lr, warm_start="bc" if (warm and not args.init) else args.init,
                     bc_heldout_eval=bc_eval, train_time_s=elapsed, heldout_eval=ev))
    with open(os.path.join(OUT_DIR, "training_history_obstacle.json"), "w") as f:
        json.dump(history, f)
    print("saved logs/trained_policy_obstacle.npz")

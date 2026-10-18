import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sim.orca_env import OrcaNavEnv
from rl.es_trainer import train_es
from rl.policy_network import MLPPolicy

OUT_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(OUT_DIR, exist_ok=True)


def env_factory(seed):
    return OrcaNavEnv(seed=seed)


if __name__ == "__main__":
    t0 = time.time()
    policy, history = train_es(
        env_factory, obs_dim=12, act_dim=3,
        n_generations=120,      # increased from 70 - more time to escape local optima
        pop_size=40,            # increased from 24 - more exploration per generation
        n_eval_seeds=3,         # increased from 2 - less noisy fitness estimate
        sigma=0.15,             # increased from 0.08 - more exploration noise, helps
                                 # escape the "stand still to avoid penalties" trap
        lr=0.03, seed=42, hidden=(64, 64),
        log_every=5, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"TRAINING COMPLETE in {elapsed:.1f}s")

    np.savez(os.path.join(OUT_DIR, "trained_policy.npz"),
             theta=policy.get_flat_params(),
             sizes=np.array(policy.sizes))

    with open(os.path.join(OUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f)

    print("Saved trained_policy.npz and training_history.json to", OUT_DIR)

    # Quick sanity check right after training: confirm the policy actually
    # moves rather than converging to "stand still". If avg_speed is near 0
    # across a few episodes, the idle_penalty in orca_env.py or the ES
    # hyperparameters above likely need further tuning.
    print("\n--- post-training sanity check ---")
    for seed in range(3):
        env = OrcaNavEnv(seed=seed)
        obs, _ = env.reset()
        done = False
        speeds = []
        while not done:
            action = policy.forward(obs)
            obs, r, done, info = env.step(action)
            speeds.append(env.true_state.speed)
        print(f"seed {seed}: mean_speed={np.mean(speeds):.2f} m/s, "
              f"reached_goal={info['reached_goal']}, steps={len(speeds)}")

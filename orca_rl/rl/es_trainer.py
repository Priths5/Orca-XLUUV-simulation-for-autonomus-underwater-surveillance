"""
es_trainer.py
--------------
Evolution Strategies (ES) training loop, used here in place of PPO because
this sandboxed environment has no network access to install torch,
gymnasium, or stable-baselines3 (verified: pip cannot reach the package
index for anything not already pre-installed).

ES is a legitimate, published alternative to policy-gradient RL for
continuous control (Salimans et al., "Evolution Strategies as a Scalable
Alternative to Reinforcement Learning", OpenAI, 2017) and requires no
autodiff/backprop, making it tractable to implement correctly in pure
NumPy. On a machine with full internet access, this module's role is a
drop-in placeholder for `stable_baselines3.PPO` - the environment
(`orca_env.py`) already exposes a standard reset/step API compatible with
wrapping into a real Gymnasium env for that purpose.
"""

import numpy as np
import time


def evaluate_policy(env_factory, policy, seed, max_steps_override=None):
    env = env_factory(seed=seed)
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = policy.forward(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward, env.log


# ---- optional parallel evaluation (same algorithm, just spread over processes) ----
_WORKER = {}


def _worker_init(env_factory, obs_dim, act_dim, hidden):
    from .policy_network import MLPPolicy
    _WORKER["env_factory"] = env_factory
    _WORKER["policy"] = MLPPolicy(obs_dim, act_dim, hidden=hidden)


def _worker_eval(args):
    theta, seeds = args
    pol = _WORKER["policy"]
    pol.set_flat_params(theta)
    return float(np.mean([evaluate_policy(_WORKER["env_factory"], pol, seed=s)[0] for s in seeds]))


def train_es(env_factory, obs_dim, act_dim, n_generations=60, pop_size=48,
             sigma=0.08, lr=0.03, seed=42, hidden=(64, 64), n_eval_seeds=3,
             log_every=5, verbose=True, n_workers=1, common_seeds=False,
             init_theta=None, on_generation=None):
    """
    Standard ES loop: sample population of parameter perturbations,
    evaluate fitness (mean episodic return across a few seeds), update
    parameters in the direction of the fitness-weighted average
    perturbation (antithetic sampling for variance reduction).

    Optional (defaults reproduce the original behaviour exactly):
      n_workers     evaluate the population in that many processes
      common_seeds  every candidate in a generation is scored on the SAME
                    evaluation seeds (common random numbers), so rank-based
                    fitness compares policies, not episode luck
      init_theta    warm-start parameters
      on_generation callback(gen, theta, history) e.g. for checkpoints
    """
    from .policy_network import MLPPolicy

    rng = np.random.default_rng(seed)
    base_policy = MLPPolicy(obs_dim, act_dim, hidden=hidden, seed=seed)
    if init_theta is not None:
        base_policy.set_flat_params(np.asarray(init_theta, dtype=float).copy())
    theta = base_policy.get_flat_params()
    n_params = theta.size
    pool = None
    if n_workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(n_workers, initializer=_worker_init,
                                           initargs=(env_factory, obs_dim, act_dim, hidden))

    history = {"generation": [], "mean_fitness": [], "best_fitness": [], "time_s": []}
    t0 = time.time()

    half_pop = pop_size // 2
    for gen in range(n_generations):
        noise = rng.normal(0, 1, size=(half_pop, n_params))
        noise_full = np.concatenate([noise, -noise], axis=0)  # antithetic pairs
        fitnesses = np.zeros(pop_size)

        def seeds_for(i):
            if common_seeds:
                return [100000 + 1000 * gen + k for k in range(n_eval_seeds)]
            return [1000 * gen + 10 * i + k for k in range(n_eval_seeds)]

        if pool is not None:
            jobs = [(theta + sigma * noise_full[i], seeds_for(i)) for i in range(pop_size)]
            fitnesses[:] = pool.map(_worker_eval, jobs)
        else:
            for i in range(pop_size):
                candidate_theta = theta + sigma * noise_full[i]
                base_policy.set_flat_params(candidate_theta)
                rets = []
                for k in seeds_for(i):
                    r, _ = evaluate_policy(env_factory, base_policy, seed=k)
                    rets.append(r)
                fitnesses[i] = np.mean(rets)

        # Fitness shaping: rank-based standardization (robust to outliers)
        ranks = np.argsort(np.argsort(fitnesses))
        shaped = (ranks / (pop_size - 1)) - 0.5

        grad_est = (noise_full * shaped[:, None]).mean(axis=0)
        theta = theta + lr * grad_est / (sigma + 1e-8)

        best_fitness = fitnesses.max()
        mean_fitness = fitnesses.mean()
        history["generation"].append(gen)
        history["mean_fitness"].append(mean_fitness)
        history["best_fitness"].append(best_fitness)
        history["time_s"].append(time.time() - t0)

        if verbose and (gen % log_every == 0 or gen == n_generations - 1):
            print(f"[ES] gen {gen:3d}/{n_generations}  mean_fit={mean_fitness:8.2f}  "
                  f"best_fit={best_fitness:8.2f}  elapsed={time.time()-t0:5.1f}s")

        if on_generation is not None:
            on_generation(gen, theta, history)

    if pool is not None:
        pool.close()
        pool.join()
    base_policy.set_flat_params(theta)
    return base_policy, history

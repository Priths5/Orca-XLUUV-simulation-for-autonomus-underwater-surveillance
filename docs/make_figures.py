#!/usr/bin/env python3
"""
make_figures.py - regenerates every figure used in docs/Orca_RL_Stonefish_Review.pptx
and docs/report/ from the logged data (no simulator needed; the training-env
example episode is re-simulated with the saved policies).

    python3 docs/make_figures.py            # -> docs/figures/*.png

Inputs (all inside this folder tree):
    orca_rl/logs/training_history.json            original ES training (legacy 12-d policy)
    orca_rl/logs/eval_summary.json                original RL vs baseline evaluation
    orca_rl/logs/trained_policy.npz               original policy (probed here)
    orca_rl/logs/training_history_obstacle.json   BC-warm-started ES training (obstacle policy)
    orca_rl/logs/eval_obstacle_summary.json       100-seed held-out test
    orca_rl/logs/sysid_stonefish.json             Stonefish system identification
    orca_rl/logs/stonefish_runs/*.jsonl|_summary  closed-loop Stonefish runs
    docs/data/*.log|jsonl                         capsize before/after, setpoint tracking, legacy run
    stonefish_sim/resources/orca_rl_world_obstacles.json
"""
import fnmatch
import glob
import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Wedge

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RL = os.path.join(ROOT, "orca_rl")
LOGS = os.path.join(RL, "logs")
DATA = os.path.join(ROOT, "docs", "data")
OUT = os.path.join(ROOT, "docs", "figures")
sys.path.insert(0, RL)
os.makedirs(OUT, exist_ok=True)

# one palette for every figure
C_RL = "#0F766E"        # learned policy
C_BASE = "#C2410C"      # classical planner
C_BC = "#64748B"        # behaviour cloning
C_EST = "#1D4ED8"       # estimate
C_GT = "#16A34A"        # ground truth
C_ROCK = "#8A6D3B"
C_GRID = "#E5E7EB"
C_DARK = "#0F172A"
C_MUTED = "#475569"
WORLD_NAME = {"rl": "obstacle world", "survey": "survey world", "multisub": "multi-sub world"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": C_GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def load_json(p):
    with open(p) as f:
        return json.load(f)


def latest(pattern):
    """Pinned evaluation run (docs/data/final_runs.json) if one matches, else the newest file."""
    runs = os.path.join(LOGS, "stonefish_runs")
    pinned = os.path.join(DATA, "final_runs.json")
    if os.path.exists(pinned):
        for stem in load_json(pinned).values():
            for suffix in (".jsonl", "_summary.json", ".png"):
                if fnmatch.fnmatch(stem + suffix, pattern) and os.path.exists(os.path.join(runs, stem + suffix)):
                    return os.path.join(runs, stem + suffix)
    fs = sorted(glob.glob(os.path.join(runs, pattern)))
    return fs[-1] if fs else None


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def smooth(y, k=9):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


# --------------------------------------------------------------------------- RL part
def fig_original_training():
    h = load_json(os.path.join(LOGS, "training_history.json"))
    g = np.array(h["generation"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(g, h["mean_fitness"], color=C_RL, alpha=0.25, lw=1)
    ax.plot(g, smooth(h["mean_fitness"]), color=C_RL, lw=2.4, label="population mean (smoothed)")
    ax.plot(g, smooth(h["best_fitness"]), color=C_BASE, lw=1.8, ls="--", label="best candidate (smoothed)")
    ax.set_xlabel("ES generation")
    ax.set_ylabel("episode return")
    ax.set_title("Original policy: Evolution Strategies training (12-d observation)")
    ax.legend(loc="lower right")
    save(fig, "01_original_es_training.png")


def fig_original_policy_probe():
    from integration.policy_io import load_policy
    pol, _, _, _ = load_policy(os.path.join(LOGS, "trained_policy.npz"))
    hs = np.linspace(-math.pi, math.pi, 73)
    acts = []
    for h in hs:
        o = np.array([0, 0, 0.04, math.sin(h), math.cos(h), 0.0, 1, 0, 0, 0.96, 0.3, 0.2], dtype=np.float32)
        acts.append(pol.forward(o))
    acts = np.array(acts)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    ax.plot(np.degrees(hs), acts[:, 0], color=C_EST, lw=2.2, label="a0  speed")
    ax.plot(np.degrees(hs), acts[:, 1], color=C_BASE, lw=2.6, label="a1  yaw rate")
    ax.plot(np.degrees(hs), acts[:, 2], color=C_RL, lw=2.2, label="a2  depth rate")
    ax.axhline(1, color=C_MUTED, lw=0.8, ls=":")
    ax.axhline(-1, color=C_MUTED, lw=0.8, ls=":")
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("vehicle heading fed to the policy [deg]")
    ax.set_ylabel("policy output")
    ax.set_title("Probe of the shipped policy: yaw output saturated at +1")
    ax.legend(loc="lower left", ncol=3, fontsize=9.5)
    # legacy run in Stonefish
    recs = read_jsonl(os.path.join(DATA, "legacy_policy_rl_run.jsonl"))
    gt = np.array([r["gt_eval_only"] for r in recs if "gt_eval_only" in r])
    est = np.array([r["est"] for r in recs])
    ax = axs[1]
    ax.plot(gt[:, 1], gt[:, 0], color=C_GT, lw=2.5, label="ground truth")
    ax.plot(est[:, 1], est[:, 0], color=C_EST, lw=1.2, ls="--", label="estimate")
    ax.scatter([-60], [-80], marker="*", s=180, color=C_BASE, zorder=5, label="first survey waypoint")
    ax.scatter([gt[0, 1]], [gt[0, 0]], s=40, color=C_DARK, zorder=5, label="start")
    ax.set_aspect("equal")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_title("Same policy in Stonefish (200 s): it circles")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "02_original_policy_probe.png")


def fig_original_eval():
    s = load_json(os.path.join(LOGS, "eval_summary.json"))
    keys = [("goal_reach_rate", "goal-reach rate"), ("mean_final_battery_frac", "final battery"),
            ("mean_risk_exposure", "risk exposure")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(keys))
    b = [s["baseline"][k] for k, _ in keys]
    r = [s["rl"][k] for k, _ in keys]
    ax.bar(x - 0.2, b, 0.38, color=C_BASE, label="classical planner")
    ax.bar(x + 0.2, r, 0.38, color=C_RL, label="original RL policy")
    for i in range(len(keys)):
        ax.text(x[i] - 0.2, b[i] + 0.02, f"{b[i]:.2f}", ha="center", fontsize=10)
        ax.text(x[i] + 0.2, r[i] + 0.02, f"{r[i]:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x, [l for _, l in keys])
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Shipped evaluation (fixed goal, {s['n_seeds']} seeds): RL reaches 0 goals")
    ax.set_ylim(0, 1.3)
    ax.legend(loc="upper right", ncol=2, fontsize=10)
    save(fig, "03_original_eval.png")


def fig_observation_layout():
    groups = [
        ("original 12 features (kept, same order)", C_EST, [
            "north est.", "east est.", "depth", "sin hdg", "cos hdg", "uncert.", "battery", "thr. wear",
            "nav resid.", "depth margin", "gradient", "risk"]),
        ("goal", C_BASE, ["sin brg", "cos brg", "goal dist"]),
        ("sonar sectors", C_RL, ["-64°", "-32°", "0°", "+32°", "+64°"]),
        ("flanks", "#7C3AED", ["port", "stbd"]),
        ("depth", C_MUTED, ["Δdepth"]),
    ]
    fig, ax = plt.subplots(figsize=(13, 2.9))
    ax.axis("off")
    x = 0
    for title, col, items in groups:
        x0 = x
        for it in items:
            ax.add_patch(FancyBboxPatch((x, 0), 0.92, 1, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc=col, ec="white", alpha=0.9))
            ax.text(x + 0.46, 0.5, it, ha="center", va="center", color="white", fontsize=8.6,
                    rotation=90, fontweight="bold")
            x += 1
        ax.text((x0 + x - 0.08) / 2, 1.25, title, ha="center", va="bottom", fontsize=10, color=col,
                fontweight="bold")
        x += 0.6
    ax.set_xlim(-0.2, x)
    ax.set_ylim(-0.25, 1.75)
    ax.text(0, -0.2, "Obstacle-aware observation: 23 inputs to the same 2 x 64 tanh network (the original policy used the first 12)",
            fontsize=10, color=C_MUTED, va="top")
    save(fig, "04_observation_layout.png")


def _run_env_episode(act_fn, seed):
    from sim.obstacle_env import OrcaObstacleNavEnv
    env = OrcaObstacleNavEnv(seed=seed)
    obs, _ = env.reset()
    sx, sy, sr = env.sx.copy(), env.sy.copy(), env.sr.copy()
    m0 = (env.mx.copy(), env.my.copy(), env.mr.copy())
    goals = [g.copy() for g in env.goals]
    traj, mtraj = [(env.x, env.y)], [(env.mx.copy(), env.my.copy())]
    done = False
    while not done:
        obs, r, done, info = env.step(act_fn(obs))
        traj.append((env.x, env.y))
        mtraj.append((env.mx.copy(), env.my.copy()))
    return dict(static=(sx, sy, sr), movers0=m0, mtraj=mtraj, goals=goals, traj=np.array(traj), info=info,
                start=traj[0])


def fig_training_env_episode():
    from integration.policy_io import load_policy
    from sim.obstacle_baseline import ObstaclePotentialFieldPlanner
    pol = load_policy(os.path.join(LOGS, "trained_policy_obstacle.npz"))[0]
    planner = ObstaclePotentialFieldPlanner()
    # pick a held-out seed where the two differ in an instructive way
    chosen = None
    for seed in range(7000, 7060):
        a = _run_env_episode(pol.forward, seed)
        b = _run_env_episode(planner.act, seed)
        if a["info"]["goals_reached"] == a["info"]["n_goals"] and not a["info"]["collided"] and b["info"]["collided"]:
            chosen = (seed, a, b)
            break
    if chosen is None:
        chosen = (7000, _run_env_episode(pol.forward, 7000), _run_env_episode(planner.act, 7000))
    seed, a, b = chosen
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 6))
    for ax, ep, name, col in [(axs[0], b, "classical planner", C_BASE), (axs[1], a, "RL policy (BC + ES)", C_RL)]:
        sx, sy, sr = ep["static"]
        for x, y, r in zip(sx, sy, sr):
            ax.add_patch(Circle((y, x), r, color=C_ROCK, alpha=0.85))
        mx = np.array([m[0] for m in ep["mtraj"]])
        my = np.array([m[1] for m in ep["mtraj"]])
        for k in range(mx.shape[1] if mx.ndim == 2 else 0):
            ax.plot(my[:, k], mx[:, k], color="#38BDF8", lw=1, ls=":")
            ax.add_patch(Circle((my[-1, k], mx[-1, k]), ep["movers0"][2][k], color="#38BDF8", alpha=0.6))
        g = np.array(ep["goals"])
        ax.scatter(g[:, 1], g[:, 0], marker="*", s=220, color=C_DARK, zorder=5, label="goals")
        ax.scatter([ep["start"][1]], [ep["start"][0]], s=60, color=C_DARK, marker="o", zorder=5, label="start")
        t = ep["traj"]
        ax.plot(t[:, 1], t[:, 0], color=col, lw=2.4, label="hull centre track")
        info = ep["info"]
        status = "COLLIDED" if info["collided"] else ("all goals" if info["goals_reached"] == info["n_goals"] else "timeout")
        ax.set_title(f"{name}: {info['goals_reached']}/{info['n_goals']} goals, {status}")
        ax.set_aspect("equal")
        ax.set_xlabel("east [m]")
        ax.set_ylabel("north [m]")
        pad = 25
        allx = np.concatenate([t[:, 1], g[:, 1]])
        ally = np.concatenate([t[:, 0], g[:, 0]])
        ax.set_xlim(allx.min() - pad, allx.max() + pad)
        ax.set_ylim(ally.min() - pad, ally.max() + pad)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(f"Training environment, held-out seed {seed} (rocks brown, moving fish/subs blue)",
                 fontsize=12, color=C_MUTED)
    save(fig, "05_training_env_episode.png")


def fig_perception():
    from integration import perception as P
    from integration.policy_io import load_policy
    from sim.obstacle_env import OrcaObstacleNavEnv
    pol = load_policy(os.path.join(LOGS, "trained_policy_obstacle.npz"))[0]
    env = None
    for seed in range(7000, 7040):
        env = OrcaObstacleNavEnv(seed=seed)
        obs, _ = env.reset()
        found = False
        for _ in range(300):
            obs, r, d, info = env.step(pol.forward(obs))
            if d:
                break
            near = (obs[15:20] * 30 < 22).sum()
            if near >= 2 and env.memory.pts.shape[0] > 15 and min(obs[20:22]) * 30 < 20:
                found = True
                break
        if found:
            break
    fig, ax = plt.subplots(figsize=(8.5, 7))
    c, s = math.cos(env.heading), math.sin(env.heading)
    bx, by = env.x + P.PROFILER_MOUNT_X * c, env.y + P.PROFILER_MOUNT_X * s
    edges = np.degrees(np.linspace(-0.5 * P.PROFILER_FOV, 0.5 * P.PROFILER_FOV, P.N_SECTORS + 1))
    cl = obs[15:20] * 30
    hdg_plot = 90 - math.degrees(env.heading)   # plot angle (east=x, north=y)
    for k in range(P.N_SECTORS):
        a0 = hdg_plot - edges[k + 1]
        a1 = hdg_plot - edges[k]
        ax.add_patch(Wedge((by, bx), 30, a0, a1, color="#99F6E4", alpha=0.35, ec="white"))
        ax.add_patch(Wedge((by, bx), cl[k], a0, a1, color=C_RL, alpha=0.55, ec="white"))
        mid = math.radians((a0 + a1) / 2)
        ax.text(by + 33 * math.cos(mid), bx + 33 * math.sin(mid), f"{cl[k]:.0f} m", ha="center",
                va="center", fontsize=9, color=C_RL, fontweight="bold")
    for x, y, r in zip(*env._circles()):
        ax.add_patch(Circle((y, x), r, color=C_ROCK, alpha=0.85))
    if env.memory.pts.shape[0]:
        ax.scatter(env.memory.pts[:, 1], env.memory.pts[:, 0], s=8, color="#7C3AED", zorder=4,
                   label="remembered sonar returns (estimated pose)")
    L1, L2 = 13.3, 13.85
    ax.plot([env.y - L1 * s, env.y + L2 * s], [env.x - L1 * c, env.x + L2 * c], color=C_DARK, lw=7,
            solid_capstyle="round", label="Orca hull (27 m)")
    ang = np.radians(np.linspace(-80, 80, 33))
    for a in ang:
        a_world = env.heading + a
        d = env._raycast(bx, by, a_world)
        ax.plot([by, by + d * math.sin(a_world)], [bx, bx + d * math.cos(a_world)], color="#0EA5E9", lw=0.5,
                alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(by - 45, by + 45)
    ax.set_ylim(bx - 40, bx + 45)
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_title("Sonar perception: swept profiling sonar -> 5 sector clearances + flank memory")
    ax.legend(loc="lower left", fontsize=9)
    save(fig, "06_perception.png")


def _box(ax, xy, w, h, text, fc, fs=10, tc="white"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="none"))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold", wrap=True)


def _arrow(ax, p0, p1, col=C_MUTED, text=None, fs=9, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, color=col, lw=1.6,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.12, text, ha="center", va="bottom", fontsize=fs,
                color=col)


def fig_training_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.axis("off")
    _box(ax, (0, 1), 2.5, 1.1, "Classical\nplanner", C_BASE, fs=9.5)
    _box(ax, (3.3, 1), 2.5, 1.1, "Behaviour\ncloning (111k)", C_BC, fs=9.5)
    _box(ax, (6.6, 1), 2.5, 1.1, "DAgger round\n(+25k)", C_BC, fs=9.5)
    _box(ax, (9.9, 1), 2.5, 1.1, "ES, 200 gen\n48 x 4 seeds", C_RL, fs=9.5)
    _box(ax, (13.2, 1), 2.3, 1.1, "Validation\n100 seeds", C_DARK, fs=9.5)
    for x0, x1, t in [(2.5, 3.3, "demos"), (5.8, 6.6, "policy"), (9.1, 9.9, "init θ"), (12.4, 13.2, "θ")]:
        _arrow(ax, (x0, 1.55), (x1, 1.55), text=t)
    ax.text(0, 0.45, "same MLP (23-64-64-3, tanh), same ES trainer, same reward terms + collision penalty, "
                     "safety shield in the loop, 14 worker processes (65 min)", fontsize=10, color=C_MUTED)
    ax.set_xlim(-0.2, 15.7)
    ax.set_ylim(0.2, 2.4)
    save(fig, "07_training_pipeline.png")


def fig_es_obstacle_training():
    h = load_json(os.path.join(LOGS, "training_history_obstacle.json"))
    ev = load_json(os.path.join(LOGS, "eval_obstacle_summary.json"))
    g = np.array(h["generation"])
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.plot(g, h["mean_fitness"], color=C_RL, alpha=0.25, lw=1)
    ax.plot(g, smooth(h["mean_fitness"], 15), color=C_RL, lw=2.5, label="population mean (smoothed)")
    ax.plot(g, smooth(h["best_fitness"], 15), color=C_BASE, lw=1.8, ls="--", label="best candidate (smoothed)")
    ax.axhline(ev["classical_planner"]["mean_reward"], color=C_BASE, lw=1, ls=":",
               label=f"planner, held-out ({ev['classical_planner']['mean_reward']:.0f})")
    ax.axhline(ev["trained_policy_obstacle.npz"]["mean_reward"], color=C_RL, lw=1, ls=":",
               label=f"selected RL policy, held-out ({ev['trained_policy_obstacle.npz']['mean_reward']:.0f})")
    ax.set_xlabel("ES generation (after the behaviour-cloning warm start)")
    ax.set_ylabel("mean episode return")
    ax.set_title("Obstacle-aware policy: ES fine-tuning of the cloned network")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "08_es_obstacle_training.png")


def fig_offline_eval():
    ev = load_json(os.path.join(LOGS, "eval_obstacle_summary.json"))
    names = [("classical_planner", "Classical\nplanner", C_BASE), ("trained_policy_obstacle_bc.npz", "BC only", C_BC),
             ("trained_policy_obstacle.npz", "RL\n(BC + ES)", C_RL)]
    metrics = [("goal_fraction", "goals reached", 100, "%"), ("all_goals_rate", "all 3 goals", 100, "%"),
               ("collision_rate", "collision rate", 100, "%"), ("median_min_clearance_m", "median min clearance", 1, " m")]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.9))
    for ax, (k, lab, sc, unit) in zip(axs, metrics):
        vals = [ev[n][k] * sc for n, _, _ in names]
        bars = ax.bar(range(3), vals, color=[c for _, _, c in names], width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.03, f"{v:.0f}{unit}" if unit == "%" else f"{v:.1f}{unit}",
                    ha="center", fontsize=10.5, fontweight="bold")
        ax.set_xticks(range(3), [l for _, l, _ in names], fontsize=9.5)
        ax.set_title(lab)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Held-out test: {ev['classical_planner']['n']} unseen obstacle episodes (seeds {ev['seeds'][0]}+)",
                 fontsize=12, color=C_MUTED, y=1.03)
    save(fig, "09_offline_eval.png")


# --------------------------------------------------------------------------- Stonefish part
def fig_sysid():
    s = load_json(os.path.join(LOGS, "sysid_stonefish.json"))
    sur = [x for x in s["surge"] if x["force"] <= 30000]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    F = np.array([x["force"] for x in sur]) / 1000
    U = np.array([x["u_ss"] for x in sur])
    uu = np.linspace(0, 1.2, 100)
    ax = axs[0]
    ax.plot(9000 * uu ** 2 / 1000 + 23500 * uu / 1000, uu, color=C_RL, lw=2.2,
            label="fit F = 9000 u² + 23500 u")
    ax.scatter(F, U, s=70, color=C_BASE, zorder=5, label="measured steady state")
    ax.axvline(40, color=C_MUTED, ls=":", lw=1)
    ax.text(39, 0.1, "40 kN limit\n-> ~1.15 m/s", fontsize=9, color=C_MUTED, ha="right")
    ax.set_xlabel("main thruster force [kN]")
    ax.set_ylabel("surge speed [m/s]")
    ax.set_title("Surge: identified on the Stonefish Orca")
    ax.legend(loc="upper left", fontsize=9)
    y = [x for x in s["yaw"] if x["surge_force"] > 0]
    ax = axs[1]
    Fy = np.array([x["yaw_force"] for x in y]) / 1000
    R = np.degrees([x["r_ss"] for x in y])
    ax.plot([0, 8.5], [0, math.degrees(8500 / 150000)], color=C_RL, lw=2.2, label="F ≈ 150 000 · r")
    ax.scatter(Fy, R, s=70, color=C_BASE, zorder=5, label="measured (surge 20 kN)")
    ax.scatter([8], [math.degrees(0.037)], s=70, marker="s", color=C_EST, zorder=5,
               label="closed loop at 0.87 m/s")
    ax.set_xlabel("yaw thruster force [kN]")
    ax.set_ylabel("yaw rate [deg/s]")
    ax.set_title("Yaw: authority drops with speed")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "10_sysid.png")


def _parse_attitude(path):
    t, roll, pitch, z = [], [], [], []
    for line in open(path):
        m = re.search(r"t=(\d+).*?xyz=\S+ \S+ (\S+) rpy=(\S+) (\S+)", line)
        if m:
            t.append(float(m.group(1))); z.append(float(m.group(2)))
            roll.append(float(m.group(3))); pitch.append(float(m.group(4)))
    return np.array(t), np.array(roll), np.array(pitch), np.array(z)


def fig_trim():
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axs[0]
    for f, lab, col, ls in [("attitude_before_trim.log", "original model", C_BASE, "-"),
                            ("attitude_after_trim.log", "trimmed model", C_RL, "-")]:
        t, roll, pitch, _ = _parse_attitude(os.path.join(DATA, f))
        m = t <= 200
        ax.plot(t[m], np.abs(roll[m]), color=col, lw=2.4, ls=ls, label=f"|roll|  {lab}")
        ax.plot(t[m], np.abs(pitch[m]), color=col, lw=1.4, ls="--", label=f"|pitch| {lab}")
    ax.set_xlabel("simulation time [s]  (C++ survey autopilot, unchanged)")
    ax.set_ylabel("attitude [deg]")
    ax.set_title("The original Orca capsized in every scenario")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
    ax = axs[1]
    ax.axis("off")
    rows = [("", "original", "trimmed"), ("mass [kg]", "53 370", "53 625"), ("buoyancy [kg]", "53 411", "53 667"),
            ("CG below CB [m]", "-0.05", "+0.08"), ("CG - CB fore-aft [m]", "+0.05", "0.00"),
            ("sail / mast", "Composite", "Syntactic foam"), ("keel ballast", "-", "2 769 kg @ z=+1.05")]
    for i, r in enumerate(rows):
        for j, cell in enumerate(r):
            ax.text([0.0, 0.48, 0.78][j], 0.92 - i * 0.13, cell, fontsize=10,
                    fontweight="bold" if i == 0 or j == 0 else "normal",
                    color=[C_DARK, C_BASE, C_RL][j] if i > 0 and j > 0 else C_DARK)
    ax.set_title("Mass properties read from Stonefish")
    save(fig, "11_trim.png")


def fig_tracking():
    t, u = [], []
    for line in open(os.path.join(DATA, "setpoint_tracking.log")):
        m = re.search(r"t=(\d+) src=(\d).*? u=(\S+)", line)
        if m:
            t.append(float(m.group(1))); u.append(float(m.group(3)))
    t, u = np.array(t), np.array(u)
    m = t <= 61
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(t[m], u[m], color=C_RL, lw=2.4, label="measured surge speed (DVL)")
    ax.step([0, 13, 13, 61], [0, 0, 0.8, 0.8], color=C_BASE, lw=1.8, ls="--", where="post",
            label="speed setpoint sent by the RL node")
    ax.set_xlabel("simulation time [s]")
    ax.set_ylabel("speed [m/s]")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("C++ command adapter: 0.8 m/s setpoint held at 0.81-0.85 m/s")
    ax.legend(loc="lower right", fontsize=9.5)
    save(fig, "12_setpoint_tracking.png")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 3.3), 6.2, 3.3, boxstyle="round,pad=0.05,rounding_size=0.2", fc="#ECFDF5",
                                ec=C_RL, lw=1.5))
    ax.text(0.2, 6.35, "Python RL node  (1 Hz, simulation time)", fontsize=11.5, fontweight="bold", color=C_RL)
    _box(ax, (0.3, 5.15), 2.8, 0.85, "Navigation filter\nDVL x IMU + particle filter", C_EST, fs=9)
    _box(ax, (3.25, 5.15), 2.7, 0.85, "Sonar perception\nsectors + flank memory", "#7C3AED", fs=9)
    _box(ax, (0.3, 4.05), 2.8, 0.85, "VehicleHealth +\nEnvironmentHealth", C_MUTED, fs=9)
    _box(ax, (3.25, 4.05), 2.7, 0.85, "Observation builder\n23 inputs", C_EST, fs=9)
    _box(ax, (0.3, 3.45), 5.65, 0.5, "Policy network -> scaling -> safety shield -> setpoint", C_RL, fs=9)
    ax.add_patch(FancyBboxPatch((7.3, 0.2), 6.0, 6.4, boxstyle="round,pad=0.05,rounding_size=0.2", fc="#EFF6FF",
                                ec=C_EST, lw=1.5))
    ax.text(7.5, 6.35, "C++ Stonefish application (ROS 2-aware)", fontsize=11.5, fontweight="bold", color=C_EST)
    _box(ax, (7.6, 5.15), 5.4, 0.85, "ROS 2 executor thread\nsubscribers + services -> mutex slots", C_MUTED, fs=9)
    _box(ax, (7.6, 3.95), 2.6, 0.95, "Command adapter\n(300 Hz)", C_EST, fs=9)
    _box(ax, (10.4, 3.95), 2.6, 0.95, "Vehicle controller\n(autopilot off)", C_EST, fs=9)
    _box(ax, (7.6, 2.75), 5.4, 0.9, "Main thruster · yaw thruster · heave thruster", C_DARK, fs=9.5)
    _box(ax, (7.6, 1.55), 5.4, 0.9, "Stonefish physics (Bullet, 300 Hz)\nOrca, rocks, fish, other subs", "#0E7490", fs=9)
    _box(ax, (7.6, 0.4), 5.4, 0.85, "IMU, DVL, pressure, GPS, profiling sonar,\nmultibeam sonar, contacts", C_GT, fs=9)
    _arrow(ax, (6.2, 3.7), (7.6, 4.4), col=C_RL, rad=-0.1)
    ax.text(6.85, 4.55, "motion\nsetpoint", fontsize=9, color=C_RL, ha="center")
    _arrow(ax, (7.6, 0.8), (4.6, 3.35), col=C_GT, rad=0.3)
    ax.text(4.3, 0.25, "inertial, velocity, depth and sonar data,\nsimulation clock", fontsize=9,
            color=C_GT, ha="center")
    _arrow(ax, (10.2, 4.4), (10.4, 4.4))
    _arrow(ax, (11.7, 3.95), (11.7, 3.65))
    _arrow(ax, (10.3, 2.75), (10.3, 2.45))
    _arrow(ax, (10.3, 1.55), (10.3, 1.25))
    ax.text(0.2, 2.9, "Only Stonefish integrates vehicle physics at run time.\n"
                      "The fast NumPy dynamics are used for offline training only.\n"
                      "The simulator's ground-truth pose is logged for evaluation\n"
                      "and used as the estimate only in a debug mode.",
            fontsize=10, color=C_MUTED, va="top")
    ax.set_xlim(-0.2, 13.5)
    ax.set_ylim(0, 6.8)
    save(fig, "13_architecture.png")


def fig_rl_world_layout():
    from integration.missions import Mission
    ob = load_json(os.path.join(ROOT, "stonefish_sim", "resources", "orca_rl_world_obstacles.json"))
    fig, ax = plt.subplots(figsize=(8, 7.5))
    for p in ob["pinnacles"]:
        ax.add_patch(Circle((p["y"], p["x"]), p["r"], color=C_ROCK))
    for w in ob["walls"]:
        c, s = math.cos(w["yaw"]), math.sin(w["yaw"])
        hx, hy = -s * w["length"] / 2, c * w["length"] / 2
        ax.plot([w["y"] - hy, w["y"] + hy], [w["x"] - hx, w["x"] + hx], color="#374151", lw=5)
    for b in ob["boulders"]:
        ax.add_patch(Circle((b["y"], b["x"]), b["approx_r"], color="#B39B72", alpha=0.7))
    for f in ob["fish"]:
        ax.add_patch(Ellipse((f["cy"], f["cx"]), 2 * f["ry"], 2 * f["rx"], fill=False, ls=":", lw=1.6,
                             color="#0EA5E9"))
    legs = [(-50, -40), (50, -40), (50, 25), (-50, 25), (-50, -40)]
    ax.plot([l[0] for l in legs], [l[1] for l in legs], color="#7C3AED", lw=1.5, ls="--", label="TrafficSub legs")
    wps = Mission("rl").waypoints
    ax.plot([w[1] for w in wps], [w[0] for w in wps], color=C_MUTED, lw=1.2, ls="--", label="Orca survey legs")
    ax.scatter([w[1] for w in wps], [w[0] for w in wps], s=22, color=C_MUTED, zorder=4)
    ax.scatter([-60], [-95], marker="o", s=70, color=C_DARK, zorder=5, label="Orca start")
    ax.plot([], [], color=C_ROCK, lw=8, label="pinnacles (seabed -> 4 m depth)")
    ax.plot([], [], color="#0EA5E9", ls=":", label="fish-school loops (~2 m/s)")
    ax.set_aspect("equal")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_title("The base world with obstacles")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    save(fig, "14_rl_world_layout.png")


def fig_stonefish_runs():
    """top-down map of each final Stonefish run, sized for slides and the report."""
    from integration.missions import Mission
    ob = load_json(os.path.join(ROOT, "stonefish_sim", "resources", "orca_rl_world_obstacles.json"))
    for w in ["rl", "survey", "multisub"]:
        f = latest(f"{w}_obstacle_v1_*.jsonl")
        if not f:
            continue
        recs = read_jsonl(f)
        summ = load_json(f.replace(".jsonl", "_summary.json"))
        fig, ax = plt.subplots(figsize=(7.2, 7.6))
        wps = Mission(w).waypoints
        ax.plot([p[1] for p in wps], [p[0] for p in wps], "--", color="0.6", lw=1.2, label="survey legs")
        ax.scatter([p[1] for p in wps], [p[0] for p in wps], s=22, color="0.4", zorder=3)
        if w == "rl":
            for p in ob["pinnacles"]:
                ax.add_patch(Circle((p["y"], p["x"]), p["r"], color=C_ROCK))
            for wl in ob["walls"]:
                c, s_ = math.cos(wl["yaw"]), math.sin(wl["yaw"])
                hx, hy = -s_ * wl["length"] / 2, c * wl["length"] / 2
                ax.plot([wl["y"] - hy, wl["y"] + hy], [wl["x"] - hx, wl["x"] + hx], color="#374151", lw=5)
            for b in ob["boulders"]:
                ax.add_patch(Circle((b["y"], b["x"]), b["approx_r"], color="#B39B72", alpha=0.6))
            for fsch in ob["fish"]:
                ax.add_patch(Ellipse((fsch["cy"], fsch["cx"]), 2 * fsch["ry"], 2 * fsch["rx"], fill=False, ls=":",
                                     color="#0EA5E9"))
        est = np.array([r["est"] for r in recs])
        gt = np.array([r["gt_eval_only"][:2] for r in recs])
        ax.plot(gt[:, 1], gt[:, 0], color=C_GT, lw=2.6, alpha=0.9, label="ground truth")
        ax.plot(est[:, 1], est[:, 0], color=C_EST, lw=1.4, label="estimate (policy input)")
        sh = np.array([r["shield"] for r in recs])
        if sh.any():
            ax.scatter(gt[sh, 1], gt[sh, 0], s=10, color=C_BASE, zorder=4, label="shield active")
        ct = np.array([r["contact"] for r in recs])
        if ct.any():
            ax.scatter(gt[ct, 1], gt[ct, 0], s=55, marker="x", color="red", zorder=5, label="contact")
        ax.set_aspect("equal")
        ax.set_xlabel("east [m]")
        ax.set_ylabel("north [m]")
        ax.set_title(f"{WORLD_NAME[w].capitalize()}: {summ['waypoints_reached']}/{summ['waypoints_total']} waypoints, "
                     f"{summ['sim_time_s']:.0f} s")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=10)
        save(fig, f"15_stonefish_run_{w}.png")


def fig_nav_error():
    fig, ax = plt.subplots(figsize=(9, 3.8))
    cols = {"rl": C_RL, "multisub": C_EST, "survey": C_BASE}
    for w in ["rl", "multisub", "survey"]:
        f = latest(f"{w}_obstacle_v1_*.jsonl")
        if not f:
            continue
        recs = read_jsonl(f)
        t = np.array([r["t"] for r in recs]); e = np.array([r["nav_err"] for r in recs])
        ax.plot(t - t[0], e, color=cols[w], lw=1.8, label=f"{WORLD_NAME[w]} (mean {e.mean():.2f} m)")
    ax.set_yscale("log")
    ax.set_xlabel("mission time [s]")
    ax.set_ylabel("|estimate - ground truth| [m]")
    ax.set_title("Navigation error with realistic onboard sensing (Stonefish)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10)
    save(fig, "16_nav_error.png")


def fig_clearance_timeline():
    f = latest("rl_obstacle_v1_*.jsonl")
    recs = read_jsonl(f)
    t = np.array([r["t"] for r in recs]); t -= t[0]
    front = np.array([min(r["clear"][1:4]) for r in recs])
    flank = np.array([min(r["flank"]) for r in recs])
    shield = np.array([r["shield"] for r in recs])
    contact = np.array([r["contact"] for r in recs])
    spd = np.array([r["cmd"][0] for r in recs])
    fig, axs = plt.subplots(2, 1, figsize=(12, 5.0), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    ax = axs[0]
    ax.plot(t, front, color=C_RL, lw=1.2, label="forward sonar clearance (min of 3 front sectors)")
    ax.plot(t, flank, color="#7C3AED", lw=1.2, label="hull flank clearance (memory)")
    ax.scatter(t[shield], np.full(shield.sum(), 31.5), s=8, color=C_BASE, label="safety shield active", marker="|")
    ax.scatter(t[contact], np.full(contact.sum(), 33), s=20, color="red", marker="x", label="contact")
    ax.set_ylabel("clearance [m]")
    ax.set_ylim(0, 35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=10)
    ax = axs[1]
    ax.plot(t, spd, color=C_EST, lw=1.2)
    ax.set_ylabel("speed cmd [m/s]")
    ax.set_xlabel("mission time [s]")
    save(fig, "17_clearance_timeline.png")


def autopilot_baseline():
    """vijay's C++ survey autopilot (ground-truth odometry navigation) in the same worlds,
    parsed from orca_rl/logs/stonefish_runs/autopilot_<world>.log."""
    out = {}
    for w in ["rl", "multisub", "survey"]:
        p = os.path.join(LOGS, "stonefish_runs", f"autopilot_{w}.log")
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        wp = re.findall(r"OrcaXLUUV t=\d+ .*? wp (\d+)/(\d+) .*?(DONE|surveying)", txt)
        reached, total, state = wp[-1] if wp else ("0", "0", "")
        ents = re.findall(r"CONTACT: OrcaXLUUV touched (\S+)", txt)
        fish = sum(e.startswith("Fish") for e in ents)
        sub = sum("Sub" in e for e in ents)
        out[w] = dict(reached=int(reached) + (1 if state == "DONE" else 0), total=int(total),
                      static=len(ents) - fish - sub, fish=fish, sub=sub,
                      resets=len(re.findall("PHYSICS DIVERGENCE", txt)))
    with open(os.path.join(DATA, "autopilot_baseline.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def fig_stonefish_results():
    ap = autopilot_baseline()
    rows = []
    for w in ["rl", "multisub", "survey"]:
        f = latest(f"{w}_obstacle_v1_*_summary.json")
        simlog = os.path.join(LOGS, "stonefish_runs", f"sim_{w}.log")
        if f and os.path.exists(simlog):
            s = load_json(f)
            ents = re.findall(r"CONTACT: OrcaXLUUV touched (\S+)", open(simlog).read())
            fish = sum(e.startswith("Fish") for e in ents)
            sub = sum("Sub" in e for e in ents)
            rows.append(dict(w=w, reached=s["waypoints_reached"], skipped=len(s["waypoints_skipped"]),
                             total=s["waypoints_total"], static=len(ents) - fish - sub, fish=fish, sub=sub,
                             resets=s["physics_resets"]))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    x = np.arange(len(rows))
    ax = axs[0]
    rl_pct = [r["reached"] / r["total"] * 100 for r in rows]
    ap_pct = [ap[r["w"]]["reached"] / ap[r["w"]]["total"] * 100 if r["w"] in ap else 0 for r in rows]
    b1 = ax.bar(x - 0.2, rl_pct, 0.38, color=C_RL, label="RL policy (realistic navigation)")
    b2 = ax.bar(x + 0.2, ap_pct, 0.38, color=C_BASE, label="original autopilot (ground-truth navigation)")
    for i, r in enumerate(rows):
        ax.text(i - 0.2, rl_pct[i] + 2, f"{r['reached']}/{r['total']}", ha="center", fontsize=9.5, fontweight="bold")
        a = ap.get(r["w"])
        if a:
            ax.text(i + 0.2, ap_pct[i] + 2, f"{a['reached']}/{a['total']}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x, [WORLD_NAME[r["w"]] for r in rows])
    ax.set_ylim(0, 125)
    ax.set_ylabel("% of survey waypoints reached")
    ax.set_title("Survey completion in Stonefish")
    ax.legend(loc="upper right", fontsize=9)
    ax = axs[1]
    w = 0.38
    rl_tot = [r["static"] + r["sub"] for r in rows]
    ap_tot = [ap[r["w"]]["static"] + ap[r["w"]]["sub"] if r["w"] in ap else 0 for r in rows]
    rl_f = [r["fish"] for r in rows]
    ap_f = [ap[r["w"]]["fish"] if r["w"] in ap else 0 for r in rows]
    ax.bar(x - 0.2, rl_tot, w, color=C_RL, label="RL: rocks / seabed / subs")
    ax.bar(x - 0.2, rl_f, w, bottom=rl_tot, color="#5EEAD4", label="RL: scripted fish")
    ax.bar(x + 0.2, ap_tot, w, color=C_BASE, label="autopilot: rocks / seabed / subs")
    ax.bar(x + 0.2, ap_f, w, bottom=ap_tot, color="#FDBA74", label="autopilot: scripted fish")
    ax.set_xticks(x, [WORLD_NAME[r["w"]] for r in rows])
    ax.set_ylabel("contact events (sim log)")
    ax.set_title("Contacts (fish are kinematic and never yield)")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "18_stonefish_results.png")


# --------------------------------------------------------------------------- objective function
EQUATIONS = {
    "eq_objective": r"$\theta^{*}=\arg\max_{\theta}\;J(\theta),\qquad J(\theta)=\mathbb{E}_{\,\mathrm{world}\sim\mathcal{D}}\left[\sum_{t=0}^{T} r_t\right],\qquad a_t=\pi_{\theta}(o_t)$",
    "eq_reward": r"$r_t = w_p\,\Delta d_t \;-\; w_u\,\frac{\sigma_t^{2}}{100} \;-\; w_r\,\rho_t \;-\; w_e\,\frac{E_t}{100} \;-\; w_h\,\mathbf{1}[\mathrm{shield}] \;-\; w_i\,\mathbf{1}[u_t<0.2] \;+\; w_g\,\mathbf{1}[\mathrm{goal}] \;-\; w_c\,\mathbf{1}[\mathrm{collision}]$",
    "eq_policy": r"$\pi_{\theta}(o)=\tanh\!\left(W_3\,\tanh\left(W_2\,\tanh\left(W_1 o+b_1\right)+b_2\right)+b_3\right),\qquad \theta=\{W_k,b_k\}\in\mathbb{R}^{5891}$",
    "eq_action": r"$v=\frac{a_0+1}{2}\,v_{\max},\qquad \dot\psi=a_1\,\dot\psi_{\max},\qquad \dot z=a_2\,\dot z_{\max},\qquad (v,\dot\psi,\dot z)\leftarrow\mathrm{Shield}(v,\dot\psi,\dot z\mid \mathrm{sonar},\mathrm{altitude},\mathrm{battery},\sigma^2)$",
    "eq_es": r"$\hat g=\frac{1}{n\,\sigma}\sum_{i=1}^{n} \mathcal{R}\!\left(J(\theta+\sigma\epsilon_i)\right)\,\epsilon_i,\qquad \epsilon_i\sim\mathcal{N}(0,I),\qquad \theta\leftarrow\theta+\alpha\,\hat g$",
    "eq_bc": r"$\theta_0=\arg\min_{\theta}\;\frac{1}{N}\sum_{j=1}^{N}\Vert\pi_{\theta}(o_j)-a^{\mathrm{planner}}(o_j)\Vert^{2}$",
}


def fig_equations():
    for name, tex in EQUATIONS.items():
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, tex, fontsize=24, color=C_DARK)
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight", pad_inches=0.08,
                    facecolor="white")
        plt.close(fig)
        print("wrote", f"docs/figures/{name}.png")


def _reward_terms(act_fn, seeds):
    """Decompose the episode return into the reward terms (energy approximated from speed)."""
    from sim.obstacle_env import OrcaObstacleNavEnv
    keys = ["progress", "goal bonus", "uncertainty", "risk", "energy", "shield", "idle", "collision"]
    tot = {k: 0.0 for k in keys}
    for sd in seeds:
        env = OrcaObstacleNavEnv(seed=sd)
        obs, _ = env.reset()
        w, p = env.w, env.params
        done = False
        while not done:
            g = env._goal().copy()
            d0 = math.hypot(g[0] - env.est_x, g[1] - env.est_y)
            obs, r, done, info = env.step(act_fn(obs))
            d1 = math.hypot(g[0] - env.est_x, g[1] - env.est_y)
            u = max(env.u, 0.0)
            e_wh = (abs(p.thrust_for_speed(u) * u) / 0.55 + p.hotel_load_w) * env.dt / 3600.0
            tot["progress"] += w["progress"] * (d0 - d1)
            tot["goal bonus"] += w["goal_bonus"] * info["reached_goal"]
            tot["uncertainty"] -= w["uncertainty"] * env.uncertainty / 100.0
            tot["risk"] -= w["risk"] * info["risk"]
            tot["energy"] -= w["energy"] * e_wh / 100.0
            tot["shield"] -= w["health"] * info["overridden"]
            tot["idle"] -= w["idle"] * (env.u < 0.2)
            tot["collision"] -= w["collision"] * info["collided"]
    return {k: v / len(seeds) for k, v in tot.items()}


def fig_reward_breakdown():
    from integration.policy_io import load_policy
    from sim.obstacle_baseline import ObstaclePotentialFieldPlanner
    pol = load_policy(os.path.join(LOGS, "trained_policy_obstacle.npz"))[0]
    seeds = range(7000, 7040)
    rl = _reward_terms(pol.forward, seeds)
    pl = _reward_terms(ObstaclePotentialFieldPlanner().act, seeds)
    keys = list(rl.keys())
    fig, ax = plt.subplots(figsize=(11, 4.6))
    y = np.arange(len(keys))
    ax.barh(y + 0.2, [pl[k] for k in keys], 0.38, color=C_BASE, label="classical planner")
    ax.barh(y - 0.2, [rl[k] for k in keys], 0.38, color=C_RL, label="RL policy")
    for i, k in enumerate(keys):
        for val, off in [(rl[k], -0.2), (pl[k], 0.2)]:
            ax.text(val + (6 if val >= 0 else -6), i + off, "≈0" if abs(val) < 0.5 else f"{val:+.0f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=9.5)
    ax.set_yticks(y, keys)
    ax.invert_yaxis()
    ax.axvline(0, color=C_DARK, lw=0.8)
    lim = max(abs(v) for v in list(rl.values()) + list(pl.values())) * 1.25
    ax.set_xlim(-lim * 0.75, lim)
    ax.set_xlabel("mean contribution to the episode return (40 held-out episodes)")
    ax.set_title(f"Where the return comes from: RL {sum(rl.values()):.0f} vs planner {sum(pl.values()):.0f}")
    ax.legend(loc="lower right")
    with open(os.path.join(DATA, "reward_breakdown.json"), "w") as f:
        json.dump(dict(rl=rl, planner=pl, seeds=[7000, 40]), f, indent=1)
    save(fig, "19_reward_breakdown.png")


def fig_motivation():
    rows = [
        ("No GPS under water: position\nfrom dead reckoning drifts",
         "Navigate on the estimate, not the truth:\nDVL + IMU + particle filter in the loop", C_EST),
        ("Charts are incomplete: subs have hit\nuncharted seamounts (2005, 2021)",
         "Sense and avoid on board: forward sonar\nsectors + flank memory in the observation", C_RL),
        ("Shared, moving waters: fish, other\nAUVs, traffic crossing the route",
         "Learn reactive trade-offs end to end\ninstead of hand-tuning a planner", "#7C3AED"),
        ("Hard limits must never be broken:\ncollision, depth, seabed, battery",
         "Safety shield with hard rules around the\nlearned policy; overrides are penalised", C_BASE),
        ("Long missions on a battery budget,\nno operator in the loop",
         "Energy and idling in the reward;\nsmall network that runs on board", C_MUTED),
    ]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.axis("off")
    ax.text(0.1, 5.82, "Real-world situation", fontsize=15, fontweight="bold", color=C_DARK)
    ax.text(7.3, 5.82, "Design response in this work", fontsize=15, fontweight="bold", color=C_DARK)
    for i, (a, b, col) in enumerate(rows):
        y = 4.75 - i * 1.12
        _box(ax, (0, y), 5.9, 0.98, a, col, fs=13)
        _box(ax, (7.2, y), 5.9, 0.98, b, "#F1F5F9", fs=13, tc=C_DARK)
        _arrow(ax, (5.95, y + 0.46), (7.15, y + 0.46), col=col)
    ax.set_xlim(-0.1, 13.2)
    ax.set_ylim(0.0, 6.1)
    save(fig, "20_motivation_map.png")


if __name__ == "__main__":
    for fn in [fig_original_training, fig_original_policy_probe, fig_original_eval, fig_observation_layout,
               fig_training_env_episode, fig_perception, fig_training_pipeline, fig_es_obstacle_training,
               fig_offline_eval, fig_sysid, fig_trim, fig_tracking, fig_architecture, fig_rl_world_layout,
               fig_stonefish_runs, fig_nav_error, fig_clearance_timeline, fig_stonefish_results,
               fig_equations, fig_reward_breakdown, fig_motivation]:
        try:
            fn()
        except Exception as e:      # keep going; report which figure failed
            print(f"FAILED {fn.__name__}: {e!r}")
            raise

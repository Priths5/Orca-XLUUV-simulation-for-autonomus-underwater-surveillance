#!/usr/bin/env python3
"""
plot_stonefish_run.py - top-down plot of one Stonefish RL run.

Draws the survey legs, the estimated track (what the policy believed), the
ground-truth track (logged for evaluation only), points where the safety
shield intervened and where Stonefish reported contact, and - for the "rl"
world - the obstacle layout from resources/orca_rl_world_obstacles.json.

    python3 orca_rl/tools/plot_stonefish_run.py orca_rl/logs/stonefish_runs/<run>.jsonl [out.png]
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from integration.missions import WORLDS  # noqa: E402

OBST = os.path.join(HERE, "..", "..", "stonefish_sim", "resources", "orca_rl_world_obstacles.json")


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".jsonl", ".png")
    recs = [json.loads(l) for l in open(path)]
    world = os.path.basename(path).split("_")[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    # NED: plot east on x, north on y so it reads like a map
    from integration.missions import Mission
    wps = Mission(world).waypoints
    ax.plot([w[1] for w in wps], [w[0] for w in wps], "--", color="0.6", lw=1, label="survey legs")
    ax.scatter([w[1] for w in wps], [w[0] for w in wps], s=18, color="0.4", zorder=3)
    if world == "rl" and os.path.exists(OBST):
        ob = json.load(open(OBST))
        for p in ob["pinnacles"]:
            ax.add_patch(plt.Circle((p["y"], p["x"]), p["r"], color="#8a6d3b"))
        for w in ob["walls"]:
            c, s = math.cos(w["yaw"]), math.sin(w["yaw"])
            hx, hy = -s * w["length"] / 2, c * w["length"] / 2   # box long side is its local y
            ax.plot([w["y"] - hy, w["y"] + hy], [w["x"] - hx, w["x"] + hx], color="#555", lw=4)
        for b in ob["boulders"]:
            ax.add_patch(plt.Circle((b["y"], b["x"]), b["approx_r"], color="#b39b72", alpha=0.6))
        for f in ob["fish"]:
            ax.add_patch(matplotlib.patches.Ellipse((f["cy"], f["cx"]), 2 * f["ry"], 2 * f["rx"],
                                                    fill=False, ls=":", color="tab:cyan"))
    ax.plot([r["est"][1] for r in recs], [r["est"][0] for r in recs], color="tab:blue", lw=1.6,
            label="estimated track (policy input)")
    gt = [r for r in recs if "gt_eval_only" in r]
    if gt:
        ax.plot([r["gt_eval_only"][1] for r in gt], [r["gt_eval_only"][0] for r in gt], color="tab:green",
                lw=1, alpha=0.8, label="ground truth (evaluation only)")
    sh = [r for r in gt if r["shield"]]
    if sh:
        ax.scatter([r["gt_eval_only"][1] for r in sh], [r["gt_eval_only"][0] for r in sh], s=10,
                   color="tab:orange", label="safety shield active", zorder=4)
    ct = [r for r in gt if r["contact"]]
    if ct:
        ax.scatter([r["gt_eval_only"][1] for r in ct], [r["gt_eval_only"][0] for r in ct], s=40,
                   marker="x", color="red", label="contact", zorder=5)
    summ = path.replace(".jsonl", "_summary.json")
    title = os.path.basename(path)
    if os.path.exists(summ):
        s = json.load(open(summ))
        title += (f"\nwaypoints reached {s['waypoints_reached']}/{s['waypoints_total']}"
                  f" (skipped {len(s.get('waypoints_skipped', []))}), contacts {len(s['contacts'])}, "
                  f"shield steps {s['shield_overrides']}, {s['sim_time_s']:.0f} s sim")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
build_report.py - writes the project report in the same IEEE two-column
conference format as orca_xluuv_report_updated.pdf:

    docs/report/Orca_RL_Stonefish_Report.html   (source of the PDF)
    docs/report/Orca_RL_Stonefish_Report.pdf    (printed with headless Brave/Chrome)
    docs/report/Orca_RL_Stonefish_Report.tex    (IEEEtran, compile on Overleaf / pdflatex)

All numbers come from the logged results (orca_rl/logs, docs/data); figures
from docs/figures (python3 docs/make_figures.py).

    python3 docs/report/build_report.py [--browser /snap/bin/brave]
"""
import argparse
import html
import json
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(ROOT, "docs", "figures")
LOGS = os.path.join(ROOT, "orca_rl", "logs")
DATA = os.path.join(ROOT, "docs", "data")
NAME = "Orca_RL_Stonefish_Report"


def load(p):
    with open(p) as f:
        return json.load(f)


def latest_summary(world):
    d = os.path.join(LOGS, "stonefish_runs")
    pinned = os.path.join(DATA, "final_runs.json")
    if os.path.exists(pinned):
        stem = load(pinned).get(world)
        if stem and os.path.exists(os.path.join(d, stem + "_summary.json")):
            return load(os.path.join(d, stem + "_summary.json"))
    fs = sorted(f for f in os.listdir(d) if f.startswith(f"{world}_obstacle_v1_") and f.endswith("_summary.json"))
    return load(os.path.join(d, fs[-1]))


# Each equation: (lines for the HTML/PNG rendering with matplotlib mathtext, LaTeX for the .tex), in order of use.
EQ = {
    "objective": ([r"$\theta^{*}=\arg\max_{\theta}\,J(\theta),\quad J(\theta)=\mathbb{E}_{\,\mathrm{world}\sim\mathcal{D}}"
                   r"\left[\sum_{t=0}^{T} r_t\right]$"],
                  r"\theta^{*}=\arg\max_{\theta} J(\theta),\quad J(\theta)=\mathbb{E}_{\mathrm{world}\sim\mathcal{D}}"
                  r"\left[\sum_{t=0}^{T} r_t\right]"),
    "policy": ([r"$a_t=\pi_{\theta}(o_t)=\tanh\left(W_3\tanh\left(W_2\tanh\left(W_1 o_t+b_1\right)+b_2\right)+b_3\right)$"],
               r"a_t=\pi_{\theta}(o_t)=\tanh\big(W_3\tanh(W_2\tanh(W_1 o_t+b_1)+b_2)+b_3\big)"),
    "action": ([r"$v=\frac{a_0+1}{2}\,v_{\max},\quad \dot\psi=a_1\,\dot\psi_{\max},\quad \dot z=a_2\,\dot z_{\max}$",
                r"$(v,\dot\psi,\dot z)\leftarrow\mathrm{Shield}\left(v,\dot\psi,\dot z\mid\mathrm{sonar},h,\mathrm{battery},\sigma^{2}\right)$"],
               r"\begin{split}&v=\tfrac{a_0+1}{2}\,v_{\max},\quad \dot\psi=a_1\,\dot\psi_{\max},\quad \dot z=a_2\,\dot z_{\max}\\"
               r"&(v,\dot\psi,\dot z)\leftarrow\mathrm{Shield}\big(v,\dot\psi,\dot z\mid\mathrm{sonar},h,\mathrm{battery},\sigma^{2}\big)\end{split}"),
    "reward": ([r"$r_t=w_p\,\Delta d_t+w_g\,\mathbf{1}[\mathrm{goal}]-w_c\,\mathbf{1}[\mathrm{collision}]$",
                r"$\qquad-\,w_u\,\sigma_t^{2}/100-w_r\,\rho_t-w_e\,E_t/100$",
                r"$\qquad-\,w_h\,\mathbf{1}[\mathrm{shield}]-w_i\,\mathbf{1}[u_t<0.2]$"],
               r"\begin{split}r_t={}&w_p\,\Delta d_t+w_g\,\mathbf{1}[\mathrm{goal}]-w_c\,\mathbf{1}[\mathrm{collision}]\\"
               r"&-w_u\,\sigma_t^{2}/100-w_r\,\rho_t-w_e\,E_t/100\\"
               r"&-w_h\,\mathbf{1}[\mathrm{shield}]-w_i\,\mathbf{1}[u_t<0.2]\end{split}"),
    "bc": ([r"$\theta_0=\arg\min_{\theta}\;\frac{1}{N}\sum_{j=1}^{N}\Vert\pi_{\theta}(o_j)-a^{\mathrm{planner}}(o_j)\Vert^{2}$"],
           r"\theta_0=\arg\min_{\theta}\frac{1}{N}\sum_{j=1}^{N}\big\lVert\pi_{\theta}(o_j)-a^{\mathrm{planner}}(o_j)\big\rVert^{2}"),
    "es": ([r"$\hat g=\frac{1}{n\,\sigma}\sum_{i=1}^{n}\mathcal{R}\left(J(\theta+\sigma\epsilon_i)\right)\epsilon_i,\quad"
            r"\epsilon_i\sim\mathcal{N}(0,I)$", r"$\theta\leftarrow\theta+\alpha\,\hat g$"],
           r"\begin{split}&\hat g=\frac{1}{n\sigma}\sum_{i=1}^{n}\mathcal{R}\big(J(\theta+\sigma\epsilon_i)\big)\,\epsilon_i,"
           r"\quad\epsilon_i\sim\mathcal{N}(0,I)\\&\theta\leftarrow\theta+\alpha\,\hat g\end{split}"),
}
EQ_DPI = 300


def render_equations():
    """Equation images for the HTML report, at the 10 pt body size (written to docs/figures/eq_r_*.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["mathtext.fontset"] = "stix"
    for name, (lines, _) in EQ.items():
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, "\n".join(lines), fontsize=10, linespacing=1.6)
        fig.savefig(os.path.join(FIG, f"eq_r_{name}.png"), dpi=EQ_DPI, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
        plt.close(fig)


def number_figures(B):
    """Replace "@<figure stem>" in the text blocks with that figure's number (order of appearance)."""
    order = [b[1].rsplit(".", 1)[0] for b in B if b[0] == "fig"]
    num = {stem: str(i + 1) for i, stem in enumerate(order)}
    sub = lambda t: re.sub(r"@([0-9A-Za-z_]+)", lambda m: num[m.group(1)], t)
    out = []
    for blk in B:
        if blk[0] in ("p", "abstract"):
            blk = (blk[0], sub(blk[1]))
        elif blk[0] in ("enum", "bullets"):
            blk = (blk[0], [sub(x) for x in blk[1]])
        out.append(blk)
    return out


def tex_figure_refs(t):
    """LaTeX: "@<figure stem>" (already escaped) becomes a \\ref to that figure's label."""
    return re.sub(r"@((?:[0-9A-Za-z]|\\_)+)", lambda m: r"\ref{fig:" + m.group(1).replace("\\_", "_") + "}", t)


def pct(x):
    return f"{x * 100:.0f}%"


# --------------------------------------------------------------------- content
def content():
    ev = load(os.path.join(LOGS, "eval_obstacle_summary.json"))
    val = load(os.path.join(DATA, "validation_selection.json"))
    orig = load(os.path.join(LOGS, "eval_summary.json"))
    ap = load(os.path.join(DATA, "autopilot_baseline.json"))
    sf = {w: latest_summary(w) for w in ["rl", "multisub", "survey"]}
    rb = load(os.path.join(DATA, "reward_breakdown.json"))
    rl, pl, bc = ev["trained_policy_obstacle.npz"], ev["classical_planner"], ev["trained_policy_obstacle_bc.npz"]
    v1, v2 = val["policy_v1.npz"], val["policy_v2.npz"]

    def ctype(s):
        fish = sum(1 for e in s["contacts"] if (e["entity"] or "").startswith("Fish"))
        sub = sum(1 for e in s["contacts"] if "Sub" in (e["entity"] or ""))
        return len(s["contacts"]) - fish - sub, fish, sub

    B = []
    B.append(("title", "Obstacle-Aware Reinforcement Learning Navigation for an ORCA-Inspired XLUUV in Stonefish over ROS 2: "
                       "Integration, Training, and Closed-Loop Evaluation"))
    B.append(("authors", [("Vijay Srinivas P", "CB.SC.U4AIE23263"), ("Joel Sahay D", "CB.SC.U4AIE23234"),
                          ("DK Preetham Reddy", "CB.SC.U4AIE23222"), ("A Prithvi", "CB.SC.U4AIE23201")],
              "Department of Artificial Intelligence and Electronics", "Project Guide: Dr. D. Palmani"))
    B.append(("abstract",
              "The previous phase of this project produced two disconnected tracks for an Orca-inspired Extra-Large "
              "Unmanned Underwater Vehicle (XLUUV): a NumPy reinforcement-learning (RL) stack trained with Evolution "
              "Strategies (ES), and a Stonefish-based high-fidelity simulator whose scenarios were driven by a hand-written "
              "reactive controller. This paper reports the work that connects them. The Stonefish application was made "
              "ROS 2-aware: it subscribes to the policy's physical setpoints, converts them into forces on the model's "
              "actual main, yaw and heave thrusters through an identified feed-forward plus PI inner loop, "
              "and publishes the vehicle's real sensors. The Python bridge, previously a skeleton written for Gazebo and the UUV Simulator, was "
              "rewritten to build the policy observation from those sensors through the existing particle-filter "
              "localization, health monitors and safety shield. Analysis showed that the shipped policy could not "
              "navigate (it saturates its yaw output and reaches 0 of 5 goals), because its observation contains neither "
              "the goal nor any obstacle information. A 23-dimensional sonar-aware observation was therefore introduced and "
              "a new policy of the same architecture was trained, warm-started by behaviour cloning of a classical "
              f"planner and refined with ES. On 100 unseen episodes it reaches {pct(rl['goal_fraction'])} of goals with a "
              f"{pct(rl['collision_rate'])} collision rate, against {pct(pl['goal_fraction'])} and "
              f"{pct(pl['collision_rate'])} for the planner. In Stonefish, using only onboard sensors for navigation, it "
              "completes the survey missions of the obstacle world and the multi-submarine world. Three defects of the "
              "original vehicle model (static capsizing, a DVL that never achieved bottom lock, and multibeam coverage "
              "that stopped after one sample) were found and fixed along the way. The dense seabed-survey scenario remains "
              "an open problem."))
    B.append(("keywords", "reinforcement learning, XLUUV, Stonefish, ROS 2, obstacle avoidance, Evolution Strategies, "
                          "behaviour cloning, safety shield, sim-to-sim transfer"))

    # ---------------------------------------------------------------- I
    B.append(("section", "Introduction"))
    B.append(("p", "A long-endurance XLUUV such as the Boeing Orca must survey large areas without GPS, while its position "
                   "estimate degrades and while it shares the water with seabed obstacles, marine life and other vehicles. "
                   "Our earlier report described a layered NumPy RL stack for this problem (physics, INS/DVL/depth sensing, "
                   "particle-filter localization, a safety shield and an ES-trained policy) and, separately, Stonefish [4] "
                   "scenarios in which the Orca surveys a rock field or crosses the lanes of other survey submarines under a "
                   "reactive controller. Its future-work section named the missing link: the ROS 2 bridge existed only as a "
                   "skeleton written for Gazebo and the UUV Simulator [5], and the RL policy had never driven the "
                   "high-fidelity vehicle. Section II explains, with real incidents, why this problem calls for a learned "
                   "policy that senses obstacles itself, wrapped in hard safety rules and tested in simulation first."))
    B.append(("p", "This paper closes that gap. Its contributions are:"))
    B.append(("enum", [
        "A ROS 2-aware Stonefish application that receives physical setpoints, drives the model's real "
        "actuators through an identified inner loop, and publishes inertial, DVL, pressure, GPS, profiling-sonar and multibeam data, "
        "contact events and the simulation clock.",
        "A rewritten Python RL node that removes every Gazebo assumption and constructs the observation from Stonefish "
        "sensors with the existing localization, health and safety modules, with a realistic onboard-navigation "
        "mode kept strictly separate from a ground-truth debug mode.",
        "A diagnosis of the shipped policy and a sonar-aware policy of the same architecture, trained with behaviour "
        "cloning plus ES on an explicit mission objective (Section VI-D) in a training environment calibrated to the "
        "identified Stonefish vehicle.",
        "Obstacles, fish schools and a traffic submarine in the base world, and closed-loop evaluation in three worlds "
        "against the original autopilot.",
        "Fixes for three defects in the vehicle model that affected every earlier Stonefish scenario."]))
    B.append(("p", "Section II gives the real-world motivation, Section III the architecture and Section IV the Gazebo-to-Stonefish "
                   "migration. Section V covers the vehicle model, Section VI the RL formulation, objective and training, "
                   "Sections VII and VIII the offline and closed-loop evaluation, and Sections IX-XI limitations, future "
                   "work and conclusions."))

    # ---------------------------------------------------------------- II
    B.append(("section", "Real-World Situations and Why This Method"))
    B.append(("p", "Four kinds of real situation shape the problem. Each one is answered by a specific element of the "
                   "proposed method (Fig. @20_motivation_map)."))
    B.append(("bullets", [
        "**Charts are incomplete.** In 2005 the submarine USS San Francisco struck an undersea mountain near Guam that "
        "was missing from its charts, and in 2021 USS Connecticut struck an uncharted seamount in the South China Sea. "
        "A route planned on a chart is therefore not enough: the vehicle must sense obstacles itself and react. "
        "This is why the policy observes forward sonar sector clearances and a memory of obstacles along the hull.",
        "**No GPS under water.** Below the surface, position comes from inertial and Doppler dead reckoning, and the "
        "error grows with distance travelled. A long-range vehicle cannot keep surfacing for a fix. This is why the "
        "policy acts on the particle-filter estimate rather than on the true position, and why localisation "
        "uncertainty appears in the reward.",
        "**Long, unattended survey missions.** Seabed mapping, pipeline and cable routes, offshore wind-farm sites "
        "and search operations (for example the 2014 deep-sea search for flight MH370) run for hours in lawnmower "
        "lanes on a battery budget, without an operator. This is why progress, energy and idling are terms of the "
        "objective, and why the policy is a small network that can run on board.",
        "**Shared, moving water.** Fish schools, marine mammals, other survey vehicles and ship traffic cross the "
        "route without warning. Losing a vehicle is costly: Autosub2 was lost under an Antarctic ice shelf in 2005. "
        "Hand-tuning a planner for every such trade-off is fragile, which is why the trade-off between progress and "
        "clearance is learned from reward, while hard limits (collision, depth rating, seabed, battery) are enforced "
        "by a safety shield that the learned policy cannot override."]))
    B.append(("fig", "20_motivation_map.png", "Real-world situations (left) and the design element of this work that "
                                              "answers each (right).", False))
    B.append(("p", "Three choices follow. First, reinforcement learning: a classical potential-field planner on the "
                   f"same inputs collides in {pct(pl['collision_rate'])} of held-out missions, against "
                   f"{pct(rl['collision_rate'])} for the learned policy (Section VII). Second, a safety shield around "
                   "the policy, because learned behaviour is not a guarantee. Third, simulation first: collisions cannot "
                   "be tested on a 50-tonne vehicle, and Stonefish provides realistic hydrodynamics and the vehicle's "
                   "own sonar, DVL and pressure sensors, so the policy can be judged on the sensing it would really have."))

    # ---------------------------------------------------------------- III
    B.append(("section", "System Architecture"))
    B.append(("p", "Fig. @13_architecture shows the integrated system. The Stonefish C++ application owns the physics: it loads the Orca "
                   "and the world from the scenario files, steps Bullet physics at 300 Hz and runs a ROS 2 node "
                   "of its own. The Python RL node runs at a fixed 1 Hz on the simulation clock that Stonefish publishes. "
                   "Each step it updates the navigation estimate, computes sonar features, builds the "
                   "observation, evaluates the policy, applies the original action scaling and the safety shield, and "
                   "publishes a physical setpoint (surge speed, yaw rate, depth rate)."))
    B.append(("fig", "13_architecture.png", "Integrated architecture. The policy reaches the actuators only through ROS 2 "
                                            "setpoints, and sees the world only through Stonefish sensor topics.", True))
    B.append(("p", "Inside the executable, the ROS 2 executor spins on its own thread; its callbacks only copy data into "
                   "mutex-protected slots. Stonefish already runs physics on a separate thread from rendering, and the "
                   "physics callback reads those slots, runs the command adapter and the existing vehicle controller (with its "
                   "survey autopilot switched off), and publishes sensor data. No Stonefish object is touched from the ROS "
                   "thread, and ROS 2 never blocks the simulator. The NumPy vehicle dynamics are used only for "
                   "offline training; at run time Stonefish is the only physics."))

    # ---------------------------------------------------------------- IV
    B.append(("section", "From Gazebo Assumptions to Stonefish Interfaces"))
    B.append(("p", "Every Gazebo-specific concept in the original bridge was traced to its source and replaced with the "
                   "corresponding Stonefish mechanism rather than renamed (Table I). Two examples show why renaming would "
                   "have been wrong. The old bridge computed depth as (P - 101325)/(rho g), but the Stonefish pressure "
                   "sensor reports gauge pressure, so the vehicle would have appeared about 10 m too shallow. And a "
                   "velocity command message has no consumer in Stonefish, because the Orca has no velocity controller, "
                   "only force actuators."))
    B.append(("table", "Gazebo / UUV Simulator Assumptions and Their Stonefish Replacements",
              ["Old assumption", "Stonefish replacement", "Data over ROS 2"],
              [["Ground-truth pose fed into the control loop", "simulator odometry, labelled ground truth; used only for "
                "evaluation or in debug mode", "ground-truth odometry (100 Hz)"],
               ["Depth = (pressure - 1 atm)/ρg", "gauge pressure: depth = pressure/ρg", "pressure (10 Hz)"],
               ["Velocity command to a simulator plugin", "speed / yaw-rate / depth-rate setpoint, converted to thruster "
                "forces in C++", "motion setpoint (1 Hz)"],
               ["Per-thruster plugin messages", "direct thruster forces, kept for manual tests", "thruster forces"],
               ["Fixed heading, zero-velocity placeholders", "measured heading and DVL velocity into the particle filter",
                "attitude (100 Hz), DVL (10 Hz)"],
               ["Simulator reset / pause services", "the robot is re-spawned at its start pose on the physics thread",
                "reset service"],
               ["Spawning the model into the world", "the Orca is loaded from the scenario description", "-"],
               ["Procedural risk map", "risk from the profiling sonar and DVL altitude; terrain richness from the "
                "multibeam", "sonar scans (10 / 2 Hz)"]],
              True))
    B.append(("p", "Frames were verified in the Stonefish source rather than taken from comments. The world frame is NED "
                   "and the body frame FRD. Odometry and DVL velocities are body-frame, and the DVL is mounted pitched by "
                   "180 degrees, so its x and z axes are reversed relative to the body. Topics are published in these NED/FRD "
                   "frames. Because the Python environment already integrates x += v cos(psi) and y += v sin(psi), its "
                   "coordinates are NED as well, and no conversion is needed between the simulator and the policy. "
                   "Stonefish's full scenario restart rebuilds the scene from the physics thread and is not safe at "
                   "run time, so a reset re-spawns the robot at its start pose and clears the adapter's integrators."))

    # ---------------------------------------------------------------- V
    B.append(("section", "The Stonefish Vehicle Model"))
    B.append(("subsection", "Hydrostatic Trim"))
    B.append(("p", "The first closed-loop test showed the Orca pitching and then rolling over even while stationary. "
                   "Reading Stonefish's own mass properties exposed the cause: with a solid composite sail, the centre of "
                   "gravity sat 0.05 m above the centre of buoyancy, so the hull was statically unstable. The original "
                   "survey demonstration capsized the same way within about 35 s (Fig. @11_trim). We made the sail and mast "
                   "syntactic foam and moved the removed mass into an internal keel ballast, positioned so that CG and CB "
                   "align fore-aft. Net buoyancy is unchanged (+42 kg), and the CG now lies 0.084 m below the CB. The same "
                   "procedure made the smaller survey submarines neutral; they had been 237 kg heavy."))
    B.append(("fig", "11_trim.png", "Roll and pitch of the original and trimmed Orca under the unchanged C++ survey "
                                    "autopilot, with the mass properties reported by Stonefish.", True))
    B.append(("subsection", "Further Defects"))
    B.append(("p", "The DVL never achieved bottom lock. Specifying only a velocity range leaves the altitude limit at "
                   "Stonefish's default of about 1e30, and the sensor then ray-marches ceil(1e30) steps, which overflows to "
                   "zero. An explicit altitude range fixes it. Stonefish's sensor interface returns sample copies whose "
                   "identifier is always zero, so the controller's multibeam coverage accepted only its first fan; samples "
                   "are now de-duplicated by timestamp. Headless runs aborted because console mode refuses lights, so lights "
                   "are stripped after include processing. Finally, Stonefish's scripted fish are kinematic bodies of "
                   "infinite mass. When one squeezes the hull into the seabed or another hull, Bullet can diverge. A "
                   "watchdog detects non-finite or implausible states, returns the Orca to its pose of about 2 s earlier, "
                   "and reports the event."))
    B.append(("subsection", "System Identification and Command Adapter"))
    B.append(("p", "Rather than assuming that an action of +1 corresponds to some force, the vehicle was identified with "
                   "raw thrust steps (Fig. @10_sysid). Steady speeds of 0.37, 0.68 and 0.93 m/s at 10, 20 and 30 kN fit "
                   "F = 9000 u|u| + 23500 u, which gives about 1.15 m/s at the 40 kN thruster limit. The yaw rate follows "
                   "F_yaw of about 150000 r, with yaw authority falling at higher speed. The published Orca envelope "
                   "(4.1 m/s, 3 deg/s) is therefore not achievable by this model, and the new policy uses the identified "
                   "limits. The C++ adapter adds these feed-forwards to PI loops on DVL speed and IMU yaw rate. It "
                   "integrates the depth-rate command into a depth reference, holds it with the autopilot's own gains and a "
                   "DVL floor guard, and stops the vehicle if commands are older than 3 s. A 0.8 m/s setpoint is held at "
                   "0.81-0.85 m/s."))
    B.append(("fig", "10_sysid.png", "System identification of the Stonefish Orca: surge speed against thrust, and yaw "
                                     "rate against yaw force.", False))

    # ---------------------------------------------------------------- VI
    B.append(("section", "RL Formulation and Training"))
    B.append(("subsection", "Diagnosis of the Shipped Policy"))
    B.append(("p", f"The shipped policy reaches 0 of {orig['n_seeds']} goals in its own evaluation. Probing it shows a yaw "
                   "output of +1 for every heading (Fig. @02_original_policy_probe), and in Stonefish it simply circles near the start. The cause is "
                   "structural: the 12-dimensional observation contains the position estimate, heading, health and context "
                   "terms, but neither the goal nor any obstacle information, so no policy trained on it can steer to a "
                   "waypoint or around a rock. It is kept unchanged and can still be selected at run time."))
    B.append(("fig", "02_original_policy_probe.png", "Left: outputs of the shipped policy as the heading input varies. "
                                                    "Right: the same policy in Stonefish.", True))
    B.append(("subsection", "Sonar-Aware Observation"))
    B.append(("p", "The new layout keeps the original 12 features in the same order and adds 11: the sine and cosine of the "
                   "bearing to the next waypoint and its distance (from the estimated position); five profiling-sonar sector "
                   "clearances measured from the bow sensor; port and starboard hull-flank clearances; and the depth error "
                   "relative to the mission depth (Fig. @04_observation_layout). The network class is unchanged (23-64-64-3, tanh). The flank "
                   "terms come from an obstacle memory. Sonar returns are placed in the world with the estimated pose and "
                   "are removed only when a later beam looks through them (free-space carving) or when the vehicle is more "
                   "than 60 m away. An earlier 30 s time-out caused stern strikes on rocks that had left the forward fan "
                   "while the 27 m hull was still swinging past them."))
    B.append(("fig", "04_observation_layout.png", "The 23-dimensional sonar-aware observation.", True))
    B.append(("subsection", "Training Environment and Safety Shield"))
    B.append(("p", "Training uses a NumPy environment at Stonefish scale, with the identified speed and speed-dependent "
                   "yaw-rate limits and time constants. The hull is a 27 m x 1.3 m capsule. The world contains rock "
                   "pinnacles, ridge walls, fish and traffic submarines, and the same mechanically swept profiling sonar as the "
                   "model (64 steps over 160 degrees at 10 Hz, binned to 5 degrees as the ROS node publishes it). Navigation "
                   "is DVL dead reckoning into the existing particle filter. Missions are chains of three goals that include "
                   "short lane changes. The safety shield keeps the original depth, battery and uncertainty rules and adds "
                   "obstacle-stop, obstacle-slow, seabed-floor and depth-band rules computed only from sensor-derived "
                   "quantities. Each override is penalised, so the policy learns to avoid needing it (Fig. @06_perception)."))
    B.append(("fig", "06_perception.png", "Training-environment perception: sonar sector clearances (shaded), ray "
                                          "returns and remembered obstacle points around the hull.", False))
    B.append(("subsection", "Objective Function: What Is Optimised"))
    B.append(("p", "Training searches for the policy parameters that maximise the expected total reward of one mission:"))
    B.append(("eq", "objective"))
    B.append(("p", "Here θ = {W_k, b_k} is the set of all 5,891 weights and biases of the policy network, r_t the reward at step t, "
                   "and the expectation is taken over the distribution D of randomly generated worlds: start pose, "
                   "obstacle field, fish and traffic paths, currents and waypoints. One step lasts 1 s; an episode has "
                   "three waypoints and ends at the last one, on a collision, or after T = 420 steps (7 min). Because "
                   "missions are finite, the return is undiscounted. The policy is a deterministic network with two hidden "
                   "layers of 64 tanh units:"))
    B.append(("eq", "policy"))
    B.append(("p", "Its three outputs, each in [-1, 1], are scaled to a physical setpoint using the identified limits "
                   "of the Stonefish Orca (top speed 1.1 m/s, a yaw-rate limit of 0.045 rad/s that shrinks with speed, and "
                   "a depth-rate limit of 0.3 m/s), and the safety shield then has the last word:"))
    B.append(("eq", "action"))
    B.append(("p", "The reward (4) rewards mission progress and penalises everything an operator would not want. Table II "
                   "lists each term, its weight and its real-world meaning."))
    B.append(("eq", "reward"))
    B.append(("table", "Reward Terms and Weights",
              ["Term", "Weight", "Measures", "Real-world meaning"],
              [["Progress (Δd)", "w_p = 1.0 per m", "drop in estimated distance to the waypoint", "complete the survey"],
               ["Waypoint bonus", "w_g = 200", "waypoint reached (within 10 m)", "mission completion"],
               ["Collision", "w_c = 300", "hull contact; episode ends", "losing the vehicle"],
               ["Uncertainty (σ²/100)", "w_u = 0.0025", "particle-filter spread (m²)", "stay localisable without GPS"],
               ["Risk (ρ)", "w_r = 0.4", "sonar proximity and low altitude, 0 to 1", "keep a safe margin"],
               ["Energy (E/100)", "w_e = 0.02", "watt-hours used in the step", "battery endurance"],
               ["Shield override", "w_h = 1.5", "the shield had to change the command", "do not rely on the safety net"],
               ["Idle", "w_i = 0.5", "speed below 0.2 m/s", "do not stall to dodge penalties"]],
              False))
    B.append(("p", "The weights encode priorities. A collision costs 300, more than a waypoint is worth, so no shortcut "
                   "through an obstacle can pay off. The progress term gives a dense signal between waypoints, while the "
                   "risk and shield terms make keeping a margin cheaper than relying on the shield. The idle penalty "
                   "removes the trivial solution of stopping to avoid every other penalty. Only sensor-derived quantities "
                   "enter the reward terms that depend on the vehicle's state, apart from the collision flag, so the "
                   "policy is rewarded for what it can actually perceive."))
    B.append(("subsection", "How the Objective Is Optimised"))
    B.append(("p", "ES from a random network learned too slowly, reaching 8% of goals after 50 generations. The network was "
                   "therefore first trained by behaviour cloning on 111k samples from a classical potential-field planner "
                   "that reads the same observation, followed by one DAgger round [6] of 25k relabelled states. This warm "
                   "start minimises the mean-squared difference between the policy's actions and the planner's:"))
    B.append(("eq", "bc"))
    B.append(("p", "ES [1] then maximises (1) directly. Each generation samples n Gaussian perturbations of the parameters, "
                   "runs one set of missions per perturbed policy, and moves the parameters along the rank-weighted "
                   "average of the perturbations:"))
    B.append(("eq", "es"))
    B.append(("p", "Here R is a centred rank transform to [-0.5, 0.5], which makes the update insensitive to the scale "
                   "of the large collision and waypoint terms. ES needs no gradient through the simulator, copes with the "
                   "abrupt collision penalty, and evaluates every candidate as an independent mission, so 14 processor "
                   "cores run in parallel. Settings were 200 generations, n = 48 (24 mirrored pairs), σ = 0.03, "
                   "α = 0.01 and four common seeds per generation (65 min in total) (Fig. @08_es_obstacle_training). Model selection used 100 validation seeds. "
                   f"An additional 80-generation fine-tune reached {pct(v2['goal_fraction'])} of goals but had "
                   f"{pct(v2['collision_rate'])} collisions, against {pct(v1['collision_rate'])} for the selected policy, "
                   "and was rejected."))
    B.append(("fig", "08_es_obstacle_training.png", "ES fine-tuning of the behaviour-cloned network (population mean and "
                                                    "best candidate, smoothed).", False))

    # ---------------------------------------------------------------- VII
    B.append(("section", "Offline Evaluation"))
    B.append(("p", "Table III compares the three controllers on 100 episodes whose seeds were used for neither training nor "
                   "model selection. All three use the same observation and the same safety shield."))
    B.append(("table", "Held-Out Test, 100 Episodes of Three Goals",
              ["Metric", "Planner", "BC only", "RL (BC+ES)"],
              [["Goals reached", pct(pl["goal_fraction"]), pct(bc["goal_fraction"]), pct(rl["goal_fraction"])],
               ["All three goals", pct(pl["all_goals_rate"]), pct(bc["all_goals_rate"]), pct(rl["all_goals_rate"])],
               ["Collision rate", pct(pl["collision_rate"]), pct(bc["collision_rate"]), pct(rl["collision_rate"])],
               ["Median min. clearance (m)", f"{pl['median_min_clearance_m']:.1f}", f"{bc['median_min_clearance_m']:.1f}",
                f"{rl['median_min_clearance_m']:.1f}"],
               ["Mean episode reward", f"{pl['mean_reward']:.0f}", f"{bc['mean_reward']:.0f}", f"{rl['mean_reward']:.0f}"],
               ["Mean shield steps", f"{pl['mean_shield_steps']:.1f}", f"{bc['mean_shield_steps']:.1f}",
                f"{rl['mean_shield_steps']:.1f}"]],
              False))
    B.append(("p", "Behaviour cloning reproduces the planner, collisions included. ES then cuts the collision rate by more "
                   "than half and raises goal completion, mainly by keeping larger margins: the median closest approach "
                   "grows from about 1.7 m to 5.9 m, and the policy accepts more brief shield interventions while doing so. "
                   "Fig. @05_training_env_episode shows a typical case: the planner cuts a corner into a rock cluster, while the RL policy slows, "
                   "swings wide and completes all three goals."))
    B.append(("p", f"Fig. @19_reward_breakdown breaks the return into the reward terms over 40 held-out missions. The RL policy earns more "
                   f"waypoint bonus ({rb['rl']['goal bonus']:.0f} against {rb['planner']['goal bonus']:.0f}) and more "
                   f"progress ({rb['rl']['progress']:.0f} against {rb['planner']['progress']:.0f}), and halves the "
                   f"collision penalty ({rb['rl']['collision']:.0f} against {rb['planner']['collision']:.0f}). It pays "
                   "somewhat more for shield overrides, because it accepts brief interventions to keep moving. The "
                   "uncertainty and energy terms are small at this scale (flat seabeds and seven-minute missions), which "
                   "is consistent with their role as tie-breakers rather than drivers."))
    B.append(("fig", "05_training_env_episode.png", "The same held-out episode under the classical planner (left) and the RL "
                                                    "policy (right).", True))
    B.append(("fig", "19_reward_breakdown.png", "Mean contribution of each reward term per mission, RL policy vs. "
                                                "classical planner (40 held-out missions; energy approximated from "
                                                "speed).", False))

    # ---------------------------------------------------------------- VIII
    B.append(("section", "Closed-Loop Evaluation in Stonefish"))
    B.append(("p", "The base world, which previously contained only the vehicle, now has 12 rock pinnacles and two ridge "
                   "walls rising from a 30 m seabed through the 12 m cruise depth, two boulders, three fish schools of "
                   "three fish each looping at about 2 m/s, and a traffic submarine running its own avoidance across the "
                   "Orca's lanes (Fig. @14_rl_world_layout). The survey world (rocks, pillars, wrecks and fish over an 18 m seabed) and the "
                   "multi-submarine world are unchanged apart from the model fixes. Long survey legs are split into "
                   "waypoints at most 80 m apart. A waypoint that the sonar shows to be occupied, or whose leg times out, "
                   "is skipped, as an AUV mission executive would do."))
    B.append(("fig", "14_rl_world_layout.png", "The base world with obstacles and the Orca's survey legs.", False))
    sr, sm, ss = sf["rl"], sf["multisub"], sf["survey"]
    c_rl, c_ms, c_sv = ctype(sr), ctype(sm), ctype(ss)
    B.append(("table", "Stonefish Closed Loop: RL Policy (Realistic Navigation) vs. the Original Autopilot (Ground-Truth Navigation)",
              ["World", "RL waypoints", "RL contacts (static/fish/sub)", "RL resets", "Autopilot waypoints",
               "Autopilot contacts"],
              [["Obstacle", f"{sr['waypoints_reached']}/{sr['waypoints_total']} (+{len(sr['waypoints_skipped'])} skipped)",
                f"{c_rl[0]}/{c_rl[1]}/{c_rl[2]}", str(sr["physics_resets"]), f"{ap['rl']['reached']}/{ap['rl']['total']}",
                str(ap["rl"]["static"] + ap["rl"]["fish"] + ap["rl"]["sub"])],
               ["Multi-sub", f"{sm['waypoints_reached']}/{sm['waypoints_total']} (+{len(sm['waypoints_skipped'])} skipped)",
                f"{c_ms[0]}/{c_ms[1]}/{c_ms[2]}", str(sm["physics_resets"]),
                f"{ap['multisub']['reached']}/{ap['multisub']['total']}",
                str(ap["multisub"]["static"] + ap["multisub"]["fish"] + ap["multisub"]["sub"])],
               ["Survey", f"{ss['waypoints_reached']}/{ss['waypoints_total']} (+{len(ss['waypoints_skipped'])} skipped)",
                f"{c_sv[0]}/{c_sv[1]}/{c_sv[2]}", str(ss["physics_resets"]),
                f"{ap['survey']['reached']}/{ap['survey']['total']}",
                str(ap["survey"]["static"] + ap["survey"]["fish"] + ap["survey"]["sub"])]],
              True))
    B.append(("p", f"Table IV summarises the closed-loop runs (1800 s limit, three times real time, realistic onboard "
                   f"navigation). In the obstacle world the policy completes the survey in {sr['sim_time_s']:.0f} s, reaching "
                   f"{sr['waypoints_reached']} of {sr['waypoints_total']} waypoints and skipping the one that lies on a "
                   "pinnacle. Its track detours around pinnacles, ridges and boulders on every lane (Figs. @15_stonefish_run_rl and @17_clearance_timeline). Mean "
                   "navigation error is 0.64 m over 25 minutes (Fig. @16_nav_error). Of its contacts, the static ones are grazes in "
                   "tight passes and the rest are scripted fish crossing the hull. In the multi-submarine world it reaches "
                   f"{sm['waypoints_reached']} of {sm['waypoints_total']} waypoints with four brief touches of another "
                   "submarine. The original autopilot, which navigates on ground-truth odometry, reaches comparable "
                   "waypoint counts, with 15 and 17 contact events in the same two worlds."))
    B.append(("fig", "15_stonefish_run_rl.png", "Obstacle world: estimated (blue) and true (green) track of the RL-driven "
                                                "Orca, with shield interventions and contacts.", False))
    B.append(("fig", "17_clearance_timeline.png", "Obstacle-world run: forward sonar clearance, flank clearance from the "
                                                  "obstacle memory, shield activity, contacts and commanded speed.", True))
    B.append(("fig", "16_nav_error.png", "Navigation error of the realistic estimate in Stonefish.", False))
    B.append(("p", f"The survey world remains unsolved. The policy reaches {ss['waypoints_reached']} of "
                   f"{ss['waypoints_total']} waypoints and the autopilot {ap['survey']['reached']}. The world is a 90 m x "
                   "70 m box with lanes 23 m apart for a 26 m hull, and its large kinematic fish repeatedly press the hull "
                   f"into the seabed, causing {ss['physics_resets']} physics resets. Because each reset moves the vehicle "
                   "without any sensor seeing it, the realistic estimate drifts and later legs time out. This is a property "
                   "of the scripted scene rather than of the integration, but it marks the limit of the current policy and "
                   "sensing (Fig. @18_stonefish_results)."))
    B.append(("fig", "18_stonefish_results.png", "Survey completion and contacts by type for the RL policy and the original "
                                                 "autopilot in the three worlds.", True))
    B.append(("fig", "stonefish_orca_view.png", "Stonefish view of the obstacle world: the Orca XLUUV among the rock "
                                                "pinnacles and ridge walls.", False))

    # ---------------------------------------------------------------- IX-XI
    B.append(("section", "Current Limitations"))
    B.append(("bullets", [
        "The policy is trained in a NumPy environment calibrated to Stonefish, not in Stonefish itself, so some sim-to-sim "
        "gap remains (for example in turning dynamics near obstacles).",
        "Every scenario has a flat seabed, so terrain-aided navigation carries no position information and the estimate "
        "is DVL dead reckoning.",
        "Stonefish's scripted fish are kinematic and never yield; they can strike the hull from outside the forward sonar "
        "fan and can force physics resets.",
        "The mechanically swept profiling sonar refreshes each bearing only about every 6.4 s, which limits reaction to fast "
        "movers.",
        "Contacts still occur: six static grazes in the obstacle world, and the dense survey world is not completed."]))
    B.append(("section", "Future Work"))
    B.append(("enum", [
        "Fine-tune the selected policy with ES directly in windowless Stonefish (three times real time), resetting "
        "the vehicle over ROS 2 between episodes, to close the remaining sim-to-sim gap.",
        "Randomise the obstacle field per episode with the scenario generator, and randomise traffic submarines and "
        "fish paths.",
        "Replace the flat seabeds with charted bathymetry so that the particle filter's terrain update becomes "
        "informative.",
        "Increase the flank safety margin in tight passes, and evaluate PPO [2] on the same environment once a deep-learning "
        "toolchain is available."]))
    B.append(("section", "Conclusion"))
    B.append(("p", "The RL navigation stack and the Stonefish simulator are now one system. The policy drives the real "
                   "Stonefish actuators through ROS 2, using only real Stonefish sensors, while the existing localization, "
                   "health and safety layers are preserved. The shipped policy was shown to be structurally unable to "
                   "navigate, and a sonar-aware policy of the same architecture, trained with behaviour cloning and ES, "
                   f"reaches {pct(rl['goal_fraction'])} of goals with {pct(rl['collision_rate'])} collisions offline, "
                   "against 68% and 28% for a classical planner. In Stonefish it completes the obstacle-world and "
                   "multi-submarine surveys under realistic navigation. Three defects that affected every earlier Stonefish "
                   "scenario were fixed along the way. The dense survey field, with its non-yielding scripted fish, "
                   "remains the next target."))
    B.append(("ack", "The authors thank their Project Guide, Dr. D. Palmani, for guidance and feedback throughout the design, "
                     "implementation and evaluation of this project."))
    B.append(("refs", [
        "T. Salimans, J. Ho, X. Chen, S. Sidor, and I. Sutskever, “Evolution Strategies as a Scalable Alternative to "
        "Reinforcement Learning,” arXiv preprint arXiv:1703.03864, 2017.",
        "J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal Policy Optimization Algorithms,” "
        "arXiv preprint arXiv:1707.06347, 2017.",
        "T. I. Fossen, Handbook of Marine Craft Hydrodynamics and Motion Control. Chichester, U.K.: Wiley, 2011.",
        "P. Cieślak, “Stonefish: An Advanced Open-Source Simulation Tool Designed for Marine Robotics, with a ROS "
        "Interface,” in Proc. IEEE/OES OCEANS, Marseille, France, 2019.",
        "M. M. M. Manhães, S. A. Scherer, M. Voss, L. R. Douat, and T. Rauschenbach, “UUV Simulator: A Gazebo-Based "
        "Package for Underwater Intervention and Multi-Robot Simulation,” in Proc. IEEE/MTS OCEANS, Monterey, CA, "
        "USA, 2016.",
        "S. Ross, G. Gordon, and D. Bagnell, “A Reduction of Imitation Learning and Structured Prediction to No-Regret "
        "Online Learning,” in Proc. AISTATS, 2011.",
        "S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall, “Robot Operating System 2: Design, "
        "Architecture, and Uses in the Wild,” Science Robotics, vol. 7, no. 66, 2022."]))
    return B


# --------------------------------------------------------------------- HTML
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


def inline_html(t):
    t = html.escape(t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w/.])([A-Za-z])_([A-Za-z0-9]+)\b", r"<i>\1</i><sub>\2</sub>", t)   # w_p, r_t, F_yaw
    return t


CSS = """
@page { size: letter; margin: 0.75in 0.62in 0.9in 0.62in; }
body { font-family: 'Liberation Serif','Nimbus Roman','Times New Roman',serif; font-size: 10pt; line-height: 1.18;
       color: #000; margin: 0; text-align: justify; hyphens: auto; }
.title { font-size: 22pt; text-align: center; line-height: 1.15; margin: 0 0 12pt 0; }
.authors { display: flex; justify-content: space-around; text-align: center; font-size: 10.5pt; margin-bottom: 4pt; }
.affil { text-align: center; font-size: 9.5pt; margin-bottom: 14pt; }
.cols { column-count: 2; column-gap: 0.24in; }
.abstract, .keywords { font-size: 9pt; font-weight: bold; text-align: justify; margin: 0 0 6pt 0; }
.abstract i, .keywords i { font-weight: bold; }
h2 { font-size: 10pt; font-weight: normal; text-align: center; font-variant: small-caps; margin: 10pt 0 5pt 0;
     break-after: avoid; letter-spacing: 0.3px; }
h3 { font-size: 10pt; font-weight: normal; font-style: italic; margin: 7pt 0 3pt 0; break-after: avoid; }
p { margin: 0 0 4pt 0; text-indent: 1em; }
ol, ul { margin: 2pt 0 5pt 0; padding-left: 1.6em; }
li { margin-bottom: 2pt; }
code { font-family: 'Liberation Mono','DejaVu Sans Mono',monospace; font-size: 8.6pt; }
figure { margin: 6pt 0 8pt 0; break-inside: avoid; text-align: center; }
figure img { width: 100%; }
figure.wide { margin: 6pt 0 8pt 0; }
figcaption { font-size: 8pt; text-align: justify; margin-top: 3pt; }
.tbl { break-inside: avoid; margin: 6pt 0 8pt 0; }
.tbl.wide table { font-size: 7pt; }
.tbl.wide th, .tbl.wide td { padding: 1.5pt 2.5pt; }
.tcap { text-align: center; font-size: 8pt; font-variant: small-caps; margin-bottom: 3pt; }
.tcap .num { display: block; font-variant: normal; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; text-align: left; }
th, td { padding: 2pt 4pt; vertical-align: top; }
thead tr { border-top: 1px solid #000; border-bottom: 1px solid #000; }
tbody tr:last-child { border-bottom: 1px solid #000; }
.eq { display: flex; align-items: center; margin: 4pt 0 6pt 0; break-inside: avoid; }
.eq .m { flex: 1; text-align: center; }
.eq img { max-width: 100%; vertical-align: middle; }
.eq .n { width: 18pt; text-align: right; }
.refs { font-size: 8pt; padding-left: 0; list-style: none; }
.refs li { display: flex; gap: 6pt; }
.refs li span:first-child { min-width: 18pt; }
"""


def render_html(B):
    head, body = [], []
    fig_n = tab_n = sec_n = eq_n = 0
    from PIL import Image
    for blk in number_figures(B):
        k = blk[0]
        if k == "title":
            head.append(f'<div class="title">{inline_html(blk[1])}</div>')
        elif k == "authors":
            head.append('<div class="authors">' + "".join(
                f"<div>{html.escape(n)}<br>{html.escape(r)}</div>" for n, r in blk[1]) + "</div>")
            head.append(f'<div class="affil">{html.escape(blk[2])}<br>{html.escape(blk[3])}</div>')
        elif k == "abstract":
            body.append(f'<div class="abstract"><i>Abstract</i>—{inline_html(blk[1])}</div>')
        elif k == "keywords":
            body.append(f'<div class="keywords"><i>Index Terms</i>—{inline_html(blk[1])}</div>')
        elif k == "section":
            sec_n += 1
            body.append(f"<h2>{ROMAN[sec_n - 1]}. {html.escape(blk[1])}</h2>")
            sub_n = 0
        elif k == "subsection":
            sub_n += 1
            body.append(f"<h3>{chr(64 + sub_n)}. {html.escape(blk[1])}</h3>")
        elif k == "p":
            body.append(f"<p>{inline_html(blk[1])}</p>")
        elif k in ("enum", "bullets"):
            tag = "ol" if k == "enum" else "ul"
            body.append(f"<{tag}>" + "".join(f"<li>{inline_html(x)}</li>" for x in blk[1]) + f"</{tag}>")
        elif k == "eq":
            eq_n += 1
            img = f"eq_r_{blk[1]}.png"
            w_in = Image.open(os.path.join(FIG, img)).size[0] / EQ_DPI
            body.append(f'<div class="eq"><span class="m"><img src="../figures/{img}" style="width:{w_in:.3f}in">'
                        f'</span><span class="n">({eq_n})</span></div>')
        elif k == "fig":
            fig_n += 1
            _, fname, cap, wide = blk
            body.append(f'<figure class="{"wide" if wide else ""}"><img src="../figures/{fname}">'
                        f"<figcaption>Fig. {fig_n}. {inline_html(cap)}</figcaption></figure>")
        elif k == "table":
            tab_n += 1
            _, cap, hdr, rows, wide = blk
            t = ("<table><thead><tr>" + "".join(f"<th>{inline_html(h)}</th>" for h in hdr) + "</tr></thead><tbody>"
                 + "".join("<tr>" + "".join(f"<td>{inline_html(c)}</td>" for c in r) + "</tr>" for r in rows)
                 + "</tbody></table>")
            body.append(f'<div class="tbl {"wide" if wide else ""}"><div class="tcap"><span class="num">TABLE {ROMAN[tab_n - 1]}</span>'
                        f"{html.escape(cap)}</div>{t}</div>")
        elif k == "ack":
            body.append(f'<h2>Acknowledgment</h2><p>{inline_html(blk[1])}</p>')
        elif k == "refs":
            body.append('<h2>References</h2><ol class="refs">' + "".join(
                f"<li><span>[{i + 1}]</span><span>{html.escape(r)}</span></li>" for i, r in enumerate(blk[1])) + "</ol>")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Orca-RL in Stonefish</title><style>{CSS}</style></head>"
            f"<body>{''.join(head)}<div class='cols'>{''.join(body)}</div></body></html>")


# --------------------------------------------------------------------- LaTeX
def tex_escape(t):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in t)


def inline_tex(t):
    parts = re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", t)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`"):
            out.append(r"\texttt{" + tex_escape(p[1:-1]) + "}")
        elif p.startswith("**") and p.endswith("**"):
            out.append(r"\textbf{" + tex_escape(p[2:-2]) + "}")
        else:
            out.append(re.sub(r"(?<![A-Za-z0-9\\])([A-Za-z])\\_([A-Za-z0-9]+)\b", r"$\1_{\2}$", tex_escape(p)))
    s = "".join(out)
    return s.replace("—", "---").replace("“", "``").replace("”", "''")


def render_tex(B):
    L = [r"\documentclass[conference]{IEEEtran}", r"\usepackage[utf8]{inputenc}", r"\usepackage[T1]{fontenc}",
         r"\usepackage{graphicx}", r"\usepackage{booktabs}", r"\usepackage{amsmath}", r"\usepackage{amssymb}", r"\usepackage{hyperref}", r"\usepackage{newunicodechar}",
         r"\newunicodechar{°}{\ensuremath{^\circ}}", r"\newunicodechar{ś}{\'s}", r"\newunicodechar{ã}{\~a}",
         r"\newunicodechar{θ}{\ensuremath{\theta}}", r"\newunicodechar{σ}{\ensuremath{\sigma}}",
         r"\newunicodechar{α}{\ensuremath{\alpha}}", r"\newunicodechar{ρ}{\ensuremath{\rho}}",
         r"\newunicodechar{Δ}{\ensuremath{\Delta}}", r"\newunicodechar{²}{\ensuremath{^2}}",
         r"\graphicspath{{../figures/}}", r"\begin{document}"]
    sec = False
    for blk in B:
        k = blk[0]
        if k == "title":
            L.append(r"\title{" + inline_tex(blk[1]) + "}")
        elif k == "authors":
            names = r" \and ".join(r"\IEEEauthorblockN{" + tex_escape(n) + r"}\IEEEauthorblockA{" + tex_escape(r) + "}"
                                   for n, r in blk[1])
            L.append(r"\author{" + names + r" \and \IEEEauthorblockA{" + tex_escape(blk[2]) + r"\\" + tex_escape(blk[3])
                     + "}}")
            L.append(r"\maketitle")
        elif k == "abstract":
            L.append(r"\begin{abstract}" + inline_tex(blk[1]) + r"\end{abstract}")
        elif k == "keywords":
            L.append(r"\begin{IEEEkeywords}" + inline_tex(blk[1]) + r"\end{IEEEkeywords}")
        elif k == "section":
            L.append(r"\section{" + inline_tex(blk[1]) + "}")
        elif k == "subsection":
            L.append(r"\subsection{" + inline_tex(blk[1]) + "}")
        elif k == "p":
            L.append(inline_tex(blk[1]) + "\n")
        elif k in ("enum", "bullets"):
            env = "enumerate" if k == "enum" else "itemize"
            L.append(r"\begin{" + env + "}" + "".join(r"\item " + inline_tex(x) for x in blk[1]) + r"\end{" + env + "}")
        elif k == "eq":
            L.append(r"\begin{equation}" + EQ[blk[1]][1] + r"\end{equation}")
        elif k == "fig":
            _, fname, cap, wide = blk
            env = "figure*" if wide else "figure"
            w = r"0.9\textwidth" if wide else r"\columnwidth"
            L.append(r"\begin{" + env + r"}[!t]\centering\includegraphics[width=" + w + "]{" + fname + r"}\caption{"
                     + inline_tex(cap) + r"}\label{fig:" + fname.rsplit(".", 1)[0] + r"}\end{" + env + "}")
        elif k == "table":
            _, cap, hdr, rows, wide = blk
            env = "table*" if wide else "table"
            cols = "l" * len(hdr)
            body = " & ".join(r"\textbf{" + inline_tex(h) + "}" for h in hdr) + r" \\ \midrule " + " ".join(
                " & ".join(inline_tex(c) for c in r) + r" \\" for r in rows)
            L.append(r"\begin{" + env + r"}[!t]\centering\caption{" + inline_tex(cap) + r"}\footnotesize"
                     + r"\resizebox{" + (r"\textwidth" if wide else r"\columnwidth") + r"}{!}{\begin{tabular}{" + cols
                     + r"}\toprule " + body + r" \bottomrule\end{tabular}}\end{" + env + "}")
        elif k == "ack":
            L.append(r"\section*{Acknowledgment}" + inline_tex(blk[1]))
        elif k == "refs":
            L.append(r"\begin{thebibliography}{9}" + "".join(r"\bibitem{r" + str(i) + "} " + inline_tex(r)
                                                            for i, r in enumerate(blk[1])) + r"\end{thebibliography}")
    L.append(r"\end{document}")
    return tex_figure_refs("\n".join(L))


# --------------------------------------------------------------------- main
def find_browser(pref):
    for b in [pref, "/snap/bin/brave", "brave-browser", "google-chrome", "chromium", "chromium-browser"]:
        if b and shutil.which(b):
            return shutil.which(b)
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", default=None)
    args = ap.parse_args()
    B = content()
    render_equations()
    html_path = os.path.join(HERE, NAME + ".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_html(B))
    with open(os.path.join(HERE, NAME + ".tex"), "w", encoding="utf-8") as f:
        f.write(render_tex(B))
    print("wrote", html_path, "and", NAME + ".tex")
    br = find_browser(args.browser)
    if br is None:
        print("no Chromium-based browser found: open the .html and print to PDF, or compile the .tex")
    else:
        pdf = os.path.join(HERE, NAME + ".pdf")
        subprocess.run([br, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                        f"--print-to-pdf={pdf}", "file://" + html_path], check=True, capture_output=True, timeout=180)
        print("wrote", pdf)

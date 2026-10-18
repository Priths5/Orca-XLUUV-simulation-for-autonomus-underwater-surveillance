# Orca-RL: Deep RL for Detectability-Aware, Energy-Constrained Passive Navigation
### Full Implementation Manual

This package implements the full technical proposal: a physically-grounded
simulation of the Boeing Orca XLUUV (real public specs), a deep-RL
navigation policy trained on it, a classical baseline for comparison, a
live evaluation dashboard, and - since the integration in ../README.md -
a working ROS 2 connection to the Stonefish Orca simulation
(../stonefish_sim, executable `orca_rl_sim`). The earlier Gazebo /
uuv_simulator path has been removed.

---

## 1. What's actually in this package

```
orca_rl/
  sim/
    vehicle_dynamics.py     # Fossen-style physics using Orca's real public specs
    sensors.py               # INS/DVL/depth sensor noise models
    ocean_environment.py     # Procedural bathymetry, currents, detection-risk map
    localization.py          # Particle-filter localization backend
    health_monitor.py        # Vehicle + environment health tracking
    safety_shield.py         # Hard physical/health constraint enforcement
    orca_env.py              # Full RL environment (reset/step API), ties it all together
    baseline_planner.py      # Classical entropy/potential-field planner (comparison baseline)
  rl/
    policy_network.py        # Small NumPy MLP policy (no torch dependency)
    es_trainer.py             # Evolution Strategies trainer (NumPy-only PPO substitute)
  ros2_bridge/
    orca_rl_bridge_node.py   # ROS 2 node: runs the policy against Stonefish (orca_rl_sim)
    package.xml               # ROS 2 manifest (standard msgs only, no Gazebo)
    launch_orca_bridge.launch.py  # launches orca_rl_sim + this node
  integration/                # Stonefish frames, sonar perception, navigation, observation layouts
  sim/obstacle_env.py         # Stonefish-scale obstacle/goal training env (obstacle-aware policy)
  train_obstacle.py           # trains logs/trained_policy_obstacle.npz (same ES/MLP methodology)
  tools/                      # manual_cmd.py, check_stonefish_topics.py, sysid_stonefish.py
  tests/                      # offline tests for the integration layer
  train.py                    # Trains via Evolution Strategies (works in this sandbox)
  ppo_train.py                 # Trains via REAL PPO/Stable-Baselines3 (needs internet access)
  evaluate.py                  # Runs RL vs baseline, writes logs for dashboard + report
  dashboard.html               # Live mission-playback evaluation console
  requirements.txt
  README.md                    # This file
```

**Nothing here was re-generated for this manual** - it documents the exact
code already built. If you already have these files from earlier in this
conversation, this manual is your reference for how they fit together and
how to run them.

---

## 2. Environment notes - read this first

This codebase was originally built inside a sandboxed environment with
**no network access to PyPI** beyond a small pre-installed set of packages
(numpy, scipy, matplotlib, pandas). `gymnasium`, `stable-baselines3`, and
`torch` could not be installed there. Two consequences:

1. **`train.py` uses Evolution Strategies (ES), not PPO.** ES (Salimans et
   al., 2017, OpenAI) is a legitimate, published, gradient-free RL
   optimization method that needs no autodiff/backprop, so it's fully
   implementable in plain NumPy. It is used here as a working substitute,
   not a compromise on the RL formulation itself (state/action/reward/
   safety-shield design is identical either way).
2. **`ppo_train.py` is provided as the "real" path.** On any machine with
   normal internet access, install `requirements.txt` and run
   `ppo_train.py` instead - it wraps the exact same `sim/orca_env.py`
   physics/sensor/health/safety stack into a standard `gymnasium.Env` and
   trains it with real PPO via Stable-Baselines3. Nothing about the
   environment changes; only the optimizer does.

If you're running this on your own machine (not the sandbox this was
built in), you likely have full internet access - in that case, just use
`ppo_train.py` from the start and treat `train.py`/`es_trainer.py` as a
reference/fallback.

---

## 3. Quick start (works fully offline, no extra installs)

```bash
cd orca_rl
pip install -r requirements.txt      # numpy/scipy/matplotlib/pandas only, for this path

# 1. Train (Evolution Strategies - ~15-20 min on a laptop CPU)
python3 train.py
# -> writes logs/trained_policy.npz and logs/training_history.json

# 2. Evaluate RL vs classical baseline across multiple seeds
python3 evaluate.py --n_seeds 10
# -> writes logs/eval_summary.json (aggregate metrics)
# -> writes logs/eval_trajectories.json (full step-by-step trajectory for the dashboard)

# 3. Open the live dashboard
# Just open dashboard.html directly in a browser, then use "LOAD JSON FILE"
# to load logs/eval_trajectories.json (browsers block local fetch() of
# file:// JSON for security, so manual load is the reliable path - or serve
# the folder locally to enable auto-load):
python3 -m http.server 8000
# then visit http://localhost:8000/dashboard.html - it will auto-load
# logs/eval_trajectories.json if present.
```

### What the dashboard shows
- Live playback of both the RL policy's and classical baseline's missions,
  side by side or toggled individually.
- Vehicle trajectory (estimated position) with a live uncertainty ellipse
  (grows/shrinks as localization confidence changes).
- Vehicle health panel: battery, depth, uncertainty, risk exposure, and
  safety-shield override events, updated per playback step.
- Scrubber + play/pause + speed controls for step-by-step inspection.

---

## 4. Training with real PPO (recommended once you have internet access)

```bash
pip install gymnasium stable-baselines3 torch
python3 ppo_train.py
# -> trains 2M timesteps across 8 parallel envs (adjust N_ENVS/TOTAL_TIMESTEPS
#    at the top of the script based on your hardware)
# -> saves logs/orca_ppo_final.zip and logs/ppo_best/best_model.zip
```

To evaluate a PPO-trained model instead of the ES one, load it with
Stable-Baselines3's `PPO.load(...)` and call `.predict(obs)` in place of
`policy.forward(obs)` in `evaluate.py` - the rest of the evaluation/
dashboard pipeline is unchanged, since both produce the same action format.

---

## 5. Architecture - how the pieces cooperate

This mirrors the layered design from the technical proposal, specifically
so that swapping fidelity levels (NumPy physics -> Stonefish physics; ES ->
PPO) only touches one layer at a time:

```
 Layer 1: Physics engine        (sim/vehicle_dynamics.py for training, OR Stonefish via ROS 2)
              |
 Layer 2: Sensors + Estimation  (sim/sensors.py, sim/localization.py)
              |
 Layer 3: RL Policy             (rl/policy_network.py + es_trainer.py, OR ppo_train.py)
              |
 Layer 4: Safety Shield         (sim/safety_shield.py)  -- enforces hard limits
              |
        (action applied back to Layer 1, loop closes)
```

`sim/orca_env.py` is the orchestrator that wires all four layers together
with the standard `reset()`/`step()` API. `sim/health_monitor.py` feeds
both Layer 3 (as part of the observation) and Layer 4 (as a safety
trigger) independently.

---

## 6. High-fidelity simulation: Stonefish over ROS 2

The Gazebo / uuv_simulator instructions that used to be here are obsolete.
The policy now drives the Stonefish Orca XLUUV model through ROS 2 - see
**../README.md** for the architecture, the Gazebo -> Stonefish migration
table, the ROS 2 interface, build/run commands and the test sequence.
Short version:

```bash
source /opt/ros/jazzy/setup.bash
cd ../stonefish_sim && mkdir -p build && cd build && cmake .. && make -j
ros2 launch ../../orca_rl/ros2_bridge/launch_orca_bridge.launch.py world:=rl
```

Real bathymetry/current data (GEBCO, HYCOM/Copernicus) can still replace
`OceanEnvironment` for the NumPy training env; for Stonefish the chart
lives in `integration/world_model.py` (currently the scenarios' flat
seabed, which makes terrain-aided navigation uninformative there).

---

## 7. Known limitations / what to check before presenting

- **Vehicle dynamics parameters not covered by public Orca specs are
  clearly marked `[ESTIMATED]`** in `vehicle_dynamics.py` (drag
  coefficients, added mass, battery capacity, structural margins). These
  are realistic engineering estimates for a comparable-class hull, not
  real Orca figures - be upfront about this if presenting the numbers.
- **The ES-trained policy in this sandbox is a proof-of-concept, not a
  fully converged model.** Expect noticeably better, more stable results
  from `ppo_train.py` given proper compute (GPU, longer training, larger
  population/parallel environments than were tractable in this sandbox).
- **The novelty claim (no prior work jointly optimizing localization,
  energy, and detectability via deep RL for passive-navigation UUVs) was
  verified by reading full papers, not just abstracts, but is not an
  exhaustive systematic literature search.** Confirm with your advisor or
  a thorough Scholar/arXiv search before final submission.
- The `OceanEnvironment`'s bathymetry/currents/risk-zones are procedurally
  generated for training speed and reproducibility. Swap in real GEBCO/
  HYCOM data (Section 6) before drawing conclusions meant to generalize
  to real ocean conditions.
- **The shipped `logs/trained_policy.npz` does not navigate:** in its own
  environment it saturates yaw at +1 (circles) and commands a negative
  depth rate (surfaces), reaching the goal in 0/5 evaluation seeds
  (`logs/eval_summary.json`). Its 12-d observation has no goal or obstacle
  information, so it cannot steer to a goal or around obstacles. See
  ../README.md for the obstacle-aware policy trained for Stonefish.

---

## 8. Suggested order of work if extending this further

1. Run the offline quick-start (Section 3) to get a working baseline
   result end-to-end.
2. Move to `ppo_train.py` on a machine with GPU access for a real,
   properly-converged policy.
3. Swap in real GEBCO/HYCOM data (Section 6.4).
4. Validate in Stonefish over ROS 2 (Section 6 / ../README.md) for a
   high-fidelity run and demo-quality visuals.
5. Re-run comparisons with the Stonefish-validated policy and regenerate
   the dashboard + report with those numbers.

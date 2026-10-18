# integrated_Orca_RL - the Orca RL policy driving the Stonefish Orca over ROS 2

This folder combines

* `orca_rl/` - the Python RL project from `orca_rl.zip` (sim, localisation,
  health, safety shield, ES-trained MLP policy), with its Gazebo/uuv_simulator
  bridge replaced by a Stonefish one, and
* `stonefish_sim/` - the Stonefish Orca XLUUV scenarios from `vijay_orca.zip`
  (`orca_simul_scenarios/simul`, without its machine-specific `build/` and the
  300 MB vendored `env/` toolchain - this machine has Stonefish 1.6 in `/usr/local`),
  plus a new ROS 2-aware executable `orca_rl_sim`,

following `claude_stonefish_ros2_integration_prompt.md`.

```
integrated_Orca_RL/
  README.md                        <- this file (deliverables A-K)
  run_orca_rl.sh                   <- every command (build / run / watch / eval / train / docs ...)
  scripts/closed_loop_eval.sh      <- headless Stonefish + RL run per world, writes summaries
  docs/                            <- slide deck, report, figures and the scripts that generate them
  stonefish_sim/                   <- C++ (Stonefish + rclcpp)
    src/main_rl.cpp                   orca_rl_sim: scenario loading, sensors -> ROS 2, ROS 2 -> actuators
    src/OrcaStonefishRos.{hpp,cpp}    ROS 2 node "orca_stonefish" (topics/services, thread-safe slots)
    src/OrcaRLCommandAdapter.hpp      speed / yaw-rate / depth-rate setpoint -> thruster forces (300 Hz)
    src/OrcaController.{hpp,cpp}      original autopilot (RL mode = autopilot off + SetAction)
    resources/orca_rl_world.scn       base open-ocean world WITH obstacles, fish and traffic
    tools/make_rl_world.py, tools/mass_probe.cpp, test/test_command_adapter.cpp
  orca_rl/                         <- Python
    ros2_bridge/orca_rl_bridge_node.py   RL node "orca_rl_bridge" (rewritten, no Gazebo)
    ros2_bridge/launch_orca_bridge.launch.py
    integration/                         frames, sonar perception, navigation, observation, missions
    sim/obstacle_env.py, train_obstacle.py   obstacle-aware training (same ES/MLP methodology)
    logs/trained_policy.npz              original policy (unchanged)
    logs/trained_policy_obstacle.npz     new obstacle-aware policy
    tools/manual_cmd.py, check_stonefish_topics.py, sysid_stonefish.py
    tests/test_stonefish_integration.py
```

---

## Quick start

`./run_orca_rl.sh` wraps every command below (`build`, `run [world]`, `watch [world]` (chase camera),
`headless`, `autopilot`, `legacy`, `debug`, `manual`, `topics`, `eval`, `train`, `docs`, `stop`).


```bash
# 1. build (once)
source /opt/ros/jazzy/setup.bash
cd ~/Documents/integrated_Orca_RL/stonefish_sim
mkdir -p build && cd build && cmake .. && make -j$(nproc)

# 2. run the Orca under the RL policy in the obstacle world (window opens)
cd ~/Documents/integrated_Orca_RL
ros2 launch orca_rl/ros2_bridge/launch_orca_bridge.launch.py world:=rl
#    other scenarios:  world:=survey   world:=multisub
#    original C++ autopilot instead of RL:  mode:=autopilot
```

---

## A. Architecture

```
            Python  (orca_rl/ros2_bridge/orca_rl_bridge_node.py, node /orca_rl_bridge)
  +---------------------------------------------------------------------------+
  | IMU, DVL, pressure, GPS --> StonefishNavigator (DVL dead reckoning ->      |
  |                             existing ParticleFilterLocalizer)  est x,y,unc |
  | Profiler --> sector clearances + ObstacleMemory flanks --> risk            |
  | Multibeam --> seabed gradient richness                                     |
  | existing VehicleHealth / EnvironmentHealth --> ObservationBuilder (23-d)   |
  | existing MLPPolicy.forward --> orca_env action scaling --> safety shield   |
  +--------------------------------|------------------------------------------+
            ROS 2  /orca/cmd/setpoint [speed m/s, yaw rate rad/s, depth rate m/s]     (1 Hz, sim time)
                                   v
            C++  stonefish_sim/build/orca_rl_sim  (node /orca_stonefish)
  +---------------------------------------------------------------------------+
  | ROS 2 executor thread: subscriber/services -> mutex-protected slots        |
  | Stonefish sim thread, SimulationStepCompleted() @ 300 Hz:                  |
  |   latest setpoint (held; 3 s watchdog) -> OrcaRLCommandAdapter             |
  |   -> OrcaController (autopilot OFF) -> MainThruster.setSetpoint,           |
  |      YawThruster.setForce, HeaveThruster.setForce                          |
  |   -> Stonefish physics -> IMU/DVL/Pressure/GPS/Odometry/Profiler/Multibeam |
  |   -> publishers (+ /clock, contacts, applied forces)                       |
  | Main thread: Stonefish rendering (or idle loop with --headless)            |
  +--------------------------------|------------------------------------------+
            ROS 2  /orca/imu /orca/dvl/* /orca/pressure /orca/gps /orca/profiler /orca/multibeam ...
                                   v
                              back to Python
```

Only Stonefish integrates vehicle physics while the system runs;
`sim/vehicle_dynamics.py` / `sim/obstacle_env.py` are used only for offline training.

Threading (`main_rl.cpp`): Stonefish's `GraphicalSimulationApp` owns the main
thread (render loop) and runs physics on its own SDL thread, which calls
`SimulationStepCompleted`. `rclcpp::executors::SingleThreadedExecutor::spin()`
runs on a separate `std::thread`. Callbacks only copy into mutex-protected slots;
all Stonefish calls happen on the physics thread; `rclcpp` publishers are
thread-safe. The app subclasses override `LoopInternal()` so Ctrl-C
(`rclcpp::ok() == false`) quits Stonefish; closing the window cancels the
executor and shuts ROS 2 down.

Timing: physics 300 Hz; sensors at their `.scn` rates (IMU 100, Odometry 20,
DVL/Pressure/Profiler 10, Multibeam 2, GPS 1 Hz); `/clock` 100 Hz; RL control
**1 Hz on simulation time** (the original bridge's rate; the obstacle policy was
trained with dt = 1 s); dead reckoning 10 Hz. The C++ adapter holds the last
setpoint and closes its speed/yaw-rate/depth loops at 300 Hz.

---

## B. Gazebo / uuv_simulator -> Stonefish migration table

| Existing Gazebo/UUV assumption (old `orca_rl_bridge_node.py`, launch, package.xml, README s.6) | Actual Stonefish replacement | File changed | ROS 2 interface |
|---|---|---|---|
| `/orca/pose_gt` (nav_msgs/Odometry "ground truth") fed straight into the loop | Stonefish `Odometry` sensor, published **explicitly as ground truth**; used for the estimate only with `nav_mode:=DEBUG_GROUND_TRUTH`; in `REALISTIC_SENSOR` only logged as `gt_eval_only` / `nav_err` | `main_rl.cpp`, `orca_rl_bridge_node.py`, `integration/navigation.py` | `/orca/ground_truth/odometry` nav_msgs/Odometry (world_ned / body FRD) |
| `/orca/pressure` FluidPressure, depth = (P - 101325)/(rho g) | Stonefish `Pressure` sensor reports **gauge** pressure (verified in `Pressure.cpp`): depth = P/(rho g); subtracting 1 atm would put the Orca ~10 m too shallow | `main_rl.cpp`, `integration/frames.py` | `/orca/pressure` sensor_msgs/FluidPressure |
| `/orca/cmd_vel` geometry_msgs/Twist (linear.x, angular.z, linear.z) to a velocity-controller plugin | No velocity API exists. Setpoint [speed, yaw rate, depth rate] -> `OrcaRLCommandAdapter` (identified feed-forward + PI, 300 Hz) -> `SimpleThruster::setSetpoint`, `Push::setForce` x2 via `OrcaController` RL mode | `OrcaRLCommandAdapter.hpp`, `main_rl.cpp`, `orca_rl_bridge_node.py` | `/orca/cmd/setpoint` std_msgs/Float64MultiArray; raw forces on `/orca/cmd/thrust` |
| `/orca/thruster_command` uuv_gazebo_ros_plugins_msgs/FloatStamped | Raw actuator path for manual tests, straight to MainThruster/YawThruster/HeaveThruster | `OrcaStonefishRos.cpp`, `main_rl.cpp` | `/orca/cmd/thrust` Float64MultiArray [surge N, yaw N, heave N] |
| `/orca/dvl` uuv_sensor_msgs/DVL (subscribed in docs, never used) | Stonefish `DVL`: velocity rotated from the DVL frame (mounted pitched 180 deg) into body FRD; altitude; published only with bottom lock (status 0/2). The model's DVL could never lock (see Fixes) | `main_rl.cpp`, `orca_xluuv_survey.scn` | `/orca/dvl/twist` TwistWithCovarianceStamped, `/orca/dvl/altitude` sensor_msgs/Range (+inf = no lock) |
| heading placeholder `sin=0, cos=1` | Stonefish `IMU` roll/pitch/yaw -> quaternion, body rates, specific force | `main_rl.cpp`, `OrcaStonefishRos.cpp` | `/orca/imu` sensor_msgs/Imu |
| DVL dx,dy placeholder `predict(dx=0, dy=0)` | DVL body velocity x IMU yaw, integrated at 10 Hz -> existing `ParticleFilterLocalizer.predict` | `integration/navigation.py` | - |
| GPS (not handled) | Stonefish `GPS` (no fix under water) -> NavSatFix; a surface fix re-centres the filter | `main_rl.cpp`, `navigation.py`, `missions.py` | `/orca/gps` sensor_msgs/NavSatFix |
| Gazebo model state / `gazebo_msgs` | not needed; pose from navigation (or GT in debug) | bridge rewritten | - |
| Gazebo reset / pause / unpause services | `SimulationManager::RestartScenario` is not safe at runtime (rebuilds the GL scene from the physics thread), so reset = `FeatherstoneRobot::Respawn(startPose)` executed on the physics thread at the next step, adapter integrators reset. Static obstacles stay; animated fish keep their clock. No pause service (not needed for inference); use `--rtf` | `OrcaStonefishRos.cpp`, `main_rl.cpp` | `/orca/reset` std_srvs/Trigger |
| Gazebo spawn / `spawn_entity` / `robot_description` | none - the Orca comes from `<include file="data/orca_xluuv_survey.scn">` in the world file, parsed by `sf::ScenarioParser` in `BuildScenario()` | `main_rl.cpp` | - |
| Gazebo sensors (pose_gt, pressure, DVL plugins) | Stonefish IMU, DVL, Pressure, GPS, Odometry, Profiler, Multibeam actually mounted on `orca_xluuv_survey.scn` | `main_rl.cpp` | see E |
| Gazebo plugins (hydrodynamics, thruster) | Stonefish built-in `submerged` physics + `simple_thruster`/`push` actuators already in the model | - | - |
| Gazebo world / obstacles / risk map | Stonefish worlds; obstacle info only via Profiler -> `perception.py`; `OceanEnvironment` replaced at runtime by `StonefishOceanModel` (chart depth + sonar measurements) | `integration/world_model.py`, `perception.py` | `/orca/profiler`, `/orca/multibeam` LaserScan |
| collision information (none) | Bullet contact manifolds between Orca links and any other entity | `main_rl.cpp` | `/orca/contact` Bool, `/orca/contact/entity` String |
| simulator clock (wall time, `create_timer(1.0)`) | Stonefish simulation time published on `/clock`; Python node runs with `use_sim_time=true` | `OrcaStonefishRos.cpp`, bridge | `/clock` rosgraph_msgs/Clock |
| launch file "adjust for uuv_simulator / stonefish_ros2" | launches `orca_rl_sim` + RL node; Stonefish exit -> shutdown | `launch_orca_bridge.launch.py` | - |
| `package.xml` / `requirements.txt` mention uuv_simulator / Gazebo | standard ROS 2 messages only | `package.xml`, `requirements.txt` | - |

Remaining occurrences of the word "Gazebo" (repository-wide `grep -ri gazebo`):
historical notes only - `orca_rl/README.md` (states the Gazebo path was removed),
the header of `orca_rl_bridge_node.py` (states what it replaces), a comment in
`integration/frames.py` (why 101325 Pa is no longer subtracted), and this file.
Hits for "uuv" are otherwise the vehicle name "XL**UUV**". No runtime dependency
on Gazebo, gazebo_ros, gazebo_msgs, uuv_*, or stonefish_ros2 remains.

---

## C. Files changed / created

**stonefish_sim/** (from vijay_orca.zip)
* created: `src/main_rl.cpp`, `src/OrcaStonefishRos.hpp`, `src/OrcaStonefishRos.cpp`,
  `src/OrcaRLCommandAdapter.hpp`, `test/test_command_adapter.cpp`,
  `resources/orca_rl_world.scn` + `orca_rl_world_obstacles.json` (generated by
  `tools/make_rl_world.py`), `tools/mass_probe.cpp`
* changed: `CMakeLists.txt` (optional ROS 2 target, tests, probe, resource copy target),
  `src/OrcaController.{hpp,cpp}` (multibeam de-duplication bug),
  `resources/data/orca_xluuv_survey.scn`, `orca_xluuv.scn`, `survey_sub.scn`
  (hydrostatic trim, DVL range), `resources/orca_world.scn`, `orca_survey_world.scn`,
  `orca_multisub_world.scn` (added `SyntacticFoam` material), `README.md`
* unchanged: `src/main.cpp`, `src/main_survey.cpp`, `src/main_multisub.cpp`

**orca_rl/** (from orca_rl.zip)
* rewritten: `ros2_bridge/orca_rl_bridge_node.py`, `ros2_bridge/launch_orca_bridge.launch.py`,
  `ros2_bridge/package.xml`
* created: `integration/` (`frames.py`, `perception.py`, `world_model.py`, `navigation.py`,
  `observation.py`, `missions.py`, `policy_io.py`), `sim/obstacle_env.py`,
  `train_obstacle.py`, `tools/manual_cmd.py`, `tools/check_stonefish_topics.py`,
  `tools/sysid_stonefish.py`, `tests/test_stonefish_integration.py`,
  `logs/trained_policy_obstacle.npz`, `logs/training_history_obstacle.json`,
  `logs/sysid_stonefish.json`
* extended (additive, defaults unchanged): `sim/vehicle_dynamics.py` (+`StonefishOrcaParams`),
  `sim/safety_shield.py` (+`ObstacleAwareSafetyShield`, original rules applied first),
  `rl/es_trainer.py` (+optional `n_workers`, `common_seeds`, `init_theta`, `on_generation`)
* edited docs: `README.md` (section 6 now points here), `requirements.txt`
* unchanged: `sim/orca_env.py`, `sensors.py`, `localization.py`, `health_monitor.py`,
  `ocean_environment.py`, `baseline_planner.py`, `rl/policy_network.py`, `train.py`,
  `ppo_train.py`, `evaluate.py`, `logs/trained_policy.npz`

**scripts/closed_loop_eval.sh** (created)

---

## D. Code

All code is in the files above (complete, not patches). Starting points:
`stonefish_sim/src/main_rl.cpp`, `orca_rl/ros2_bridge/orca_rl_bridge_node.py`.

---

## E. ROS 2 interfaces

Nodes: `/orca_stonefish` (inside `orca_rl_sim`), `/orca_rl_bridge` (Python).

| Topic / service | Type | Pub / Sub | Content |
|---|---|---|---|
| `/orca/cmd/setpoint` | std_msgs/Float64MultiArray | bridge -> sim | `[surge_speed m/s, yaw_rate rad/s (+ = bow to starboard), depth_rate m/s (+ = down)]`; held, 3 s sim-time watchdog -> stop & hold depth |
| `/orca/cmd/thrust` | std_msgs/Float64MultiArray | manual -> sim | `[MainThruster N, YawThruster N, HeaveThruster N]` (bypasses the adapter) |
| `/orca/imu` | sensor_msgs/Imu | sim | 100 Hz, orientation world_ned<-body_frd, body rates, specific force |
| `/orca/dvl/twist` | geometry_msgs/TwistWithCovarianceStamped | sim | 10 Hz, body-frame velocity, only with bottom lock |
| `/orca/dvl/altitude` | sensor_msgs/Range | sim | 10 Hz, altitude, +inf = no lock |
| `/orca/pressure` | sensor_msgs/FluidPressure | sim | 10 Hz, gauge Pa |
| `/orca/gps` | sensor_msgs/NavSatFix | sim | 1 Hz, NO_FIX + NaN when submerged |
| `/orca/profiler` | sensor_msgs/LaserScan | sim | 10 Hz, 33 bins x 5 deg over +-80 deg, newest return per bin from the swept beam; +inf = nothing within 30 m, NaN = not swept yet; frame `OrcaXLUUV/profiler_frd` (angle + = starboard) |
| `/orca/multibeam` | sensor_msgs/LaserScan | sim | 2 Hz, 129 beams over 120 deg across-track, pointing down |
| `/orca/ground_truth/odometry` | nav_msgs/Odometry | sim | 20 Hz, **ground truth** (debug/evaluation only) |
| `/orca/actuators` | std_msgs/Float64MultiArray | sim | 20 Hz, `[F_surge, F_yaw, F_heave, main thrust produced, setpoint(3), depth ref, source 0 none/1 setpoint/2 thrust/3 autopilot]` |
| `/orca/contact`, `/orca/contact/entity` | std_msgs/Bool, std_msgs/String | sim | 10 Hz, Orca touching another entity, and which |
| `/orca/status` | std_msgs/String (transient local) | sim | world and mode |
| `/clock` | rosgraph_msgs/Clock | sim | 100 Hz simulation time |
| `/orca/rl/observation`, `/orca/rl/action` | std_msgs/Float64MultiArray | bridge | the policy input / raw output each step |
| `/orca/rl/estimate` | nav_msgs/Odometry | bridge | navigation estimate + covariance |
| `/orca/rl/status` | std_msgs/String | bridge | JSON run summary at the end |
| `/orca/reset` | std_srvs/Trigger | sim service | respawn the Orca at its start pose |
| `/orca/set_autopilot` | std_srvs/SetBool | sim service | true = original lawnmower autopilot, false = external RL |

Parameters - `orca_stonefish`: `command_timeout` (3.0 s), `world_frame`, `body_frame`.
`orca_rl_bridge`: `world` (rl|survey|multisub), `nav_mode` (REALISTIC_SENSOR|DEBUG_GROUND_TRUTH),
`policy_path`, `control_rate_hz` (1.0), `use_safety_shield` (true), `max_mission_time`,
`loop_mission`, `exit_on_finish`, `log_dir`; `use_sim_time` defaults to true.

### Coordinate frames (verified in the Stonefish source, not from comments)

| | convention |
|---|---|
| Stonefish world | NED: x north, y east, z down (depth = z) |
| Stonefish / Orca body | FRD: x forward, y starboard, z down |
| ROS 2 topics | published in the same NED / FRD frames (`world_ned`, `orca/base_link_frd`, REP-105 `_ned` style); `integration/frames.ned_to_enu` for RViz |
| Python RL | x north, y east, heading = NED yaw, depth + down. `orca_env.py` integrates `x += v cos h, y += v sin h` - already NED, so no conversion |
| yaw / yaw rate | radians; + = clockwise from above (bow to starboard), IMU gyro z = NED yaw rate when level |
| quaternion | ROS (x, y, z, w), ZYX Euler from the IMU |
| velocities | Odometry twist and DVL are **body** frame (`Odometry.cpp`, `DVL.cpp`); DVL additionally rotated from its pitched-180 deg mount into the body frame in C++ |
| actuators | MainThruster +X forward; YawThruster at the bow, + pushes the bow to starboard (+ yaw); HeaveThruster + = down |

### Where the 12 original observation features come from in Stonefish

| feature | Stonefish source | note |
|---|---|---|
| est_x_norm, est_y_norm | particle filter fed by DVL x IMU dead reckoning (+GPS at surface) | GT only in DEBUG mode |
| depth_norm | Pressure (gauge) / rho g / 3400 | |
| heading sin/cos | IMU yaw | |
| uncertainty_norm | particle-filter covariance trace / 20000 | flat seabed -> grows slowly (TAN uninformative) |
| battery_frac, thruster_wear, nav_residual, depth_margin | existing `VehicleHealth`, energy from applied MainThruster force x DVL speed | commanded vs DVL-measured speed for the residual |
| gradient_richness | Multibeam across-track slope + depth spread, in [0, 1] | was the bathymetry-gradient grid |
| risk | max(Profiler obstacle proximity, DVL-altitude risk) | was the procedural shallow/shipping-lane map; no such map exists in Stonefish |

The obstacle policy appends: goal bearing (sin, cos) and distance, 5 Profiler
sector clearances, port/starboard hull clearance from a 30 s sonar memory, and
depth error w.r.t. the mission depth (23-d total, `integration/observation.py`).

### Thrust scaling (identified, not assumed)

`orca_rl/tools/sysid_stonefish.py` (raw thrust steps, results in `logs/sysid_stonefish.json`):

| test | Stonefish result |
|---|---|
| MainThruster 10 / 20 / 30 kN | 0.37 / 0.68 / 0.93 m/s steady, tau63 ~6.5 s -> F = 9000 u|u| + 23500 u, 40 kN limit -> ~1.15 m/s |
| YawThruster 2 / 4 / 8 kN at ~0.5 m/s | 0.014 / 0.028 / 0.048 rad/s, tau63 < 1 s -> F ~= 150000 r |
| 8 kN at 0.87 m/s (closed loop) | 0.037 rad/s (yaw authority falls with speed) |

So the original `OrcaVehicleParams` envelope (4.11 m/s, 3 deg/s) is not achievable
by the Stonefish model. `StonefishOrcaParams` (1.1 m/s, 0.045 rad/s, speed-dependent
yaw limit) is used for the obstacle policy; the C++ adapter additionally clips to
1.1 m/s / 0.048 rad/s / 0.3 m/s whatever it receives. Depth rate is real: the
survey model has a HeaveThruster (+-55 kN as clamped by OrcaController); the
adapter integrates the depth-rate command into a depth reference held with the
autopilot's own PD gains and an altitude floor guard.

---

## F. Build

```bash
source /opt/ros/jazzy/setup.bash           # ROS_DISTRO detected by CMake; any ROS 2 with rclcpp works
cd ~/Documents/integrated_Orca_RL/stonefish_sim
mkdir -p build && cd build
cmake ..                                   # "-- ROS 2 found (jazzy) - building orca_rl_sim"
make -j$(nproc)
./orca_rl_adapter_test                     # "all checks passed"
cd ~/Documents/integrated_Orca_RL/orca_rl && python3 -m pytest -q tests
```

## G. Run

```bash
cd ~/Documents/integrated_Orca_RL
# everything at once (Stonefish window + RL node):
ros2 launch orca_rl/ros2_bridge/launch_orca_bridge.launch.py world:=rl
ros2 launch orca_rl/ros2_bridge/launch_orca_bridge.launch.py world:=survey
ros2 launch orca_rl/ros2_bridge/launch_orca_bridge.launch.py world:=multisub headless:=true rtf:=3

# or two terminals:
cd stonefish_sim/build && ./orca_rl_sim --world rl
python3 orca_rl/ros2_bridge/orca_rl_bridge_node.py --ros-args -p world:=rl

# original policy instead of the obstacle-aware one:
python3 orca_rl/ros2_bridge/orca_rl_bridge_node.py --ros-args -p world:=rl \
    -p policy_path:=$HOME/Documents/integrated_Orca_RL/orca_rl/logs/trained_policy.npz

# batch evaluation (headless, 3x real time, writes logs/stonefish_runs/*_summary.json)
scripts/closed_loop_eval.sh orca_rl/logs/trained_policy_obstacle.npz 3 900 rl survey multisub
```

## H. Manual Orca movement test (no RL)

```bash
cd ~/Documents/integrated_Orca_RL/stonefish_sim/build && ./orca_rl_sim --world rl     # terminal 1
# terminal 2:
python3 ~/Documents/integrated_Orca_RL/orca_rl/tools/manual_cmd.py --speed 0.6                   # forward
python3 ~/Documents/integrated_Orca_RL/orca_rl/tools/manual_cmd.py --speed 0.5 --yaw-rate 0.03   # turn starboard
python3 ~/Documents/integrated_Orca_RL/orca_rl/tools/manual_cmd.py --thrust 20000 0 0            # raw 20 kN
# plain CLI equivalent:
ros2 topic pub -r 2 /orca/cmd/setpoint std_msgs/msg/Float64MultiArray "{data: [0.6, 0.0, 0.0]}"
ros2 topic pub -r 2 /orca/cmd/setpoint std_msgs/msg/Float64MultiArray "{data: [0.5, 0.03, 0.0]}"
ros2 service call /orca/reset std_srvs/srv/Trigger
```
(In `world:=rl` the first pinnacle is 55 m dead ahead of the start - a manual
straight run hits it after ~70 s; that is the contact detector working.)

## I. Sensor / observation test

```bash
ros2 node list                    # /orca_stonefish
ros2 topic list
python3 ~/Documents/integrated_Orca_RL/orca_rl/tools/check_stonefish_topics.py --duration 8
ros2 topic echo /orca/profiler --once
ros2 topic echo /orca/rl/observation          # while the RL node runs
```

## J. RL inference test

```bash
# debug connectivity first (ground truth position, flagged in the log):
python3 orca_rl/ros2_bridge/orca_rl_bridge_node.py --ros-args -p world:=rl -p nav_mode:=DEBUG_GROUND_TRUTH
# then the real thing:
python3 orca_rl/ros2_bridge/orca_rl_bridge_node.py --ros-args -p world:=rl -p nav_mode:=REALISTIC_SENSOR
```
Each run writes `orca_rl/logs/stonefish_runs/<world>_<layout>_<mode>_<time>.jsonl`
(per-step estimate, ground truth for evaluation, clearances, action, setpoint,
shield, reward, contact) and a `_summary.json`. Training against Stonefish itself
was not started (inference only, as the prompt's Test 8 requires closed-loop
inference first).

---

## Results and verification status

### What was verified at runtime on this machine (Ubuntu 24.04, ROS 2 Jazzy, Stonefish 1.6, RTX 3050)

| test (prompt section 27) | result |
|---|---|
| 1. Stonefish alone | original `orca_sim` / `orca_survey_sim` / `orca_multisub_sim` build and run against `/usr/local` Stonefish |
| 2. Stonefish ROS 2 executable | `ros2 node list` -> `/orca_stonefish`; all topics/services above listed |
| 3. ROS 2 action -> Orca | `/orca/cmd/thrust` 20 kN -> 0.55-0.68 m/s; `/orca/cmd/setpoint` 0.8 m/s -> 0.81-0.85 m/s tracked; yaw / depth respond with the documented signs |
| 4. Stonefish -> ROS 2 | IMU 100 Hz, pressure 10 Hz (120.5 kPa = 11.99 m), DVL 10 Hz (16.0 m altitude after the DVL fix), GPS NO_FIX submerged, profiler/multibeam scans, contact events, /clock |
| 5. Python receives Stonefish data | REALISTIC_SENSOR estimate vs ground truth: 0.3-0.7 m error over 600 s runs |
| 6. existing trained policy (inference) | `trained_policy.npz` runs in closed loop; it circles (yaw saturated at +1) and drifts deeper until the shield's depth band stops it - the same behaviour it shows in its own NumPy env (0/5 goals in `eval_summary.json`). It reached 1 of 10 survey waypoints in 200 s |
| 7. closed loop | Stonefish -> ROS 2 -> Python observation -> policy -> ROS 2 -> Stonefish, 1 Hz on sim time, see table below |
| 8. training | done offline in `sim/obstacle_env.py` (fast NumPy env calibrated to the Stonefish Orca); no training against Stonefish itself yet |
| reset | `/orca/reset` (Robot::Respawn) used between system-identification steps |
| launch | `ros2 launch .../launch_orca_bridge.launch.py` starts both processes; mode:=autopilot verified |
| unit tests | `orca_rl_adapter_test` all checks passed; `pytest tests` 18 passed |

### The original policy and why a second policy was trained

`logs/trained_policy.npz` is kept unchanged and still loads and runs
(`policy_path:=.../trained_policy.npz`). But it cannot do what you asked for:
its 12-d observation contains no goal and no obstacle information, and in its
own environment it saturates yaw at +1 and dives/surfaces (0 % goals in the
shipped `eval_summary.json`; reproduced here). So the obstacle-aware task needs
a policy that sees the sonar and the goal. `logs/trained_policy_obstacle.npz`
keeps the methodology - same `MLPPolicy` class (2 x 64 tanh), same ES trainer,
same reward terms (+ collision penalty and per-goal bonus), the safety shield in
the loop, the same localisation / health modules - and changes only the input
(23-d "obstacle_v1") and the task. ES from a random network learned too slowly
(8 % of goals after 50 generations), so it is warm-started by behaviour
cloning the classical planner (`sim/obstacle_baseline.py`, one DAgger round)
and then optimised with ES for 200 generations (65 min on 14 cores).
Flank awareness comes from `ObstacleMemory`: Profiler returns placed in the
world with the *estimated* pose, cleared only when a later beam looks
through them (free-space carving) or the vehicle is > 60 m away, so a rock
that slides out of the forward fan is not forgotten while the 27 m hull
swings past it (an earlier 30 s time-out caused stern strikes on pinnacles).

Selection: the BC+ES policy was compared with an 80-generation ES fine-tune
(after the obstacle memory was changed to free-space carving) on 100
validation seeds (6000-6099); the fine-tune was worse (16 % vs 8 % collisions)
and was rejected (`logs/train_obstacle_finetune_REJECTED.log`).

Offline test, 100 seeds never used for training or selection
(`orca_rl/logs/eval_obstacle_summary.json`, seeds 7000-7099, 3 goals per episode):

| | goals reached | all 3 goals | collision rate | median min hull clearance |
|---|---|---|---|---|
| classical potential-field planner | 68 % | 46 % | 28 % | 1.7 m |
| behaviour cloning only | 67 % | 47 % | 32 % | 1.7 m |
| **RL (BC + ES), deployed** | **82 %** | **63 %** | **13 %** | **5.9 m** |

Stonefish closed loop with the final policy (`scripts/closed_loop_eval.sh`, headless, 3x real time,
REALISTIC_SENSOR navigation, logs + plots in `orca_rl/logs/stonefish_runs/`):

| world | result | waypoints | sim time | contacts (static / fish / subs) | physics resets | mean nav error |
|---|---|---|---|---|---|---|
| `rl` (base world + obstacles) | mission complete | 14 reached, 1 skipped (waypoint on Pinnacle05) | 1485 s | 6 / 5 / 0 | 0 | 0.64 m |
| `multisub` (2 other subs) | mission complete | 8 reached, 1 timed out | 858 s | 0 / 0 / 4 | 1 | 0.69 m |
| `survey` (vijay's dense field) | 1800 s limit | 4 reached, 6 timed out | 1800 s | 23 (incl. 14 seabed) / 44 / 0 | 15 | 14.1 m |

The `survey` world is the open problem: a 90 x 70 m box for a 27 m hull, with
large scripted (non-yielding) fish that push the hull into the seabed and make
Bullet diverge; every reset puts the Orca back on its track, which realistic
navigation cannot observe, so the estimate drifts and legs time out.


### Things to know

* The Stonefish Orca cannot reach the published Orca envelope: 1.15 m/s max
  (40 kN thruster), ~2.7 deg/s yaw. All policies are scaled to what the model
  can do (see Thrust scaling).
* TAN is uninformative in these worlds (flat seabed planes); the estimate is
  DVL dead reckoning, still well under 1 m error over 10-30 minutes.
* Stonefish's scripted fish (`<animated>`) are kinematic, infinite-mass bodies
  that never yield. They can strike the hull from outside the forward sonar
  fan, and can squeeze the hull into the seabed or another hull hard enough to
  make Bullet diverge. `orca_rl_sim` detects such divergences (non-finite or >
  10 m/s state), puts the Orca back on its track ~2 s earlier and reports it
  (`physics_resets` in the run summary). In training the movers are evasive.
* Contacts are reported, not hidden: every run summary lists them with the
  entity names.
* What was NOT done: training directly against Stonefish (Test 8 is offline,
  as the prompt asks to stop at inference); a ROS 2 package build (the Python
  node runs directly with python3; package.xml records its dependencies).


---

## Documentation (slides, report, figures)

| file | what |
|---|---|
| `docs/Orca_RL_Stonefish_Review.pptx` (+ `.pdf`) | 36-slide review deck in the college template (`docs/templates/PPT_format_22AIE450.pptx`): Part A the RL policy, Part B the Stonefish simulation, Part C how they are connected + results |
| `docs/report/Orca_RL_Stonefish_Report.pdf` | 6-page report in the IEEE two-column format of the earlier `orca_xluuv_report_updated.pdf` |
| `docs/report/Orca_RL_Stonefish_Report.tex` | the same report as IEEEtran LaTeX (compile on Overleaf; figures from `docs/figures/`) |
| `docs/figures/*.png` | every figure (training curves, policy probe, evaluation, system ID, trim, architecture, Stonefish runs, screenshots) |
| `docs/data/` | logs the figures are made from (capsize before/after, setpoint tracking, original-policy run, validation selection, autopilot baseline, reward breakdown); `final_runs.json` pins the three Stonefish evaluation runs used for the results, so later runs in `orca_rl/logs/stonefish_runs` do not change the slides or report (edit it to point at new runs) |

Regenerate everything from the logged data (no simulator needed):

```bash
./run_orca_rl.sh docs
# = python3 docs/make_figures.py && python3 docs/build_presentation.py && python3 docs/report/build_report.py
docs/capture_screenshots.sh rl 20 45     # new Stonefish screenshots (opens the window; X11)
```
`build_presentation.py` writes the slides straight into the template's XML (standard library + Pillow);
`build_report.py` prints the HTML with headless Brave/Chrome and also writes the `.tex`.

---

## K. Debugging guide

| symptom | check |
|---|---|
| Orca doesn't move | `ros2 topic echo /orca/actuators`: last field 0 = no command / watchdog (republish at >= 0.5 Hz sim time), 3 = autopilot on (`ros2 service call /orca/set_autopilot std_srvs/srv/SetBool "{data: false}"`). Forces non-zero but no motion -> `/orca/contact/entity` (pressed against something) |
| ROS 2 topic missing | was cmake run with ROS sourced ("building orca_rl_sim")? same `ROS_DOMAIN_ID` in every terminal? `ros2 node list` |
| callback doesn't fire | sensor topics are best-effort (SensorDataQoS): subscribers must use `qos_profile_sensor_data`; `/orca/status` is transient-local |
| wrong thrust scaling | `tools/sysid_stonefish.py` against your build, update `OrcaRLCommandAdapter.hpp` and `StonefishOrcaParams` |
| wrong yaw direction | + yaw rate must turn the bow to starboard (NED). `manual_cmd.py --speed 0.5 --yaw-rate 0.03`: IMU yaw must increase |
| wrong depth sign | `/orca/pressure` is gauge; depth = P/(1025*9.81) (~120 kPa at 12 m). + depth rate = deeper |
| NED/ENU mismatch | every topic is NED/FRD; convert with `integration/frames.ned_to_enu` for RViz; never mix ROS ENU tools' yaw with these |
| Python receives no sensor data | `check_stonefish_topics.py`; the bridge waits for `/orca/imu` and `/orca/pressure`; `use_sim_time` needs `/clock` |
| Stonefish freezes / black window | try `--headless`; GPU: OpenGL 4.3+ required; wave rendering is disabled in the .scn (flat ocean) for Mesa |
| ROS 2 blocks Stonefish | never call `rclcpp::spin` on the main thread (it runs on its own thread here); callbacks must not touch Stonefish objects |
| reset problems | `/orca/reset` = `Robot::Respawn` at the next physics step; it does not restart the fish clocks or traffic sub; restart the executable for a full reset |
| build/linker problems | `Max` macro clash: include ROS headers before Stonefish headers (done in `main_rl.cpp`); Stonefish needs `BT_USE_DOUBLE_PRECISION` (comes from its CMake target); `libStonefish.so` must be found at runtime (`/usr/local/lib`) |
| headless crash "Lights not supported in console mode" | handled: `--headless` strips `<light>` elements after parsing includes |
| position estimate jumps / ground truth alternates between two places / many odd contacts | two `orca_rl_sim` instances on the same `ROS_DOMAIN_ID` (e.g. a headless one still running): `ros2 node list` shows `/orca_stonefish` twice. `pkill -x orca_rl_sim` or use a different `ROS_DOMAIN_ID` per run |
| hull rolls over | use the trimmed models in this folder (see stonefish_sim/README "Fixes"); check with `./orca_mass_probe <world>.scn OrcaXLUUV ./resources/` |

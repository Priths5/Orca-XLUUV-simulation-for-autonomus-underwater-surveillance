#!/usr/bin/env python3
"""
orca_rl_bridge_node.py
------------------------
ROS 2 node that runs the trained RL policy against the Stonefish Orca
(stonefish_sim -> executable orca_rl_sim). Replaces the old Gazebo /
uuv_simulator skeleton: no gazebo_msgs, no uuv_* messages, no spawn / pause /
reset services, no Twist on /orca/cmd_vel, no /orca/pose_gt.

    Stonefish sensors --ROS 2--> this node:
        /orca/imu (sensor_msgs/Imu)                   heading, yaw rate
        /orca/dvl/twist (TwistWithCovarianceStamped)  body velocity (bottom lock)
        /orca/dvl/altitude (sensor_msgs/Range)        altitude above seabed
        /orca/pressure (sensor_msgs/FluidPressure)    depth (gauge Pa / rho g)
        /orca/gps (sensor_msgs/NavSatFix)             surface fixes only
        /orca/profiler (sensor_msgs/LaserScan)        obstacle sonar
        /orca/multibeam (sensor_msgs/LaserScan)       seabed sonar
        /orca/actuators (std_msgs/Float64MultiArray)  applied thrust (energy)
        /orca/contact (std_msgs/Bool)                 collisions (evaluation)
        /orca/ground_truth/odometry (nav_msgs/Odometry)
              DEBUG_GROUND_TRUTH mode: used as the position estimate.
              REALISTIC_SENSOR mode: only logged next to the estimate for
              evaluation (navigation error); never enters the observation.
    -> existing processing (sim/ + integration/):
        StonefishNavigator (DVL dead reckoning + ParticleFilterLocalizer),
        VehicleHealth, EnvironmentHealth, sonar perception, ObservationBuilder
    -> MLPPolicy.forward(obs) -> action in [-1, 1]^3
    -> orca_env.py action scaling -> safety shield -> physical setpoint
    -> /orca/cmd/setpoint (std_msgs/Float64MultiArray)
           [surge_speed m/s, yaw_rate rad/s (NED, + = starboard), depth_rate m/s (+ = down)]
       which orca_rl_sim turns into MainThruster / YawThruster / HeaveThruster forces.

Control runs at a fixed rate on SIMULATION time (/clock), default 1 Hz.

Run (after `source /opt/ros/<distro>/setup.bash`):
    python3 orca_rl/ros2_bridge/orca_rl_bridge_node.py --ros-args \
        -p world:=rl -p nav_mode:=REALISTIC_SENSOR
"""

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import numpy as np

try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rcl_interfaces.msg import ParameterDescriptor
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import TwistWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import FluidPressure, Imu, LaserScan, NavSatFix, Range
    from std_msgs.msg import Bool, Float64MultiArray, String
except ImportError:
    raise SystemExit("This node needs a sourced ROS 2 environment (rclpy, sensor_msgs, nav_msgs, "
                     "geometry_msgs, std_msgs): source /opt/ros/<distro>/setup.bash")

from integration import perception as P
from integration.frames import depth_from_gauge_pressure, yaw_from_quaternion
from integration.missions import Mission, geodetic_to_ned
from integration.navigation import DEBUG_GROUND_TRUTH, REALISTIC_SENSOR, StonefishNavigator
from integration.observation import ObservationBuilder, action_to_command
from integration.policy_io import load_policy
from integration.world_model import StonefishOceanModel
from sim.health_monitor import EnvironmentHealth, VehicleHealth
from sim.safety_shield import ObstacleAwareSafetyShield

DVL_OFFSET_Z = 1.45      # DVL is 1.45 m below the hull axis (orca_xluuv_survey.scn)
LEGACY_W = dict(progress=1.0, uncertainty=0.0025, risk=0.4, energy=0.02, health=3.0, idle=0.5)
OBSTACLE_W = dict(progress=1.0, uncertainty=0.0025, risk=0.4, energy=0.02, health=1.5, idle=0.5)


class OrcaRLBridge(Node):
    def __init__(self):
        super().__init__("orca_rl_bridge",
                         parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        default_policy = os.path.join(ROOT, "logs", "trained_policy_obstacle.npz")
        if not os.path.exists(default_policy):
            default_policy = os.path.join(ROOT, "logs", "trained_policy.npz")
        self.world = self.declare_parameter("world", "rl").value
        self.nav_mode = self.declare_parameter("nav_mode", REALISTIC_SENSOR).value
        policy_path = self.declare_parameter("policy_path", default_policy).value
        num = ParameterDescriptor(dynamic_typing=True)   # accept 1800 as well as 1800.0
        self.rate_hz = float(self.declare_parameter("control_rate_hz", 1.0, num).value)
        self.use_shield = bool(self.declare_parameter("use_safety_shield", True).value)
        self.loop_mission = bool(self.declare_parameter("loop_mission", False).value)
        self.log_dir = self.declare_parameter("log_dir", os.path.join(ROOT, "logs", "stonefish_runs")).value
        self.exit_on_finish = bool(self.declare_parameter("exit_on_finish", False).value)
        self.exit_now = False
        max_time = float(self.declare_parameter("max_mission_time", 0.0, num).value)
        self.max_time = max_time if max_time > 0 else None

        if self.nav_mode not in (REALISTIC_SENSOR, DEBUG_GROUND_TRUTH):
            raise SystemExit(f"nav_mode must be {REALISTIC_SENSOR} or {DEBUG_GROUND_TRUTH}")

        # ---- the existing RL stack ----
        self.policy, self.layout, self.params, meta = load_policy(policy_path)
        self.obs_builder = ObservationBuilder(self.layout)
        self.mission = Mission(self.world, loop=self.loop_mission)
        self.ocean = StonefishOceanModel(self.mission.seabed_depth)
        self.rng = np.random.default_rng(0)
        self.nav = StonefishNavigator(self.ocean, self.nav_mode, rng=self.rng)
        self.nav.reset(*self.mission.start)
        self.origin = self.mission.start
        self.env_health = EnvironmentHealth(self.ocean)
        self.vehicle_health = VehicleHealth(self.params)
        self.shield = ObstacleAwareSafetyShield(self.params)
        self.memory = P.ObstacleMemory()
        self.w = OBSTACLE_W if self.layout == "obstacle_v1" else LEGACY_W

        # ---- latest sensor data ----
        self.yaw = None
        self.yaw_rate = 0.0
        self.depth = None
        self.altitude = None
        self.dvl_uv = None
        self.dvl_stamp = -1.0
        self.gps_ne = None
        self.scan_angles = P.profiler_bin_angles()
        self.scan = np.full(P.PROFILER_BINS, np.nan)
        self.prev_scan = None
        self.mb_ranges = None
        self.surge_force = 0.0
        self.contact = False
        self.contact_events = []
        self.physics_resets = 0
        self.gt = None
        self.last_cmd = (0.0, 0.0, 0.0)
        self.last_integrate = None
        self.prev_goal_dist = None
        self.total_reward = 0.0
        self.steps = 0
        self.shield_overrides = 0
        self.min_profiler = P.PROFILER_RANGE
        self.t_start = None
        self.finished = False

        sq = qos_profile_sensor_data
        self.create_subscription(Imu, "/orca/imu", self.on_imu, sq)
        self.create_subscription(TwistWithCovarianceStamped, "/orca/dvl/twist", self.on_dvl, sq)
        self.create_subscription(Range, "/orca/dvl/altitude", self.on_altitude, sq)
        self.create_subscription(FluidPressure, "/orca/pressure", self.on_pressure, sq)
        self.create_subscription(NavSatFix, "/orca/gps", self.on_gps, sq)
        self.create_subscription(LaserScan, "/orca/profiler", self.on_profiler, sq)
        self.create_subscription(LaserScan, "/orca/multibeam", self.on_multibeam, sq)
        self.create_subscription(Float64MultiArray, "/orca/actuators", self.on_actuators, 10)
        self.create_subscription(Bool, "/orca/contact", self.on_contact, 10)
        self.create_subscription(String, "/orca/contact/entity", self.on_contact_entity, 10)
        self.create_subscription(Odometry, "/orca/ground_truth/odometry", self.on_ground_truth, sq)
        from rclpy.qos import DurabilityPolicy, QoSProfile
        self.create_subscription(String, "/orca/status", self.on_sim_status,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self.cmd_pub = self.create_publisher(Float64MultiArray, "/orca/cmd/setpoint", 10)
        self.obs_pub = self.create_publisher(Float64MultiArray, "/orca/rl/observation", 10)
        self.action_pub = self.create_publisher(Float64MultiArray, "/orca/rl/action", 10)
        self.est_pub = self.create_publisher(Odometry, "/orca/rl/estimate", 10)
        self.status_pub = self.create_publisher(String, "/orca/rl/status", 10)

        self.create_timer(0.1, self.integrate_step)               # 10 Hz dead reckoning (sim time)
        self.create_timer(1.0 / self.rate_hz, self.control_step)  # RL control rate (sim time)

        os.makedirs(self.log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"{self.world}_{self.layout}_{self.nav_mode.lower()}_{stamp}.jsonl")
        self.log_file = open(self.log_path, "w")
        self.get_logger().info(
            f"policy {os.path.basename(policy_path)} (layout {self.layout}, {type(self.params).__name__}), "
            f"world '{self.world}', nav {self.nav_mode}, control {self.rate_hz:g} Hz on sim time, "
            f"{len(self.mission.waypoints)} survey waypoints. Log: {self.log_path}")
        if self.nav_mode == DEBUG_GROUND_TRUTH:
            self.get_logger().warn("DEBUG_GROUND_TRUTH: the policy is being fed Stonefish ground-truth "
                                   "position. Connectivity testing only - not a valid RL experiment.")

    # ---------------------------------------------------------------- sensors
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_imu(self, m):
        q = m.orientation
        self.yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.yaw_rate = m.angular_velocity.z

    def on_dvl(self, m):
        self.dvl_uv = (m.twist.twist.linear.x, m.twist.twist.linear.y)
        self.dvl_stamp = self.now()

    def on_altitude(self, m):
        self.altitude = float(m.range) if math.isfinite(m.range) else None

    def on_pressure(self, m):
        self.depth = depth_from_gauge_pressure(m.fluid_pressure)

    def on_gps(self, m):
        # status < 0 = no fix (antenna under water)
        self.gps_ne = geodetic_to_ned(m.latitude, m.longitude) if m.status.status >= 0 else None

    def on_profiler(self, m):
        n = len(m.ranges)
        self.scan_angles = m.angle_min + m.angle_increment * np.arange(n)
        self.scan = np.array(m.ranges, dtype=float)
        if self.yaw is None:
            return
        changed = np.ones(n, bool) if self.prev_scan is None or self.prev_scan.size != n else \
            ~((self.scan == self.prev_scan) | (np.isnan(self.scan) & np.isnan(self.prev_scan)))
        self.prev_scan = self.scan.copy()
        ex, ey = self.current_estimate()
        self.memory.add_scan(self.now(), ex, ey, self.yaw, self.scan_angles[changed], self.scan[changed])

    def on_multibeam(self, m):
        self.mb_ranges = np.array(m.ranges, dtype=float)

    def on_actuators(self, m):
        if len(m.data) >= 1:
            self.surge_force = m.data[0]

    def on_contact(self, m):
        if m.data and not self.contact:
            self.contact_events.append(dict(t=self.now(), entity=None))
        self.contact = m.data

    def on_contact_entity(self, m):
        if m.data and self.contact_events and self.contact_events[-1]["entity"] is None:
            self.contact_events[-1]["entity"] = m.data
            self.get_logger().warn(f"CONTACT with {m.data}")

    def on_sim_status(self, m):
        if "physics_reset=" in m.data:
            n = int(m.data.split("physics_reset=")[1].split()[0])
            if n > self.physics_resets:
                self.physics_resets = n
                self.get_logger().warn(f"Stonefish physics divergence #{n}: Orca put back ~2 s along its track")

    def on_ground_truth(self, m):
        p = m.pose.pose.position
        self.gt = (p.x, p.y, p.z)
        self.nav.set_ground_truth(p.x, p.y)

    # -------------------------------------------------------------- helpers
    def current_estimate(self):
        return self.nav.est[0] + self.nav.acc_dx, self.nav.est[1] + self.nav.acc_dy

    def integrate_step(self):
        t = self.now()
        if self.yaw is None or self.last_integrate is None:
            self.last_integrate = t
            return
        dt = t - self.last_integrate
        self.last_integrate = t
        dvl = self.dvl_uv if (t - self.dvl_stamp) < 0.35 else None
        self.nav.integrate(dt, self.yaw, dvl_uv=dvl, commanded_speed=self.last_cmd[0])

    def publish_setpoint(self, speed, yaw_rate, depth_rate):
        msg = Float64MultiArray()
        msg.data = [float(speed), float(yaw_rate), float(depth_rate)]
        self.cmd_pub.publish(msg)
        self.last_cmd = (float(speed), float(yaw_rate), float(depth_rate))

    # ------------------------------------------------------------ RL step
    def control_step(self):
        t = self.now()
        if self.yaw is None or self.depth is None:
            self.get_logger().info("waiting for /orca/imu and /orca/pressure from orca_rl_sim ...",
                                   throttle_duration_sec=5.0)
            return
        if self.t_start is None:
            self.t_start = t
        if self.finished:
            self.publish_setpoint(0.0, 0.0, 0.0)
            return
        dt = 1.0 / self.rate_hz

        # ---- navigation (realistic: DVL/IMU/pressure/GPS; debug: ground truth) ----
        seabed = None
        if self.altitude is not None:
            seabed = self.depth + DVL_OFFSET_Z + self.altitude
        est_x, est_y, unc = self.nav.update(measured_seabed_depth=seabed, gps_ne=self.gps_ne)

        # ---- sonar perception -> environment context ----
        clearances = P.sector_clearances(self.scan_angles, self.scan)
        flank = self.memory.flank_clearances(t, est_x, est_y, self.yaw)
        risk = max(P.obstacle_risk(clearances), P.altitude_risk(self.altitude))
        grad = P.seabed_gradient_richness(self.mb_ranges) if self.mb_ranges is not None else 0.0
        self.ocean.set_measurements(gradient_richness=grad, risk=risk)
        env_ctx = self.env_health.sample(est_x, est_y, t)
        self.min_profiler = min(self.min_profiler, float(np.min(clearances)))

        # ---- mission bookkeeping (on the ESTIMATED position) ----
        g0 = self.mission.goal()
        rel = math.atan2(g0[1] - est_y, g0[0] - est_x) - self.yaw
        rel = math.atan2(math.sin(rel), math.cos(rel))
        gdist = math.hypot(g0[0] - est_x, g0[1] - est_y)
        blocked = False
        if abs(rel) < 0.5 * P.PROFILER_FOV:      # goal inside the sonar fan: is it occupied?
            k = int(np.clip(np.searchsorted(np.linspace(-0.5 * P.PROFILER_FOV, 0.5 * P.PROFILER_FOV,
                                                        P.N_SECTORS + 1), rel) - 1, 0, P.N_SECTORS - 1))
            blocked = clearances[k] + P.PROFILER_MOUNT_X < gdist - 2.0 and clearances[k] < 15.0
        event = self.mission.update(est_x, est_y, t, goal_blocked=blocked)
        reached = event is not None
        if event:
            self.get_logger().info(f"waypoint {self.mission.idx}/{len(self.mission.waypoints)} {event}")
        if self.mission.done:
            self.finish("mission complete")
            return
        if self.max_time is not None and t - self.t_start > self.max_time:
            self.finish("max_mission_time reached")
            return
        goal = self.mission.goal()

        # ---- observation -> policy -> action ----
        obs = self.obs_builder.build(est_x, est_y, self.origin, self.depth, self.yaw, unc,
                                     self.vehicle_health.as_vector(), env_ctx, goal=goal,
                                     clearances=clearances, flank=flank,
                                     mission_depth=self.mission.mission_depth)
        action = self.policy.forward(obs)
        cmd_speed, cmd_yaw, cmd_depth = action_to_command(action, self.params)
        overridden = False
        if self.use_shield:
            cmd_speed, cmd_yaw, cmd_depth, overridden = self.shield.filter_action(
                cmd_speed, cmd_yaw, cmd_depth, self.depth, self.vehicle_health.battery_frac(), unc,
                clearances=clearances, altitude=self.altitude,
                depth_error=self.depth - self.mission.mission_depth)
        self.shield_overrides += int(overridden)
        self.publish_setpoint(cmd_speed, cmd_yaw, cmd_depth)

        # ---- health + reward (same terms as orca_env.py; logged for evaluation) ----
        u = self.dvl_uv[0] if self.dvl_uv is not None else self.last_cmd[0]
        energy_wh = (abs(self.surge_force * u) / 0.55 + self.params.hotel_load_w) * dt / 3600.0
        self.vehicle_health.update(energy_wh, cmd_speed, u, self.depth,
                                   dvl_dropout=(self.now() - self.dvl_stamp) > 0.5)
        dist = math.hypot(goal[0] - est_x, goal[1] - est_y)
        progress = 0.0 if (self.prev_goal_dist is None or reached) else self.prev_goal_dist - dist
        self.prev_goal_dist = dist
        health_pen = 1.0 if (overridden or self.vehicle_health.is_critical()) else 0.0
        reward = (self.w["progress"] * progress - self.w["uncertainty"] * unc / 100.0
                  - self.w["risk"] * env_ctx["risk"] - self.w["energy"] * energy_wh / 100.0
                  - self.w["health"] * health_pen - (self.w["idle"] if u < 0.2 else 0.0))
        self.total_reward += reward
        self.steps += 1

        self.obs_pub.publish(Float64MultiArray(data=[float(v) for v in obs]))
        self.action_pub.publish(Float64MultiArray(data=[float(v) for v in action]))
        est = Odometry()
        est.header.stamp = self.get_clock().now().to_msg()
        est.header.frame_id = "world_ned"
        est.child_frame_id = "orca/base_link_frd"
        est.pose.pose.position.x, est.pose.pose.position.y, est.pose.pose.position.z = est_x, est_y, self.depth
        est.pose.pose.orientation.z = math.sin(self.yaw / 2)
        est.pose.pose.orientation.w = math.cos(self.yaw / 2)
        est.pose.covariance[0] = est.pose.covariance[7] = unc / 2.0
        self.est_pub.publish(est)

        rec = dict(t=round(t, 2), est=[round(est_x, 2), round(est_y, 2)], depth=round(self.depth, 2),
                   yaw=round(self.yaw, 4), unc=round(unc, 2), goal=list(goal), wp=self.mission.idx,
                   clear=[round(float(c), 1) for c in clearances], flank=[round(f, 1) for f in flank],
                   action=[round(float(a), 3) for a in action],
                   cmd=[round(cmd_speed, 3), round(cmd_yaw, 4), round(cmd_depth, 3)],
                   shield=bool(overridden), reward=round(reward, 3), contact=bool(self.contact),
                   u=round(u, 3), alt=self.altitude)
        if self.gt is not None:
            rec["gt_eval_only"] = [round(v, 2) for v in self.gt]
            rec["nav_err"] = round(math.hypot(self.gt[0] - est_x, self.gt[1] - est_y), 2)
        self.log_file.write(json.dumps(rec) + "\n")
        if self.steps % 10 == 0:
            self.get_logger().info(
                f"t={t - self.t_start:5.0f}s wp {self.mission.idx}/{len(self.mission.waypoints)} "
                f"est=({est_x:6.1f},{est_y:6.1f}) d={self.depth:4.1f} goal_dist={dist:5.1f} "
                f"front_clear={clearances[2]:4.1f}m cmd=[{cmd_speed:.2f} {cmd_yaw:+.3f} {cmd_depth:+.2f}]"
                f"{' SHIELD' if overridden else ''} contacts={len(self.contact_events)}"
                + (f" nav_err={rec['nav_err']:.1f}m" if 'nav_err' in rec else ""))
        if self.nav_mode == DEBUG_GROUND_TRUTH:
            self.get_logger().warn("DEBUG_GROUND_TRUTH in use", throttle_duration_sec=10.0)

    def finish(self, why):
        if self.finished:
            return
        self.finished = True
        summary = self.summary(why)
        with open(self.log_path.replace(".jsonl", "_summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print("RUN SUMMARY " + json.dumps(summary), flush=True)
        if rclpy.ok():            # context may already be gone on Ctrl-C / SIGTERM
            try:
                self.publish_setpoint(0.0, 0.0, 0.0)
                self.status_pub.publish(String(data=json.dumps(summary)))
            except Exception:
                pass
        self.exit_now = self.exit_on_finish

    def summary(self, why):
        return dict(reason=why, world=self.world, layout=self.layout, nav_mode=self.nav_mode,
                    waypoints_reached=len(self.mission.reached),
                    waypoints_skipped=[dict(index=i, t=round(tt, 1), reason=r) for i, tt, r in self.mission.skipped],
                    waypoints_done=self.mission.idx, waypoints_total=len(self.mission.waypoints),
                    sim_time_s=round(self.now() - (self.t_start or self.now()), 1), steps=self.steps,
                    total_reward=round(self.total_reward, 1), shield_overrides=self.shield_overrides,
                    contacts=self.contact_events, physics_resets=self.physics_resets,
                    min_profiler_clearance_m=round(self.min_profiler, 2),
                    log=self.log_path)


def main(args=None):
    rclpy.init(args=args)
    node = OrcaRLBridge()
    try:
        while rclpy.ok() and not node.exit_now:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if not node.finished:
            node.finish("interrupted")
        node.log_file.close()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

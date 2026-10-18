#!/usr/bin/env python3
"""
check_stonefish_topics.py - Test 4 helper ("Stonefish -> ROS 2").

Subscribes to every topic orca_rl_sim publishes for a few seconds and prints
message rate + the latest value of each, so you can see real Stonefish data
arriving without the RL node running.

    python3 orca_rl/tools/check_stonefish_topics.py --duration 8
"""
import argparse
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import FluidPressure, Imu, LaserScan, NavSatFix, Range
from std_msgs.msg import Bool, Float64MultiArray, String


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def summary(name, m):
    if name == "/clock":
        return f"sim t = {m.clock.sec + m.clock.nanosec * 1e-9:.2f} s"
    if name == "/orca/imu":
        return (f"yaw = {math.degrees(yaw_of(m.orientation)):7.2f} deg, "
                f"r = {m.angular_velocity.z:+.4f} rad/s")
    if name == "/orca/pressure":
        return f"{m.fluid_pressure:9.1f} Pa gauge -> depth {m.fluid_pressure / (1025 * 9.81):6.2f} m"
    if name == "/orca/dvl/twist":
        v = m.twist.twist.linear
        return f"body u,v,w = {v.x:+.3f} {v.y:+.3f} {v.z:+.3f} m/s"
    if name == "/orca/dvl/altitude":
        return f"altitude = {m.range:.2f} m"
    if name == "/orca/gps":
        return "NO FIX (submerged)" if m.status.status < 0 else f"lat {m.latitude:.6f} lon {m.longitude:.6f}"
    if name == "/orca/ground_truth/odometry":
        p = m.pose.pose.position
        return (f"[GROUND TRUTH] N,E,D = {p.x:7.2f} {p.y:7.2f} {p.z:6.2f}, "
                f"yaw {math.degrees(yaw_of(m.pose.pose.orientation)):7.2f} deg, u {m.twist.twist.linear.x:+.3f}")
    if name in ("/orca/profiler", "/orca/multibeam"):
        finite = [r for r in m.ranges if math.isfinite(r)]
        return (f"{len(m.ranges)} bins, {len(finite)} returns, "
                f"min {min(finite):.2f} m" if finite else f"{len(m.ranges)} bins, no returns")
    if name == "/orca/actuators":
        d = m.data
        return f"F = [{d[0]:.0f}, {d[1]:.0f}, {d[2]:.0f}] N  setpoint [{d[4]:.2f}, {d[5]:.3f}, {d[6]:.2f}]  src {int(d[8])}"
    if name in ("/orca/contact",):
        return f"in contact = {m.data}"
    if name in ("/orca/contact/entity", "/orca/status"):
        return repr(m.data)
    return ""


TOPICS = [
    ("/clock", Clock), ("/orca/imu", Imu), ("/orca/pressure", FluidPressure),
    ("/orca/dvl/twist", TwistWithCovarianceStamped), ("/orca/dvl/altitude", Range),
    ("/orca/gps", NavSatFix), ("/orca/ground_truth/odometry", Odometry),
    ("/orca/profiler", LaserScan), ("/orca/multibeam", LaserScan),
    ("/orca/actuators", Float64MultiArray), ("/orca/contact", Bool),
    ("/orca/contact/entity", String), ("/orca/status", String),
]


class Probe(Node):
    def __init__(self):
        super().__init__("orca_topic_probe")
        self.count = {t: 0 for t, _ in TOPICS}
        self.last = {}
        for topic, typ in TOPICS:
            qos = qos_profile_sensor_data if typ in (Imu, FluidPressure, TwistWithCovarianceStamped,
                                                    Range, NavSatFix, Odometry, LaserScan) else 10
            if topic == "/orca/status":
                from rclpy.qos import QoSProfile, DurabilityPolicy
                qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            if topic == "/clock":
                qos = qos_profile_sensor_data
            self.create_subscription(typ, topic, self._cb(topic), qos)

    def _cb(self, topic):
        def f(msg):
            self.count[topic] += 1
            self.last[topic] = msg
        return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=6.0)
    args = ap.parse_args()
    rclpy.init()
    node = Probe()
    t0 = time.time()
    while time.time() - t0 < args.duration:
        rclpy.spin_once(node, timeout_sec=0.05)
    dt = time.time() - t0
    print(f"{'topic':32s} {'Hz(wall)':>9s}  latest")
    for topic, _ in TOPICS:
        n = node.count[topic]
        latest = summary(topic, node.last[topic]) if topic in node.last else "-- nothing received --"
        print(f"{topic:32s} {n / dt:9.1f}  {latest}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

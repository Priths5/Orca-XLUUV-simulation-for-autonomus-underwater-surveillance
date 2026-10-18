#!/usr/bin/env python3
"""
manual_cmd.py - drive the Stonefish Orca by hand over ROS 2 (no RL policy).

This is the easiest way to verify  ROS 2 -> Stonefish -> Orca  (Test 3).
It republishes the command at 2 Hz (orca_rl_sim drops a command that is
older than its command_timeout, 3 s of sim time by default) and prints the
Orca's motion from the ground-truth odometry so you can see it respond.

Setpoint mode (what the RL bridge uses; converted to forces in C++):
    python3 orca_rl/tools/manual_cmd.py --speed 0.8                  # forward 0.8 m/s
    python3 orca_rl/tools/manual_cmd.py --speed 0.6 --yaw-rate 0.03  # turn to starboard
    python3 orca_rl/tools/manual_cmd.py --depth-rate 0.2             # go deeper

Raw thrust mode (bypasses the converter, straight to the actuators):
    python3 orca_rl/tools/manual_cmd.py --thrust 20000 0 0           # MainThruster 20 kN
    python3 orca_rl/tools/manual_cmd.py --thrust 0 4000 0            # YawThruster 4 kN

Units / signs (NED world, FRD body):
    speed      m/s,   + forward
    yaw-rate   rad/s, + clockwise seen from above (bow to starboard)
    depth-rate m/s,   + down
"""
import argparse
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class ManualCommander(Node):
    def __init__(self, args):
        super().__init__("orca_manual_cmd")
        self.args = args
        if args.thrust is not None:
            self.pub = self.create_publisher(Float64MultiArray, "/orca/cmd/thrust", 10)
            self.data = [float(v) for v in args.thrust]
        else:
            self.pub = self.create_publisher(Float64MultiArray, "/orca/cmd/setpoint", 10)
            self.data = [args.speed, args.yaw_rate, args.depth_rate]
        self.odom = None
        self.create_subscription(Odometry, "/orca/ground_truth/odometry", self._odom, qos_profile_sensor_data)

    def _odom(self, msg):
        self.odom = msg

    def send(self, data=None):
        m = Float64MultiArray()
        m.data = list(data if data is not None else self.data)
        self.pub.publish(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=0.0)
    ap.add_argument("--yaw-rate", type=float, default=0.0)
    ap.add_argument("--depth-rate", type=float, default=0.0)
    ap.add_argument("--thrust", type=float, nargs=3, metavar=("SURGE_N", "YAW_N", "HEAVE_N"))
    ap.add_argument("--duration", type=float, default=30.0, help="wall-clock seconds to keep commanding")
    args = ap.parse_args()

    rclpy.init()
    node = ManualCommander(args)
    topic = "/orca/cmd/thrust" if args.thrust is not None else "/orca/cmd/setpoint"
    print(f"publishing {node.data} on {topic} for {args.duration:.0f} s (Ctrl-C to stop)")
    t0 = time.time()
    last_pub = last_print = 0.0
    try:
        while time.time() - t0 < args.duration:
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.time()
            if now - last_pub >= 0.5:
                node.send()
                last_pub = now
            if now - last_print >= 2.0 and node.odom is not None:
                p = node.odom.pose.pose.position
                v = node.odom.twist.twist
                print(f"  N {p.x:8.2f}  E {p.y:8.2f}  D {p.z:6.2f}  "
                      f"yaw {math.degrees(yaw_of(node.odom.pose.pose.orientation)):7.1f} deg  "
                      f"u {v.linear.x:+.2f} m/s  r {v.angular.z:+.4f} rad/s   [ground truth]")
                last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        node.send([0.0, 0.0, 0.0])   # leave the Orca with a stop command
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
sysid_stonefish.py - measure how the Stonefish Orca actually responds to
MainThruster / YawThruster forces, to calibrate

  * OrcaRLCommandAdapter (C++): surge drag feed-forward, yaw feed-forward,
    achievable max speed / yaw rate, and
  * the Python training dynamics (sim/obstacle_env.py): max speed, speed and
    yaw-rate time constants,

instead of assuming "[-1, 1] == Newtons".

Uses /orca/cmd/thrust (raw forces, no inner loop) and the ground-truth
odometry - this is system identification, not an RL observation.
Between steps it calls /orca/reset (Robot::Respawn).

    ./orca_rl_sim --world rl --headless --rtf 4 &
    python3 orca_rl/tools/sysid_stonefish.py --out logs/sysid.json
"""
import argparse
import json
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger


class SysId(Node):
    def __init__(self):
        super().__init__("orca_sysid")
        self.pub = self.create_publisher(Float64MultiArray, "/orca/cmd/thrust", 10)
        self.reset_cli = self.create_client(Trigger, "/orca/reset")
        self.sim_t = 0.0
        self.log = []
        self.create_subscription(Clock, "/clock", self._on_clock, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/orca/ground_truth/odometry", self._on_odom, qos_profile_sensor_data)

    def _on_clock(self, m):
        self.sim_t = m.clock.sec + m.clock.nanosec * 1e-9

    def _on_odom(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.log.append((t, m.twist.twist.linear.x, m.twist.twist.angular.z, m.pose.pose.position.z, yaw))

    def wait_sim(self, dt, cmd=None):
        t_end = self.sim_t + dt
        last = 0.0
        while rclpy.ok() and self.sim_t < t_end:
            rclpy.spin_once(self, timeout_sec=0.02)
            if cmd is not None and time.time() - last > 0.2:
                m = Float64MultiArray(); m.data = [float(c) for c in cmd]
                self.pub.publish(m); last = time.time()

    def reset(self):
        self.reset_cli.wait_for_service(timeout_sec=5.0)
        fut = self.reset_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        self.wait_sim(8.0, [0, 0, 0])   # let it settle after the respawn

    def run_step(self, cmd, duration):
        self.reset()
        self.log = []
        t0 = self.sim_t
        self.wait_sim(duration, cmd)
        arr = np.array([r for r in self.log if r[0] >= t0])
        arr[:, 0] -= t0
        return arr


def time_constant(t, y):
    """time to reach 63% of the final value (final = mean of last 20%)."""
    yf = np.mean(y[int(0.8 * len(y)):])
    if abs(yf) < 1e-6:
        return float("nan"), yf
    idx = np.argmax(np.abs(y) >= 0.632 * abs(yf))
    return float(t[idx]), float(yf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/sysid_stonefish.json")
    ap.add_argument("--duration", type=float, default=90.0)
    args = ap.parse_args()
    rclpy.init()
    n = SysId()
    while n.sim_t == 0.0:
        rclpy.spin_once(n, timeout_sec=0.1)

    res = {"surge": [], "yaw": []}
    for f in [10000, 20000, 30000, 40000]:
        a = n.run_step([f, 0, 0], args.duration)
        tau, uf = time_constant(a[:, 0], a[:, 1])
        res["surge"].append(dict(force=f, u_ss=uf, tau=tau, depth_drift=float(a[-1, 3] - a[0, 3])))
        print(f"surge {f:6d} N -> u_ss {uf:.3f} m/s, tau63 {tau:.1f} s, depth drift {a[-1, 3] - a[0, 3]:+.2f} m")
    for fy in [2000, 4000, 8000]:
        a = n.run_step([20000, fy, 0], args.duration)
        tau, rf = time_constant(a[:, 0], a[:, 2])
        res["yaw"].append(dict(surge_force=20000, yaw_force=fy, r_ss=rf, tau=tau,
                               u_ss=float(np.mean(a[int(0.8 * len(a)):, 1]))))
        print(f"yaw  {fy:6d} N (surge 20 kN) -> r_ss {rf:+.4f} rad/s ({math.degrees(rf):+.2f} deg/s), "
              f"tau63 {tau:.1f} s, u_ss {res['yaw'][-1]['u_ss']:.2f}")
    a = n.run_step([0, 4000, 0], args.duration)
    tau, rf = time_constant(a[:, 0], a[:, 2])
    res["yaw"].append(dict(surge_force=0, yaw_force=4000, r_ss=rf, tau=tau, u_ss=0.0))
    print(f"yaw   4000 N (no surge)    -> r_ss {rf:+.4f} rad/s, tau63 {tau:.1f} s")

    # fits:  F = kq u|u| + kl u  ;  F_yaw = c r (at cruise)
    U = np.array([s["u_ss"] for s in res["surge"]]); F = np.array([s["force"] for s in res["surge"]])
    A = np.stack([U * np.abs(U), U], 1)
    kq, kl = np.linalg.lstsq(A, F, rcond=None)[0]
    res["fit"] = dict(drag_quad=float(kq), drag_lin=float(kl), u_max_40kN=float(U[-1]),
                      yaw_ff=float(np.mean([y["yaw_force"] / y["r_ss"] for y in res["yaw"][:3] if abs(y["r_ss"]) > 1e-4])),
                      r_max_8kN=float(res["yaw"][2]["r_ss"]))
    print("fit:", res["fit"])
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

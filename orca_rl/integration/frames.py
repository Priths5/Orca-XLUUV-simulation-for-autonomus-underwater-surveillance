"""
frames.py
---------
Coordinate conventions shared by the Stonefish executable, the ROS 2 topics
and the Python RL code. Verified against the Stonefish library source
(Odometry/IMU/DVL/Pressure/GPS InternalUpdate) - not taken from comments.

  Stonefish world  : NED  (x north, y east, z down)         frame "world_ned"
  Stonefish body   : FRD  (x forward, y starboard, z down)  frame "orca/base_link_frd"
  ROS 2 topics     : published in those same NED/FRD frames (REP-105 "_ned"
                     style), NOT converted to ENU/FLU. Use ned_to_enu() if
                     you need RViz/ENU.
  Python RL (sim/) : x = north, y = east, heading = NED yaw, depth = +down.
                     orca_env.py integrates  x += v cos(h), y += v sin(h),
                     h += yaw_rate  - exactly NED, so no conversion is
                     needed between Stonefish and the RL code.

  yaw / yaw rate   : radians, + = clockwise seen from above (bow to starboard)
  quaternion       : ROS (x, y, z, w) order, rotation world_ned <- body_frd
  velocities       : Odometry twist and DVL are BODY-frame (FRD); rotate by
                     yaw to get north/east velocity
  pressure         : Stonefish Pressure sensor reports GAUGE pressure (Pa) -
                     no 101325 Pa atmosphere to subtract (the old Gazebo
                     bridge did subtract it; that would be wrong here)
"""

import math

import numpy as np

RHO_SEAWATER = 1025.0
G = 9.81


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(x, y, z, w):
    """NED yaw (ZYX convention) from a world_ned <- body_frd quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def depth_from_gauge_pressure(pa):
    return float(pa) / (RHO_SEAWATER * G)


def body_to_ned_velocity(u, v, yaw):
    """Body (forward, starboard) velocity -> (north, east) velocity."""
    c, s = math.cos(yaw), math.sin(yaw)
    return u * c - v * s, u * s + v * c


def ned_to_enu(n, e, d):
    return e, n, -d


def ned_yaw_to_enu_yaw(yaw_ned):
    return wrap_angle(math.pi / 2.0 - yaw_ned)


def bearing_to(x, y, heading, gx, gy):
    """(relative bearing to goal in [-pi, pi], distance). + = goal to starboard."""
    dx, dy = gx - x, gy - y
    return wrap_angle(math.atan2(dy, dx) - heading), float(np.hypot(dx, dy))

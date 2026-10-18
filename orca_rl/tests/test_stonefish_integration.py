"""
Offline tests for the Stonefish integration layer (no ROS, no Stonefish):

    cd orca_rl && python3 -m pytest -q tests
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from integration import perception as P
from integration.frames import (body_to_ned_velocity, bearing_to, depth_from_gauge_pressure,
                                ned_to_enu, yaw_from_quaternion)
from integration.missions import Mission, geodetic_to_ned, lawnmower
from integration.navigation import DEBUG_GROUND_TRUTH, REALISTIC_SENSOR, StonefishNavigator
from integration.observation import ObservationBuilder, action_to_command
from integration.policy_io import load_policy
from integration.world_model import StonefishOceanModel
from sim.health_monitor import EnvironmentHealth, VehicleHealth
from sim.orca_env import OrcaNavEnv
from sim.safety_shield import ObstacleAwareSafetyShield, SafetyShield
from sim.vehicle_dynamics import OrcaVehicleParams, StonefishOrcaParams


# ---------------------------------------------------------------- frames
def test_yaw_from_quaternion_ned():
    for yaw in (-2.5, -0.3, 0.0, 0.7, 3.0):
        q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        assert abs(yaw_from_quaternion(*q) - yaw) < 1e-9


def test_body_to_ned_heading_east():
    vn, ve = body_to_ned_velocity(1.0, 0.0, math.pi / 2)   # heading east (NED yaw +90)
    assert abs(vn) < 1e-9 and abs(ve - 1.0) < 1e-9


def test_python_env_uses_ned_heading():
    """orca_env.py integrates x += v cos(h), y += v sin(h): same as NED."""
    vn, ve = body_to_ned_velocity(2.0, 0.0, 0.4)
    assert abs(vn - 2.0 * math.cos(0.4)) < 1e-12 and abs(ve - 2.0 * math.sin(0.4)) < 1e-12


def test_gauge_pressure_depth():
    assert abs(depth_from_gauge_pressure(1025 * 9.81 * 12.0) - 12.0) < 1e-9
    # value seen on /orca/pressure at the 12 m start depth in orca_rl_world
    assert abs(depth_from_gauge_pressure(120519.8) - 11.99) < 0.01


def test_ned_to_enu():
    assert ned_to_enu(1.0, 2.0, 3.0) == (2.0, 1.0, -3.0)


def test_bearing_starboard_positive():
    rel, d = bearing_to(0, 0, 0.0, 0, 10)       # heading north, goal due east
    assert rel > 0 and abs(rel - math.pi / 2) < 1e-9 and abs(d - 10) < 1e-9


# ------------------------------------------------------------ perception
def test_sector_clearances_and_inf_nan():
    ang = P.profiler_bin_angles()
    r = np.full(ang.size, np.inf)
    r[:3] = np.nan                     # not swept yet
    r[16] = 6.0                        # dead ahead
    c = P.sector_clearances(ang, r)
    assert c.shape == (5,) and c[2] == 6.0 and np.all(c[[0, 1, 3, 4]] == P.PROFILER_RANGE)
    assert P.obstacle_risk(c) == 1.0
    assert P.obstacle_risk(np.full(5, 30.0)) == 0.0


def test_obstacle_memory_flank():
    m = P.ObstacleMemory()
    # a return 20 m dead ahead of the bow sensor while heading north at (0,0)
    m.add_scan(0.0, 0.0, 0.0, 0.0, [0.0], [20.0])
    # vehicle offset 5 m west and 30 m further north: the rock is now beside the hull, to starboard
    port, stbd = m.flank_clearances(100.0, 30.0, -5.0, 0.0)
    assert stbd < 5.0 and port == P.PROFILER_RANGE          # out of view, but NOT forgotten
    # a later beam that looks through that spot with no return clears it (it was a fish)
    m.add_scan(101.0, 0.0, 0.0, 0.0, [0.0], [np.inf])
    assert m.flank_clearances(101.0, 30.0, -5.0, 0.0) == (P.PROFILER_RANGE, P.PROFILER_RANGE)
    # far away -> pruned
    m.add_scan(0.0, 0.0, 0.0, 0.0, [0.0], [20.0])
    assert m.flank_clearances(1.0, 200.0, 0.0, 0.0) == (P.PROFILER_RANGE, P.PROFILER_RANGE)


def test_gradient_richness_flat_vs_rough():
    n = 129
    ang = np.linspace(-math.radians(60), math.radians(60), n)
    flat = 16.0 / np.cos(ang)
    assert P.seabed_gradient_richness(flat) < 0.05
    rough = flat.copy()
    rough[40:60] *= 0.4
    assert P.seabed_gradient_richness(rough) > 0.2


# ------------------------------------------------------ observation layout
def _ctx():
    ocean = StonefishOceanModel(30.0)
    ocean.set_measurements(gradient_richness=0.1, risk=0.2)
    return ocean, EnvironmentHealth(ocean).sample(0, 0, 0)


def test_legacy12_matches_orca_env_layout():
    env = OrcaNavEnv(seed=3)
    o_env, _ = env.reset()
    ocean, _ = _ctx()
    env_ctx = env.env_health.sample(env.est_x, env.est_y, env.t)
    b = ObservationBuilder("legacy12")
    o = b.build(env.est_x, env.est_y, (0.0, 0.0), env.true_state.depth, env.true_state.heading,
                env.uncertainty, env.vehicle_health.as_vector(), env_ctx)
    assert o.shape == (12,)
    assert np.allclose(o, o_env, atol=1e-6)


def test_obstacle_v1_layout():
    _, ctx = _ctx()
    vh = VehicleHealth(StonefishOrcaParams()).as_vector()
    b = ObservationBuilder("obstacle_v1")
    o = b.build(10, 0, (0, 0), 13.0, 0.0, 8.0, vh, ctx, goal=(10, 50),
                clearances=[30, 30, 6, 30, 30], flank=(30, 4), mission_depth=12.0)
    assert o.shape == (23,)
    assert o[12] > 0.99 and abs(o[13]) < 1e-6      # goal 90 deg to starboard
    assert abs(o[17] - 0.2) < 1e-6                  # 6 m dead ahead / 30
    assert abs(o[21] - 4 / 30) < 1e-6
    assert abs(o[22] - 0.2) < 1e-6                  # 1 m deeper than mission depth / 5


def test_action_scaling_is_orca_env_scaling():
    p = OrcaVehicleParams()
    s, y, d = action_to_command([1, -1, 0.5], p)
    assert abs(s - p.max_speed) < 1e-12 and abs(y + p.max_yaw_rate) < 1e-12 and abs(d - 0.5 * p.max_depth_rate) < 1e-12
    q = StonefishOrcaParams()
    assert action_to_command([1, 1, 1], q)[0] == q.max_speed == 1.1


# ---------------------------------------------------------------- shield
def test_original_shield_rules_unchanged():
    p = OrcaVehicleParams()
    a = SafetyShield(p).filter_action(2.0, 0.05, 0.1, 100.0, 0.5, 100.0)
    b = ObstacleAwareSafetyShield(p).filter_action(2.0, 0.05, 0.1, 100.0, 0.5, 100.0)
    assert a == b
    s, y, d, o = ObstacleAwareSafetyShield(p).filter_action(2.0, 0.05, 0.1, 100.0, 0.05, 100.0)
    assert o and s <= p.min_cruise_speed          # battery reserve rule still there


def test_obstacle_shield_turns_to_open_side():
    p = StonefishOrcaParams()
    sh = ObstacleAwareSafetyShield(p)
    s, y, d, o = sh.filter_action(1.0, 0.0, 0.0, 12, 1.0, 10, clearances=[30, 30, 5, 10, 8])
    assert o and s <= 0.15 and y < 0          # starboard cluttered -> turn to port
    s, y, d, o = sh.filter_action(1.0, 0.0, 0.0, 12, 1.0, 10, clearances=[8, 9, 5, 30, 30])
    assert o and y > 0
    s, y, d, o = sh.filter_action(1.0, 0.0, 0.2, 12, 1.0, 10, clearances=[30] * 5, altitude=2.0)
    assert o and d < 0                        # floor guard


# ------------------------------------------------------------ navigation
def test_realistic_dead_reckoning_and_debug_mode():
    ocean = StonefishOceanModel(30.0)
    nav = StonefishNavigator(ocean, REALISTIC_SENSOR, rng=np.random.default_rng(1))
    nav.reset(0.0, 0.0)
    for _ in range(100):                      # 10 s at 1 m/s heading east
        nav.integrate(0.1, math.pi / 2, dvl_uv=(1.0, 0.0))
    nav.set_ground_truth(1000.0, 1000.0)      # must be ignored in realistic mode
    x, y, unc = nav.update(measured_seabed_depth=30.0)
    assert abs(x) < 3 and abs(y - 10) < 3 and unc > 0
    dbg = StonefishNavigator(ocean, DEBUG_GROUND_TRUTH)
    dbg.reset(0, 0)
    dbg.set_ground_truth(5.0, 6.0)
    assert dbg.update()[:2] == (5.0, 6.0)


def test_gps_recentres_filter():
    nav = StonefishNavigator(StonefishOceanModel(30.0), rng=np.random.default_rng(2))
    nav.reset(0, 0)
    x, y, _ = nav.update(gps_ne=(50.0, -20.0))
    assert abs(x - 50) < 1 and abs(y + 20) < 1


# ------------------------------------------------------------- missions
def test_lawnmower_and_geodetic():
    wps = lawnmower(-80, -60, 80, 60, 30)
    assert wps[0] == (-80, -60) and wps[1] == (80, -60) and wps[2] == (80, -30) and len(wps) == 10
    n, e = geodetic_to_ned(36.6 + 100 / 111319.49, -121.9)
    assert abs(n - 100) < 0.5 and abs(e) < 1e-6
    m = Mission("rl")
    assert m.goal() == (-80.0, -60.0)
    assert m.update(-79, -59, t=0.0) == "reached" and m.idx == 1
    assert m.goal() == (0.0, -60.0)                      # 160 m leg split at 80 m
    assert m.update(-20, -60, t=1.0, goal_blocked=True) == "blocked"
    assert m.update(-200, -200, t=1.0) is None
    assert m.update(-200, -200, t=1e4) == "timeout"


# ---------------------------------------------------------------- policies
def test_policy_files_load():
    pol, layout, params, _ = load_policy(os.path.join(ROOT, "logs", "trained_policy.npz"))
    assert layout == "legacy12" and isinstance(params, OrcaVehicleParams) and pol.sizes[0] == 12
    p2 = os.path.join(ROOT, "logs", "trained_policy_obstacle.npz")
    if os.path.exists(p2):
        pol, layout, params, _ = load_policy(p2)
        assert layout == "obstacle_v1" and isinstance(params, StonefishOrcaParams) and pol.sizes[0] == 23
        a = pol.forward(np.zeros(23, dtype=np.float32))
        assert a.shape == (3,) and np.all(np.abs(a) <= 1)

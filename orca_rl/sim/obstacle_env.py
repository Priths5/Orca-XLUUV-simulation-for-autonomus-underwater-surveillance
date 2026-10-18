"""
obstacle_env.py
----------------
Training environment for an obstacle-aware Orca policy that will drive the
STONEFISH Orca. Same reset/step API, same layered design and the same
modules as orca_env.py (VehicleHealth, EnvironmentHealth,
ParticleFilterLocalizer, safety shield, reward structure), at the scale of
the Stonefish scenarios:

  * vehicle: StonefishOrcaParams - limits and time constants identified on
    the Stonefish Orca (tools/sysid_stonefish.py); hull = 27 m x 1.3 m
    capsule used for collision checks
  * world: flat 30 m seabed, rock pinnacles / ridge walls (circle chains),
    fish schools and traffic subs that move
  * perception: the SAME single swept profiler beam as the Stonefish model
    (bow mount, 160 deg, 64 steps, 10 Hz, 30 m) binned into 33 x 5 deg bins
    exactly as orca_rl_sim publishes them, then integration/perception.py
  * navigation: DVL dead reckoning into the existing particle filter via
    integration/navigation.py (REALISTIC mode; the flat chart makes TAN
    uninformative, as in Stonefish)
  * mission: a chain of goals (survey-leg style); the goal is part of the
    observation (layout "obstacle_v1", integration/observation.py)

Physics here is a light kinematic model for fast ES training only. When the
policy runs in Stonefish this file is not used at all - Stonefish is the
only physics then.
"""

import math

import numpy as np

from integration import perception as P
from integration.frames import bearing_to, wrap_angle
from integration.navigation import REALISTIC_SENSOR, StonefishNavigator
from integration.observation import ObservationBuilder, action_to_command
from integration.world_model import StonefishOceanModel
from .health_monitor import EnvironmentHealth, VehicleHealth
from .safety_shield import ObstacleAwareSafetyShield
from .vehicle_dynamics import StonefishOrcaParams

SEABED_Z = 30.0
MISSION_DEPTH = 12.0


class OrcaObstacleNavEnv:
    def __init__(self, seed=0, dt=1.0, substeps=10, max_steps=420, n_goals=3,
                 use_safety_shield=True, reward_weights=None, difficulty=1.0):
        self.rng = np.random.default_rng(seed)
        self.params = StonefishOrcaParams()
        self.shield = ObstacleAwareSafetyShield(self.params)
        self.use_shield = use_safety_shield
        self.obs_builder = ObservationBuilder("obstacle_v1")
        self.dt = dt
        self.substeps = substeps
        self.max_steps = max_steps
        self.n_goals = n_goals
        self.difficulty = difficulty
        self.w = reward_weights or dict(progress=1.0, uncertainty=0.0025, risk=0.4, energy=0.02,
                                        health=1.5, idle=0.5, goal_bonus=200.0, collision=300.0)
        self.action_dim = 3
        self.obs_dim = self.obs_builder.obs_dim
        self.bin_angles = P.profiler_bin_angles()
        self.beam_angles = np.linspace(-0.5 * P.PROFILER_FOV, 0.5 * P.PROFILER_FOV, 65)
        self.log = []

    # ------------------------------------------------------------------ world
    def _place_world(self):
        rng = self.rng
        self.box = 125.0
        h0 = rng.uniform(-math.pi, math.pi)
        self.x, self.y = rng.uniform(-60, 60), rng.uniform(-60, 60)
        self.heading = h0
        pts = [(self.x, self.y)]
        d = h0 + rng.uniform(-1.2, 1.2) if rng.random() < 0.7 else rng.uniform(-math.pi, math.pi)
        for _ in range(self.n_goals):
            for _try in range(20):
                L = rng.uniform(20, 40) if rng.random() < 0.3 else rng.uniform(50, 120)  # lane changes + legs
                gx, gy = pts[-1][0] + L * math.cos(d), pts[-1][1] + L * math.sin(d)
                if abs(gx) < self.box - 15 and abs(gy) < self.box - 15:
                    break
                d += rng.uniform(1.0, 2.5)
            gx, gy = float(np.clip(gx, -self.box + 15, self.box - 15)), float(np.clip(gy, -self.box + 15, self.box - 15))
            pts.append((gx, gy))
            d += rng.choice([-1, 1]) * rng.uniform(1.2, 2.6)
        self.goals = [np.array(p) for p in pts[1:]]

        cx, cy, cr = [], [], []

        # keep obstacles >= 15 m from the whole starting hull (not just its
        # origin: the hull reaches 13.85 m ahead / 13.3 m behind) and 12 m
        # from every goal, so no episode starts beside or inside a rock
        hc, hs = math.cos(h0), math.sin(h0)

        def free(x, y, r):
            t = float(np.clip((x - self.x) * hc + (y - self.y) * hs, -13.3, 13.85))
            if math.hypot(x - (self.x + t * hc), y - (self.y + t * hs)) < r + 1.3 + 15.0:
                return False
            for (px, py) in pts[1:]:
                if math.hypot(px - x, py - y) < r + 12.0:
                    return False
            return True

        n_on_path = int(round(self.difficulty * rng.integers(2, 6)))
        for _ in range(n_on_path):          # obstacles sitting on the legs -> must be avoided
            k = rng.integers(0, len(pts) - 1)
            a, b = np.array(pts[k]), np.array(pts[k + 1])
            t = rng.uniform(0.25, 0.75)
            p = a + t * (b - a)
            n = np.array([-(b - a)[1], (b - a)[0]]) / (np.linalg.norm(b - a) + 1e-9)
            p = p + n * rng.uniform(-3, 3)
            r = rng.uniform(1.5, 4.0)
            if free(p[0], p[1], r) and np.linalg.norm(b - a) > 30:
                cx.append(p[0]); cy.append(p[1]); cr.append(r)
        for _ in range(int(round(self.difficulty * rng.integers(4, 12)))):
            x, y, r = rng.uniform(-self.box, self.box), rng.uniform(-self.box, self.box), rng.uniform(1.5, 4.0)
            if free(x, y, r):
                cx.append(x); cy.append(y); cr.append(r)
        for _ in range(int(rng.integers(0, 3) * self.difficulty)):   # ridge walls as circle chains
            k = rng.integers(0, len(pts) - 1)
            a, b = np.array(pts[k]), np.array(pts[k + 1])
            mid = a + rng.uniform(0.3, 0.7) * (b - a)
            ang = math.atan2((b - a)[1], (b - a)[0]) + math.pi / 2 + rng.uniform(-0.5, 0.5)
            L = rng.uniform(8, 20)
            off = rng.uniform(-L / 2, L / 2) * 0.6
            for s in np.arange(-L / 2, L / 2 + 0.1, 1.5):
                x = mid[0] + (s + off) * math.cos(ang)
                y = mid[1] + (s + off) * math.sin(ang)
                if free(x, y, 1.5):
                    cx.append(x); cy.append(y); cr.append(1.5)
        self.sx, self.sy, self.sr = np.array(cx), np.array(cy), np.array(cr)

        # moving: fish schools (fast, small) and traffic subs (slow, bigger)
        n_fish = int(rng.integers(0, 5) * self.difficulty)
        n_sub = int(rng.integers(0, 2) * self.difficulty)
        mx, my, mr, mvx, mvy = [], [], [], [], []
        for i in range(n_fish + n_sub):
            fish = i < n_fish
            r = rng.uniform(1.0, 2.5) if fish else rng.uniform(2.0, 3.0)
            spd = rng.uniform(0.3, 2.0) if fish else rng.uniform(0.3, 0.7)   # Stonefish schools ~1.8-2 m/s
            k = rng.integers(0, len(pts) - 1)
            a, b = np.array(pts[k]), np.array(pts[k + 1])
            p = a + rng.uniform(0.2, 0.8) * (b - a)
            ang = rng.uniform(-math.pi, math.pi)
            # start 20-40 m upstream so it crosses the leg while the Orca is there
            p = p - np.array([math.cos(ang), math.sin(ang)]) * rng.uniform(15, 40)
            if not free(p[0], p[1], r):
                continue
            mx.append(p[0]); my.append(p[1]); mr.append(r)
            mvx.append(spd * math.cos(ang)); mvy.append(spd * math.sin(ang))
        self.mx, self.my, self.mr = np.array(mx), np.array(my), np.array(mr)
        self.mvx, self.mvy = np.array(mvx), np.array(mvy)

    def _circles(self):
        return (np.concatenate([self.sx, self.mx]), np.concatenate([self.sy, self.my]),
                np.concatenate([self.sr, self.mr]))

    # ------------------------------------------------------------ sensing
    def _raycast(self, ox, oy, ang):
        cx, cy, cr = self._circles()
        if cx.size == 0:
            return P.PROFILER_RANGE
        dx, dy = math.cos(ang), math.sin(ang)
        wx, wy = cx - ox, cy - oy
        t = wx * dx + wy * dy
        d2 = wx * wx + wy * wy - t * t
        hit = d2 <= cr * cr
        if not np.any(hit):
            return P.PROFILER_RANGE
        th = t[hit] - np.sqrt(np.maximum(cr[hit] ** 2 - d2[hit], 0.0))
        inside = (wx[hit] ** 2 + wy[hit] ** 2) < cr[hit] ** 2
        th = np.where(inside, 0.3, th)
        th = th[(th > 0.0) & (th < P.PROFILER_RANGE)]
        return float(np.min(th)) if th.size else P.PROFILER_RANGE

    def _sweep_profiler(self):
        """one beam step of the mechanically swept profiler (10 Hz)."""
        a = self.beam_angles[self.beam_idx]
        ox = self.x + P.PROFILER_MOUNT_X * math.cos(self.heading)
        oy = self.y + P.PROFILER_MOUNT_X * math.sin(self.heading)
        d = self._raycast(ox, oy, self.heading + a)
        d = d + self.rng.normal(0, 0.05) if d < P.PROFILER_RANGE else math.inf
        b = int(round((a - self.bin_angles[0]) / (self.bin_angles[1] - self.bin_angles[0])))
        self.bins[b] = d
        if hasattr(self, "memory"):            # hits are stored, misses carve free space
            # placed with the ESTIMATED pose (true pose + current nav error), as on the vehicle
            ex, ey = self.x + self.nav_err[0], self.y + self.nav_err[1]
            self.memory.add_scan(self.t_fine, ex, ey, self.heading, [a], [d])
        self.beam_idx += self.beam_dir
        if self.beam_idx >= len(self.beam_angles) - 1 or self.beam_idx <= 0:
            self.beam_dir *= -1

    def _hull_clearance(self):
        """min distance between the hull capsule and any obstacle surface."""
        cx, cy, cr = self._circles()
        if cx.size == 0:
            return 1e3
        c, s = math.cos(self.heading), math.sin(self.heading)
        ax, ay = self.x - self.params.hull_half_len_aft * c, self.y - self.params.hull_half_len_aft * s
        L = self.params.hull_half_len_aft + self.params.hull_half_len_fwd
        t = np.clip((cx - ax) * c + (cy - ay) * s, 0.0, L)
        px, py = ax + t * c, ay + t * s
        d = np.hypot(cx - px, cy - py) - cr - self.params.hull_radius
        return float(np.min(d))

    def _multibeam_richness(self):
        """crude multibeam: flat seabed + any obstacle under the fan."""
        alt = SEABED_Z - self.depth
        ang = np.linspace(-0.5 * P.MULTIBEAM_FOV, 0.5 * P.MULTIBEAM_FOV, 129)
        lateral = alt * np.tan(ang)
        r = alt / np.cos(ang)
        c, s = math.cos(self.heading), math.sin(self.heading)
        px, py = self.x - s * lateral, self.y + c * lateral
        cx, cy, cr = self._circles()
        if cx.size:
            inside = (np.hypot(px[:, None] - cx[None], py[:, None] - cy[None]) < cr[None]).any(1)
            r = np.where(inside, np.maximum(0.5, r * 0.3), r)
        return P.seabed_gradient_richness(r)

    # ------------------------------------------------------------ API
    def reset(self):
        self.step_count = 0
        self.t = 0.0
        self.log = []
        self._place_world()
        self.goal_idx = 0
        self.u = self.rng.uniform(0.0, 0.5)
        self.r = 0.0
        self.depth = MISSION_DEPTH + self.rng.normal(0, 0.3)
        self.depth_ref = self.depth
        self.current = self.rng.normal(0, 0.04, size=2)
        self.bins = np.full(P.PROFILER_BINS, np.nan)
        self.memory = P.ObstacleMemory()
        self.nav_err = (0.0, 0.0)
        self.t_fine = 0.0
        self.beam_idx = int(self.rng.integers(0, 64))
        self.beam_dir = 1 if self.rng.random() < 0.5 else -1
        for _ in range(int(self.rng.integers(30, 70))):     # pre-fill a partial sweep
            self._sweep_profiler()
        self.est_x, self.est_y = self.x, self.y
        self.ocean = StonefishOceanModel(SEABED_Z)
        self.env_health = EnvironmentHealth(self.ocean)
        self.vehicle_health = VehicleHealth(self.params)
        self.nav = StonefishNavigator(self.ocean, REALISTIC_SENSOR, n_particles=60, rng=self.rng)
        self.origin = (self.x, self.y)
        self.nav.reset(self.x, self.y)
        self.est_x, self.est_y = self.x, self.y
        self.uncertainty = self.nav.uncertainty
        self.prev_dist = self._goal_dist()
        self.min_clearance = 1e3
        return self._obs(), {}

    def _goal(self):
        return self.goals[min(self.goal_idx, len(self.goals) - 1)]

    def _goal_dist(self):
        g = self._goal()
        return float(math.hypot(g[0] - self.est_x, g[1] - self.est_y))

    def _perceive(self):
        self.clearances = P.sector_clearances(self.bin_angles, self.bins)
        self.flank = self.memory.flank_clearances(self.t_fine, self.est_x, self.est_y, self.heading)
        alt = SEABED_Z - self.depth - 1.45
        risk = max(P.obstacle_risk(self.clearances), P.altitude_risk(alt))
        self.ocean.set_measurements(gradient_richness=self._multibeam_richness(), risk=risk)
        return alt

    def _obs(self):
        self._perceive()
        env_ctx = self.env_health.sample(self.est_x, self.est_y, self.t)
        return self.obs_builder.build(self.est_x, self.est_y, self.origin, self.depth, self.heading,
                                      self.uncertainty, self.vehicle_health.as_vector(), env_ctx,
                                      goal=self._goal(), clearances=self.clearances,
                                      flank=self.flank, mission_depth=MISSION_DEPTH)

    def step(self, action):
        p = self.params
        cmd_speed, cmd_yaw, cmd_depth = action_to_command(action, p)
        alt = self._perceive()
        overridden = False
        if self.use_shield:
            cmd_speed, cmd_yaw, cmd_depth, overridden = self.shield.filter_action(
                cmd_speed, cmd_yaw, cmd_depth, self.depth, self.vehicle_health.battery_frac(),
                self.uncertainty, clearances=self.clearances, altitude=alt,
                depth_error=self.depth - MISSION_DEPTH)

        h = self.dt / self.substeps
        energy_wh = 0.0
        collided = False
        dist_step = 0.0
        for _ in range(self.substeps):
            self.u += (cmd_speed - self.u) * h / p.tau_speed + self.rng.normal(0, 0.003)
            r_lim = p.yaw_rate_limit(self.u)
            self.r += (float(np.clip(cmd_yaw, -r_lim, r_lim)) - self.r) * h / p.tau_yaw + self.rng.normal(0, 0.0005)
            self.u = float(np.clip(self.u, -0.3, p.max_speed * 1.05))
            self.heading = wrap_angle(self.heading + self.r * h)
            self.depth_ref = float(np.clip(self.depth_ref + cmd_depth * h, 2.5, SEABED_Z - 4.0))
            self.depth += (self.depth_ref - self.depth) * h / p.tau_depth
            vx = self.u * math.cos(self.heading) + self.current[0]
            vy = self.u * math.sin(self.heading) + self.current[1]
            self.x += vx * h
            self.y += vy * h
            dist_step += math.hypot(vx, vy) * h
            # moving obstacles (bounce off the box). Like real fish schools and
            # the Stonefish traffic sub (which runs its own avoidance), they veer
            # away once within 8 m of the hull instead of ramming its flank -
            # those strikes come from outside the forward sonar fan and could
            # not be avoided, so they would only add noise to the fitness.
            if self.mx.size:
                c, s_ = math.cos(self.heading), math.sin(self.heading)
                ax = self.x - p.hull_half_len_aft * c
                ay = self.y - p.hull_half_len_aft * s_
                L = p.hull_half_len_aft + p.hull_half_len_fwd
                tt = np.clip((self.mx - ax) * c + (self.my - ay) * s_, 0.0, L)
                dx, dy = self.mx - (ax + tt * c), self.my - (ay + tt * s_)
                dd = np.hypot(dx, dy)
                near = (dd - self.mr - p.hull_radius) < 8.0
                if np.any(near):
                    spd = np.hypot(self.mvx[near], self.mvy[near])
                    self.mvx[near] = spd * dx[near] / np.maximum(dd[near], 1e-6)
                    self.mvy[near] = spd * dy[near] / np.maximum(dd[near], 1e-6)
                self.mx += self.mvx * h
                self.my += self.mvy * h
                out_x = np.abs(self.mx) > self.box + 20
                out_y = np.abs(self.my) > self.box + 20
                self.mvx[out_x] *= -1
                self.mvy[out_y] *= -1
            self.t_fine += h
            self._sweep_profiler()
            # navigation integrates DVL (bottom lock: seabed in range)
            dvl = (self.u + self.rng.normal(0, 0.01), self.rng.normal(0, 0.01))
            self.nav.integrate(h, self.heading + self.rng.normal(0, 0.002), dvl_uv=dvl)
            thrust = p.thrust_for_speed(max(self.u, 0.0))
            power = abs(thrust * self.u) / 0.55 + p.hotel_load_w
            energy_wh += power * h / 3600.0
            clr = self._hull_clearance()
            self.min_clearance = min(self.min_clearance, clr)
            if clr <= 0.0:
                collided = True
                break

        self.vehicle_health.update(energy_wh, cmd_speed, self.u, self.depth, dvl_dropout=False)
        seabed_meas = self.depth + (SEABED_Z - self.depth) + self.rng.normal(0, 0.1)
        self.est_x, self.est_y, self.uncertainty = self.nav.update(measured_seabed_depth=seabed_meas)
        self.nav_err = (self.est_x - self.x, self.est_y - self.y)

        # ---- reward: same terms as orca_env.py (+ goal chain + collision) ----
        dist = self._goal_dist()
        progress = self.prev_dist - dist
        self.prev_dist = dist
        alt = self._perceive()
        env_ctx = self.env_health.sample(self.est_x, self.est_y, self.t)
        risk = env_ctx["risk"]
        health_penalty = 1.0 if (overridden or self.vehicle_health.is_critical()) else 0.0
        idle_penalty = self.w["idle"] if self.u < 0.2 else 0.0
        reward = (self.w["progress"] * progress
                  - self.w["uncertainty"] * self.uncertainty / 100.0
                  - self.w["risk"] * risk
                  - self.w["energy"] * (energy_wh / 100.0)
                  - self.w["health"] * health_penalty
                  - idle_penalty)

        self.t += self.dt
        self.step_count += 1
        done = False
        reached = False
        if dist < 10.0:
            reward += self.w["goal_bonus"]
            reached = True
            self.goal_idx += 1
            if self.goal_idx >= len(self.goals):
                done = True
            else:
                self.prev_dist = self._goal_dist()
        if collided:
            reward -= self.w["collision"]
            done = True
        if self.step_count >= self.max_steps:
            done = True

        g = self._goal()
        info = dict(true_x=self.x, true_y=self.y, est_x=self.est_x, est_y=self.est_y,
                    heading=self.heading, depth=self.depth, speed=self.u,
                    uncertainty=self.uncertainty, risk=risk, reached_goal=reached,
                    goals_reached=self.goal_idx, n_goals=len(self.goals), collided=collided,
                    overridden=bool(overridden), min_clearance=self.min_clearance,
                    goal_x=float(g[0]), goal_y=float(g[1]), t=self.t,
                    battery_frac=self.vehicle_health.battery_frac())
        self.log.append(info)
        return self._obs(), reward, done, info

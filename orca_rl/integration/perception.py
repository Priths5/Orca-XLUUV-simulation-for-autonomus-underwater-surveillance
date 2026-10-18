"""
perception.py
-------------
Obstacle and seabed features computed ONLY from the Orca's own sonars:

  * Profiler  (/orca/profiler, sensor_msgs/LaserScan): a single mechanically
    swept beam on the bow (x = +14.05 m), 160 deg fan, 30 m range. orca_rl_sim
    publishes the newest return per 5-deg bin; +inf = beam saw nothing out to
    30 m, NaN = that bin has not been swept yet.
  * Multibeam (/orca/multibeam, LaserScan): 120 deg across-track fan pointed
    down; used for seabed slope ("gradient richness").

Exactly the same functions are used by the training environment
(sim/obstacle_env.py, which simulates the same swept beam) and by the ROS 2
node, so the policy sees identically-computed features in both.
"""

import math

import numpy as np

PROFILER_FOV = math.radians(160.0)
PROFILER_RANGE = 30.0
PROFILER_BINS = 33                      # 5 deg bins, as published by orca_rl_sim
PROFILER_MOUNT_X = 14.05                # m ahead of the hull origin (orca_xluuv_survey.scn)
N_SECTORS = 5                           # [-80,-48] [-48,-16] [-16,16] [16,48] [48,80] deg

MULTIBEAM_FOV = math.radians(120.0)


def profiler_bin_angles(n_bins=PROFILER_BINS, fov=PROFILER_FOV):
    return np.linspace(-0.5 * fov, 0.5 * fov, n_bins)


def clean_ranges(ranges, max_range=PROFILER_RANGE):
    """inf (no return) and NaN (not swept yet) -> max_range; clip to range."""
    r = np.asarray(ranges, dtype=float).copy()
    r[~np.isfinite(r)] = max_range
    return np.clip(r, 0.0, max_range)


def sector_clearances(angles, ranges, n_sectors=N_SECTORS, fov=PROFILER_FOV, max_range=PROFILER_RANGE):
    """Minimum range (m, measured from the bow sensor) in each angular sector,
    port (-) to starboard (+)."""
    angles = np.asarray(angles, dtype=float)
    r = clean_ranges(ranges, max_range)
    edges = np.linspace(-0.5 * fov, 0.5 * fov, n_sectors + 1)
    out = np.full(n_sectors, max_range)
    idx = np.clip(np.searchsorted(edges, angles, side="right") - 1, 0, n_sectors - 1)
    for k in range(n_sectors):
        m = idx == k
        if np.any(m):
            out[k] = float(np.min(r[m]))
    return out


def nearest_return(angles, ranges, max_range=PROFILER_RANGE):
    r = clean_ranges(ranges, max_range)
    i = int(np.argmin(r))
    return float(r[i]), float(np.asarray(angles)[i])


def obstacle_risk(clearances, safety_distance=8.0, horizon=20.0):
    """0 = nothing inside `horizon` m, 1 = something within `safety_distance`
    of the bow. Side sectors count half as much as the forward three."""
    c = np.asarray(clearances, dtype=float)
    w = np.ones_like(c)
    if c.size >= 5:
        w[0] = w[-1] = 0.5
    closeness = np.clip((horizon - c) / (horizon - safety_distance), 0.0, 1.0)
    return float(np.max(w * closeness))


def altitude_risk(altitude, min_altitude=3.0, horizon=8.0):
    """Risk from being close to the seabed (DVL altitude). Unknown -> 0."""
    if altitude is None or not np.isfinite(altitude) or altitude <= 0:
        return 0.0
    return float(np.clip((horizon - altitude) / (horizon - min_altitude), 0.0, 1.0))


def seabed_gradient_richness(ranges, fov=MULTIBEAM_FOV, max_slope=0.6):
    """Across-track seabed roughness from one multibeam fan, in [0, 1].

    The original env's gradient_richness is the normalised bathymetric
    gradient magnitude (information for terrain-aided navigation). Here it
    is measured directly: convert each beam to (across-track offset, depth
    below sensor), then average two normalised terms - the 90th-percentile
    |slope| between neighbouring beams (/ max_slope, 0.6 ~ 31 deg) and the
    across-track spread of seabed depth (std / 3 m) - so a single boulder
    under the fan registers, a flat plane gives 0."""
    r = np.asarray(ranges, dtype=float)
    n = r.size
    if n < 3:
        return 0.0
    ang = np.linspace(-0.5 * fov, 0.5 * fov, n)
    ok = np.isfinite(r) & (r > 0.3)
    if ok.sum() < 3:
        return 0.0
    yb = r[ok] * np.sin(ang[ok])
    zb = r[ok] * np.cos(ang[ok])
    dy = np.diff(yb)
    dz = np.diff(zb)
    good = np.abs(dy) > 1e-3
    if not np.any(good):
        return 0.0
    slope = np.abs(dz[good] / dy[good])
    slope_term = np.clip(np.percentile(slope, 90) / max_slope, 0.0, 1.0)
    spread_term = np.clip(np.std(zb) / 3.0, 0.0, 1.0)
    return float(0.5 * slope_term + 0.5 * spread_term)


def seabed_depth_from_multibeam(vehicle_depth, ranges, fov=MULTIBEAM_FOV, sensor_offset_z=1.48):
    """Seabed depth under the vehicle from the near-nadir multibeam beams."""
    r = np.asarray(ranges, dtype=float)
    n = r.size
    if n == 0:
        return None
    ang = np.linspace(-0.5 * fov, 0.5 * fov, n)
    nadir = (np.abs(ang) < math.radians(10.0)) & np.isfinite(r) & (r > 0.3)
    if not np.any(nadir):
        return None
    return float(vehicle_depth + sensor_offset_z + np.median(r[nadir] * np.cos(ang[nadir])))


class ObstacleMemory:
    """
    Short-term map of Profiler returns in the world frame, so obstacles that
    have slid out of the +-80 deg forward fan (alongside / behind the bow)
    still count while the 27 m hull swings past them. Points are placed with
    the vehicle's ESTIMATED pose (navigation output, not ground truth).

    Forgetting works like free-space carving in a sonar occupancy map: a
    remembered point is deleted when a later beam looks through it (same
    bearing, longer range or no return) - so a fish that has swum away is
    cleared - or when the vehicle is more than `max_dist` away. A rock that is
    only out of view (beside or behind the hull) is therefore NOT forgotten,
    however long the vehicle takes to get past it. `horizon_s` is only a
    backstop.
    """

    def __init__(self, horizon_s=600.0, max_points=1500, max_dist=60.0,
                 beam_halfwidth=math.radians(3.0), carve_margin=1.0):
        self.horizon = horizon_s
        self.max_points = max_points
        self.max_dist = max_dist
        self.beam_halfwidth = beam_halfwidth
        self.carve_margin = carve_margin
        self.pts = np.zeros((0, 3))       # x, y, t

    def clear(self):
        self.pts = np.zeros((0, 3))

    def add_scan(self, t, est_x, est_y, heading, angles, ranges, max_range=PROFILER_RANGE):
        """angles/ranges: beams just measured (+inf = no return within range,
        NaN = no measurement - ignored)."""
        a = np.asarray(angles, dtype=float)
        r = np.asarray(ranges, dtype=float)
        meas = ~np.isnan(r)
        if not np.any(meas):
            return
        a, r = a[meas], np.where(np.isfinite(r[meas]), r[meas], max_range)
        c, s = math.cos(heading), math.sin(heading)
        sx = est_x + PROFILER_MOUNT_X * c
        sy = est_y + PROFILER_MOUNT_X * s
        if self.pts.shape[0]:
            dx, dy = self.pts[:, 0] - sx, self.pts[:, 1] - sy
            dist = np.hypot(dx, dy)
            bear = np.arctan2(dy, dx) - heading
            bear = (bear + math.pi) % (2 * math.pi) - math.pi
            seen_through = np.zeros(len(self.pts), bool)
            for ai, ri in zip(a, r):
                seen_through |= (np.abs(bear - ai) < self.beam_halfwidth) & (dist < ri - self.carve_margin)
            self.pts = self.pts[~seen_through]
        hit = (r < max_range - 1e-3) & (r > 0.2)
        if np.any(hit):
            px = sx + r[hit] * np.cos(heading + a[hit])
            py = sy + r[hit] * np.sin(heading + a[hit])
            new = np.stack([px, py, np.full(px.shape, t)], 1)
            self.pts = np.concatenate([self.pts, new])[-self.max_points:]

    def flank_clearances(self, t, est_x, est_y, heading, half_fwd=13.85, half_aft=13.3,
                         radius=1.3, cap=PROFILER_RANGE):
        """(port, starboard) distance from the hull surface to the nearest
        remembered return on each side, capped at `cap`."""
        if self.pts.shape[0]:
            keep = (self.pts[:, 2] >= t - self.horizon) & \
                   (np.hypot(self.pts[:, 0] - est_x, self.pts[:, 1] - est_y) < self.max_dist)
            self.pts = self.pts[keep]
        if self.pts.shape[0] == 0:
            return cap, cap
        c, s = math.cos(heading), math.sin(heading)
        dx = self.pts[:, 0] - est_x
        dy = self.pts[:, 1] - est_y
        bx = dx * c + dy * s            # body forward
        by = -dx * s + dy * c           # body starboard
        tx = np.clip(bx, -half_aft, half_fwd)
        d = np.hypot(bx - tx, by) - radius
        port = d[by < 0]
        stbd = d[by >= 0]
        return (float(np.clip(port.min(), 0, cap)) if port.size else cap,
                float(np.clip(stbd.min(), 0, cap)) if stbd.size else cap)

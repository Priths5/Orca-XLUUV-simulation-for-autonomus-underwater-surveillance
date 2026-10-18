"""
missions.py
-----------
What the Orca is asked to do in each Stonefish world: scout the region on
survey legs (a boustrophedon / lawnmower pattern, like the C++ autopilot's
BuildLawnmowerPattern), at a mission depth, starting from the world's known
launch point.

What is legitimately known before the dive and therefore allowed here:
launch point, chart seabed depth, NED origin of the scenario, the legs to
fly. Obstacle positions are NOT here - the Orca has to find them with its
Profiler.
"""

import math

# NED origin of every scenario (<ned latitude="36.6" longitude="-121.9"/>)
NED_ORIGIN_LAT = 36.6
NED_ORIGIN_LON = -121.9


def lawnmower(x0, y0, x1, y1, spacing):
    wps, fwd, y = [], True, y0
    while y <= y1 + 1e-6:
        wps += [(x0, y), (x1, y)] if fwd else [(x1, y), (x0, y)]
        fwd = not fwd
        y += spacing
    return wps


WORLDS = {
    # orca_rl_world.scn: open ocean, pinnacles + ridges + fish + TrafficSub
    "rl": dict(start=(-95.0, -60.0), seabed_depth=30.0, mission_depth=12.0,
               waypoints=lawnmower(-80.0, -60.0, 80.0, 60.0, 30.0)),
    # orca_survey_world.scn: rocks, pillars, wrecks, boulders, fish schools
    "survey": dict(start=(-45.0, -35.0), seabed_depth=18.0, mission_depth=12.0,
                   waypoints=lawnmower(-45.0, -35.0, 45.0, 35.0, 70.0 / 3.0)),
    # orca_multisub_world.scn: two other survey subs crossing the lanes
    "multisub": dict(start=(-50.0, -10.0), seabed_depth=18.0, mission_depth=12.0,
                     waypoints=lawnmower(-50.0, -15.0, 50.0, 25.0, 20.0)),
}


def geodetic_to_ned(lat, lon, lat0=NED_ORIGIN_LAT, lon0=NED_ORIGIN_LON):
    """Small-area (equirectangular) conversion, metres. Good to < 1 m over
    the few hundred metres of these scenarios."""
    R = 6378137.0
    n = math.radians(lat - lat0) * R
    e = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    return n, e


def subdivide(waypoints, start, max_leg):
    """Insert evenly spaced intermediate waypoints so no leg is longer than
    max_leg (the obstacle policy was trained on 20-120 m legs; the rl world's
    survey legs are 160 m)."""
    out, prev = [], start
    for wp in waypoints:
        L = math.hypot(wp[0] - prev[0], wp[1] - prev[1])
        n = max(1, int(math.ceil(L / max_leg)))
        for k in range(1, n + 1):
            out.append((prev[0] + (wp[0] - prev[0]) * k / n, prev[1] + (wp[1] - prev[1]) * k / n))
        prev = wp
    return out


class Mission:
    def __init__(self, world, reach_radius=10.0, loop=False, max_leg=80.0):
        spec = WORLDS[world]
        self.world = world
        self.start = spec["start"]
        self.seabed_depth = spec["seabed_depth"]
        self.mission_depth = spec["mission_depth"]
        self.survey_waypoints = list(spec["waypoints"])
        self.waypoints = subdivide(self.survey_waypoints, self.start, max_leg) if max_leg else list(self.survey_waypoints)
        self.reach_radius = reach_radius
        self.loop = loop
        self.idx = 0
        self.wp_start_t = None
        self.reached = []          # (index, t)
        self.skipped = []          # (index, t, reason)

    @property
    def done(self):
        return self.idx >= len(self.waypoints)

    def goal(self):
        return self.waypoints[min(self.idx, len(self.waypoints) - 1)]

    def leg_timeout(self):
        prev = self.start if self.idx == 0 else self.waypoints[self.idx - 1]
        g = self.goal()
        return max(150.0, 3.0 * math.hypot(g[0] - prev[0], g[1] - prev[1]) / 0.5)

    def _advance(self, t):
        self.idx += 1
        self.wp_start_t = t
        if self.done and self.loop:
            self.idx = 0

    def update(self, est_x, est_y, t=0.0, goal_blocked=False):
        """Advance when the ESTIMATED position is within reach of the goal.
        Like a real AUV mission executive, a waypoint is skipped when the
        sonar shows it is occupied (goal_blocked while < 25 m away) or when
        its leg times out. Returns "reached", "blocked", "timeout" or None."""
        if self.done:
            return None
        if self.wp_start_t is None:
            self.wp_start_t = t
        gx, gy = self.goal()
        d = math.hypot(gx - est_x, gy - est_y)
        if d < self.reach_radius:
            self.reached.append((self.idx, t))
            self._advance(t)
            return "reached"
        if goal_blocked and d < 25.0:
            self.skipped.append((self.idx, t, "blocked"))
            self._advance(t)
            return "blocked"
        if t - self.wp_start_t > self.leg_timeout():
            self.skipped.append((self.idx, t, "timeout"))
            self._advance(t)
            return "timeout"
        return None

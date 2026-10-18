#!/usr/bin/env python3
"""
Generates resources/orca_rl_world.scn - the open-ocean "base" world used by
orca_rl_sim --world rl. Same environment/material conventions as
orca_world.scn, plus things the Orca has to detect and avoid while it
scouts the area:

  * rock pinnacles and two ridge walls rising from the 30 m seabed to
    within a few metres of the surface (they cross the cruise depth),
  * two large boulders between survey lanes,
  * three fish schools (animated, collidable) looping across the lanes,
  * TrafficSub - an independent survey submarine running its own
    lawnmower (driven by OrcaController in main_rl.cpp) across the lanes.

Run:  python3 tools/make_rl_world.py   (writes ../resources/orca_rl_world.scn)
The obstacle list is also written to resources/orca_rl_world_obstacles.json
so the Python side can draw/evaluate (never used as a sensor).
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "resources")

SEABED_Z = 30.0
PINNACLE_TOP_Z = 4.0

PINNACLES = [  # x, y, radius
    (-40.0, -60.0, 3.0), (22.0, -57.0, 2.5), (-10.0, -31.0, 3.5), (52.0, -28.0, 2.2),
    (2.0, 3.0, 3.0), (-55.0, -1.0, 2.5), (30.0, 31.0, 3.0), (-25.0, 33.0, 2.2),
    (62.0, 60.0, 3.0), (-50.0, 58.0, 2.5), (75.0, -2.0, 2.0), (-78.0, 28.0, 2.0),
]
WALLS = [  # x, y, length (along yaw), thickness, yaw
    (38.0, -2.0, 18.0, 3.0, 0.0),
    (-12.0, 62.0, 16.0, 3.0, 0.6),
]
BOULDERS = [  # x, y, mesh, scale, yaw
    (68.0, -45.0, "rock_a", 9.0, 0.7),
    (-68.0, 44.0, "rock_c", 7.0, 2.0),
]
LOOKS = ["rock_sand", "rock_olive", "rock_slate"]


def fish_school(name, cx, cy, rx, ry, z0, period, phase, n=3, scale=3.0):
    """n fish following an ellipse (cx,cy,rx,ry) at ~z0, slightly offset."""
    out = []
    looks = ["fish_silver", "fish_dark", "fish_amber"]
    for k in range(n):
        dx, dy, dz = 2.5 * k, -1.5 * k, 0.8 * (k - 1)
        kps = []
        steps = 12
        for i in range(steps + 1):
            a = phase + 2 * math.pi * i / steps
            x = cx + rx * math.cos(a) + dx
            y = cy + ry * math.sin(a) + dy
            z = z0 + dz + 1.2 * math.sin(2 * a)
            # heading = direction of motion along the ellipse
            yaw = math.atan2(ry * math.cos(a), -rx * math.sin(a))
            kps.append(f'            <keypoint time="{period * i / steps:.2f}" xyz="{x:.2f} {y:.2f} {z:.2f}" rpy="0.0 0.0 {yaw:.3f}"/>')
        out.append(f'''    <animated name="{name}{k + 1}" type="model" collisions="true">
        <physical>
            <mesh filename="data/fish.obj" scale="{scale - 0.5 * k:.2f}"/>
            <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
        </physical>
        <material name="Fish"/>
        <look name="{looks[k % 3]}"/>
        <trajectory type="spline" playback="repeat">
{chr(10).join(kps)}
        </trajectory>
    </animated>''')
    return "\n".join(out), dict(name=name, cx=cx, cy=cy, rx=rx, ry=ry, z=z0, period=period)


def main():
    parts = []
    obstacles = {"seabed_z": SEABED_Z, "pinnacles": [], "walls": [], "boulders": [], "fish": []}
    h = SEABED_Z - PINNACLE_TOP_Z
    for i, (x, y, r) in enumerate(PINNACLES):
        parts.append(f'''    <static name="Pinnacle{i + 1:02d}" type="cylinder">
        <material name="Rock"/>
        <look name="{LOOKS[i % 3]}"/>
        <dimensions radius="{r:.2f}" height="{h:.2f}"/>
        <world_transform xyz="{x:.2f} {y:.2f} {SEABED_Z - h / 2:.2f}" rpy="0.0 0.0 {0.3 * i:.2f}"/>
    </static>''')
        obstacles["pinnacles"].append(dict(x=x, y=y, r=r))
    for i, (x, y, length, thick, yaw) in enumerate(WALLS):
        parts.append(f'''    <static name="Ridge{i + 1:02d}" type="box">
        <material name="Rock"/>
        <look name="rock_slate"/>
        <dimensions xyz="{thick:.2f} {length:.2f} {h:.2f}"/>
        <world_transform xyz="{x:.2f} {y:.2f} {SEABED_Z - h / 2:.2f}" rpy="0.0 0.0 {yaw:.2f}"/>
    </static>''')
        obstacles["walls"].append(dict(x=x, y=y, length=length, thickness=thick, yaw=yaw))
    for i, (x, y, mesh, sc, yaw) in enumerate(BOULDERS):
        parts.append(f'''    <static name="Boulder{i + 1:02d}" type="model">
        <material name="Rock"/>
        <look name="{LOOKS[(i + 1) % 3]}"/>
        <physical>
            <mesh filename="data/{mesh}.obj" scale="{sc:.2f}" convex="true"/>
            <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
        </physical>
        <world_transform xyz="{x:.2f} {y:.2f} {SEABED_Z - 0.15:.2f}" rpy="0.05 -0.05 {yaw:.2f}"/>
    </static>''')
        obstacles["boulders"].append(dict(x=x, y=y, approx_r=1.2 * sc))
    for args in [("FishA", -30.0, -45.0, 22.0, 14.0, 12.0, 60.0, 0.0),
                 ("FishB", 30.0, 15.0, 20.0, 18.0, 11.5, 70.0, 1.5),
                 ("FishC", -40.0, 15.0, 16.0, 22.0, 12.5, 65.0, 3.0)]:
        xml, meta = fish_school(*args)
        parts.append(xml)
        obstacles["fish"].append(meta)

    body = "\n\n".join(parts)
    scn = f'''<?xml version="1.0"?>
<!--
    orca_rl_world.scn  (generated by tools/make_rl_world.py: edit that, not this)

    Base open-ocean world for the RL integration (orca_rl_sim, world "rl").
    Same environment and single-<materials>-block rule as orca_world.scn,
    but the ocean is no longer empty: pinnacles and ridges rise from a
    {SEABED_Z:.0f} m seabed through the 12 m cruise depth, fish schools loop
    across the survey lanes, and TrafficSub runs its own survey through
    the area. The Orca only learns about any of this from its own
    Profiler / Multibeam / DVL - nothing is handed to the policy directly.
-->
<scenario>
    <environment>
        <ned latitude="36.6" longitude="-121.9"/>
        <ocean>
            <water density="1025.0" jerlov="0.12"/>
            <waves height="0.0"/>
            <current type="uniform">
                <velocity xyz="0.0 0.0 0.0"/>
            </current>
        </ocean>
        <atmosphere>
            <sun azimuth="20.0" elevation="55.0"/>
        </atmosphere>
    </environment>

    <light name="FillWest">
        <specs illuminance="90000.0" radius="160.0"/>
        <color rgb="0.72 0.86 1.0"/>
        <world_transform xyz="-45.0 0.0 3.0" rpy="0.0 0.0 0.0"/>
    </light>
    <light name="FillEast">
        <specs illuminance="90000.0" radius="160.0"/>
        <color rgb="0.72 0.86 1.0"/>
        <world_transform xyz="45.0 0.0 3.0" rpy="0.0 0.0 0.0"/>
    </light>

    <materials>
        <material name="Rock" density="2700.0" restitution="0.6"/>
        <material name="Sand" density="1800.0" restitution="0.3"/>
        <material name="HullShell" density="4100.0" restitution="0.3"/>
        <material name="SubHull" density="960.0" restitution="0.3"/>
        <material name="Composite" density="1800.0" restitution="0.4"/>
        <material name="SyntacticFoam" density="500.0" restitution="0.3"/>
        <material name="Fish" density="1025.0" restitution="0.2"/>
        <friction_table>
            <friction material1="Rock" material2="Rock" static="0.7" dynamic="0.5"/>
            <friction material1="Sand" material2="Sand" static="0.5" dynamic="0.3"/>
            <friction material1="Rock" material2="Sand" static="0.6" dynamic="0.4"/>
        </friction_table>
    </materials>

    <looks>
        <look name="seabed_look" rgb="0.62 0.58 0.36" roughness="0.92" metalness="0.0"/>
        <look name="rock_sand" rgb="0.78 0.62 0.38" roughness="0.85" metalness="0.02"/>
        <look name="rock_olive" rgb="0.42 0.58 0.36" roughness="0.8" metalness="0.02"/>
        <look name="rock_slate" rgb="0.55 0.62 0.68" roughness="0.7" metalness="0.05"/>
        <look name="fish_silver" rgb="0.92 0.94 0.78" roughness="0.35" metalness="0.05"/>
        <look name="fish_dark" rgb="0.15 0.72 0.62" roughness="0.4" metalness="0.04"/>
        <look name="fish_amber" rgb="0.95 0.62 0.18" roughness="0.4" metalness="0.04"/>
    </looks>

    <static name="Seabed" type="plane">
        <material name="Sand"/>
        <look name="seabed_look"/>
        <world_transform xyz="0.0 0.0 {SEABED_Z:.2f}" rpy="0.0 0.0 0.0"/>
    </static>

{body}

    <include file="data/orca_xluuv_survey.scn">
        <arg name="name" value="OrcaXLUUV"/>
        <arg name="start_x" value="-95.0"/>
        <arg name="start_y" value="-60.0"/>
        <arg name="start_z" value="12.0"/>
    </include>

    <include file="data/survey_sub.scn">
        <arg name="name" value="TrafficSub"/>
        <arg name="start_x" value="-40.0"/>
        <arg name="start_y" value="-50.0"/>
        <arg name="start_z" value="11.0"/>
        <arg name="start_yaw" value="1.5708"/>
    </include>
</scenario>
'''
    with open(os.path.join(RES, "orca_rl_world.scn"), "w") as f:
        f.write(scn)
    with open(os.path.join(RES, "orca_rl_world_obstacles.json"), "w") as f:
        json.dump(obstacles, f, indent=1)
    print("wrote orca_rl_world.scn and orca_rl_world_obstacles.json")


if __name__ == "__main__":
    main()
